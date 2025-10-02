# phase_analyzer.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import curve_fit
import platonic  # The existing PRH module

class RealAGOPAnalyzer:
    """
    Computes real AGOP by collecting gradients during model forward passes
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def compute_real_agop(self, model, dataloader, loss_fn, batch_size=32):
        """
        Compute real AGOP by accumulating gradient outer products
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with (inputs, labels)
            loss_fn: Loss function
            batch_size: Batch size for efficiency
        
        Returns:
            Dict with AGOP analysis
        """
        model.eval()  # Important: eval mode for consistent gradients
        model.to(self.device)
        
        # Initialize AGOP accumulator
        agop_accumulator = None
        total_samples = 0
        
        print("Computing real AGOP from gradients...")
        
        with tqdm(dataloader, desc="Processing batches") as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                if batch_idx * batch_size >= 1000:  # Limit to 1000 samples for efficiency
                    break
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                model.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                # Backward pass to compute gradients
                loss.backward()
                
                # Collect and flatten gradients
                gradients = []
                for param in model.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.detach().cpu().numpy().flatten())
                
                if gradients:
                    # Concatenate all gradients
                    grad_vector = np.concatenate(gradients)
                    
                    # Compute outer product for this batch
                    # For memory efficiency, we can subsample gradient dimensions
                    if len(grad_vector) > 10000:
                        # Random subsample for large models
                        indices = np.random.choice(len(grad_vector), 10000, replace=False)
                        grad_vector = grad_vector[indices]
                    
                    batch_agop = np.outer(grad_vector, grad_vector)
                    
                    # Accumulate
                    if agop_accumulator is None:
                        agop_accumulator = batch_agop
                    else:
                        agop_accumulator += batch_agop
                    
                    total_samples += len(inputs)
                    
                    # Update progress
                    pbar.set_postfix({'samples': total_samples, 
                                     'grad_dim': len(grad_vector)})
                
                # Clear gradients to save memory
                model.zero_grad()
                torch.cuda.empty_cache()
        
        # Normalize by number of samples
        agop = agop_accumulator / total_samples
        
        # Analyze AGOP
        return self._analyze_agop(agop)
    
    def _analyze_agop(self, agop):
        """Analyze AGOP matrix"""
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(agop)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Key metrics
        results = {
            'eigenvalues': eigenvalues[:100],  # Top 100
            'agop_ratio': eigenvalues[0] / (eigenvalues[-1] + 1e-10),
            'effective_rank': np.sum(eigenvalues)**2 / np.sum(eigenvalues**2),
            'top10_concentration': np.sum(eigenvalues[:10]) / np.sum(eigenvalues),
            'spectral_decay': self._fit_spectral_decay(eigenvalues),
            'phase': self._determine_phase(eigenvalues)
        }
        
        return results
    
    def _fit_spectral_decay(self, eigenvalues):
        """Fit power law to eigenvalues"""
        def power_law(x, a, b):
            return a * x**(-b)
        
        k = np.arange(1, min(len(eigenvalues), 50) + 1)
        try:
            popt, _ = curve_fit(power_law, k, eigenvalues[:len(k)])
            return {'type': 'power_law', 'exponent': popt[1]}
        except:
            return {'type': 'unknown', 'exponent': None}
    
    def _determine_phase(self, eigenvalues):
        """Determine phase from eigenvalue spectrum"""
        top10_conc = np.sum(eigenvalues[:10]) / np.sum(eigenvalues)
        
        if top10_conc > 0.9:
            return "lazy"
        elif top10_conc > 0.7:
            return "critical" 
        else:
            return "chaotic"

def create_data_loader(embeddings, labels=None, batch_size=32):
    """
    Create a DataLoader from numpy embeddings
    """
    import torch.utils.data as data
    
    # Create synthetic labels if not provided
    if labels is None:
        labels = np.random.randint(0, 10, size=len(embeddings))
    
    # Convert to tensors
    embeddings_tensor = torch.FloatTensor(embeddings)
    labels_tensor = torch.LongTensor(labels)
    
    # Create dataset
    dataset = data.TensorDataset(embeddings_tensor, labels_tensor)
    
    # Create dataloader
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

class PhaseAnalyzer:
    """
    Analyzes phase transitions in pre-trained models by computing AGOP, NTK, and IB metrics
    from features extracted by the PRH framework.
    """

    def __init__(self, dataset="minhuh/prh", subset="wit_1024"):
        self.dataset = dataset
        self.subset = subset
        self.results_dir = Path("./results/phase_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def compute_agop_from_features(self, features: np.ndarray, n_samples: int = 1000) -> Dict:
        """
        Compute AGOP metrics from extracted features using real gradient computation.
        """
        # Features shape: (n_samples, n_features)  
        n_features = features.shape[-1]

        # Create a probe model for real gradient computation
        probe_model = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 classes
        )
        
        # Create dataloader
        dataloader = create_data_loader(features, batch_size=32)
        
        # Define loss function
        loss_fn = nn.CrossEntropyLoss()
        
        # Initialize AGOP analyzer
        agop_analyzer = RealAGOPAnalyzer()
        
        # Compute real AGOP
        agop_results = agop_analyzer.compute_real_agop(probe_model, dataloader, loss_fn)
        
        # Return results in expected format
        results = {
            'eigenvalues': agop_results['eigenvalues'],
            'top_eigenvalue_ratio': agop_results['agop_ratio'],
            'effective_rank': agop_results['effective_rank'],
            'eigenvalue_concentration': agop_results['top10_concentration'],
            'phase': agop_results['phase']
        }

        return results

    def _determine_phase_from_eigenvalues(self, eigenvalues: np.ndarray) -> str:
        """
        Determine phase based on eigenvalue distribution.
        This implements the phase criteria from your paper.
        """
        # Normalized eigenvalues
        eigenvalues = eigenvalues / np.sum(eigenvalues)

        # Phase determination based on concentration
        top_10_concentration = np.sum(eigenvalues[:10])

        if top_10_concentration > 0.9:
            return "stable"
        elif top_10_concentration > 0.7:
            return "critical"
        else:
            return "chaotic"

    def compute_ntk_stability_from_features(self, features: np.ndarray,
                                           layer_name: str = None) -> Dict:
        """
        Estimate NTK stability from features by measuring local changes.
        """
        # Compute feature similarity matrix (proxy for NTK)
        n_samples = min(1000, len(features))
        features_subset = features[:n_samples]

        # Normalize features
        features_norm = features_subset / (np.linalg.norm(features_subset, axis=1, keepdims=True) + 1e-8)

        # Compute kernel
        kernel = features_norm @ features_norm.T

        # Estimate stability by local perturbation analysis
        stability_scores = []
        for i in range(100):  # Sample some points
            # Add small noise to features
            noise_scale = 0.01
            perturbed = features_norm + np.random.randn(*features_norm.shape) * noise_scale
            perturbed = perturbed / (np.linalg.norm(perturbed, axis=1, keepdims=True) + 1e-8)

            # Compute perturbed kernel
            perturbed_kernel = perturbed @ perturbed.T

            # Measure change
            kernel_change = np.linalg.norm(kernel - perturbed_kernel) / np.linalg.norm(kernel)
            stability = 1 - kernel_change
            stability_scores.append(stability)

        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'kernel_eigenvalues': np.linalg.eigvalsh(kernel)[::-1][:20],  # Top 20
            'layer': layer_name
        }

    def analyze_model_phases(self, model_name: str, modality: str) -> Dict:
        """
        Complete phase analysis for a single model using PRH infrastructure.
        """
        print(f"Analyzing {model_name} ({modality})...")

        # Use PRH to load features
        if modality == "vision":
            pool_type = "cls"
        else:  # language
            pool_type = "avg"

        # Load pre-extracted features using PRH file structure
        feature_path = f"./results/features/{self.dataset}/{self.subset}/{model_name}_pool-{pool_type}.npy"

        try:
            features = np.load(feature_path, allow_pickle=True)
            if isinstance(features, dict):
                # Handle multi-layer features
                features = features['features']  # Get the actual feature array
        except FileNotFoundError:
            print(f"Features not found for {model_name}. Run extract_features.py first.")
            return None

        # Compute phase metrics
        agop_results = self.compute_agop_from_features(features)
        ntk_results = self.compute_ntk_stability_from_features(features, model_name)

        # Combine results
        return {
            'model': model_name,
            'modality': modality,
            'agop': agop_results,
            'ntk': ntk_results,
            'phase': agop_results['phase'],
            'phase_score': self._compute_phase_score(agop_results, ntk_results)
        }

    def _compute_phase_score(self, agop_results: Dict, ntk_results: Dict) -> float:
        """
        Combine AGOP and NTK metrics into a single phase score.
        Higher scores indicate more stable/organized phases.
        """
        agop_score = agop_results['eigenvalue_concentration']
        ntk_score = ntk_results['mean_stability']

        # Weighted combination
        return 0.6 * agop_score + 0.4 * ntk_score