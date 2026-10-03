import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
from functorch import make_functional, vmap, jacrev

class RealAGOPAnalyzer:
    """
    Computes real AGOP by collecting gradients during model forward passes
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def compute_real_agop(self, model, dataloader, loss_fn, batch_size=32):
        """
        Compute real AGOP by accumulating gradient outer products
        """
        model.eval()
        model.to(self.device)
        
        agop_accumulator = None
        total_samples = 0
        
        print("Computing real AGOP from gradients...")
        
        with tqdm(dataloader, desc="Processing batches") as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                if batch_idx * batch_size >= 1000:
                    break
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                model.zero_grad()
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                loss.backward()
                
                gradients = []
                for param in model.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.detach().cpu().numpy().flatten())
                
                if gradients:
                    grad_vector = np.concatenate(gradients)
                    
                    if len(grad_vector) > 10000:
                        indices = np.random.choice(len(grad_vector), 10000, replace=False)
                        grad_vector = grad_vector[indices]
                    
                    batch_agop = np.outer(grad_vector, grad_vector)
                    
                    if agop_accumulator is None:
                        agop_accumulator = batch_agop
                    else:
                        agop_accumulator += batch_agop
                    
                    total_samples += len(inputs)
                    
                    pbar.set_postfix({'samples': total_samples, 'grad_dim': len(grad_vector)})
                
                model.zero_grad()
                torch.cuda.empty_cache()
        
        if agop_accumulator is None:
            return self._analyze_agop(np.zeros((1,1)))

        agop = agop_accumulator / total_samples
        
        return self._analyze_agop(agop)
    
    def _analyze_agop(self, agop):
        """Analyze AGOP matrix"""
        eigenvalues = np.linalg.eigvalsh(agop)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) == 0:
            return {
                'eigenvalues': [], 'agop_ratio': 0, 'effective_rank': 0,
                'top10_concentration': 0, 'spectral_decay': {'type': 'unknown', 'exponent': None},
                'phase': 'unknown'
            }

        results = {
            'eigenvalues': eigenvalues[:100],
            'agop_ratio': eigenvalues[0] / (eigenvalues[-1] + 1e-10) if len(eigenvalues) > 1 else 1.0,
            'effective_rank': np.sum(eigenvalues)**2 / np.sum(eigenvalues**2),
            'top10_concentration': np.sum(eigenvalues[:10]) / np.sum(eigenvalues),
            'spectral_decay': self._fit_spectral_decay(eigenvalues),
            'phase': self._determine_phase(eigenvalues)
        }
        
        return results
    
    def _fit_spectral_decay(self, eigenvalues):
        """Fit power law to eigenvalues"""
        from scipy.optimize import curve_fit
        
        def power_law(x, a, b):
            return a * x**(-b)
        
        k = np.arange(1, min(len(eigenvalues), 50) + 1)
        if len(k) < 2:
            return {'type': 'unknown', 'exponent': None}
        try:
            popt, _ = curve_fit(power_law, k, eigenvalues[:len(k)])
            return {'type': 'power_law', 'exponent': popt[1]}
        except:
            return {'type': 'unknown', 'exponent': None}
    
    def _determine_phase(self, eigenvalues):
        """Determine phase from eigenvalue spectrum"""
        if len(eigenvalues) == 0:
            return 'unknown'
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
    
    if labels is None:
        labels = np.random.randint(0, 10, size=len(embeddings))
    
    embeddings_tensor = torch.FloatTensor(embeddings)
    labels_tensor = torch.LongTensor(labels)
    
    dataset = data.TensorDataset(embeddings_tensor, labels_tensor)
    
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

def compute_real_ntk_stability(fnet_single, params, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Computes NTK stability using functorch.
    """
    print("Computing real NTK stability...")
    data = data.to(device)

    def get_ntk_kernel(d):
        jac = vmap(jacrev(fnet_single), (None, 0))(params, d)
        jac_flat = [j.reshape(j.shape[0], j.shape[1], -1) for j in jac]
        J = torch.cat(jac_flat, dim=2)
        ntk = torch.einsum('Naf,Maf->NM', J, J)
        return ntk

    ntk = get_ntk_kernel(data)

    # --- Perturbation Analysis for Stability ---
    noise_scale = 0.01
    perturbed_data = data + torch.randn_like(data) * noise_scale
    
    ntk_perturbed = get_ntk_kernel(perturbed_data)
    
    # Measure change
    kernel_change = torch.linalg.norm(ntk - ntk_perturbed) / torch.linalg.norm(ntk)
    stability = 1 - kernel_change.item()

    # Eigenvalue analysis of the original NTK
    eigenvalues = torch.linalg.eigvalsh(ntk).detach().cpu().numpy()[::-1]

    return {
        'mean_stability': stability,
        'kernel_eigenvalues': eigenvalues[:20]
    }

def analyze_pretrained_models_with_real_agop_and_ntk(vision_embeddings, language_embeddings):
    """
    Analyze pretrained models using real AGOP and NTK computation.
    """
    analyzer = RealAGOPAnalyzer()
    results = {}
    loss_fn = nn.CrossEntropyLoss()

    # For vision embeddings
    print("\n=== Vision Model Analysis ===")
    for layer_name, embeddings in vision_embeddings.items():
        print(f"\nAnalyzing {layer_name}...")
        
        input_dim = embeddings.shape[1]
        probe_model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(analyzer.device)
        
        dataloader = create_data_loader(embeddings)
        
        agop_results = analyzer.compute_real_agop(probe_model, dataloader, loss_fn)
        
        fnet, params = make_functional(probe_model)
        def fnet_single(p, x): return fnet(p, x.unsqueeze(0)).squeeze(0)
        ntk_results = compute_real_ntk_stability(fnet_single, params, next(iter(dataloader))[0])

        results[f"vision_{layer_name}"] = {"agop": agop_results, "ntk": ntk_results}
        
        print(f"Phase (AGOP): {agop_results['phase']}")
        print(f"AGOP ratio: {agop_results['agop_ratio']:.2e}")
        print(f"Effective rank: {agop_results['effective_rank']:.2f}")
        print(f"Real NTK Mean Stability: {ntk_results['mean_stability']:.4f}")

    # For language embeddings
    print("\n=== Language Model Analysis ===")
    for layer_name, embeddings in language_embeddings.items():
        print(f"\nAnalyzing {layer_name}...")
        
        input_dim = embeddings.shape[1]
        probe_model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(analyzer.device)
        
        dataloader = create_data_loader(embeddings)
        
        agop_results = analyzer.compute_real_agop(probe_model, dataloader, loss_fn)

        fnet, params = make_functional(probe_model)
        def fnet_single(p, x): return fnet(p, x.unsqueeze(0)).squeeze(0)
        ntk_results = compute_real_ntk_stability(fnet_single, params, next(iter(dataloader))[0])

        results[f"language_{layer_name}"] = {"agop": agop_results, "ntk": ntk_results}
        
        print(f"Phase (AGOP): {agop_results['phase']}")
        print(f"AGOP ratio: {agop_results['agop_ratio']:.2e}")
        print(f"Effective rank: {agop_results['effective_rank']:.2f}")
        print(f"Real NTK Mean Stability: {ntk_results['mean_stability']:.4f}")
    
    return results

if __name__ == "__main__":
    base_path = "/Users/tanmoy/research/Perceptual_Features/platonic-rep/complete_features"
    
    vision_embeddings = {}
    language_embeddings = {}
    
    vision_model = "efficientnet_b0"
    vision_path = Path(base_path) / vision_model
    print(f"Loading embeddings from {vision_path}...")
    for npy_file in vision_path.glob("*.npy"):
        try:
            layer_name = npy_file.stem
            vision_embeddings[layer_name] = np.load(npy_file)
            print(f"  - Loaded {layer_name}")
        except Exception as e:
            print(f"  - Could not load {npy_file}: {e}")

    language_model = "albert"
    language_path = Path(base_path) / language_model
    print(f"Loading embeddings from {language_path}...")
    for npy_file in language_path.glob("*.npy"):
        try:
            layer_name = npy_file.stem
            language_embeddings[layer_name] = np.load(npy_file)
            print(f"  - Loaded {layer_name}")
        except Exception as e:
            print(f"  - Could not load {npy_file}: {e}")

    if vision_embeddings and language_embeddings:
        results = analyze_pretrained_models_with_real_agop_and_ntk(vision_embeddings, language_embeddings)
    
        print("\n=== Cross-Modal Phase Comparison ===")
        for v_layer in vision_embeddings.keys():
            for l_layer in language_embeddings.keys():
                v_phase = results[f"vision_{v_layer}"]['agop']['phase']
                l_phase = results[f"language_{l_layer}"]['agop']['phase']
                v_ntk = results[f"vision_{v_layer}"]['ntk']['mean_stability']
                l_ntk = results[f"language_{l_layer}"]['ntk']['mean_stability']
                print(f"{v_layer} (AGOP: {v_phase}, NTK: {v_ntk:.3f}) vs {l_layer} (AGOP: {l_phase}, NTK: {l_ntk:.3f})")
    else:
        print("Could not load embeddings. Skipping analysis.")
