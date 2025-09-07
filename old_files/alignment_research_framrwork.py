"""
Universal Geometry and Alignment Research Framework
===================================================

A comprehensive experimental framework for investigating emergent representation 
alignment through NTK-AGOP-Information theory lens.

Author: [Your Name]
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb
import json
import os
from datetime import datetime
from tqdm import tqdm
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Configuration and Loaders
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for datasets"""
    name: str
    num_classes: int
    input_dim: int
    transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    
DATASET_CONFIGS = {
    # Vision datasets
    'mnist': DatasetConfig('mnist', 10, 784),
    'fashion_mnist': DatasetConfig('fashion_mnist', 10, 784),
    'cifar10': DatasetConfig('cifar10', 10, 3072),
    'cifar100': DatasetConfig('cifar100', 100, 3072),
    'svhn': DatasetConfig('svhn', 10, 3072),
    'emnist': DatasetConfig('emnist', 47, 784),
    'kmnist': DatasetConfig('kmnist', 10, 784),
    
    # Medical/Scientific datasets
    'medmnist_path': DatasetConfig('medmnist_path', 9, 784),  # PathMNIST
    'medmnist_chest': DatasetConfig('medmnist_chest', 14, 784),  # ChestMNIST
    'medmnist_derma': DatasetConfig('medmnist_derma', 7, 784),  # DermaMNIST
    
    # Synthetic datasets for controlled experiments
    'synthetic_gaussian': DatasetConfig('synthetic_gaussian', 5, 100),
    'synthetic_manifold': DatasetConfig('synthetic_manifold', 3, 50),
    'synthetic_hierarchical': DatasetConfig('synthetic_hierarchical', 8, 128),
}

class DatasetLoader:
    """Unified dataset loader with support for multiple datasets"""
    
    @staticmethod
    def get_dataset(dataset_name: str, 
                    n_samples: Optional[int] = None,
                    split: str = 'train') -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Load dataset with optional subsampling"""
        
        if dataset_name.startswith('synthetic_'):
            return DatasetLoader._get_synthetic_dataset(dataset_name, n_samples)
        
        # Standard vision datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        if dataset_name == 'mnist':
            from torchvision.datasets import MNIST
            train_data = MNIST('./data', train=True, download=True, transform=transform)
            test_data = MNIST('./data', train=False, download=True, transform=transform)
            
        elif dataset_name == 'fashion_mnist':
            from torchvision.datasets import FashionMNIST
            train_data = FashionMNIST('./data', train=True, download=True, transform=transform)
            test_data = FashionMNIST('./data', train=False, download=True, transform=transform)
            
        elif dataset_name == 'cifar10':
            from torchvision.datasets import CIFAR10
            train_data = CIFAR10('./data', train=True, download=True, transform=transform)
            test_data = CIFAR10('./data', train=False, download=True, transform=transform)
            
        elif dataset_name == 'cifar100':
            from torchvision.datasets import CIFAR100
            train_data = CIFAR100('./data', train=True, download=True, transform=transform)
            test_data = CIFAR100('./data', train=False, download=True, transform=transform)
            
        elif dataset_name.startswith('medmnist_'):
            # MedMNIST datasets - install with: pip install medmnist
            import medmnist
            from medmnist import INFO
            
            dataset_type = dataset_name.split('_')[1]
            data_class = getattr(medmnist, f'{dataset_type.capitalize()}MNIST')
            
            train_data = data_class(split='train', transform=transform, download=True)
            test_data = data_class(split='test', transform=transform, download=True)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Subsample if requested
        if n_samples is not None and n_samples < len(train_data):
            indices = torch.randperm(len(train_data))[:n_samples]
            train_data = torch.utils.data.Subset(train_data, indices)
        
        # Create loaders with smaller batch sizes for memory efficiency
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=min(64, len(train_data)), shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=min(64, len(test_data)), shuffle=False, num_workers=2
        )
        
        return train_loader, test_loader
    
    @staticmethod
    def _get_synthetic_dataset(dataset_name: str, n_samples: int = 10000):
        """Generate synthetic datasets for controlled experiments"""
        
        if dataset_name == 'synthetic_gaussian':
            # Gaussian clusters
            n_classes = 5
            samples_per_class = n_samples // n_classes
            dim = 100
            
            X, y = [], []
            for i in range(n_classes):
                # Each class has a different mean
                mean = np.random.randn(dim) * 5
                cov = np.eye(dim) * (0.5 + i * 0.1)
                
                X_class = np.random.multivariate_normal(mean, cov, samples_per_class)
                y_class = np.full(samples_per_class, i)
                
                X.append(X_class)
                y.append(y_class)
            
            X = np.vstack(X).astype(np.float32)
            y = np.hstack(y).astype(np.int64)
            
        elif dataset_name == 'synthetic_manifold':
            # Data on a low-dimensional manifold
            n_samples = n_samples
            intrinsic_dim = 3
            ambient_dim = 50
            
            # Generate on manifold
            t = np.linspace(0, 4*np.pi, n_samples)
            manifold_data = np.column_stack([
                np.sin(t),
                np.cos(t),
                t / (4*np.pi)
            ])
            
            # Embed in higher dimension with random projection
            projection = np.random.randn(intrinsic_dim, ambient_dim)
            projection = projection / np.linalg.norm(projection, axis=0)
            
            X = manifold_data @ projection
            X += np.random.randn(n_samples, ambient_dim) * 0.1  # Add noise
            
            # Create classes based on position on manifold
            y = (t / (4*np.pi) * 3).astype(np.int64)
            
            X = X.astype(np.float32)
            
        elif dataset_name == 'synthetic_hierarchical':
            # Hierarchical structure
            n_samples = n_samples
            dim = 128
            
            # Create hierarchical clustering
            X, y = [], []
            
            # Top level: 2 super-clusters
            for super_cluster in range(2):
                super_mean = np.random.randn(dim) * 10
                
                # Mid level: 4 clusters per super-cluster
                for cluster in range(4):
                    cluster_mean = super_mean + np.random.randn(dim) * 3
                    
                    # Generate samples
                    samples = np.random.multivariate_normal(
                        cluster_mean, 
                        np.eye(dim) * 0.5,
                        n_samples // 8
                    )
                    
                    X.append(samples)
                    y.append(super_cluster * 4 + cluster)
            
            X = np.vstack(X).astype(np.float32)
            y = np.hstack(y).astype(np.int64)
        
        # Convert to PyTorch dataset
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X),
            torch.tensor(y)
        )
        
        # Split into train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=128, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=128, shuffle=False
        )
        
        return train_loader, test_loader

# ============================================================================
# Experiment Tracking and Logging
# ============================================================================

class ExperimentTracker:
    """Track experiments with comprehensive logging"""
    
    def __init__(self, experiment_name: str, use_wandb: bool = True):
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"experiments/{experiment_name}_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_file = self.exp_dir / "experiment.log"
        file_handler = logging.FileHandler(self.log_file)
        logger.addHandler(file_handler)
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(project="alignment-research", name=experiment_name)
        
        # Results storage
        self.results = {
            'config': {},
            'metrics': {},
            'phase_diagram': None,
            'scaling_laws': {},
            'application_results': {}
        }
    
    def log_config(self, config: Dict):
        """Log experiment configuration"""
        self.results['config'] = config
        
        # Save to file
        with open(self.exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Log to wandb
        if self.use_wandb:
            wandb.config.update(config)
        
        logger.info(f"Experiment configuration: {config}")
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics"""
        if step is not None:
            if step not in self.results['metrics']:
                self.results['metrics'][step] = {}
            self.results['metrics'][step].update(metrics)
        else:
            self.results['metrics'].update(metrics)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        logger.info(f"Step {step}: {metrics}")
    
    def save_figure(self, fig, name: str):
        """Save matplotlib figure"""
        fig_path = self.exp_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({name: wandb.Image(fig)})
    
    def save_results(self):
        """Save all results"""
        results_path = self.exp_dir / "results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")

# ============================================================================
# Core Analysis Components
# ============================================================================

class NTKAnalyzer:
    """Neural Tangent Kernel analysis"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_empirical_ntk(self, model, x1, x2):
        """Compute empirical NTK between two inputs"""
        model.zero_grad()
        
        # Get outputs
        y1 = model(x1)
        y2 = model(x2)
        
        # Compute kernel
        kernel = 0
        
        for i in range(y1.shape[1]):  # For each output dimension
            # Gradient for x1
            model.zero_grad()
            y1[:, i].sum().backward(retain_graph=True)
            grads1 = []
            for param in model.parameters():
                grads1.append(param.grad.view(-1) if param.grad is not None else torch.zeros_like(param).view(-1))
            grad1 = torch.cat(grads1)
            
            # Gradient for x2
            model.zero_grad()
            y2[:, i].sum().backward(retain_graph=True)
            grads2 = []
            for param in model.parameters():
                grads2.append(param.grad.view(-1) if param.grad is not None else torch.zeros_like(param).view(-1))
            grad2 = torch.cat(grads2)
            
            kernel += torch.dot(grad1, grad2)
        
        return kernel.item()
    
    def compute_ntk_spectrum(self, model, data_loader, n_samples=50):
        """Compute NTK eigenspectrum with memory optimization"""
        # Sample data more efficiently
        x_sample = []
        for x, _ in data_loader:
            x_sample.append(x)
            if len(torch.cat(x_sample)) >= n_samples:
                break
        
        x_sample = torch.cat(x_sample)[:n_samples].to(self.device)
        
        # Limit sample size to prevent memory issues
        max_samples = min(n_samples, 100)  # Cap at 100 samples
        x_sample = x_sample[:max_samples]
        n = len(x_sample)
        
        logger.info(f"Computing NTK for {n} samples...")
        
        # Compute NTK matrix in batches to save memory
        batch_size = min(10, n)  # Process in small batches
        
        ntk_matrix = torch.zeros(n, n, device=self.device)
        
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            for j in range(i, n, batch_size):  # Only compute upper triangle
                j_end = min(j + batch_size, n)
                
                # Compute batch of NTK values
                for ii in range(i, i_end):
                    for jj in range(max(ii, j), j_end):  # Avoid redundant computations
                        k_ij = self.compute_empirical_ntk(model, x_sample[ii:ii+1], x_sample[jj:jj+1])
                        ntk_matrix[ii, jj] = k_ij
                        ntk_matrix[jj, ii] = k_ij
        
        # Compute eigenvalues efficiently
        try:
            eigenvalues = torch.linalg.eigvalsh(ntk_matrix)
            eigenvalues = eigenvalues.cpu().numpy()
        except RuntimeError as e:
            logger.warning(f"Failed to compute eigenvalues: {e}")
            # Fallback: use SVD for numerical stability
            try:
                U, s, V = torch.svd(ntk_matrix)
                eigenvalues = s.cpu().numpy()
            except RuntimeError:
                logger.error("SVD also failed, returning zeros")
                eigenvalues = np.zeros(n)
        
        # Filter out very small eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        if len(eigenvalues) == 0:
            return {
                'eigenvalues': np.array([0.0]),
                'condition_number': float('inf'),
                'effective_rank': 0.0,
                'trace': ntk_matrix.trace().item()
            }
        
        return {
            'eigenvalues': eigenvalues,
            'condition_number': eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else float('inf'),
            'effective_rank': (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item(),
            'trace': ntk_matrix.trace().item()
        }
    
    def _compute_effective_rank(self, eigenvalues):
        """Compute effective rank from eigenvalues"""
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 0
        return (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()

class AGOPAnalyzer:
    """Activation Gradient Outer Product analysis"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_agop(self, model, data_loader, layer_names=None):
        """Compute AGOP for specified layers with memory optimization"""
        if layer_names is None:
            layer_names = [name for name, _ in model.named_modules() 
                          if isinstance(_, (nn.Linear, nn.Conv2d))]
        
        # Setup hooks
        activations = {}
        gradients = {}
        
        def forward_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0].detach()
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(forward_hook(name)))
                hooks.append(module.register_full_backward_hook(backward_hook(name)))
        
        # Compute AGOP with limited batches
        agop_results = {}
        max_batches = 3  # Limit batches to save memory
        
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Compute AGOP for each layer
            for name in layer_names:
                if name in activations and name in gradients:
                    act = activations[name]
                    grad = gradients[name]
                    
                    # Flatten spatial dimensions
                    if len(act.shape) > 2:
                        act = act.view(act.size(0), -1)
                        grad = grad.view(grad.size(0), -1)
                    
                    # Compute outer product
                    agop = torch.einsum('bi,bj->ij', grad, grad) / grad.size(0)
                    
                    if name not in agop_results:
                        agop_results[name] = agop
                    else:
                        agop_results[name] += agop
            
            # Clear cache periodically
            if batch_idx % 2 == 0:
                torch.cuda.empty_cache()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Normalize and compute metrics
        metrics = {}
        for name, agop in agop_results.items():
            agop /= (batch_idx + 1)
            
            # Compute eigenvalues with error handling
            try:
                eigenvalues = torch.linalg.eigvalsh(agop)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                
                if len(eigenvalues) > 0:
                    metrics[name] = {
                        'mean_agop': agop.mean().item(),
                        'trace': agop.trace().item(),
                        'max_eigenvalue': eigenvalues[-1].item(),
                        'effective_rank': (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()
                    }
                else:
                    metrics[name] = {
                        'mean_agop': agop.mean().item(),
                        'trace': agop.trace().item(),
                        'max_eigenvalue': 0.0,
                        'effective_rank': 0.0
                    }
            except RuntimeError as e:
                logger.warning(f"Failed to compute AGOP metrics for {name}: {e}")
                metrics[name] = {
                    'mean_agop': agop.mean().item(),
                    'trace': agop.trace().item(),
                    'max_eigenvalue': 0.0,
                    'effective_rank': 0.0
                }
        
        return metrics
    
    def _compute_effective_rank(self, eigenvalues):
        """Compute effective rank from eigenvalues"""
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 0
        return (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()

class AlignmentAnalyzer:
    """Analyze representation alignment between models"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_cka(self, features1, features2):
        """Compute Centered Kernel Alignment with memory optimization"""
        # Limit sample size to prevent memory issues
        max_samples = 1000
        if features1.shape[0] > max_samples:
            indices = torch.randperm(features1.shape[0])[:max_samples]
            features1 = features1[indices]
            features2 = features2[indices]
        
        n = features1.shape[0]
        
        # Center the features
        features1 = features1 - features1.mean(dim=0, keepdim=True)
        features2 = features2 - features2.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices in batches to save memory
        batch_size = min(500, n)
        
        K = torch.zeros(n, n, device=features1.device)
        L = torch.zeros(n, n, device=features2.device)
        
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            for j in range(0, n, batch_size):
                j_end = min(j + batch_size, n)
                
                batch1_i = features1[i:i_end]
                batch1_j = features1[j:j_end]
                batch2_i = features2[i:i_end]
                batch2_j = features2[j:j_end]
                
                K[i:i_end, j:j_end] = torch.mm(batch1_i, batch1_j.t())
                L[i:i_end, j:j_end] = torch.mm(batch2_i, batch2_j.t())
        
        # Center Gram matrices
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        K_c = H @ K @ H
        L_c = H @ L @ H
        
        # Compute CKA
        hsic = torch.trace(K_c @ L_c)
        var_k = torch.sqrt(torch.trace(K_c @ K_c))
        var_l = torch.sqrt(torch.trace(L_c @ L_c))
        
        cka = hsic / (var_k * var_l)
        
        return cka.item()
    
    def compute_model_alignment(self, model1, model2, data_loader):
        """Compute alignment between two models"""
        representations1 = self._extract_representations(model1, data_loader)
        representations2 = self._extract_representations(model2, data_loader)
        
        alignment_scores = {}
        
        # Find common layers
        common_layers = set(representations1.keys()) & set(representations2.keys())
        
        for layer in common_layers:
            cka_score = self.compute_cka(representations1[layer], representations2[layer])
            alignment_scores[layer] = cka_score
        
        # Overall alignment
        overall_alignment = np.mean(list(alignment_scores.values()))
        
        return {
            'layer_alignment': alignment_scores,
            'overall_alignment': overall_alignment
        }
    
    def _extract_representations(self, model, data_loader, max_samples=1000):
        """Extract representations from all layers with memory optimization"""
        representations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in representations:
                    representations[name] = []
                # Store only a limited number of samples to save memory
                if len(representations[name]) < 10:  # Limit batches stored
                    representations[name].append(output.detach().cpu())
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass with limited samples
        sample_count = 0
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                _ = model(x)
                
                sample_count += x.size(0)
                if sample_count >= max_samples:
                    break
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate and limit representations
        for name in representations:
            if representations[name]:  # Check if we have any representations
                reps = torch.cat(representations[name], dim=0)
                # Limit total samples to prevent memory issues
                if reps.shape[0] > 500:
                    indices = torch.randperm(reps.shape[0])[:500]
                    reps = reps[indices]
                representations[name] = reps
                
                # Flatten if needed
                if len(representations[name].shape) > 2:
                    representations[name] = representations[name].view(
                        representations[name].size(0), -1
                    )
            else:
                # Remove empty representations
                del representations[name]
        
        return representations

# ============================================================================
# Main Experimental Pipeline
# ============================================================================

class UniversalGeometryExperiment:
    """Main experimental framework"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.ntk_analyzer = NTKAnalyzer(self.device)
        self.agop_analyzer = AGOPAnalyzer(self.device)
        self.alignment_analyzer = AlignmentAnalyzer(self.device)
        
        # Initialize tracking
        self.tracker = ExperimentTracker(config['experiment_name'], config.get('use_wandb', True))
        self.tracker.log_config(config)
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _limit_dataset_size(self, train_loader, test_loader, max_samples=5000):
        """Limit dataset size to prevent memory issues"""
        # For training data
        train_samples = []
        train_labels = []
        count = 0
        
        for x, y in train_loader:
            train_samples.append(x)
            train_labels.append(y)
            count += x.size(0)
            if count >= max_samples:
                break
        
        if train_samples:
            train_x = torch.cat(train_samples, dim=0)[:max_samples]
            train_y = torch.cat(train_labels, dim=0)[:max_samples]
            train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=64, shuffle=True
            )
        
        # For test data
        test_samples = []
        test_labels = []
        count = 0
        
        for x, y in test_loader:
            test_samples.append(x)
            test_labels.append(y)
            count += x.size(0)
            if count >= max_samples // 5:  # Smaller test set
                break
        
        if test_samples:
            test_x = torch.cat(test_samples, dim=0)[:max_samples // 5]
            test_y = torch.cat(test_labels, dim=0)[:max_samples // 5]
            test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=64, shuffle=False
            )
        
        return train_loader, test_loader
    
    def create_model(self, architecture: str, width: int, depth: int, num_classes: int):
        """Create model based on architecture type"""
        if architecture == 'mlp':
            layers = []
            input_dim = self.config['input_dim']
            
            # Add flatten layer for image inputs
            if input_dim in [784, 3072]:  # Image datasets
                layers.append(nn.Flatten())
            
            for i in range(depth):
                if i == 0:
                    layers.append(nn.Linear(input_dim, width))
                else:
                    layers.append(nn.Linear(width, width))
                
                layers.append(nn.ReLU())
                
                if self.config.get('use_batchnorm', False):
                    layers.append(nn.BatchNorm1d(width))
            
            layers.append(nn.Linear(width, num_classes))
            model = nn.Sequential(*layers)
            
        elif architecture == 'cnn':
            # Simple CNN for image data
            model = nn.Sequential(
                nn.Conv2d(1 if self.config['input_dim'] == 784 else 3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, width),
                nn.ReLU(),
                nn.Linear(width, num_classes)
            )
            
        elif architecture == 'resnet':
            # Simplified ResNet
            from torchvision.models import resnet18
            model = resnet18(num_classes=num_classes)
            # Adjust first layer for grayscale if needed
            if self.config['input_dim'] == 784:
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return model.to(self.device)
    
    def run_phase_diagram_experiment(self):
        """Map phase diagram of NTK-AGOP space"""
        logger.info("Starting phase diagram experiment...")
        
        # Parameter grid
        widths = self.config['phase_diagram']['widths']
        learning_rates = self.config['phase_diagram']['learning_rates']
        init_scales = self.config['phase_diagram']['init_scales']
        
        results = []
        
        for width in tqdm(widths, desc="Width"):
            for lr in learning_rates:
                for init_scale in init_scales:
                    # Create models
                    dataset_config = DATASET_CONFIGS[self.config['dataset']]
                    
                    model1 = self.create_model(
                        self.config['architecture'],
                        width,
                        self.config['depth'],
                        dataset_config.num_classes
                    )
                    model2 = self.create_model(
                        self.config['architecture'],
                        width,
                        self.config['depth'],
                        dataset_config.num_classes
                    )
                    
                    # Initialize with specific scale
                    self._initialize_model(model1, init_scale)
                    self._initialize_model(model2, init_scale)
                    
                    # Get data
                    train_loader, test_loader = DatasetLoader.get_dataset(
                        self.config['dataset'],
                        self.config.get('n_samples')
                    )
                    
                    # Limit dataset size for memory efficiency
                    train_loader, test_loader = self._limit_dataset_size(train_loader, test_loader)
                    
                    # Clean memory before heavy computations
                    self._cleanup_memory()
                    
                    # Train and measure
                    metrics = self._train_and_measure(
                        model1, model2, train_loader, test_loader, lr
                    )
                    
                    metrics.update({
                        'width': width,
                        'learning_rate': lr,
                        'init_scale': init_scale
                    })
                    
                    results.append(metrics)
                    
                    # Log intermediate results
                    self.tracker.log_metrics(metrics)
                    
                    # Clean memory between iterations
                    self._cleanup_memory()
        
        # Analyze results
        df = pd.DataFrame(results)
        self.tracker.results['phase_diagram'] = df
        
        # Create visualizations
        self._plot_phase_diagram(df)
        
        return df
    
    def run_scaling_laws_experiment(self):
        """Discover scaling laws for alignment"""
        logger.info("Starting scaling laws experiment...")
        
        # Scaling parameters
        model_scales = self.config['scaling']['model_scales']
        dataset_sizes = self.config['scaling']['dataset_sizes']
        
        results = []
        
        for scale in tqdm(model_scales, desc="Model Scale"):
            for data_size in dataset_sizes:
                # Determine width and depth from scale
                width = int(np.sqrt(scale / 10))  # Heuristic
                depth = max(2, int(np.log10(scale)))
                
                # Create models
                dataset_config = DATASET_CONFIGS[self.config['dataset']]
                model1 = self.create_model(
                    self.config['architecture'],
                    width,
                    depth,
                    dataset_config.num_classes
                )
                model2 = self.create_model(
                    self.config['architecture'],
                    width,
                    depth,
                    dataset_config.num_classes
                )
                
                # Get data
                train_loader, test_loader = DatasetLoader.get_dataset(
                    self.config['dataset'],
                    n_samples=data_size
                )
                
                # Limit dataset size
                train_loader, test_loader = self._limit_dataset_size(train_loader, test_loader, max_samples=5000)
                
                # Measure critical time and final alignment
                critical_time, final_alignment = self._measure_critical_time(
                    model1, model2, train_loader, test_loader
                )
                
                results.append({
                    'model_scale': scale,
                    'width': width,
                    'depth': depth,
                    'dataset_size': data_size,
                    'critical_time': critical_time,
                    'final_alignment': final_alignment
                })
                
                self.tracker.log_metrics(results[-1])
        
        # Fit scaling laws
        df = pd.DataFrame(results)
        scaling_laws = self._fit_scaling_laws(df)
        
        self.tracker.results['scaling_laws'] = scaling_laws
        self._plot_scaling_laws(df, scaling_laws)
        
        return scaling_laws
    
    def run_cross_dataset_analysis(self):
        """Analyze alignment across different datasets"""
        logger.info("Starting cross-dataset analysis...")
        
        datasets = self.config['cross_dataset']['datasets']
        results = {}
        
        # Train models on each dataset
        models = {}
        for dataset in tqdm(datasets, desc="Training on datasets"):
            dataset_config = DATASET_CONFIGS[dataset]
            
            model = self.create_model(
                self.config['architecture'],
                self.config['cross_dataset']['width'],
                self.config['cross_dataset']['depth'],
                dataset_config.num_classes
            )
            
            train_loader, test_loader = DatasetLoader.get_dataset(dataset)
            
            # Train model
            accuracy = self._train_model(model, train_loader, test_loader)
            models[dataset] = model
            
            logger.info(f"Trained on {dataset}: accuracy = {accuracy:.3f}")
        
        # Compute cross-dataset alignments
        alignment_matrix = np.zeros((len(datasets), len(datasets)))
        
        for i, dataset1 in enumerate(datasets):
            for j, dataset2 in enumerate(datasets):
                if i <= j:
                    # Get common test set (use first dataset's test set)
                    _, test_loader = DatasetLoader.get_dataset(datasets[0], n_samples=1000)
                    
                    alignment = self.alignment_analyzer.compute_model_alignment(
                        models[dataset1],
                        models[dataset2],
                        test_loader
                    )
                    
                    alignment_matrix[i, j] = alignment['overall_alignment']
                    alignment_matrix[j, i] = alignment['overall_alignment']
        
        results['alignment_matrix'] = alignment_matrix
        results['datasets'] = datasets
        
        self._plot_cross_dataset_alignment(alignment_matrix, datasets)
        self.tracker.results['cross_dataset'] = results
        
        return results
    
    def _train_and_measure(self, model1, model2, train_loader, test_loader, lr):
        """Train models and measure key metrics with memory optimization"""
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
        
        # Initial measurements (limit samples for NTK)
        try:
            ntk_init = self.ntk_analyzer.compute_ntk_spectrum(model1, test_loader, n_samples=20)
        except Exception as e:
            logger.warning(f"Failed to compute initial NTK: {e}")
            ntk_init = {'eigenvalues': np.array([1.0]), 'condition_number': 1.0, 
                       'effective_rank': 1.0, 'trace': 1.0}
        
        # Training loop with memory cleanup
        for epoch in range(self.config['phase_diagram']['epochs']):
            # Train step
            self._train_epoch(model1, train_loader, optimizer1)
            self._train_epoch(model2, train_loader, optimizer2)
            
            # Clean memory every few epochs
            if epoch % 10 == 0:
                self._cleanup_memory()
        
        # Final measurements
        try:
            ntk_final = self.ntk_analyzer.compute_ntk_spectrum(model1, test_loader, n_samples=20)
        except Exception as e:
            logger.warning(f"Failed to compute final NTK: {e}")
            ntk_final = ntk_init
        
        try:
            agop_metrics = self.agop_analyzer.compute_agop(model1, test_loader)
        except Exception as e:
            logger.warning(f"Failed to compute AGOP: {e}")
            agop_metrics = {'layer_0': {'mean_agop': 0.0, 'trace': 0.0, 
                                       'max_eigenvalue': 0.0, 'effective_rank': 0.0}}
        
        try:
            alignment = self.alignment_analyzer.compute_model_alignment(
                model1, model2, test_loader
            )
        except Exception as e:
            logger.warning(f"Failed to compute alignment: {e}")
            alignment = {'overall_alignment': 0.0, 'layer_alignment': {}}
        
        # Compute stability and magnitude
        ntk_stability = self._compute_ntk_stability(ntk_init, ntk_final)
        agop_magnitude = np.mean([m['mean_agop'] for m in agop_metrics.values()])
        
        return {
            'ntk_stability': ntk_stability,
            'agop_magnitude': agop_magnitude,
            'alignment': alignment['overall_alignment'],
            'ntk_condition_final': ntk_final['condition_number']
        }
    
    def _train_epoch(self, model, train_loader, optimizer):
        """Single training epoch"""
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _train_model(self, model, train_loader, test_loader, epochs=50):
        """Full model training"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self._train_epoch(model, train_loader, optimizer)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return correct / total
    
    def _compute_ntk_stability(self, ntk_init, ntk_final):
        """Compute NTK stability metric"""
        # Use correlation of eigenvalue distributions
        init_eigen = ntk_init['eigenvalues']
        final_eigen = ntk_final['eigenvalues']
        
        # Normalize
        init_eigen = init_eigen / init_eigen.sum()
        final_eigen = final_eigen / final_eigen.sum()
        
        # Compute correlation
        min_len = min(len(init_eigen), len(final_eigen))
        correlation = np.corrcoef(init_eigen[:min_len], final_eigen[:min_len])[0, 1]
        
        return max(0, correlation)  # Ensure non-negative
    
    def _plot_phase_diagram(self, df):
        """Create phase diagram visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            values='alignment',
            index='agop_magnitude',
            columns='ntk_stability',
            aggfunc='mean'
        )
        
        # Heatmap
        sns.heatmap(pivot, cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Alignment Heatmap')
        
        # Scatter plot colored by phase
        scatter = axes[0, 1].scatter(
            df['ntk_stability'],
            df['agop_magnitude'],
            c=df['alignment'],
            cmap='viridis',
            s=50,
            alpha=0.6
        )
        axes[0, 1].set_xlabel('NTK Stability')
        axes[0, 1].set_ylabel('AGOP Magnitude')
        axes[0, 1].set_title('Phase Space')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Learning rate effect
        lr_effect = df.groupby('learning_rate')['alignment'].mean()
        axes[1, 0].semilogx(lr_effect.index, lr_effect.values, 'o-')
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('Average Alignment')
        axes[1, 0].set_title('Learning Rate Effect')
        
        # Width effect
        width_effect = df.groupby('width')['alignment'].mean()
        axes[1, 1].loglog(width_effect.index, width_effect.values, 'o-')
        axes[1, 1].set_xlabel('Width')
        axes[1, 1].set_ylabel('Average Alignment')
        axes[1, 1].set_title('Width Effect')
        
        plt.tight_layout()
        self.tracker.save_figure(fig, 'phase_diagram')
        plt.close()
    
    def _fit_scaling_laws(self, df):
        """Fit power law scaling relationships"""
        from scipy.optimize import curve_fit
        
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        scaling_laws = {}
        
        # Critical time vs width
        width_data = df.groupby('width')['critical_time'].mean()
        popt, _ = curve_fit(power_law, width_data.index, width_data.values)
        scaling_laws['width'] = {
            'coefficient': popt[0],
            'exponent': popt[1],
            'equation': f't_c = {popt[0]:.2f} * W^{{{popt[1]:.2f}}}'
        }
        
        # Critical time vs dataset size
        data_size = df.groupby('dataset_size')['critical_time'].mean()
        popt, _ = curve_fit(power_law, data_size.index, data_size.values)
        scaling_laws['dataset_size'] = {
            'coefficient': popt[0],
            'exponent': popt[1],
            'equation': f't_c = {popt[0]:.2f} * N^{{{popt[1]:.2f}}}'
        }
        
        return scaling_laws
    
    def _plot_scaling_laws(self, df, scaling_laws):
        """Plot scaling law results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Width scaling
        width_data = df.groupby('width').agg({
            'critical_time': ['mean', 'std']
        })
        
        axes[0].errorbar(
            width_data.index,
            width_data['critical_time']['mean'],
            yerr=width_data['critical_time']['std'],
            fmt='o',
            capsize=5
        )
        
        # Fit line
        x_fit = np.logspace(np.log10(width_data.index.min()),
                           np.log10(width_data.index.max()), 100)
        law = scaling_laws['width']
        y_fit = law['coefficient'] * np.power(x_fit, law['exponent'])
        
        axes[0].loglog(x_fit, y_fit, 'r--', label=law['equation'])
        axes[0].set_xlabel('Width')
        axes[0].set_ylabel('Critical Time')
        axes[0].set_title('Width Scaling Law')
        axes[0].legend()
        
        # Dataset size scaling
        data_size_data = df.groupby('dataset_size').agg({
            'critical_time': ['mean', 'std']
        })
        
        axes[1].errorbar(
            data_size_data.index,
            data_size_data['critical_time']['mean'],
            yerr=data_size_data['critical_time']['std'],
            fmt='o',
            capsize=5
        )
        
        # Fit line
        x_fit = np.logspace(np.log10(data_size_data.index.min()),
                           np.log10(data_size_data.index.max()), 100)
        law = scaling_laws['dataset_size']
        y_fit = law['coefficient'] * np.power(x_fit, law['exponent'])
        
        axes[1].loglog(x_fit, y_fit, 'r--', label=law['equation'])
        axes[1].set_xlabel('Dataset Size')
        axes[1].set_ylabel('Critical Time')
        axes[1].set_title('Dataset Size Scaling Law')
        axes[1].legend()
        
        plt.tight_layout()
        self.tracker.save_figure(fig, 'scaling_laws')
        plt.close()
    
    def _plot_cross_dataset_alignment(self, alignment_matrix, datasets):
        """Plot cross-dataset alignment matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            alignment_matrix,
            xticklabels=datasets,
            yticklabels=datasets,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0.5,
            vmin=0,
            vmax=1,
            ax=ax
        )
        
        ax.set_title('Cross-Dataset Model Alignment')
        plt.tight_layout()
        self.tracker.save_figure(fig, 'cross_dataset_alignment')
        plt.close()
    
    def _measure_critical_time(self, model1, model2, train_loader, test_loader):
        """Measure critical time for alignment emergence"""
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        
        critical_time = None
        max_epochs = 100
        
        for epoch in range(max_epochs):
            # Train
            self._train_epoch(model1, train_loader, optimizer1)
            self._train_epoch(model2, train_loader, optimizer2)
            
            # Measure alignment every 5 epochs
            if epoch % 5 == 0:
                alignment = self.alignment_analyzer.compute_model_alignment(
                    model1, model2, test_loader
                )
                
                if alignment['overall_alignment'] > 0.5 and critical_time is None:
                    critical_time = epoch
                    
                # Stop if well-aligned
                if alignment['overall_alignment'] > 0.8:
                    break
        
        final_alignment = alignment['overall_alignment']
        
        return critical_time if critical_time else max_epochs, final_alignment
    
    def _initialize_model(self, model, scale):
        """Initialize model with specific scale"""
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, scale / np.sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, 0, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def run_all_experiments(self):
        """Run complete experimental pipeline"""
        logger.info("Starting comprehensive experiment pipeline...")
        
        # Phase diagram
        if self.config.get('run_phase_diagram', True):
            phase_results = self.run_phase_diagram_experiment()
            logger.info("Phase diagram experiment completed")
        
        # Scaling laws
        if self.config.get('run_scaling_laws', True):
            scaling_results = self.run_scaling_laws_experiment()
            logger.info("Scaling laws experiment completed")
        
        # Cross-dataset analysis
        if self.config.get('run_cross_dataset', True):
            cross_dataset_results = self.run_cross_dataset_analysis()
            logger.info("Cross-dataset experiment completed")
        
        # Save all results
        self.tracker.save_results()
        logger.info(f"All experiments completed. Results saved to {self.tracker.exp_dir}")
        
        return self.tracker.results


# ============================================================================
# Configuration Templates
# ============================================================================

def get_default_config(experiment_type='comprehensive'):
    """Get default configuration for experiments"""
    
    base_config = {
        'experiment_name': f'universal_geometry_{experiment_type}',
        'use_wandb': False,  # Set to True if you have wandb configured
        'device': 'cuda',
        'seed': 42,
        
        # Model configuration
        'architecture': 'mlp',
        'depth': 4,
        'input_dim': 784,
        
        # Training configuration
        'batch_size': 128,
        'num_workers': 4,
        
        # Dataset configuration
        'dataset': 'mnist',
        
        # Experiment flags
        'run_phase_diagram': True,
        'run_scaling_laws': True,
        'run_cross_dataset': True,
    }
    
    if experiment_type == 'phase_diagram':
        base_config.update({
            'phase_diagram': {
                'widths': [32, 64, 128, 256, 512],
                'learning_rates': [1e-4, 1e-3, 1e-2, 1e-1],
                'init_scales': [0.1, 1.0, 10.0],
                'epochs': 50
            },
            'run_scaling_laws': False,
            'run_cross_dataset': False
        })
        
    elif experiment_type == 'scaling_laws':
        base_config.update({
            'scaling': {
                'model_scales': [1e3, 1e4, 1e5, 1e6],
                'dataset_sizes': [1000, 5000, 10000, 50000]
            },
            'run_phase_diagram': False,
            'run_cross_dataset': False
        })
        
    elif experiment_type == 'cross_dataset':
        base_config.update({
            'cross_dataset': {
                'datasets': ['mnist', 'fashion_mnist', 'kmnist', 'emnist'],
                'width': 256,
                'depth': 4
            },
            'run_phase_diagram': False,
            'run_scaling_laws': False
        })
        
    elif experiment_type == 'comprehensive':
        base_config.update({
            'phase_diagram': {
                'widths': [64, 128, 256],
                'learning_rates': [1e-3, 1e-2],
                'init_scales': [1.0],
                'epochs': 30
            },
            'scaling': {
                'model_scales': [1e4, 1e5],
                'dataset_sizes': [5000, 10000]
            },
            'cross_dataset': {
                'datasets': ['mnist', 'fashion_mnist', 'cifar10'],
                'width': 256,
                'depth': 4
            }
        })
    
    return base_config


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Geometry and Alignment Research')
    parser.add_argument('--experiment-type', type=str, default='comprehensive',
                       choices=['phase_diagram', 'scaling_laws', 'cross_dataset', 'comprehensive'])
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--architecture', type=str, default='mlp',
                       choices=['mlp', 'cnn', 'resnet'])
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get configuration
    config = get_default_config(args.experiment_type)
    config['dataset'] = args.dataset
    config['architecture'] = args.architecture
    config['use_wandb'] = args.use_wandb
    config['seed'] = args.seed
    
    # Update input dimension based on dataset
    dataset_config = DATASET_CONFIGS[args.dataset]
    config['input_dim'] = dataset_config.input_dim
    
    # Run experiments
    experiment = UniversalGeometryExperiment(config)
    results = experiment.run_all_experiments()
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)
    print(f"Results saved to: {experiment.tracker.exp_dir}")
    print("\nKey findings:")
    
    if 'phase_diagram' in results:
        print(f"- Phase diagram mapped with {len(results['phase_diagram'])} configurations")
    
    if 'scaling_laws' in results:
        print("\n- Scaling laws discovered:")
        for name, law in results['scaling_laws'].items():
            print(f"  {name}: {law['equation']}")
    
    if 'cross_dataset' in results:
        print(f"\n- Cross-dataset alignment analyzed for {len(results['cross_dataset']['datasets'])} datasets")