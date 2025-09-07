"""
Advanced Geometric Analysis Toolkit for Neural Representations
=============================================================

Comprehensive geometric analysis tools for understanding neural network representations,
including manifold analysis, curvature metrics, and topological features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.stats import entropy
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import logging
import json
import os
from datetime import datetime
from pathlib import Path
import copy
import wandb
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeometricAnalyzer:
    """
    Comprehensive geometric analysis of neural representations
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def extract_representations(self, model, data_loader, layers=None):
        """Extract representations from specified layers"""
        if 'mps' in str(self.device):
            return self._extract_representations_streaming(model, data_loader, layers)
        else:
            return self._extract_representations_standard(model, data_loader, layers)
    
    def _extract_representations_streaming(self, model, data_loader, layers=None, max_samples=200):
        """Streaming representation extraction for MPS using feature extractor - no hooks!"""
        print(f"🔍 MPS: Using feature extractor (no hooks) with max_samples={max_samples}")
        
        try:
            from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
        except ImportError:
            print("🔍 MPS: Falling back to timm forward_features")
            return self._extract_with_timm_features(model, data_loader, max_samples)
        
        # Get available nodes
        try:
            _, eval_nodes = get_graph_node_names(model)
            print(f"🔍 MPS: Found {len(eval_nodes)} available nodes")
            
            # Select only key layers for MPS to minimize memory
            return_nodes = {}
            key_patterns = ['layer1', 'layer2', 'layer3', 'layer4', 'fc', 'classifier']
            
            for node in eval_nodes:
                for pattern in key_patterns:
                    if pattern in node and len(return_nodes) < 4:  # Max 4 layers
                        return_nodes[node] = f"{pattern}_{len(return_nodes)}"
                        break
            
            print(f"🔍 MPS: Selected {len(return_nodes)} key layers: {list(return_nodes.values())}")
            
            # Create feature extractor
            feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
            
        except Exception as e:
            print(f"🔍 MPS: Feature extractor failed ({e}), using timm fallback")
            return self._extract_with_timm_features(model, data_loader, max_samples)
        
        # Extract features with minimal memory usage
        final_representations = {}
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                if batch_idx >= 5:  # Process 5 batches for more data
                    break
                
                print(f"🔍 MPS: Processing batch {batch_idx}, shape: {x.shape}")
                
                # Take more samples for richer analysis
                batch_size = min(8, x.shape[0])
                x_small = x[:batch_size].to(self.device)
                
                # Extract features
                features = feature_extractor(x_small)
                
                # Process each feature immediately
                for layer_name, feature_tensor in features.items():
                    print(f"🔍 MPS: Got {layer_name}: {feature_tensor.shape}")
                    
                    # Flatten and move to CPU immediately
                    if len(feature_tensor.shape) > 2:
                        feature_flat = feature_tensor.view(feature_tensor.shape[0], -1)
                    else:
                        feature_flat = feature_tensor
                    
                    feature_cpu = feature_flat.cpu()
                    
                    # Random projection if high-dimensional
                    if feature_cpu.shape[1] > 32:
                        proj_dim = 16
                        proj_matrix = torch.randn(feature_cpu.shape[1], proj_dim) / np.sqrt(proj_dim)
                        feature_cpu = feature_cpu @ proj_matrix
                        print(f"🔍 MPS: Applied projection {feature_flat.shape[1]} -> {proj_dim}")
                    
                    final_representations[layer_name] = feature_cpu
                
                # Clear MPS memory immediately
                del features, x_small
                torch.mps.empty_cache()
                break  # Only 1 batch
        
        print(f"🔍 MPS: Feature extraction completed with {len(final_representations)} layers")
        return final_representations
    
    def _extract_with_timm_features(self, model, data_loader, max_samples=50):
        """Fallback using timm's forward_features if available"""
        final_representations = {}
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                if batch_idx >= 5:
                    break
                
                print(f"🔍 MPS: Using timm features, batch {batch_idx}, shape: {x.shape}")
                batch_size = min(8, x.shape[0])
                x_small = x[:batch_size].to(self.device)
                
                # Try timm's forward_features
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(x_small)
                    print(f"🔍 MPS: Got features: {features.shape}")
                    
                    # Flatten
                    if len(features.shape) > 2:
                        features_flat = features.view(features.shape[0], -1)
                    else:
                        features_flat = features
                    
                    features_cpu = features_flat.cpu()
                    
                    # Random projection if needed
                    if features_cpu.shape[1] > 32:
                        proj_matrix = torch.randn(features_cpu.shape[1], 16) / np.sqrt(16)
                        features_cpu = features_cpu @ proj_matrix
                    
                    final_representations['features'] = features_cpu
                else:
                    # Just use final output
                    output = model(x_small)
                    if len(output.shape) > 2:
                        output = output.view(output.shape[0], -1)
                    final_representations['output'] = output.cpu()
                
                # Clear memory
                torch.mps.empty_cache()
                break
        
        return final_representations
    
    def _extract_representations_standard(self, model, data_loader, layers=None):
        """Standard representation extraction for CUDA/CPU"""
        representations = {}
        
        def get_hook(name):
            def hook(module, input, output):
                representations[name] = output.detach()
            return hook
        
        # Register hooks
        if 'mps' in str(self.device):
            print(f"🔍 MPS CHECKPOINT 2: Registering hooks")
            
        hooks = []
        hook_names = []
        for name, module in model.named_modules():
            if layers is None or name in layers:
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
                    hooks.append(module.register_forward_hook(get_hook(name)))
                    hook_names.append(name)
        
        if 'mps' in str(self.device):
            print(f"🔍 MPS CHECKPOINT 3: Registered {len(hooks)} hooks")
        
        # Collect representations
        all_representations = {name: [] for name in hook_names}
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(data_loader, desc="Extracting representations")):
                # More aggressive batch limiting for MPS
                max_batches = 3 if 'mps' in str(self.device) else 10
                if batch_idx >= max_batches:
                    break
                
                if 'mps' in str(self.device):
                    print(f"🔍 MPS CHECKPOINT 4: Processing batch {batch_idx}, shape: {x.shape}")
                    import psutil
                    print(f"🔍 Memory usage: {psutil.virtual_memory().percent:.1f}%")
                    
                x = x.to(self.device)
                
                if 'mps' in str(self.device):
                    print(f"🔍 MPS CHECKPOINT 5: Data moved to MPS")
                    
                _ = model(x)
                
                if 'mps' in str(self.device):
                    print(f"🔍 MPS CHECKPOINT 6: Forward pass completed")
                
                # For MPS: process immediately with random projections
                if 'mps' in str(self.device):
                    print(f"🔍 MPS CHECKPOINT 7: Processing {len(representations)} representations")
                    
                    for i, (name, rep) in enumerate(representations.items()):
                        print(f"🔍 MPS CHECKPOINT 8.{i}: Processing layer {name}, shape: {rep.shape}")
                        
                        rep_cpu = rep.cpu()
                        print(f"🔍 MPS CHECKPOINT 9.{i}: Moved to CPU")
                        
                        if len(rep_cpu.shape) > 2:
                            rep_cpu = rep_cpu.view(rep_cpu.shape[0], -1)
                            print(f"🔍 MPS CHECKPOINT 10.{i}: Flattened to {rep_cpu.shape}")
                        
                        # Apply random projection to drastically reduce dimensionality
                        if rep_cpu.shape[1] > 64:  # Only if high-dimensional
                            if not hasattr(self, '_projection_matrices'):
                                self._projection_matrices = {}
                            
                            if name not in self._projection_matrices:
                                # Create random projection matrix (original_dim -> 32)
                                proj_dim = min(32, rep_cpu.shape[1])
                                self._projection_matrices[name] = torch.randn(rep_cpu.shape[1], proj_dim) / np.sqrt(proj_dim)
                                print(f"🔍 MPS CHECKPOINT 11.{i}: Created projection matrix {rep_cpu.shape[1]} -> {proj_dim}")
                            
                            # Apply projection
                            rep_cpu = rep_cpu @ self._projection_matrices[name]
                            print(f"🔍 MPS CHECKPOINT 12.{i}: Applied projection, new shape: {rep_cpu.shape}")
                        
                        # Only keep tiny samples
                        if name not in all_representations:
                            all_representations[name] = []
                        if len(all_representations[name]) < 20:  # Even smaller limit
                            sample_size = min(5, rep_cpu.shape[0])  # Only 5 samples per batch
                            all_representations[name].append(rep_cpu[:sample_size])
                            print(f"🔍 MPS CHECKPOINT 13.{i}: Stored {sample_size} samples")
                    
                    print(f"🔍 MPS CHECKPOINT 14: Clearing MPS cache")
                    # Clear MPS cache after each batch
                    torch.mps.empty_cache()
                    print(f"🔍 MPS CHECKPOINT 15: Batch {batch_idx} completed")
                else:
                    for name, rep in representations.items():
                        all_representations[name].append(rep.cpu())
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate with size limits for MPS
        for name in all_representations:
            if all_representations[name]:
                concatenated = torch.cat(all_representations[name], dim=0)
                
                # For MPS: limit final size ultra-aggressively
                if 'mps' in str(self.device) and concatenated.shape[0] > 20:
                    indices = torch.randperm(concatenated.shape[0])[:20]
                    concatenated = concatenated[indices]
                
                all_representations[name] = concatenated
            
            # Flatten spatial dimensions if needed (standard case)
            if self.device != 'mps' and len(all_representations[name].shape) > 2:
                batch_size = all_representations[name].shape[0]
                all_representations[name] = all_representations[name].view(batch_size, -1)
        
        return all_representations

    def compute_representation_metrics(self, representations):
        """Compute comprehensive geometric metrics for representations"""
        if 'mps' in str(self.device):
            return self._compute_representation_metrics_minimal(representations)
        else:
            return self._compute_representation_metrics_standard(representations)
    
    def _compute_representation_metrics_minimal(self, representations):
        """Ultra-minimal metrics computation for MPS"""
        print(f"🔍 MPS: Computing minimal metrics for {len(representations)} layers")
        metrics = {}
        
        for layer_name, rep in representations.items():
            print(f"🔍 MPS: Processing {layer_name} with shape {rep.shape}")
            
            # Convert to numpy with minimal samples
            rep_np = rep.numpy()
            
            # Only basic statistics - no complex computations
            metrics[layer_name] = {
                'mean_activation': float(np.mean(rep_np)),
                'std_activation': float(np.std(rep_np)),
                'sparsity': float(np.mean(rep_np == 0)),
                'dimensionality': rep_np.shape[1],
                'sample_count': rep_np.shape[0],
                'intrinsic_dim': {'basic_rank': min(rep_np.shape)}
            }
            
            print(f"🔍 MPS: Computed basic stats for {layer_name}")
        
        print(f"🔍 MPS: Minimal metrics computation completed")
        return metrics
    
    def _compute_representation_metrics_standard(self, representations):
        """Standard metrics computation for CUDA/CPU"""
        metrics = {}
        layer_names = list(representations.keys())
        
        for layer_name in tqdm(layer_names, desc="Computing geometric metrics"):
            rep = representations[layer_name]
            rep_np = rep.numpy()
            
            # Basic statistics
            metrics[layer_name] = {
                'mean_activation': np.mean(rep_np),
                'std_activation': np.std(rep_np),
                'sparsity': np.mean(rep_np == 0),
                'dimensionality': rep_np.shape[1]
            }
            
            # Intrinsic dimensionality
            metrics[layer_name]['intrinsic_dim'] = self._estimate_intrinsic_dimension(rep_np)
            
            # Geometric properties
            metrics[layer_name].update(self._compute_geometric_properties(rep_np))
            metrics[layer_name].update(self._compute_topological_features(rep_np))
        
        return metrics
    
    def _estimate_intrinsic_dimension(self, X, methods=['pca', 'mle', 'correlation']):
        """Estimate intrinsic dimensionality using multiple methods"""
        results = {}
        
        # PCA-based estimation (explained variance)
        if 'pca' in methods:
            pca = PCA()
            pca.fit(X)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            # Find dimension explaining 90% variance
            dim_90 = np.argmax(cumsum >= 0.9) + 1
            # Find dimension explaining 95% variance
            dim_95 = np.argmax(cumsum >= 0.95) + 1
            results['pca_90'] = dim_90
            results['pca_95'] = dim_95
        
        # Maximum Likelihood Estimation (Levina-Bickel)
        if 'mle' in methods:
            results['mle'] = self._mle_intrinsic_dimension(X)
        
        # Correlation dimension
        if 'correlation' in methods:
            results['correlation'] = self._correlation_dimension(X)
        
        return results
    
    def _mle_intrinsic_dimension(self, X, k1=10, k2=20):
        """Maximum likelihood estimation of intrinsic dimension"""
        n = X.shape[0]
        
        # Compute pairwise distances
        dist = pairwise_distances(X)
        
        # For each point, find k nearest neighbors
        dimensions = []
        
        for i in range(n):
            # Get distances to other points
            di = dist[i, :]
            di[i] = np.inf  # Exclude self
            
            # Sort distances
            sorted_di = np.sort(di)
            
            # Estimate dimension using different k values
            for k in range(k1, min(k2, n-1)):
                # MLE estimator
                mk = np.log(sorted_di[k] / sorted_di[:k]).sum() / k
                if mk > 0:
                    dim_estimate = 1 / mk
                    dimensions.append(dim_estimate)
        
        return np.median(dimensions) if dimensions else 0
    
    def _correlation_dimension(self, X, n_samples=1000):
        """Estimate correlation dimension"""
        n = min(n_samples, X.shape[0])
        idx = np.random.choice(X.shape[0], n, replace=False)
        X_sample = X[idx]
        
        # Compute pairwise distances
        distances = pdist(X_sample)
        distances = distances[distances > 0]  # Remove zeros
        
        if len(distances) == 0:
            return 0
        
        # Log-log plot of correlation integral
        epsilons = np.logspace(np.log10(distances.min()), np.log10(distances.max()), 50)
        correlation_sum = []
        
        for eps in epsilons:
            correlation_sum.append(np.mean(distances < eps))
        
        # Fit line in log-log space
        log_eps = np.log(epsilons[10:-10])  # Avoid edges
        log_corr = np.log(np.array(correlation_sum)[10:-10] + 1e-10)
        
        # Linear regression
        if len(log_eps) > 2:
            slope, _ = np.polyfit(log_eps, log_corr, 1)
            return slope
        
        return 0
    
    def _compute_geometric_properties(self, X):
        """Compute geometric properties of the representation"""
        properties = {}
        
        # Compute covariance spectrum
        cov = np.cov(X.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) > 0:
            properties['spectral_entropy'] = entropy(eigenvalues / eigenvalues.sum())
            properties['effective_rank'] = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
            properties['condition_number'] = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else np.inf
            properties['spectral_decay_rate'] = self._fit_power_law_decay(eigenvalues)
        
        # Geometric mean of pairwise distances
        n_samples = min(1000, X.shape[0])
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        dist_matrix = pairwise_distances(X[idx])
        
        properties['mean_pairwise_distance'] = dist_matrix.mean()
        properties['std_pairwise_distance'] = dist_matrix.std()
        
        # Local geometry: average nearest neighbor distance
        k = min(10, X.shape[0] - 1)
        nn_distances = []
        for i in range(dist_matrix.shape[0]):
            sorted_dists = np.sort(dist_matrix[i])[1:k+1]  # Exclude self
            nn_distances.append(sorted_dists.mean())
        
        properties['mean_nn_distance'] = np.mean(nn_distances)
        properties['local_density_variation'] = np.std(nn_distances) / (np.mean(nn_distances) + 1e-10)
        
        return properties
        
        return properties
    
    def _fit_power_law_decay(self, eigenvalues):
        """Fit power law to eigenvalue decay"""
        log_indices = np.log(np.arange(1, len(eigenvalues) + 1))
        log_eigenvalues = np.log(eigenvalues + 1e-10)
        
        # Fit linear regression in log-log space
        if len(eigenvalues) > 2:
            slope, _ = np.polyfit(log_indices, log_eigenvalues, 1)
            return -slope  # Return positive decay rate
        
        return 0
    
    def _compute_topological_features(self, X, n_samples=500):
        """Compute topological features using persistent homology ideas"""
        features = {}
        
        # Sample for efficiency
        n = min(n_samples, X.shape[0])
        idx = np.random.choice(X.shape[0], n, replace=False)
        X_sample = X[idx]
        
        # Compute distance matrix
        dist_matrix = pairwise_distances(X_sample)
        
        # Analyze distance distribution
        distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        
        features['distance_entropy'] = entropy(np.histogram(distances, bins=50)[0] + 1)
        features['distance_uniformity'] = 1 - np.std(distances) / (np.mean(distances) + 1e-10)
        
        # Clustering coefficient analog
        # For each point, compute fraction of neighbors that are also neighbors of each other
        k = min(10, n - 1)
        clustering_coeffs = []
        
        for i in range(n):
            # Find k nearest neighbors
            neighbors = np.argsort(dist_matrix[i])[1:k+1]
            
            # Compute distances between neighbors
            neighbor_dists = dist_matrix[np.ix_(neighbors, neighbors)]
            threshold = np.median(dist_matrix[i, neighbors])
            
            # Count connections below threshold
            connections = (neighbor_dists < threshold).sum() - k
            max_connections = k * (k - 1)
            
            clustering_coeffs.append(connections / max_connections if max_connections > 0 else 0)
        
        features['clustering_coefficient'] = np.mean(clustering_coeffs)
        features['clustering_variation'] = np.std(clustering_coeffs)
        
        return features
    
    def analyze_curvature(self, representations, n_samples=100):
        """Analyze the curvature of representation manifolds"""
        curvature_results = {}
        
        for layer_name, rep in representations.items():
            rep_np = rep.numpy()
            
            # Sample points for efficiency
            n = min(n_samples, rep_np.shape[0])
            idx = np.random.choice(rep_np.shape[0], n, replace=False)
            X = rep_np[idx]
            
            # Estimate local curvature using Ricci curvature approximation
            curvatures = self._estimate_ricci_curvature(X)
            
            curvature_results[layer_name] = {
                'mean_curvature': np.mean(curvatures),
                'std_curvature': np.std(curvatures),
                'positive_curvature_fraction': np.mean(curvatures > 0),
                'negative_curvature_fraction': np.mean(curvatures < 0),
                'curvature_range': (curvatures.min(), curvatures.max())
            }
        
        return curvature_results
    
    def _estimate_ricci_curvature(self, X, k=5):
        """Estimate Ricci curvature using Ollivier-Ricci curvature approximation"""
        n = X.shape[0]
        dist_matrix = pairwise_distances(X)
        
        curvatures = []
        
        for i in range(n):
            # Find k nearest neighbors of point i
            neighbors_i = np.argsort(dist_matrix[i])[1:k+1]
            
            for j in neighbors_i:
                # Find k nearest neighbors of point j
                neighbors_j = np.argsort(dist_matrix[j])[1:k+1]
                
                # Compute Wasserstein distance between neighbor distributions
                # Simplified: use average distance
                w1_distance = 0
                for ni in neighbors_i:
                    min_dist = min(dist_matrix[ni, neighbors_j])
                    w1_distance += min_dist
                w1_distance /= k
                
                # Ollivier-Ricci curvature
                if dist_matrix[i, j] > 0:
                    curvature = 1 - w1_distance / dist_matrix[i, j]
                    curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def compute_alignment_geometry(self, model1, model2, data_loader):
        """Analyze geometric relationship between two models"""
        # Extract representations
        reps1 = self.extract_representations(model1, data_loader)
        reps2 = self.extract_representations(model2, data_loader)
        
        # Find common layers
        common_layers = set(reps1.keys()) & set(reps2.keys())
        
        alignment_geometry = {}
        
        for layer in common_layers:
            rep1 = reps1[layer].numpy()
            rep2 = reps2[layer].numpy()
            
            # Ensure same number of samples
            n = min(rep1.shape[0], rep2.shape[0])
            rep1, rep2 = rep1[:n], rep2[:n]
            
            # Compute various alignment metrics
            metrics = {
                'cka': self._compute_cka(rep1, rep2),
                'procrustes': self._compute_procrustes_distance(rep1, rep2),
                'subspace_angle': self._compute_subspace_angle(rep1, rep2),
                'shape_metric': self._compute_shape_metric(rep1, rep2),
                'topological_similarity': self._compute_topological_similarity(rep1, rep2)
            }
            
            alignment_geometry[layer] = metrics
        
        return alignment_geometry
    
    def _compute_cka(self, X, Y):
        """Compute Centered Kernel Alignment"""
        # Center the matrices
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        
        # Compute Gram matrices
        K = X @ X.T
        L = Y @ Y.T
        
        # Center Gram matrices
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ K @ H
        L_c = H @ L @ H
        
        # Compute CKA
        hsic = np.trace(K_c @ L_c)
        var1 = np.sqrt(np.trace(K_c @ K_c))
        var2 = np.sqrt(np.trace(L_c @ L_c))
        
        return hsic / (var1 * var2) if var1 * var2 > 0 else 0
    
    def _compute_procrustes_distance(self, X, Y):
        """Compute Procrustes distance after optimal alignment"""
        from scipy.spatial import procrustes
        
        # Standardize
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        Y_std = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-10)
        
        # Handle dimension mismatch
        if X_std.shape[1] != Y_std.shape[1]:
            min_dim = min(X_std.shape[1], Y_std.shape[1])
            X_std = X_std[:, :min_dim]
            Y_std = Y_std[:, :min_dim]
        
        # Compute Procrustes
        _, _, disparity = procrustes(X_std, Y_std)
        
        return 1 / (1 + disparity)  # Convert to similarity
    
    def _compute_subspace_angle(self, X, Y, k=10):
        """Compute principal angles between subspaces"""
        # Compute top k principal components
        U_x = PCA(n_components=min(k, X.shape[1])).fit(X).components_.T
        U_y = PCA(n_components=min(k, Y.shape[1])).fit(Y).components_.T
        
        # Compute principal angles
        # This is the cosine of angles between subspaces
        s = np.linalg.svd(U_x.T @ U_y, compute_uv=False)
        
        # Average cosine similarity
        return np.mean(s)
    
    def _compute_shape_metric(self, X, Y):
        """Compute shape-based similarity metric"""
        # Compute pairwise distance matrices
        D_x = pairwise_distances(X)
        D_y = pairwise_distances(Y)
        
        # Normalize by scale
        D_x = D_x / (D_x.mean() + 1e-10)
        D_y = D_y / (D_y.mean() + 1e-10)
        
        # Compute correlation of distance matrices
        return np.corrcoef(D_x.flatten(), D_y.flatten())[0, 1]
    
    def _compute_topological_similarity(self, X, Y, k=10):
        """Compute topological similarity using neighborhood preservation"""
        n = X.shape[0]
        
        # Compute k-NN graphs
        dist_x = pairwise_distances(X)
        dist_y = pairwise_distances(Y)
        
        # Find k nearest neighbors for each point
        preservation_scores = []
        
        for i in range(n):
            neighbors_x = set(np.argsort(dist_x[i])[1:k+1])
            neighbors_y = set(np.argsort(dist_y[i])[1:k+1])
            
            # Compute Jaccard similarity
            intersection = len(neighbors_x & neighbors_y)
            union = len(neighbors_x | neighbors_y)
            
            preservation_scores.append(intersection / union if union > 0 else 0)
        
        return np.mean(preservation_scores)
    
    def _create_simplified_mps_visualization(self, representations, metrics, save_path=None):
        """Simplified visualization for MPS with minimal metrics"""
        print("🔍 MPS: Creating simplified visualization")
        
        fig = plt.figure(figsize=(12, 8))
        layer_names = list(metrics.keys())
        
        # Plot 1: Basic statistics
        ax1 = plt.subplot(2, 2, 1)
        mean_activations = [metrics[l]['mean_activation'] for l in layer_names]
        x = np.arange(len(layer_names))
        ax1.bar(x, mean_activations, alpha=0.8, color='blue')
        ax1.set_title('Mean Activation per Layer')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Mean Activation')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'L{i}' for i in range(len(layer_names))])
        
        # Plot 2: Standard deviation
        ax2 = plt.subplot(2, 2, 2)
        std_activations = [metrics[l]['std_activation'] for l in layer_names]
        ax2.bar(x, std_activations, alpha=0.8, color='green')
        ax2.set_title('Activation Std per Layer')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'L{i}' for i in range(len(layer_names))])
        
        # Plot 3: Sparsity
        ax3 = plt.subplot(2, 2, 3)
        sparsities = [metrics[l]['sparsity'] for l in layer_names]
        ax3.bar(x, sparsities, alpha=0.8, color='red')
        ax3.set_title('Sparsity per Layer')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Sparsity')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'L{i}' for i in range(len(layer_names))])
        
        # Plot 4: Dimensionality
        ax4 = plt.subplot(2, 2, 4)
        dims = [metrics[l]['dimensionality'] for l in layer_names]
        ax4.bar(x, dims, alpha=0.8, color='orange')
        ax4.set_title('Feature Dimensionality per Layer')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Dimensionality')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'L{i}' for i in range(len(layer_names))])
        
        plt.tight_layout()
        plt.suptitle('MPS Simplified Geometric Analysis', fontsize=14, y=0.98)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🔍 MPS: Saved visualization to {save_path}")
        
        return fig
    
    def visualize_geometric_analysis(self, representations, metrics, save_path=None):
        """Create comprehensive geometric visualizations"""
        # For MPS: create simplified visualization
        if 'mps' in str(self.device):
            return self._create_simplified_mps_visualization(representations, metrics, save_path)
        
        n_layers = len(representations)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Intrinsic dimensionality across layers
        ax1 = plt.subplot(3, 3, 1)
        layer_names = list(metrics.keys())
        # Handle both full and minimal metrics
        if 'mps' in str(self.device):
            # MPS minimal metrics - use basic rank
            basic_ranks = [metrics[l]['intrinsic_dim']['basic_rank'] for l in layer_names]
            x = np.arange(len(layer_names))
            ax1.bar(x, basic_ranks, label='Basic Rank', alpha=0.8, color='orange')
        else:
            # Full metrics
            pca_90_dims = [metrics[l]['intrinsic_dim']['pca_90'] for l in layer_names]
            mle_dims = [metrics[l]['intrinsic_dim']['mle'] for l in layer_names]
            
            x = np.arange(len(layer_names))
            width = 0.35
            
            ax1.bar(x - width/2, pca_90_dims, width, label='PCA 90%', alpha=0.8)
            ax1.bar(x + width/2, mle_dims, width, label='MLE', alpha=0.8)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Intrinsic Dimension')
        ax1.set_title('Intrinsic Dimensionality Evolution')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'L{i}' for i in range(len(layer_names))])
        ax1.legend()
        
        # 2. Spectral decay rates
        ax2 = plt.subplot(3, 3, 2)
        decay_rates = [metrics[l]['spectral_decay_rate'] for l in layer_names]
        ax2.plot(decay_rates, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Spectral Decay Rate')
        ax2.set_title('Eigenvalue Decay Rate')
        ax2.grid(True, alpha=0.3)
        
        # 3. Effective rank vs actual dimension
        ax3 = plt.subplot(3, 3, 3)
        eff_ranks = [metrics[l]['effective_rank'] for l in layer_names]
        actual_dims = [metrics[l]['dimensionality'] for l in layer_names]
        
        ax3.scatter(actual_dims, eff_ranks, s=100, c=np.arange(len(layer_names)), cmap='viridis')
        ax3.plot([0, max(actual_dims)], [0, max(actual_dims)], 'k--', alpha=0.3)
        ax3.set_xlabel('Actual Dimension')
        ax3.set_ylabel('Effective Rank')
        ax3.set_title('Compression: Effective vs Actual Dimension')
        
        # 4. Distance distribution entropy
        ax4 = plt.subplot(3, 3, 4)
        dist_entropies = [metrics[l]['distance_entropy'] for l in layer_names]
        ax4.plot(dist_entropies, 's-', linewidth=2, markersize=8, color='red')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Distance Entropy')
        ax4.set_title('Representation Uniformity')
        ax4.grid(True, alpha=0.3)
        
        # 5. Clustering coefficient
        ax5 = plt.subplot(3, 3, 5)
        clustering_coeffs = [metrics[l]['clustering_coefficient'] for l in layer_names]
        clustering_vars = [metrics[l]['clustering_variation'] for l in layer_names]
        
        ax5.errorbar(range(len(layer_names)), clustering_coeffs, yerr=clustering_vars, 
                    fmt='o-', capsize=5, linewidth=2, markersize=8)
        ax5.set_xlabel('Layer')
        ax5.set_ylabel('Clustering Coefficient')
        ax5.set_title('Local Geometric Structure')
        ax5.grid(True, alpha=0.3)
        
        # 6. Local density variation
        ax6 = plt.subplot(3, 3, 6)
        density_vars = [metrics[l]['local_density_variation'] for l in layer_names]
        ax6.bar(range(len(layer_names)), density_vars, alpha=0.7, color='green')
        ax6.set_xlabel('Layer')
        ax6.set_ylabel('Density Variation')
        ax6.set_title('Representation Density Uniformity')
        
        # 7. 2D projection of representations (using t-SNE for selected layers)
        for i, (layer_idx, layer_name) in enumerate([(0, 'First'), 
                                                      (len(layer_names)//2, 'Middle'), 
                                                      (-1, 'Last')]):
            if layer_idx < len(layer_names):
                ax = plt.subplot(3, 3, 7 + i)
                
                rep = list(representations.values())[layer_idx].numpy()
                
                # Sample for efficiency
                n_samples = min(1000, rep.shape[0])
                idx = np.random.choice(rep.shape[0], n_samples, replace=False)
                rep_sample = rep[idx]
                
                # Apply t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                rep_2d = tsne.fit_transform(rep_sample)
                
                # Plot with density
                from scipy.stats import gaussian_kde
                
                # Calculate the point density
                xy = np.vstack([rep_2d[:, 0], rep_2d[:, 1]])
                z = gaussian_kde(xy)(xy)
                
                scatter = ax.scatter(rep_2d[:, 0], rep_2d[:, 1], c=z, s=10, 
                                   cmap='viridis', alpha=0.6)
                ax.set_title(f'{layer_name} Layer Representation')
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_3d_manifold_visualization(self, representation, labels=None):
        """Create interactive 3D visualization of representation manifold"""
        # Apply PCA for 3D projection
        pca = PCA(n_components=3)
        rep_3d = pca.fit_transform(representation.numpy())
        
        # Create plotly figure
        if labels is not None:
            fig = go.Figure(data=[go.Scatter3d(
                x=rep_3d[:, 0],
                y=rep_3d[:, 1],
                z=rep_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=labels,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'Class: {l}' for l in labels]
            )])
        else:
            # Color by local density
            from sklearn.neighbors import KernelDensity
            
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(rep_3d)
            density = np.exp(kde.score_samples(rep_3d))
            
            fig = go.Figure(data=[go.Scatter3d(
                x=rep_3d[:, 0],
                y=rep_3d[:, 1],
                z=rep_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=density,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Density")
                )
            )])
        
        fig.update_layout(
            title='3D Representation Manifold',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            width=800,
            height=800
        )
        
        return fig
    
    def analyze_layer_transitions(self, representations):
        """Analyze how representations transform between layers"""
        layer_names = list(representations.keys())
        n_layers = len(layer_names)
        
        transitions = {}
        
        for i in range(n_layers - 1):
            curr_layer = layer_names[i]
            next_layer = layer_names[i + 1]
            
            curr_rep = representations[curr_layer].numpy()
            next_rep = representations[next_layer].numpy()
            
            # Ensure same number of samples
            n = min(curr_rep.shape[0], next_rep.shape[0])
            curr_rep, next_rep = curr_rep[:n], next_rep[:n]
            
            # Analyze transformation
            transition_metrics = {
                'representation_change': np.linalg.norm(next_rep - curr_rep) / n,
                'correlation': self._compute_representation_correlation(curr_rep, next_rep),
                'mutual_information': self._estimate_mutual_information(curr_rep, next_rep),
                'geometric_distortion': self._compute_geometric_distortion(curr_rep, next_rep)
            }
            
            transitions[f'{curr_layer}->{next_layer}'] = transition_metrics
        
        return transitions
    
    def _compute_representation_correlation(self, X, Y):
        """Compute correlation between representations"""
        # Flatten and compute correlation
        return np.corrcoef(X.flatten(), Y.flatten())[0, 1]
    
    def _estimate_mutual_information(self, X, Y, n_bins=10):
        """Estimate mutual information between representations"""
        from sklearn.metrics import mutual_info_score
        
        # Discretize for MI estimation
        X_discrete = np.digitize(X.mean(axis=1), bins=np.linspace(X.min(), X.max(), n_bins))
        Y_discrete = np.digitize(Y.mean(axis=1), bins=np.linspace(Y.min(), Y.max(), n_bins))
        
        return mutual_info_score(X_discrete, Y_discrete)
    
    def _compute_geometric_distortion(self, X, Y):
        """Compute how much the geometry is distorted in transformation"""
        # Sample points for efficiency
        n_samples = min(500, X.shape[0])
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        
        # Compute pairwise distances
        dist_X = pairwise_distances(X[idx])
        dist_Y = pairwise_distances(Y[idx])
        
        # Normalize
        dist_X = dist_X / (dist_X.mean() + 1e-10)
        dist_Y = dist_Y / (dist_Y.mean() + 1e-10)
        
        # Compute distortion as relative change in distances
        distortion = np.abs(dist_X - dist_Y).mean()
        
        return distortion


# Example usage and analysis pipeline
def run_geometric_analysis(model, data_loader, save_results=True):
    """
    Run comprehensive geometric analysis on a model
    """
    device = next(model.parameters()).device
    analyzer = GeometricAnalyzer(device=device)
    
    print("Starting geometric analysis...")
    
    # Simplified pipeline for MPS
    total_steps = 3 if device == 'mps' else 5
    with tqdm(total=total_steps, desc="Geometric Analysis Pipeline") as pbar:
        print("Extracting representations...")
        representations = analyzer.extract_representations(model, data_loader)
        pbar.update(1)
        
        print("Computing geometric metrics...")
        metrics = analyzer.compute_representation_metrics(representations)
        pbar.update(1)
        
        if device != 'mps':
            print("Analyzing curvature...")
            curvature = analyzer.analyze_curvature(representations)
            pbar.update(1)
            
            print("Analyzing layer transitions...")
            transitions = analyzer.analyze_layer_transitions(representations)
            pbar.update(1)
            
            # Create visualizations
            print("Creating visualizations...")
            fig = analyzer.visualize_geometric_analysis(representations, metrics)
            pbar.update(1)
        else:
            print("Creating simplified visualizations...")
            curvature = {}
            transitions = {}
            fig = analyzer.visualize_geometric_analysis(representations, metrics)
            pbar.update(1)
    
    if save_results:
        fig.savefig('geometric_analysis.png', dpi=300)
        
        # Save metrics to file
        with open('geometric_metrics.json', 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            json.dump({
                'metrics': convert_numpy(metrics),
                'curvature': convert_numpy(curvature),
                'transitions': convert_numpy(transitions)
            }, f, indent=2)
    
    return {
        'representations': representations,
        'metrics': metrics,
        'curvature': curvature,
        'transitions': transitions,
        'figure': fig
    }


def compare_model_geometries(model1, model2, data_loader):
    """
    Compare geometric properties of two models
    """
    device = next(model1.parameters()).device
    analyzer = GeometricAnalyzer(device=device)
    
    print("Starting model comparison...")
    
    with tqdm(total=4, desc="Model Comparison Pipeline") as pbar:
        print("Analyzing alignment geometry...")
        alignment_geometry = analyzer.compute_alignment_geometry(model1, model2, data_loader)
        pbar.update(1)
        
        # Extract representations for both models
        reps1 = analyzer.extract_representations(model1, data_loader)
        reps2 = analyzer.extract_representations(model2, data_loader)
        pbar.update(1)
        
        # Compute metrics for both
        metrics1 = analyzer.compute_representation_metrics(reps1)
        metrics2 = analyzer.compute_representation_metrics(reps2)
        pbar.update(1)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Compare intrinsic dimensions
        ax = axes[0, 0]
        layer_names = list(metrics1.keys())
        dims1 = [metrics1[l]['intrinsic_dim']['pca_90'] for l in layer_names]
        dims2 = [metrics2[l]['intrinsic_dim']['pca_90'] for l in layer_names]
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        ax.bar(x - width/2, dims1, width, label='Model 1', alpha=0.8)
        ax.bar(x + width/2, dims2, width, label='Model 2', alpha=0.8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Intrinsic Dimension')
        ax.set_title('Intrinsic Dimension Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i}' for i in range(len(layer_names))])
        ax.legend()
        
        # Compare effective ranks
        ax = axes[0, 1]
        ranks1 = [metrics1[l]['effective_rank'] for l in layer_names]
        ranks2 = [metrics2[l]['effective_rank'] for l in layer_names]
        
        ax.plot(ranks1, 'o-', label='Model 1', linewidth=2, markersize=8)
        ax.plot(ranks2, 's-', label='Model 2', linewidth=2, markersize=8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Effective Rank Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compare alignment metrics
        ax = axes[0, 2]
        alignment_scores = {metric: [] for metric in ['cka', 'procrustes', 'subspace_angle']}
        
        for layer in alignment_geometry:
            for metric in alignment_scores:
                alignment_scores[metric].append(alignment_geometry[layer][metric])
        
        x = np.arange(len(alignment_geometry))
        for i, (metric, scores) in enumerate(alignment_scores.items()):
            ax.plot(x, scores, label=metric, linewidth=2, marker=['o', 's', '^'][i], markersize=8)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Alignment Score')
        ax.set_title('Layer-wise Alignment Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i}' for i in range(len(alignment_geometry))])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compare spectral properties
        ax = axes[1, 0]
        spectral_entropy1 = [metrics1[l]['spectral_entropy'] for l in layer_names]
        spectral_entropy2 = [metrics2[l]['spectral_entropy'] for l in layer_names]
        
        ax.plot(spectral_entropy1, 'o-', label='Model 1', linewidth=2, markersize=8)
        ax.plot(spectral_entropy2, 's-', label='Model 2', linewidth=2, markersize=8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spectral Entropy')
        ax.set_title('Representation Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compare clustering coefficients
        ax = axes[1, 1]
        clustering1 = [metrics1[l]['clustering_coefficient'] for l in layer_names]
        clustering2 = [metrics2[l]['clustering_coefficient'] for l in layer_names]
        
        ax.scatter(clustering1, clustering2, s=100, c=np.arange(len(layer_names)), cmap='viridis')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Model 1 Clustering')
        ax.set_ylabel('Model 2 Clustering')
        ax.set_title('Clustering Coefficient Correlation')
        
        # Add layer labels
        for i, (x, y) in enumerate(zip(clustering1, clustering2)):
            ax.annotate(f'L{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "Geometric Similarity Summary:\n\n"
        
        # Average alignment across layers
        avg_alignment = {}
        for metric in ['cka', 'procrustes', 'subspace_angle']:
            scores = [alignment_geometry[l][metric] for l in alignment_geometry]
            avg_alignment[metric] = np.mean(scores)
            summary_text += f"Average {metric}: {avg_alignment[metric]:.3f}\n"
        
        # Correlation of geometric properties
        dim_corr = np.corrcoef(dims1, dims2)[0, 1]
        rank_corr = np.corrcoef(ranks1, ranks2)[0, 1]
        
        summary_text += f"\nDimension correlation: {dim_corr:.3f}\n"
        summary_text += f"Rank correlation: {rank_corr:.3f}\n"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    pbar.update(1)
    
    return {
        'alignment_geometry': alignment_geometry,
        'metrics_model1': metrics1,
        'metrics_model2': metrics2,
        'comparison_figure': fig
    }


# ============================================================================
# Neural Tangent Kernel Analysis
# ============================================================================

class NTKAnalyzer:
    """Neural Tangent Kernel analysis for understanding training dynamics"""

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
        if 'mps' in str(self.device):
            max_samples = min(n_samples, 25)  # Smaller for MPS
        else:
            max_samples = min(n_samples, 100)  # Cap at 100 samples
        x_sample = x_sample[:max_samples]
        n = len(x_sample)

        logger.info(f"Computing NTK for {n} samples...")

        # Compute NTK matrix in batches to save memory
        batch_size = min(10, n)  # Process in small batches

        ntk_matrix = torch.zeros(n, n, device=self.device)

        with tqdm(total=n*n, desc="Computing NTK matrix") as pbar:
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
                            pbar.update(1)

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


# ============================================================================
# Activation Gradient Outer Product Analysis
# ============================================================================

class AGOPAnalyzer:
    """Activation Gradient Outer Product analysis for layer-wise dynamics"""

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

        with tqdm(total=max_batches, desc="Computing AGOP") as pbar:
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
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                pbar.update(1)

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


# ============================================================================
# Representation Alignment Analysis
# ============================================================================

class AlignmentAnalyzer:
    """Analyze representation alignment between models"""

    def __init__(self, device='cuda'):
        self.device = device

    def compute_cka(self, features1, features2):
        """Compute Centered Kernel Alignment with memory optimization"""
        # Limit sample size to prevent memory issues
        max_samples = 200 if 'mps' in str(self.device) else 1000
        if features1.shape[0] > max_samples:
            indices = torch.randperm(features1.shape[0])[:max_samples]
            features1 = features1[indices]
            features2 = features2[indices]

        n = features1.shape[0]

        # Center the features
        features1 = features1 - features1.mean(dim=0, keepdim=True)
        features2 = features2 - features2.mean(dim=0, keepdim=True)

        # Compute Gram matrices in batches to save memory
        batch_size = min(100 if 'mps' in str(self.device) else 500, n)

        K = torch.zeros(n, n, device=features1.device)
        L = torch.zeros(n, n, device=features2.device)

        with tqdm(total=(n//batch_size + 1)**2, desc="Computing CKA") as pbar:
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

                    pbar.update(1)

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

        with tqdm(total=len(common_layers), desc="Computing layer alignment") as pbar:
            for layer in common_layers:
                cka_score = self.compute_cka(representations1[layer], representations2[layer])
                alignment_scores[layer] = cka_score
                pbar.update(1)

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
            try:
                wandb.init(project="geometric-analysis", name=experiment_name)
            except:
                logger.warning("Failed to initialize wandb")

        # Results storage
        self.results = {
            'config': {},
            'metrics': {},
            'geometric_analysis': None,
            'alignment_results': {},
            'ntk_analysis': {},
            'agop_analysis': {}
        }

    def log_config(self, config: Dict):
        """Log experiment configuration"""
        self.results['config'] = config

        # Save to file
        with open(self.exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Log to wandb
        if self.use_wandb:
            try:
                wandb.config.update(config)
            except:
                pass

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
            try:
                wandb.log(metrics, step=step)
            except:
                pass

        logger.info(f"Step {step}: {metrics}")

    def save_figure(self, fig, name: str):
        """Save matplotlib figure"""
        fig_path = self.exp_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')

        # Log to wandb
        if self.use_wandb:
            try:
                wandb.log({name: wandb.Image(fig)})
            except:
                pass

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
# Enhanced Analysis Pipeline
# ============================================================================

def run_comprehensive_analysis(model, data_loader, experiment_name="comprehensive_analysis", use_wandb=True):
    """
    Run comprehensive analysis combining geometric, NTK, AGOP, and alignment analysis
    """
    tracker = ExperimentTracker(experiment_name, use_wandb)

    print("Starting comprehensive neural representation analysis...")

    # Initialize analyzers
    device = next(model.parameters()).device
    geometric_analyzer = GeometricAnalyzer(device=device)
    ntk_analyzer = NTKAnalyzer(device=device)
    agop_analyzer = AGOPAnalyzer(device=device)

    results = {}

    # 1. Geometric Analysis
    print("\n=== Geometric Analysis ===")
    with tqdm(total=4, desc="Geometric Analysis") as pbar:
        representations = geometric_analyzer.extract_representations(model, data_loader)
        pbar.update(1)

        metrics = geometric_analyzer.compute_representation_metrics(representations)
        pbar.update(1)

        curvature = geometric_analyzer.analyze_curvature(representations)
        pbar.update(1)

        transitions = geometric_analyzer.analyze_layer_transitions(representations)
        pbar.update(1)

    results['geometric'] = {
        'representations': representations,
        'metrics': metrics,
        'curvature': curvature,
        'transitions': transitions
    }

    # 2. NTK Analysis
    print("\n=== NTK Analysis ===")
    ntk_results = ntk_analyzer.compute_ntk_spectrum(model, data_loader)
    results['ntk'] = ntk_results

    # 3. AGOP Analysis
    print("\n=== AGOP Analysis ===")
    agop_results = agop_analyzer.compute_agop(model, data_loader)
    results['agop'] = agop_results

    # Log results
    tracker.results.update(results)
    tracker.save_results()

    print("\nAnalysis complete! Results saved to:", tracker.exp_dir)

    return results, tracker


def run_geometric_analysis_multiple_datasets(model_fn, dataset_loaders, save_results=True):
    """
    Run geometric analysis on multiple datasets.
    Args:
        model_fn: function that returns a model instance for a given dataset
        dataset_loaders: dict mapping dataset name to data_loader
        save_results: whether to save results
    Returns:
        results_dict: dict mapping dataset name to geometric analysis results
    """
    analyzer = GeometricAnalyzer(device='cpu')  # Default to CPU for multi-dataset analysis
    results_dict = {}
    for dataset_name, data_loader in dataset_loaders.items():
        print(f"\n=== Geometric Analysis for {dataset_name} ===")
        model = model_fn(dataset_name)
        with tqdm(total=5, desc=f"Geometric Analysis [{dataset_name}]") as pbar:
            representations = analyzer.extract_representations(model, data_loader)
            pbar.update(1)
            metrics = analyzer.compute_representation_metrics(representations)
            pbar.update(1)
            curvature = analyzer.analyze_curvature(representations)
            pbar.update(1)
            transitions = analyzer.analyze_layer_transitions(representations)
            pbar.update(1)
            fig = analyzer.visualize_geometric_analysis(representations, metrics)
            pbar.update(1)
        results_dict[dataset_name] = {
            'representations': representations,
            'metrics': metrics,
            'curvature': curvature,
            'transitions': transitions,
            'figure': fig
        }
        if save_results:
            fig.savefig(f'geometric_analysis_{dataset_name}.png', dpi=300)
            with open(f'geometric_metrics_{dataset_name}.json', 'w') as f:
                json.dump({k: v for k, v in metrics.items()}, f, indent=2)
    return results_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run geometric analysis using timm/torchvision models.")
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                       help='Model name (timm or torchvision compatible)')
    
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to model checkpoint')
    
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    parser.add_argument('--use_timm', action='store_true', default=True,
                       help='Use timm library (default: True)')
    
    parser.add_argument('--num_classes', type=int, default=1000,
                       help='Number of output classes')
    
    # Dataset arguments  
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset to use (imagenet, cifar10, cifar100, mnist, fashionmnist, svhn, stl10, places365, food101, oxford_pets, flowers102, caltech101, caltech256, dtd, or custom dataset name in data_dir)')
    
    parser.add_argument('--data_dir', type=str, default='/Users/tanmoy/research/data',
                       help='Directory containing datasets')
    
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    
    # Data loader arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for data loaders')
    
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loaders')
    
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Analysis arguments
    parser.add_argument('--save_results', action='store_true',
                       help='Flag to save the results')
    
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    
    parser.add_argument('--experiment_name', type=str, default='geometric_analysis',
                       help='Name for the experiment')
    
    # Auto-detect best available device
    if torch.backends.mps.is_available():
        default_device = 'mps'
    elif torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    
    parser.add_argument('--device', type=str, default=default_device,
                       help='Device to use (mps/cuda/cpu)')
    
    # Multi-phase analysis arguments
    parser.add_argument('--analyze_phases', action='store_true',
                       help='Run multi-phase analysis (lazy, aligned, chaotic)')
    
    parser.add_argument('--phases', nargs='+', default=['lazy', 'aligned', 'chaotic'],
                       choices=['lazy', 'aligned', 'chaotic'],
                       help='Phases to analyze')
    
    parser.add_argument('--simulate_training', action='store_true',
                       help='Simulate training steps to reach target phase')
    
    parser.add_argument('--training_steps', type=int, default=100,
                       help='Number of training simulation steps')
    
    parser.add_argument('--scale_factor', type=float, default=1.0,
                       help='Scale factor for parameter initialization')
    
    return parser.parse_args()

def load_model_with_timm(model_name, num_classes=1000, pretrained=True, checkpoint_path=None):
    """Load model using timm library"""
    try:
        import timm
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            
        return model
    except ImportError:
        raise ImportError("timm library not found. Install with: pip install timm")


def load_torchvision_model(model_name, num_classes=1000, pretrained=True, checkpoint_path=None):
    """Load model using torchvision"""
    try:
        import torchvision.models as models
        
        # Get model constructor
        model_fn = getattr(models, model_name)
        model = model_fn(pretrained=pretrained)
        
        # Modify final layer if needed
        if num_classes != 1000:
            if hasattr(model, 'fc'):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
                else:
                    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            
        return model
    except ImportError:
        raise ImportError("torchvision not found. Install with: pip install torchvision")


def load_model_from_checkpoint(checkpoint_path, model_class=None, **model_kwargs):
    """Load model from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if model_class is None:
        # Try to infer from checkpoint
        if 'model_name' in checkpoint:
            model_name = checkpoint['model_name']
            if 'timm' in checkpoint.get('source', ''):
                return load_model_with_timm(model_name, **model_kwargs)
            else:
                return load_torchvision_model(model_name, **model_kwargs)
        else:
            raise ValueError("Cannot infer model class from checkpoint. Please provide model_class.")
    
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    return model


def initialize_model_phase(model, phase='aligned', scale_factor=1.0):
    """
    Initialize model parameters for different training phases
    
    Args:
        model: PyTorch model
        phase: 'lazy', 'aligned', or 'chaotic'
        scale_factor: Scaling factor for parameter initialization
    """
    with torch.no_grad():
        if phase == 'lazy':
            # Lazy training phase: Large initialization, high NTK stability
            # Parameters stay close to initialization during training
            for param in model.parameters():
                if param.dim() >= 2:
                    # Use larger initialization for lazy regime
                    nn.init.normal_(param, mean=0.0, std=2.0 * scale_factor / np.sqrt(param.shape[1]))
                else:
                    nn.init.normal_(param, mean=0.0, std=0.1 * scale_factor)
        
        elif phase == 'aligned':
            # Aligned/Feature Learning phase: Optimal initialization for learning
            # Parameters can move significantly during training
            for param in model.parameters():
                if param.dim() >= 2:
                    # Xavier/He initialization for balanced gradients
                    nn.init.xavier_normal_(param, gain=scale_factor)
                else:
                    nn.init.zeros_(param)
        
        elif phase == 'chaotic':
            # Chaotic phase: Small initialization, unstable training
            # High sensitivity to parameter changes
            for param in model.parameters():
                if param.dim() >= 2:
                    nn.init.normal_(param, mean=0.0, std=0.01 * scale_factor / np.sqrt(param.shape[1]))
                else:
                    nn.init.normal_(param, mean=0.0, std=0.001 * scale_factor)
        
        else:
            raise ValueError(f"Unknown phase: {phase}. Choose from 'lazy', 'aligned', 'chaotic'")


def load_model_from_phase(phase, model_name='resnet18', num_classes=1000, pretrained=False, 
                         use_timm=True, scale_factor=1.0, simulate_training=False, 
                         training_steps=100, checkpoint_path=None):
    """
    Load a model initialized for a specific training phase
    
    Args:
        phase: 'lazy', 'aligned', or 'chaotic'
        model_name: Model name (timm or torchvision compatible)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (before phase initialization)
        use_timm: Whether to use timm library
        scale_factor: Initialization scaling
        simulate_training: Whether to simulate some training steps
        training_steps: Number of simulation steps
        checkpoint_path: Path to checkpoint file
    
    Returns:
        model: Initialized model in the specified phase
    """
    # Create base model
    if checkpoint_path:
        model = load_model_from_checkpoint(checkpoint_path, num_classes=num_classes)
    elif use_timm:
        model = load_model_with_timm(model_name, num_classes, pretrained)
    else:
        model = load_torchvision_model(model_name, num_classes, pretrained)
    
    # Initialize for the specific phase (reinitialize parameters)
    initialize_model_phase(model, phase, scale_factor)
    
    # Optionally simulate some training to reach the phase
    if simulate_training:
        model = simulate_training_phase(model, phase, training_steps)
    
    # Set model metadata
    model.phase = phase
    model.scale_factor = scale_factor
    
    return model


def simulate_training_phase(model, phase, steps=100, lr=0.01):
    """
    Simulate training to reach a specific phase
    
    Args:
        model: PyTorch model
        phase: Target phase
        steps: Number of training steps
        lr: Learning rate
    
    Returns:
        model: Model after simulated training
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Create dummy data for simulation
    if hasattr(model, 'conv1'):  # CNN
        dummy_input = torch.randn(32, model.conv1.in_channels, 32, 32, device=device)
    else:  # MLP
        dummy_input = torch.randn(32, model.fc1.in_features, device=device)
    
    dummy_targets = torch.randint(0, model.fc2.out_features if hasattr(model, 'fc2') else model.fc3.out_features, 
                                 (32,), device=device)
    
    model.train()
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Adjust learning rate based on phase
        if phase == 'lazy':
            # Very small updates to stay in lazy regime
            current_lr = lr * 0.001
        elif phase == 'aligned':
            # Normal learning rate for feature learning
            current_lr = lr
        elif phase == 'chaotic':
            # Large learning rate for chaotic behavior
            current_lr = lr * 10.0
        
        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Forward pass
        outputs = model(dummy_input)
        loss = F.cross_entropy(outputs, dummy_targets)
        
        # Backward pass
        loss.backward()
        
        # Add noise for chaotic phase
        if phase == 'chaotic':
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += torch.randn_like(param.grad) * 0.1
        
        optimizer.step()
    
    model.eval()
    return model




def analyze_phase_research_questions(phase_results):
    """
    Analyze key research questions about model phases
    
    Research Questions:
    1. Do aligned models have lower intrinsic dimension?
    2. Is there a characteristic curvature for aligned representations?
    3. How does spectral entropy relate to alignment?
    """
    insights = {
        'intrinsic_dimension_analysis': {},
        'curvature_analysis': {},
        'spectral_entropy_analysis': {},
        'summary': {}
    }
    
    phases = list(phase_results.keys())
    
    # Question 1: Intrinsic Dimension Analysis
    print("\n🔍 Research Question 1: Do aligned models have lower intrinsic dimension?")
    
    for phase in phases:
        metrics = phase_results[phase]['metrics']
        layer_names = list(metrics.keys())
        
        # Average intrinsic dimensions across layers
        pca_90_dims = [ metrics[l]['intrinsic_dim']['pca_90'] for l in layer_names]
        mle_dims = [metrics[l]['intrinsic_dim']['mle'] for l in layer_names]
        
        avg_pca_dim = np.mean(pca_90_dims)
        avg_mle_dim = np.mean(mle_dims)
        
        insights['intrinsic_dimension_analysis'][phase] = {
            'avg_pca_90': avg_pca_dim,
            'avg_mle': avg_mle_dim,
            'final_layer_pca_90': pca_90_dims[-1],
            'dimension_progression': pca_90_dims
        }
        
        print(f"  {phase:>8}: PCA-90={avg_pca_dim:.2f}, MLE={avg_mle_dim:.2f}")
    
    # Question 2: Curvature Analysis
    print("\n🔍 Research Question 2: Is there characteristic curvature for aligned representations?")
    
    for phase in phases:
        curvature = phase_results[phase]['curvature']
        layer_names = list(curvature.keys())
        
        # Average curvature properties
        mean_curvatures = [curvature[l]['mean_curvature'] for l in layer_names]
        pos_curvature_fracs = [curvature[l]['positive_curvature_fraction'] for l in layer_names]
        
        avg_mean_curvature = np.mean(mean_curvatures)
        avg_pos_fraction = np.mean(pos_curvature_fracs)
        
        insights['curvature_analysis'][phase] = {
            'avg_mean_curvature': avg_mean_curvature,
            'avg_positive_fraction': avg_pos_fraction,
            'curvature_progression': mean_curvatures
        }
        
        print(f"  {phase:>8}: Mean curvature={avg_mean_curvature:.4f}, Pos. fraction={avg_pos_fraction:.3f}")
    
    # Question 3: Spectral Entropy Analysis
    print("\n🔍 Research Question 3: How does spectral entropy relate to alignment?")
    
    for phase in phases:
        metrics = phase_results[phase]['metrics']
        layer_names = list(metrics.keys())
        
        # Spectral properties
        spectral_entropies = [metrics[l]['spectral_entropy'] for l in layer_names]
        effective_ranks = [metrics[l]['effective_rank'] for l in layer_names]
        
        avg_spectral_entropy = np.mean(spectral_entropies)
        avg_effective_rank = np.mean(effective_ranks)
        
        insights['spectral_entropy_analysis'][phase] = {
            'avg_spectral_entropy': avg_spectral_entropy,
            'avg_effective_rank': avg_effective_rank,
            'entropy_progression': spectral_entropies,
            'rank_progression': effective_ranks
        }
        
        print(f"  {phase:>8}: Spectral entropy={avg_spectral_entropy:.4f}, Eff. rank={avg_effective_rank:.2f}")
    
    # Generate Summary Insights
    print("\n📊 Summary Insights:")
    
    # Find phase with lowest intrinsic dimension
    phase_pca_dims = {p: insights['intrinsic_dimension_analysis'][p]['avg_pca_90'] for p in phases}
    lowest_dim_phase = min(phase_pca_dims, key=phase_pca_dims.get)
    print(f"  • Lowest intrinsic dimension: {lowest_dim_phase} ({phase_pca_dims[lowest_dim_phase]:.2f})")
    
    # Find phase with most balanced curvature
    phase_curvatures = {p: abs(insights['curvature_analysis'][p]['avg_mean_curvature']) for p in phases}
    most_balanced_phase = min(phase_curvatures, key=phase_curvatures.get)
    print(f"  • Most balanced curvature: {most_balanced_phase} ({phase_curvatures[most_balanced_phase]:.4f})")
    
    # Find phase with optimal spectral properties
    phase_entropies = {p: insights['spectral_entropy_analysis'][p]['avg_spectral_entropy'] for p in phases}
    optimal_entropy_phase = max(phase_entropies, key=phase_entropies.get)  # Higher entropy = more uniform
    print(f"  • Highest spectral entropy: {optimal_entropy_phase} ({phase_entropies[optimal_entropy_phase]:.4f})")
    
    insights['summary'] = {
        'lowest_intrinsic_dimension': lowest_dim_phase,
        'most_balanced_curvature': most_balanced_phase,
        'highest_spectral_entropy': optimal_entropy_phase,
        'phase_rankings': {
            'intrinsic_dimension': sorted(phases, key=lambda p: phase_pca_dims[p]),
            'curvature_balance': sorted(phases, key=lambda p: phase_curvatures[p]),
            'spectral_entropy': sorted(phases, key=lambda p: phase_entropies[p], reverse=True)
        }
    }
    
    return insights


def visualize_phase_comparison(phase_results, phases):
    """Create comprehensive visualization comparing different phases"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Phase Geometric Analysis Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(phases)]
    
    # 1. Intrinsic Dimension Evolution
    ax = axes[0, 0]
    for i, phase in enumerate(phases):
        metrics = phase_results[phase]['metrics']
        layer_names = list(metrics.keys())
        pca_dims = [metrics[l]['intrinsic_dim']['pca_90'] for l in layer_names]
        
        ax.plot(range(len(pca_dims)), pca_dims, 'o-', 
               color=colors[i], label=phase, linewidth=2, markersize=6)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Intrinsic Dimension (PCA 90%)')
    ax.set_title('Intrinsic Dimension Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average Curvature Comparison
    ax = axes[0, 1]
    phase_names = []
    mean_curvatures = []
    curvature_stds = []
    
    for phase in phases:
        curvature = phase_results[phase]['curvature']
        layer_names = list(curvature.keys())
        curvs = [curvature[l]['mean_curvature'] for l in layer_names]
        
        phase_names.append(phase)
        mean_curvatures.append(np.mean(curvs))
        curvature_stds.append(np.std(curvs))
    
    bars = ax.bar(phase_names, mean_curvatures, yerr=curvature_stds, 
                  color=colors[:len(phases)], alpha=0.7, capsize=5)
    ax.set_ylabel('Mean Curvature')
    ax.set_title('Average Curvature by Phase')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Spectral Entropy Comparison
    ax = axes[0, 2]
    for i, phase in enumerate(phases):
        metrics = phase_results[phase]['metrics']
        layer_names = list(metrics.keys())
        entropies = [metrics[l]['spectral_entropy'] for l in layer_names]
        
        ax.plot(range(len(entropies)), entropies, 's-', 
               color=colors[i], label=phase, linewidth=2, markersize=6)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Spectral Entropy')
    ax.set_title('Spectral Entropy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Effective Rank vs Actual Dimension
    ax = axes[1, 0]
    for i, phase in enumerate(phases):
        metrics = phase_results[phase]['metrics']
        layer_names = list(metrics.keys())
        
        eff_ranks = [metrics[l]['effective_rank'] for l in layer_names]
        actual_dims = [metrics[l]['dimensionality'] for l in layer_names]
        
        ax.scatter(actual_dims, eff_ranks, s=60, alpha=0.7, 
                  color=colors[i], label=phase)
    
    # Add diagonal line
    max_dim = max([max([metrics[l]['dimensionality'] for l in metrics]) 
                   for metrics in [phase_results[p]['metrics'] for p in phases]])
    ax.plot([0, max_dim], [0, max_dim], 'k--', alpha=0.5)
    
    ax.set_xlabel('Actual Dimension')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Dimension vs Effective Rank')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Local Density Variation
    ax = axes[1, 1]
    phase_names = []
    density_vars = []
    
    for phase in phases:
        metrics = phase_results[phase]['metrics']
        layer_names = list(metrics.keys())
        vars_list = [metrics[l]['local_density_variation'] for l in layer_names]
        
        phase_names.append(phase)
        density_vars.append(np.mean(vars_list))
    
    bars = ax.bar(phase_names, density_vars, color=colors[:len(phases)], alpha=0.7)
    ax.set_ylabel('Local Density Variation')
    ax.set_title('Representation Density Uniformity')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Phase Comparison Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create summary text
    summary_text = "Phase Analysis Summary:\n\n"
    
    for i, phase in enumerate(phases):
        metrics = phase_results[phase]['metrics']
        layer_names = list(metrics.keys())
        
        avg_dim = np.mean([metrics[l]['intrinsic_dim']['pca_90'] for l in layer_names])
        avg_entropy = np.mean([metrics[l]['spectral_entropy'] for l in layer_names])
        
        curvature = phase_results[phase]['curvature']
        avg_curvature = np.mean([curvature[l]['mean_curvature'] for l in layer_names])
        
        summary_text += f"{phase.upper()}:\n"
        summary_text += f"  Intrinsic Dim: {avg_dim:.2f}\n"
        summary_text += f"  Spectral Entropy: {avg_entropy:.3f}\n"
        summary_text += f"  Mean Curvature: {avg_curvature:.4f}\n\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig


def demo_phase_analysis():
    """
    Demo function showing how to analyze different model phases using timm
    
    This demonstrates the key research questions:
    1. Do aligned models have lower intrinsic dimension?
    2. Is there a characteristic curvature for aligned representations?
    3. How does spectral entropy relate to alignment?
    """
    print("=== Multi-Phase Analysis Demo ===")
    print("Investigating neural network training phases and their geometric properties\n")
    
    # For models in different phases using timm
    print("Loading models in different training phases...")
    lazy_model = load_model_from_phase('lazy', model_name='resnet18', num_classes=10, 
                                      use_timm=True, simulate_training=True)
    aligned_model = load_model_from_phase('aligned', model_name='resnet18', num_classes=10,
                                         use_timm=True, simulate_training=True)
    chaotic_model = load_model_from_phase('chaotic', model_name='resnet18', num_classes=10,
                                         use_timm=True, simulate_training=True)
    
    print(f"✓ Lazy model: {lazy_model.phase} phase, scale={lazy_model.scale_factor}")
    print(f"✓ Aligned model: {aligned_model.phase} phase, scale={aligned_model.scale_factor}")
    print(f"✓ Chaotic model: {chaotic_model.phase} phase, scale={chaotic_model.scale_factor}")
    
    # Create test data using CIFAR-10
    data_loader = create_standard_data_loader(
        dataset='cifar10', batch_size=32, split='val'
    )
    
    print(f"\nDataset: CIFAR-10 validation set with {len(data_loader.dataset)} samples")
    
    # Run geometric analysis on each phase
    analyzer = GeometricAnalyzer(device='cpu')  # Use CPU for demo
    
    print("\nAnalyzing geometric properties...")
    lazy_geom = run_geometric_analysis(lazy_model, data_loader, save_results=False)
    aligned_geom = run_geometric_analysis(aligned_model, data_loader, save_results=False)
    chaotic_geom = run_geometric_analysis(chaotic_model, data_loader, save_results=False)
    
    print("✓ Geometric analysis complete for all phases")
    
    # Extract key metrics for comparison
    def extract_key_metrics(geom_results):
        metrics = geom_results['metrics']
        layer_names = list(metrics.keys())
        
        avg_intrinsic_dim = np.mean([metrics[l]['intrinsic_dim']['pca_90'] for l in layer_names])
        avg_spectral_entropy = np.mean([metrics[l]['spectral_entropy'] for l in layer_names])
        avg_effective_rank = np.mean([metrics[l]['effective_rank'] for l in layer_names])
        
        return {
            'avg_intrinsic_dim': avg_intrinsic_dim,
            'avg_spectral_entropy': avg_spectral_entropy, 
            'avg_effective_rank': avg_effective_rank
        }
    
    lazy_metrics = extract_key_metrics(lazy_geom)
    aligned_metrics = extract_key_metrics(aligned_geom)
    chaotic_metrics = extract_key_metrics(chaotic_geom);
    
    # Answer key research questions:
    print("\n🔍 Key Research Questions:")
    
    print("\n1. Do aligned models have lower intrinsic dimension?")
    print(f"   Lazy:    {lazy_metrics['avg_intrinsic_dim']:.2f}")
    print(f"   Aligned: {aligned_metrics['avg_intrinsic_dim']:.2f}")
    print(f"   Chaotic: {chaotic_metrics['avg_intrinsic_dim']:.2f}")
    
    dims = {'lazy': lazy_metrics['avg_intrinsic_dim'], 
            'aligned': aligned_metrics['avg_intrinsic_dim'],
            'chaotic': chaotic_metrics['avg_intrinsic_dim']}
    lowest_dim_phase = min(dims.keys(), key=dims.get)
    print(f"   → {lowest_dim_phase.upper()} has the lowest intrinsic dimension!")
    
    print("\n2. How does spectral entropy relate to alignment?")
    print(f"   Lazy:    {lazy_metrics['avg_spectral_entropy']:.4f}")
    print(f"   Aligned: {aligned_metrics['avg_spectral_entropy']:.4f}")  
    print(f"   Chaotic: {chaotic_metrics['avg_spectral_entropy']:.4f}")
    
    entropies = {'lazy': lazy_metrics['avg_spectral_entropy'],
                'aligned': aligned_metrics['avg_spectral_entropy'],
                'chaotic': chaotic_metrics['avg_spectral_entropy']}
    highest_entropy_phase = max(entropies.keys(), key=entropies.get)
    print(f"   → {highest_entropy_phase.upper()} has the most uniform spectral distribution!")
    
    print("\n3. What about effective rank (representation efficiency)?")
    print(f"   Lazy:    {lazy_metrics['avg_effective_rank']:.2f}")
    print(f"   Aligned: {aligned_metrics['avg_effective_rank']:.2f}")
    print(f"   Chaotic: {chaotic_metrics['avg_effective_rank']:.2f}")
    
    print(f"\n📊 Summary:")
    print(f"• Most compressed representations: {lowest_dim_phase}")
    print(f"• Most uniform spectral properties: {highest_entropy_phase}")
    print(f"• This suggests different training phases have distinct geometric signatures!")
    print(f"• Use --analyze_phases flag to run comprehensive multi-phase analysis")
    
    return {
        'models': {'lazy': lazy_model, 'aligned': aligned_model, 'chaotic': chaotic_model},
        'results': {'lazy': lazy_geom, 'aligned': aligned_geom, 'chaotic': chaotic_geom},
        'metrics': {'lazy': lazy_metrics, 'aligned': aligned_metrics, 'chaotic': chaotic_metrics}
    }


def get_dataset_transforms(dataset_name, image_size=224, split="val"):
    """Get appropriate transforms for different datasets"""
    try:
        import torchvision.transforms as transforms
    except ImportError:
        raise ImportError("torchvision not found. Install with: pip install torchvision")
    
    # Dataset-specific normalization values
    normalization = {
        'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
        'cifar100': {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
        'mnist': {'mean': [0.1307], 'std': [0.3081]},
        'fashionmnist': {'mean': [0.2860], 'std': [0.3530]},
        'svhn': {'mean': [0.4377, 0.4438, 0.4728], 'std': [0.1980, 0.2010, 0.1970]},
        'stl10': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'places365': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'food101': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    }
    
    # Get normalization values (default to ImageNet)
    norm_vals = normalization.get(dataset_name.lower(), normalization['imagenet'])
    
    # Training transforms (with augmentation)
    if split == "train":
        if dataset_name.lower() in ['mnist', 'fashionmnist']:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_vals['mean'], std=norm_vals['std'])
            ])
        elif dataset_name.lower() in ['cifar10', 'cifar100', 'svhn']:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_vals['mean'], std=norm_vals['std'])
            ])
        else:  # ImageNet and similar
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_vals['mean'], std=norm_vals['std'])
            ])
    
    # Validation/test transforms (no augmentation)
    else:
        if dataset_name.lower() in ['mnist', 'fashionmnist']:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_vals['mean'], std=norm_vals['std'])
            ])
        elif dataset_name.lower() in ['cifar10', 'cifar100', 'svhn']:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_vals['mean'], std=norm_vals['std'])
            ])
        else:  # ImageNet and similar
            transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_vals['mean'], std=norm_vals['std'])
            ])
    
    return transform


def create_comprehensive_data_loader(data_dir="/Users/tanmoy/research/data", dataset="imagenet", 
                                   batch_size=64, num_workers=4, image_size=224, split="val",
                                   download=True, shuffle=None):
    """
    Create comprehensive data loader supporting many datasets
    
    Args:
        data_dir: Root data directory
        dataset: Dataset name
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        split: Dataset split ('train', 'val', 'test')
        download: Whether to download dataset if not found
        shuffle: Whether to shuffle data (None = auto based on split)
    
    Returns:
        data_loader: PyTorch DataLoader
        num_classes: Number of classes in dataset
    """
    try:
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
    except ImportError:
        raise ImportError("torchvision not found. Install with: pip install torchvision")
    
    # Auto-determine shuffle if not specified
    if shuffle is None:
        shuffle = (split == "train")
    
    # Get appropriate transforms
    transform = get_dataset_transforms(dataset, image_size, split)
    
    dataset_name = dataset.lower()
    
    # Standard torchvision datasets
    if dataset_name == "imagenet":
        if split == "val":
            split_dir = "val"
        elif split == "train":
            split_dir = "train"
        else:
            split_dir = split
            
        dataset_obj = datasets.ImageFolder(
            root=os.path.join(data_dir, "imagenet", split_dir),
            transform=transform
        )
        num_classes = 1000
    
    elif dataset_name == "cifar10":
        train = (split == "train")
        dataset_obj = datasets.CIFAR10(
            root=data_dir, train=train, download=download, transform=transform
        )
        num_classes = 10
    
    elif dataset_name == "cifar100":
        train = (split == "train")
        dataset_obj = datasets.CIFAR100(
            root=data_dir, train=train, download=download, transform=transform
        )
        num_classes = 100
    
    elif dataset_name == "mnist":
        train = (split == "train")
        dataset_obj = datasets.MNIST(
            root=data_dir, train=train, download=download, transform=transform
        )
        num_classes = 10
    
    elif dataset_name == "fashionmnist":
        train = (split == "train")
        dataset_obj = datasets.FashionMNIST(
            root=data_dir, train=train, download=download, transform=transform
        )
        num_classes = 10
    
    elif dataset_name == "svhn":
        if split == "train":
            split_name = "train"
        elif split == "val" or split == "test":
            split_name = "test"
        else:
            split_name = split
            
        dataset_obj = datasets.SVHN(
            root=data_dir, split=split_name, download=download, transform=transform
        )
        num_classes = 10
    
    elif dataset_name == "stl10":
        if split == "val":
            split_name = "test"
        else:
            split_name = split
            
        dataset_obj = datasets.STL10(
            root=data_dir, split=split_name, download=download, transform=transform
        )
        num_classes = 10
    
    elif dataset_name == "places365":
        small = True  # Use Places365-Standard (small version)
        train = (split == "train")
        dataset_obj = datasets.Places365(
            root=data_dir, split="train-standard" if train else "val", 
            small=small, download=download, transform=transform
        )
        num_classes = 365
    
    elif dataset_name == "food101":
        train = (split == "train")
        dataset_obj = datasets.Food101(
            root=data_dir, split="train" if train else "test",
            download=download, transform=transform
        )
        num_classes = 101
    
    elif dataset_name == "oxford_pets":
        train = (split == "train")
        dataset_obj = datasets.OxfordIIITPet(
            root=data_dir, split="trainval" if train else "test",
            download=download, transform=transform, target_type="category"
        )
        num_classes = 37
    
    elif dataset_name == "flowers102":
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "val" 
        else:
            split_name = "test"
            
        dataset_obj = datasets.Flowers102(
            root=data_dir, split=split_name, download=download, transform=transform
        )
        num_classes = 102
    
    elif dataset_name == "caltech101":
        dataset_obj = datasets.Caltech101(
            root=data_dir, download=download, transform=transform
        )
        num_classes = 101
    
    elif dataset_name == "caltech256":
        dataset_obj = datasets.Caltech256(
            root=data_dir, download=download, transform=transform
        )
        num_classes = 256
    
    elif dataset_name == "dtd":  # Describable Textures Dataset
        train = (split == "train")
        dataset_obj = datasets.DTD(
            root=data_dir, split="train" if train else "test",
            download=download, transform=transform
        )
        num_classes = 47
    
    # Custom dataset from directory structure
    elif os.path.isdir(os.path.join(data_dir, dataset_name)):
        dataset_path = os.path.join(data_dir, dataset_name)
        
        # Try to find split subdirectory
        if os.path.isdir(os.path.join(dataset_path, split)):
            dataset_path = os.path.join(dataset_path, split)
        elif os.path.isdir(os.path.join(dataset_path, "train")) and split == "val":
            # Check for val directory, fallback to test
            if os.path.isdir(os.path.join(dataset_path, "val")):
                dataset_path = os.path.join(dataset_path, "val")
            elif os.path.isdir(os.path.join(dataset_path, "test")):
                dataset_path = os.path.join(dataset_path, "test")
        
        dataset_obj = datasets.ImageFolder(
            root=dataset_path,
            transform=transform
        )
        num_classes = len(dataset_obj.classes)
        print(f"Found custom dataset '{dataset}' with {num_classes} classes: {dataset_obj.classes[:10]}{'...' if num_classes > 10 else ''}")
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: imagenet, cifar10, cifar100, mnist, fashionmnist, svhn, stl10, places365, food101, oxford_pets, flowers102, caltech101, caltech256, dtd, or custom ImageFolder in {data_dir}/{dataset}")
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return data_loader, num_classes


# Legacy function for backward compatibility
def create_standard_data_loader(data_dir="/Users/tanmoy/research/data", dataset="imagenet", 
                               batch_size=64, num_workers=4, image_size=224, split="val"):
    """Legacy function - use create_comprehensive_data_loader instead"""
    data_loader, _ = create_comprehensive_data_loader(
        data_dir=data_dir, dataset=dataset, batch_size=batch_size,
        num_workers=num_workers, image_size=image_size, split=split
    )
    return data_loader

if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"🔬 Geometric Analysis Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Device: {args.device}")
    print(f"  Use timm: {args.use_timm}")
    print(f"  Pretrained: {args.pretrained}")
    
    # Create data loader
    data_loader, dataset_num_classes = create_comprehensive_data_loader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        split=args.split
    )
    
    # Update num_classes from dataset if not explicitly set
    if args.num_classes == 1000 and dataset_num_classes != 1000:
        args.num_classes = dataset_num_classes
        print(f"Updated num_classes to {args.num_classes} based on dataset")
    
    # Run analysis based on mode
    if args.analyze_phases:
        print(f"\n🔬 Running Multi-Phase Analysis: {args.phases}")
        print("This will initialize models in different training phases and compare their geometry")
        
        # Initialize analyzer
        analyzer = GeometricAnalyzer(device=args.device)
        
        # Store results for each phase
        phase_results = {}
        phase_models = {}
        
        # Analyze each phase
        for phase in args.phases:
            print(f"\n--- Analyzing {phase.upper()} phase ---")
            
            # Load model for this phase
            model = load_model_from_phase(
                phase=phase,
                model_name=args.model,
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                use_timm=args.use_timm,
                scale_factor=args.scale_factor,
                simulate_training=args.simulate_training,
                training_steps=args.training_steps,
                checkpoint_path=args.checkpoint_path
            )
            
            model = model.to(args.device)
            phase_models[phase] = model
            
            print(f"Model phase: {phase}")
            print(f"Scale factor: {model.scale_factor}")
            
            # Run geometric analysis
            with tqdm(total=4, desc=f"Geometric Analysis [{phase}]") as pbar:
                representations = analyzer.extract_representations(model, data_loader)
                pbar.update(1)
                
                metrics = analyzer.compute_representation_metrics(representations)
                pbar.update(1)
                
                curvature = analyzer.analyze_curvature(representations)
                pbar.update(1)
                
                transitions = analyzer.analyze_layer_transitions(representations)
                pbar.update(1)
            
            phase_results[phase] = {
                'representations': representations,
                'metrics': metrics,
                'curvature': curvature,
                'transitions': transitions
            }
        
        # Cross-phase comparison
        print("\n=== Cross-Phase Analysis ===")
        comparison_results = {}
        
        # Compare alignment between phases
        for i, phase1 in enumerate(args.phases):
            for phase2 in args.phases[i+1:]:
                print(f"Comparing {phase1} vs {phase2}...")
                alignment_geom = analyzer.compute_alignment_geometry(
                    phase_models[phase1], 
                    phase_models[phase2], 
                    data_loader
                )
                comparison_results[f"{phase1}_vs_{phase2}"] = alignment_geom
        
        # Analyze key research questions
        research_insights = analyze_phase_research_questions(phase_results)
        
        # Create comprehensive visualization
        fig = visualize_phase_comparison(phase_results, args.phases)
        
        if args.save_results:
            # Save phase comparison results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(f"phase_analysis_{timestamp}")
            results_dir.mkdir(exist_ok=True)
            
            fig.savefig(results_dir / "phase_comparison.png", dpi=300, bbox_inches='tight')
            
            # Save detailed results
            with open(results_dir / "phase_analysis.json", 'w') as f:
                def convert_numpy(obj):
                    if isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy(v) for v in obj]
                    return obj
                
                json.dump({
                    'phase_results': convert_numpy(phase_results),
                    'comparison_results': convert_numpy(comparison_results),
                    'research_insights': convert_numpy(research_insights),
                    'config': {
                        'model': args.model,
                        'dataset': args.dataset,
                        'num_classes': args.num_classes,
                        'phases_analyzed': args.phases,
                        'use_timm': args.use_timm,
                        'pretrained': args.pretrained
                    }
                }, f, indent=2)
            
            print(f"Results saved to {results_dir}")
        
        
        
        insights = research_insights
        summary = insights['summary']
        
        
        for metric, ranking in summary['phase_rankings'].items():
            print(f"  {metric}: {' > '.join(ranking)}")
        
        results = {
            'phase_results': phase_results,
            'comparison_results': comparison_results,
            'research_insights': research_insights,
            'models': phase_models
        }
        
    else:
        # Load single model
        if args.checkpoint_path:
            model = load_model_from_checkpoint(args.checkpoint_path, num_classes=args.num_classes)
        elif args.use_timm:
            model = load_model_with_timm(args.model, args.num_classes, args.pretrained)
        else:
            model = load_torchvision_model(args.model, args.num_classes, args.pretrained)
        
        model = model.to(args.device)
        print(f"✓ Model loaded: {args.model}")
        
        if args.use_wandb or args.experiment_name != 'geometric_analysis':
            # Run comprehensive analysis with tracking
            results, tracker = run_comprehensive_analysis(
                model, 
                data_loader, 
                experiment_name=args.experiment_name,
                use_wandb=args.use_wandb
            )
            print(f"Comprehensive analysis completed. Results saved to: {tracker.exp_dir}")
        else:
            # Run basic geometric analysis
            results = run_geometric_analysis(model, data_loader, save_results=args.save_results)
            print("Geometric analysis completed.")
        
        # Analysis Summary
        print(f"\nAnalysis Summary:")
        print(f"- Dataset: {args.dataset}")
        print(f"- Model: {args.model}")
        print(f"- Samples analyzed: {len(data_loader.dataset)}")
        if 'geometric' in results:
            n_layers = len(results['geometric']['metrics'])
            print(f"- Layers analyzed: {n_layers}")
            
            # Show some key metrics
            first_layer = list(results['geometric']['metrics'].keys())[0]
            last_layer = list(results['geometric']['metrics'].keys())[-1]
            
            # Handle both full and minimal metrics  
            if 'pca_90' in results['geometric']['metrics'][first_layer]['intrinsic_dim']:
                first_dim = results['geometric']['metrics'][first_layer]['intrinsic_dim']['pca_90']
                last_dim = results['geometric']['metrics'][last_layer]['intrinsic_dim']['pca_90']
                print(f"- Intrinsic dimension (first layer): {first_dim}")
                print(f"- Intrinsic dimension (last layer): {last_dim}")
            else:
                # MPS minimal metrics
                first_rank = results['geometric']['metrics'][first_layer]['intrinsic_dim']['basic_rank']
                last_rank = results['geometric']['metrics'][last_layer]['intrinsic_dim']['basic_rank']
                print(f"- Basic rank (first layer): {first_rank}")
                print(f"- Basic rank (last layer): {last_rank}")
        
        elif 'metrics' in results:
            n_layers = len(results['metrics'])
            print(f"- Layers analyzed: {n_layers}")
            
            # Show some key metrics
            first_layer = list(results['metrics'].keys())[0]
            last_layer = list(results['metrics'].keys())[-1]
            
            # Handle both full and minimal metrics
            if 'pca_90' in results['metrics'][first_layer]['intrinsic_dim']:
                first_dim = results['metrics'][first_layer]['intrinsic_dim']['pca_90']
                last_dim = results['metrics'][last_layer]['intrinsic_dim']['pca_90']
                print(f"- Intrinsic dimension (first layer): {first_dim}")
                print(f"- Intrinsic dimension (last layer): {last_dim}")
            else:
                # MPS minimal metrics
                first_rank = results['metrics'][first_layer]['intrinsic_dim']['basic_rank']
                last_rank = results['metrics'][last_layer]['intrinsic_dim']['basic_rank']
                print(f"- Basic rank (first layer): {first_rank}")
                print(f"- Basic rank (last layer): {last_rank}")
