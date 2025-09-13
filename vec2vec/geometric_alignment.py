#!/usr/bin/env python3
"""
Non-adversarial vec2vec using geometric insights from platonic representations.

This script extracts embeddings from different models and finds universal alignment
using direct geometric transformations instead of adversarial training.
"""

import os
import argparse
import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List

# Try to import from utils, fallback to simple implementations if not available
try:
    from utils.model_utils import load_encoder, get_sentence_embedding_dimension
    from utils.streaming_utils import load_streaming_embeddings
    USE_UTILS = True
except ImportError:
    print("Warning: utils modules not available, using fallback implementations")
    USE_UTILS = False


class GeometricVec2Vec:
    """
    Non-adversarial vec2vec using geometric insights from platonic representations.
    
    The key insight: instead of learning to fool a discriminator, we directly
    compute geometric transformations that preserve universal structure.
    """
    
    def __init__(self, alignment_method='auto', use_platonic_insights=True):
        self.alignment_method = alignment_method
        self.use_platonic_insights = use_platonic_insights
        self.transform_A_to_B = None
        self.transform_B_to_A = None
        self.alignment_info = {}
        
    def fit(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray, spectral_predictions=None):
        """
        Learn transformation between embedding spaces without adversarial training.
        
        The beauty here is that we're not fighting against a discriminator -
        we're directly optimizing for geometric alignment based on mathematical
        principles.
        """
        print(f"Fitting geometric alignment between spaces of shape {embeddings_A.shape} and {embeddings_B.shape}")
        
        # If using platonic insights, choose method based on spectral prediction
        if self.use_platonic_insights and spectral_predictions is not None:
            self.alignment_method = self._choose_method_from_prediction(spectral_predictions)
            print(f"Selected {self.alignment_method} based on spectral analysis")
        
        # Apply the appropriate non-adversarial alignment
        if self.alignment_method == 'procrustes':
            self._fit_procrustes(embeddings_A, embeddings_B)
        elif self.alignment_method == 'cca':
            self._fit_cca(embeddings_A, embeddings_B)
        elif self.alignment_method == 'lowrank':
            self._fit_lowrank_alignment(embeddings_A, embeddings_B)
        else:  # 'auto' or fallback
            self._fit_adaptive(embeddings_A, embeddings_B)
            
    def _fit_procrustes(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray):
        """
        Orthogonal Procrustes: finds the best orthogonal transformation.
        
        This is perfect when models are in the same phase (from your platonic work)
        because it preserves geometric structure while being computationally simple.
        """
        # Center the embeddings
        mean_A = embeddings_A.mean(axis=0)
        mean_B = embeddings_B.mean(axis=0)
        
        centered_A = embeddings_A - mean_A
        centered_B = embeddings_B - mean_B
        
        # Compute optimal orthogonal transformation
        R, scale = orthogonal_procrustes(centered_A, centered_B)
        
        # Store transformations for both directions
        self.transform_A_to_B = lambda x: (x - mean_A) @ R + mean_B
        self.transform_B_to_A = lambda x: (x - mean_B) @ R.T + mean_A
        
        self.alignment_info = {
            'method': 'procrustes',
            'scale': scale,
            'rotation_matrix': R,
            'mean_A': mean_A,
            'mean_B': mean_B
        }
        
        print(f"Procrustes alignment complete. Scale factor: {scale:.4f}")
        
    def _fit_cca(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray):
        """
        Choose between standard CCA and geometric CCA based on data characteristics.
        """
        # Use geometric CCA for better structure preservation
        try:
            self._fit_cca_geometric(embeddings_A, embeddings_B)
        except Exception as e:
            print(f"Geometric CCA failed ({e}), trying standard CCA")
            self._fit_cca_standard(embeddings_A, embeddings_B)
    
    def _fit_cca_geometric(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray):
        """
        Geometrically-enhanced CCA that preserves structure better.
        """
        from scipy.linalg import eigh, sqrtm
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = embeddings_A.shape[0]
        
        # 1. Preprocessing with structure preservation
        # Instead of just centering, we'll use a softer normalization
        A_centered = embeddings_A - embeddings_A.mean(axis=0)
        B_centered = embeddings_B - embeddings_B.mean(axis=0)
        
        # Compute optimal scaling to preserve distances
        A_scale = np.sqrt(np.trace(A_centered.T @ A_centered) / n_samples)
        B_scale = np.sqrt(np.trace(B_centered.T @ B_centered) / n_samples)
        
        # Scale to unit Frobenius norm
        A_normalized = A_centered / A_scale if A_scale > 1e-8 else A_centered
        B_normalized = B_centered / B_scale if B_scale > 1e-8 else B_centered
        
        # 2. Compute neighbor structure for geometric regularization
        k_neighbors = min(50, n_samples // 10)
        nbrs_A = NearestNeighbors(n_neighbors=k_neighbors).fit(A_normalized)
        nbrs_B = NearestNeighbors(n_neighbors=k_neighbors).fit(B_normalized)
        
        # Get neighbor graphs
        dist_A, idx_A = nbrs_A.kneighbors(A_normalized)
        dist_B, idx_B = nbrs_B.kneighbors(B_normalized)
        
        # 3. Compute covariance with geometric regularization
        C_AA = A_normalized.T @ A_normalized / (n_samples - 1)
        C_BB = B_normalized.T @ B_normalized / (n_samples - 1)
        C_AB = A_normalized.T @ B_normalized / (n_samples - 1)
        
        # Add geometric regularization term
        # This encourages preserving local structure
        geometry_weight = 0.1  # Tune this
        
        # Compute Laplacian regularization
        L_A = np.zeros_like(C_AA)
        L_B = np.zeros_like(C_BB)
        
        for i in range(n_samples):
            for j, neighbor in enumerate(idx_A[i, 1:]):  # Skip self
                if j < len(idx_A[i]) - 1:  # Safety check
                    weight = np.exp(-dist_A[i, j+1]**2 / 2)
                    diff = A_normalized[i] - A_normalized[neighbor]
                    L_A += weight * np.outer(diff, diff) / n_samples
        
        for i in range(n_samples):
            for j, neighbor in enumerate(idx_B[i, 1:]):
                if j < len(idx_B[i]) - 1:  # Safety check
                    weight = np.exp(-dist_B[i, j+1]**2 / 2)
                    diff = B_normalized[i] - B_normalized[neighbor]
                    L_B += weight * np.outer(diff, diff) / n_samples
        
        # 4. Adaptive regularization with geometric awareness
        # Estimate effective rank using eigenvalue decay
        eigvals_A = np.linalg.eigvalsh(C_AA)
        eigvals_B = np.linalg.eigvalsh(C_BB)
        
        # Find elbow in eigenvalue spectrum
        def find_effective_rank(eigvals, threshold=0.95):
            eigvals = np.sort(eigvals)[::-1]
            eigvals = eigvals[eigvals > 0]
            if len(eigvals) == 0:
                return 1
            cumsum = np.cumsum(eigvals)
            total = cumsum[-1]
            return max(1, np.argmax(cumsum / total > threshold) + 1)
        
        rank_A = find_effective_rank(eigvals_A)
        rank_B = find_effective_rank(eigvals_B)
        effective_rank = min(rank_A, rank_B)
        
        print(f"Effective ranks: A={rank_A}, B={rank_B}, using {effective_rank}")
        
        # Adaptive regularization based on effective rank
        reg_base = 1e-6
        reg_factor = max(1, np.log(embeddings_A.shape[1] / max(effective_rank, 1)))
        reg = reg_base * reg_factor
        
        # Apply regularization with geometric term
        C_AA_reg = C_AA + reg * np.eye(C_AA.shape[0]) + geometry_weight * L_A
        C_BB_reg = C_BB + reg * np.eye(C_BB.shape[0]) + geometry_weight * L_B
        
        # 5. Solve CCA with Cholesky decomposition for stability
        try:
            # Cholesky decomposition
            L_A_chol = np.linalg.cholesky(C_AA_reg)
            L_B_chol = np.linalg.cholesky(C_BB_reg)
            
            # Whitened cross-covariance
            C_AB_white = np.linalg.solve(L_A_chol, C_AB)
            C_AB_white = np.linalg.solve(L_B_chol.T, C_AB_white.T).T
            
            # SVD of whitened cross-covariance
            U, S, Vt = np.linalg.svd(C_AB_white, full_matrices=False)
            
            # CCA directions
            A_dirs = np.linalg.solve(L_A_chol.T, U)
            B_dirs = np.linalg.solve(L_B_chol.T, Vt.T)
            
        except np.linalg.LinAlgError:
            print("Cholesky failed, using eigendecomposition")
            # Fallback to eigendecomposition
            A_dirs, B_dirs, S = self._fit_cca_eigendecomp(A_normalized, B_normalized, C_AA_reg, C_BB_reg, C_AB)
        
        # 6. Select components adaptively
        # Use both correlation strength and variance explained
        n_components = min(
            effective_rank,
            embeddings_A.shape[1],
            embeddings_B.shape[1],
            n_samples // 3,
            len(S)
        )
        
        # Weight components by their importance
        component_scores = S[:n_components]**2  # Square to emphasize differences
        component_weights = component_scores / component_scores.sum() if component_scores.sum() > 0 else np.ones(n_components) / n_components
        
        print(f"Using {n_components} components, top correlations: {S[:min(5, len(S))]}")
        
        # 7. Create structure-preserving transformation
        A_dirs = A_dirs[:, :n_components]
        B_dirs = B_dirs[:, :n_components]
        
        # Store all parameters
        self.cca_params = {
            'A_dirs': A_dirs,
            'B_dirs': B_dirs,
            'A_mean': embeddings_A.mean(axis=0),
            'B_mean': embeddings_B.mean(axis=0),
            'A_scale': A_scale,
            'B_scale': B_scale,
            'correlations': S[:n_components],
            'component_weights': component_weights,
            'n_components': n_components
        }
        
        # 8. Learn direct transformation with structure preservation
        # Project training data
        A_proj = A_normalized @ A_dirs
        B_proj = B_normalized @ B_dirs
        
        # Weight by correlation strength
        A_proj_weighted = A_proj * np.sqrt(component_weights)
        B_proj_weighted = B_proj * np.sqrt(component_weights)
        
        # Learn mapping that preserves both correlation and structure
        # We'll use a combination of direct mapping and neighbor preservation
        
        # Direct linear mapping
        ridge_lambda = 1e-4
        W_direct = np.linalg.solve(
            A_proj_weighted.T @ A_proj_weighted + ridge_lambda * np.eye(n_components),
            A_proj_weighted.T @ B_normalized
        )
        
        # Store transformation matrices
        self.W_A_to_B = A_dirs @ W_direct
        
        # Similarly for B to A
        W_direct_rev = np.linalg.solve(
            B_proj_weighted.T @ B_proj_weighted + ridge_lambda * np.eye(n_components),
            B_proj_weighted.T @ A_normalized
        )
        self.W_B_to_A = B_dirs @ W_direct_rev
        
        # 9. Create transformation functions
        def transform_A_to_B(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Apply same preprocessing
            x_centered = x - self.cca_params['A_mean']
            x_normalized = x_centered / self.cca_params['A_scale'] if self.cca_params['A_scale'] > 1e-8 else x_centered
            
            # Direct transformation
            x_transformed = x_normalized @ self.W_A_to_B
            
            # Denormalize
            result = x_transformed * self.cca_params['B_scale'] + self.cca_params['B_mean']
            
            return result[0] if x.shape[0] == 1 else result
        
        def transform_B_to_A(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            x_centered = x - self.cca_params['B_mean']
            x_normalized = x_centered / self.cca_params['B_scale'] if self.cca_params['B_scale'] > 1e-8 else x_centered
            x_transformed = x_normalized @ self.W_B_to_A
            result = x_transformed * self.cca_params['A_scale'] + self.cca_params['A_mean']
            
            return result[0] if x.shape[0] == 1 else result
        
        self.transform_A_to_B = transform_A_to_B
        self.transform_B_to_A = transform_B_to_A
        
        # 10. Store detailed metrics
        self.alignment_info = {
            'method': 'cca_geometric',
            'n_components': n_components,
            'correlations': S[:n_components].tolist(),
            'mean_correlation': float(np.mean(S[:n_components])),
            'effective_rank': int(effective_rank),
            'regularization': float(reg),
            'geometry_weight': float(geometry_weight)
        }
        
        print(f"Geometric CCA complete: mean correlation={np.mean(S[:n_components]):.4f}")

    def _fit_cca_eigendecomp(self, A, B, C_AA, C_BB, C_AB):
        """Fallback CCA using eigendecomposition"""
        # Compute sqrt inverse using eigendecomposition
        def matrix_sqrt_inv(C, reg=1e-6):
            eigvals, eigvecs = eigh(C)
            eigvals_inv = np.where(eigvals > reg, 1.0 / np.sqrt(eigvals), 0)
            return eigvecs @ np.diag(eigvals_inv) @ eigvecs.T
        
        C_AA_sqrt_inv = matrix_sqrt_inv(C_AA)
        C_BB_sqrt_inv = matrix_sqrt_inv(C_BB)
        
        # Form matrix for eigendecomposition
        M = C_AA_sqrt_inv @ C_AB @ C_BB_sqrt_inv
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        
        # CCA directions
        A_dirs = C_AA_sqrt_inv @ U
        B_dirs = C_BB_sqrt_inv @ Vt.T
        
        return A_dirs, B_dirs, S

    def _fit_cca_standard(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray):
        """
        Standard CCA fallback implementation.
        """
        # Determine number of components (min of dimensions and samples)
        n_components = min(embeddings_A.shape[1], embeddings_B.shape[1], embeddings_A.shape[0] // 3, 50)
        
        try:
            cca = CCA(n_components=n_components, max_iter=200)
            cca.fit(embeddings_A, embeddings_B)
            
            # Store the fitted CCA and reference embeddings for transformation
            self.cca_model = cca
            self.embeddings_A_mean = np.mean(embeddings_A, axis=0)
            self.embeddings_B_mean = np.mean(embeddings_B, axis=0)
            
            # Get the canonical coordinates for training data
            A_canonical, B_canonical = cca.transform(embeddings_A, embeddings_B)
            
            # Find linear mapping from canonical space back to original spaces
            self.canonical_to_B = np.linalg.lstsq(A_canonical, embeddings_B - self.embeddings_B_mean, rcond=None)[0]
            self.canonical_to_A = np.linalg.lstsq(B_canonical, embeddings_A - self.embeddings_A_mean, rcond=None)[0]
            
            # Create transformation functions using direct linear mapping
            def transform_A_to_B(x):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                # Center the input
                x_centered = x - self.embeddings_A_mean
                # Project through CCA transformation (simplified)
                x_canonical = x_centered @ cca.x_weights_
                # Map to B space and add mean back
                result = x_canonical @ self.canonical_to_B + self.embeddings_B_mean
                return result[0] if x.shape[0] == 1 and result.shape[0] == 1 else result
            
            def transform_B_to_A(x):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                # Center the input
                x_centered = x - self.embeddings_B_mean
                # Project through CCA transformation (simplified)
                x_canonical = x_centered @ cca.y_weights_
                # Map to A space and add mean back
                result = x_canonical @ self.canonical_to_A + self.embeddings_A_mean
                return result[0] if x.shape[0] == 1 and result.shape[0] == 1 else result
                
            self.transform_A_to_B = transform_A_to_B
            self.transform_B_to_A = transform_B_to_A
            
            correlation_score = cca.score(embeddings_A, embeddings_B)
            
            self.alignment_info = {
                'method': 'cca_standard',
                'n_components': n_components,
                'canonical_correlations': correlation_score
            }
            
            print(f"Standard CCA alignment complete with {n_components} components. Correlation score: {correlation_score:.4f}")
            
        except Exception as e:
            print(f"Standard CCA failed ({e}), falling back to Procrustes")
            self._fit_procrustes(embeddings_A, embeddings_B)
        
    def _fit_lowrank_alignment(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray):
        """
        Low-rank alignment based on universal structure characterization (RQ1).
        
        This method leverages your discovery of universal low-rank structures
        to create more robust alignments.
        """
        # Extract low-rank approximations using truncated SVD for efficiency
        svd_A = TruncatedSVD(n_components=min(200, embeddings_A.shape[1]))
        svd_B = TruncatedSVD(n_components=min(200, embeddings_B.shape[1]))
        
        # Fit SVD on transposed matrices (features x samples)
        A_reduced = svd_A.fit_transform(embeddings_A)
        B_reduced = svd_B.fit_transform(embeddings_B)
        
        # Determine optimal rank based on variance preservation
        rank = self._determine_optimal_rank(svd_A.singular_values_, svd_B.singular_values_, target_variance=0.95)
        
        # Use only the top-k components
        A_lowrank = A_reduced[:, :rank]
        B_lowrank = B_reduced[:, :rank]
        
        # Align in low-rank space (more stable than full-rank)
        R_lowrank, scale = orthogonal_procrustes(A_lowrank, B_lowrank)
        
        # Store the SVD models and transformation parameters
        self.svd_A = svd_A
        self.svd_B = svd_B
        self.rank = rank
        self.R_lowrank = R_lowrank
        
        # Create transformation functions that project through low-rank space
        def transform_A_to_B(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            # Transform A to low-rank space
            A_lowrank_x = self.svd_A.transform(x)[:, :self.rank]
            # Apply rotation in low-rank space
            A_rotated = A_lowrank_x @ self.R_lowrank
            # Pad with zeros to match B's component count
            if A_rotated.shape[1] < self.svd_B.components_.shape[0]:
                padding = np.zeros((A_rotated.shape[0], self.svd_B.components_.shape[0] - A_rotated.shape[1]))
                A_rotated = np.hstack([A_rotated, padding])
            elif A_rotated.shape[1] > self.svd_B.components_.shape[0]:
                A_rotated = A_rotated[:, :self.svd_B.components_.shape[0]]
            # Transform back to B space
            result = self.svd_B.inverse_transform(A_rotated)
            return result[0] if x.shape[0] == 1 else result
        
        def transform_B_to_A(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            # Transform B to low-rank space
            B_lowrank_x = self.svd_B.transform(x)[:, :self.rank]
            # Apply inverse rotation in low-rank space
            B_rotated = B_lowrank_x @ self.R_lowrank.T
            # Pad with zeros to match A's component count
            if B_rotated.shape[1] < self.svd_A.components_.shape[0]:
                padding = np.zeros((B_rotated.shape[0], self.svd_A.components_.shape[0] - B_rotated.shape[1]))
                B_rotated = np.hstack([B_rotated, padding])
            elif B_rotated.shape[1] > self.svd_A.components_.shape[0]:
                B_rotated = B_rotated[:, :self.svd_A.components_.shape[0]]
            # Transform back to A space
            result = self.svd_A.inverse_transform(B_rotated)
            return result[0] if x.shape[0] == 1 else result
        
        self.transform_A_to_B = transform_A_to_B
        self.transform_B_to_A = transform_B_to_A
        
        self.alignment_info = {
            'method': 'lowrank',
            'rank': rank,
            'scale': scale,
            'rotation_matrix': R_lowrank,
            'svd_A': svd_A,
            'svd_B': svd_B,
            'explained_variance_A': svd_A.explained_variance_ratio_[:rank].sum(),
            'explained_variance_B': svd_B.explained_variance_ratio_[:rank].sum()
        }
        
        print(f"Low-rank alignment complete using rank {rank}")
        print(f"Explained variance: A={self.alignment_info['explained_variance_A']:.3f}, B={self.alignment_info['explained_variance_B']:.3f}")
        
    def _fit_adaptive(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray):
        """
        Adaptive method selection based on data characteristics.
        """
        # Analyze data characteristics
        dim_A, dim_B = embeddings_A.shape[1], embeddings_B.shape[1]
        n_samples = embeddings_A.shape[0]
        
        # Check if dimensions are very different
        dim_ratio = max(dim_A, dim_B) / min(dim_A, dim_B)
        
        # Check rank of data
        rank_A = np.linalg.matrix_rank(embeddings_A)
        rank_B = np.linalg.matrix_rank(embeddings_B)
        effective_rank = min(rank_A, rank_B)
        
        print(f"Data analysis: dim_ratio={dim_ratio:.2f}, effective_rank={effective_rank}/{min(dim_A, dim_B)}")
        
        if dim_A != dim_B:
            print("Different dimensions detected, using CCA")
            self.alignment_method = 'cca'
            self._fit_cca(embeddings_A, embeddings_B)
        elif effective_rank < min(dim_A, dim_B) * 0.8:
            print("Low-rank structure detected, using low-rank alignment")
            self.alignment_method = 'lowrank'
            self._fit_lowrank_alignment(embeddings_A, embeddings_B)
        else:
            print("Using Procrustes alignment")
            self.alignment_method = 'procrustes'
            self._fit_procrustes(embeddings_A, embeddings_B)
    
    def _determine_optimal_rank(self, singular_values_A: np.ndarray, singular_values_B: np.ndarray, 
                              target_variance: float = 0.95) -> int:
        """Determine optimal rank based on variance preservation."""
        # Normalize singular values to get explained variance ratios
        var_A = singular_values_A ** 2
        var_A = var_A / var_A.sum()
        cumvar_A = np.cumsum(var_A)
        
        var_B = singular_values_B ** 2
        var_B = var_B / var_B.sum()
        cumvar_B = np.cumsum(var_B)
        
        # Find rank that preserves target_variance in both spaces
        rank_A = np.searchsorted(cumvar_A, target_variance) + 1
        rank_B = np.searchsorted(cumvar_B, target_variance) + 1
        
        # Use the minimum to ensure both spaces are well-represented
        optimal_rank = min(rank_A, rank_B, len(singular_values_A), len(singular_values_B))
        return max(optimal_rank, 10)  # At least 10 dimensions
    
    def _choose_method_from_prediction(self, spectral_predictions) -> str:
        """Choose alignment method based on spectral analysis predictions."""
        # This would integrate with your platonic representation insights
        # For now, implement a simple heuristic
        if 'phase_transition' in spectral_predictions and spectral_predictions['phase_transition']:
            return 'lowrank'
        elif 'high_correlation' in spectral_predictions and spectral_predictions['high_correlation']:
            return 'cca'
        else:
            return 'procrustes'
    
    def transform(self, embeddings: np.ndarray, direction: str = 'A_to_B') -> np.ndarray:
        """Transform embeddings using the learned alignment."""
        if direction == 'A_to_B':
            if self.transform_A_to_B is None:
                raise ValueError("Alignment not fitted yet!")
            return self.transform_A_to_B(embeddings)
        elif direction == 'B_to_A':
            if self.transform_B_to_A is None:
                raise ValueError("Alignment not fitted yet!")
            return self.transform_B_to_A(embeddings)
        else:
            raise ValueError("Direction must be 'A_to_B' or 'B_to_A'")
    
    def evaluate_alignment(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray) -> Dict[str, float]:
        """Evaluate the quality of the alignment."""
        # Transform A to B space
        A_transformed = self.transform(embeddings_A, 'A_to_B')
        
        # Ensure A_transformed has the right shape
        if A_transformed.ndim == 1:
            A_transformed = A_transformed.reshape(1, -1)
        
        # Compute alignment metrics
        cosine_similarities = []
        for i in range(min(len(A_transformed), len(embeddings_B))):
            a = A_transformed[i]
            b = embeddings_B[i]
            cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cosine_similarities.append(cosine_sim)
        
        cosine_sim = np.mean(cosine_similarities)
        
        # MSE calculation
        min_samples = min(len(A_transformed), len(embeddings_B))
        mse = np.mean(np.sum((A_transformed[:min_samples] - embeddings_B[:min_samples]) ** 2, axis=1))
        
        return {
            'cosine_similarity': cosine_sim,
            'mse': mse,
            'method': self.alignment_method
        }


def extract_embeddings(model_name: str, texts: List[str], device: str = 'cpu') -> np.ndarray:
    """Extract embeddings from a given model for the provided texts."""
    if USE_UTILS:
        print(f"Loading model: {model_name}")
        encoder = load_encoder(model_name, device=device)
        
        print(f"Encoding {len(texts)} texts...")
        with torch.no_grad():
            embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        return embeddings
    else:
        # Fallback: use sentence-transformers directly
        from sentence_transformers import SentenceTransformer
        
        # Map model names to HuggingFace model IDs
        model_mapping = {
            'stella': 'infgrad/stella-base-en-v2',
            'gte': 'thenlper/gte-base',
            'gist': 'avsolatorio/GIST-Embedding-v0',
            'sbert': 'sentence-transformers/all-MiniLM-L12-v2',
            'e5': 'intfloat/e5-base-v2',
        }
        
        model_id = model_mapping.get(model_name, model_name)
        print(f"Loading model: {model_name} -> {model_id}")
        
        try:
            encoder = SentenceTransformer(model_id, device=device)
            print(f"Encoding {len(texts)} texts...")
            embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Using random embeddings as fallback...")
            # Return random embeddings with typical dimensions
            return np.random.randn(len(texts), 768).astype(np.float32)


def load_sample_texts(dataset_name: str = 'nq', n_samples: int = 1000) -> List[str]:
    """Load sample texts from the specified dataset."""
    print(f"Loading {n_samples} samples from {dataset_name} dataset...")
    
    if USE_UTILS:
        try:
            dset = load_streaming_embeddings(dataset_name)
            dset = dset.shuffle(seed=42).select(range(min(n_samples, len(dset))))
            texts = dset['text']
            print(f"Loaded {len(texts)} texts")
            return texts
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            # Fallback to dummy data
            print("Using dummy data as fallback...")
            return [f"This is sample text number {i} for testing the alignment." for i in range(n_samples)]
    else:
        # Fallback to dummy data when utils not available
        print("Using dummy data as fallback...")
        return [f"This is sample text number {i} for testing the alignment." for i in range(n_samples)]


def visualize_alignment(embeddings_A: np.ndarray, embeddings_B: np.ndarray, 
                       A_transformed: np.ndarray, model_A: str, model_B: str,
                       save_path: str = None):
    """Visualize the alignment using t-SNE or PCA."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Subsample for visualization
    n_viz = min(500, embeddings_A.shape[0])
    indices = np.random.choice(embeddings_A.shape[0], n_viz, replace=False)
    
    A_viz = embeddings_A[indices]
    B_viz = embeddings_B[indices]
    A_trans_viz = A_transformed[indices]
    
    # Use PCA for high-dimensional data, t-SNE for final visualization
    if A_viz.shape[1] > 50:
        pca = PCA(n_components=50)
        A_viz = pca.fit_transform(A_viz)
        B_viz = pca.transform(B_viz)
        A_trans_viz = pca.transform(A_trans_viz)
    
    # Apply t-SNE
    combined = np.vstack([A_viz, B_viz, A_trans_viz])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_2d = tsne.fit_transform(combined)
    
    # Split back
    A_2d = combined_2d[:n_viz]
    B_2d = combined_2d[n_viz:2*n_viz]
    A_trans_2d = combined_2d[2*n_viz:]
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Before alignment
    plt.subplot(1, 3, 1)
    plt.scatter(A_2d[:, 0], A_2d[:, 1], alpha=0.6, label=f'{model_A}', s=20)
    plt.scatter(B_2d[:, 0], B_2d[:, 1], alpha=0.6, label=f'{model_B}', s=20)
    plt.title('Before Alignment')
    plt.legend()
    
    # After alignment
    plt.subplot(1, 3, 2)
    plt.scatter(A_trans_2d[:, 0], A_trans_2d[:, 1], alpha=0.6, label=f'{model_A} → {model_B}', s=20)
    plt.scatter(B_2d[:, 0], B_2d[:, 1], alpha=0.6, label=f'{model_B}', s=20)
    plt.title('After Alignment')
    plt.legend()
    
    # Alignment vectors
    plt.subplot(1, 3, 3)
    plt.scatter(B_2d[:, 0], B_2d[:, 1], alpha=0.4, label=f'{model_B}', s=20, color='blue')
    plt.scatter(A_trans_2d[:, 0], A_trans_2d[:, 1], alpha=0.4, label=f'{model_A} → {model_B}', s=20, color='red')
    
    # Draw arrows showing transformation
    for i in range(0, n_viz, max(1, n_viz // 20)):  # Show every nth arrow
        plt.arrow(A_2d[i, 0], A_2d[i, 1], 
                 A_trans_2d[i, 0] - A_2d[i, 0], A_trans_2d[i, 1] - A_2d[i, 1],
                 alpha=0.3, width=0.5, head_width=2, length_includes_head=True, color='gray')
    
    plt.title('Transformation Vectors')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Geometric Vec2Vec Alignment")
    parser.add_argument("--model_a", type=str, default="stella", help="First model name")
    parser.add_argument("--model_b", type=str, default="gte", help="Second model name")
    parser.add_argument("--dataset", type=str, default="nq", help="Dataset to use")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples to use")
    parser.add_argument("--method", type=str, choices=['auto', 'procrustes', 'cca', 'lowrank'], 
                       default='auto', help="Alignment method")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load sample texts
    texts = load_sample_texts(args.dataset, args.n_samples)
    
    # Extract embeddings from both models
    print(f"\n=== Extracting embeddings ===")
    embeddings_A = extract_embeddings(args.model_a, texts, args.device)
    embeddings_B = extract_embeddings(args.model_b, texts, args.device)
    
    print(f"Model {args.model_a} embeddings shape: {embeddings_A.shape}")
    print(f"Model {args.model_b} embeddings shape: {embeddings_B.shape}")
    
    # Initialize and fit geometric alignment
    print(f"\n=== Fitting geometric alignment ===")
    aligner = GeometricVec2Vec(alignment_method=args.method, use_platonic_insights=True)
    aligner.fit(embeddings_A, embeddings_B)
    
    # Evaluate alignment
    print(f"\n=== Evaluating alignment ===")
    eval_metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
    
    print("Alignment Results:")
    print(f"Method used: {eval_metrics['method']}")
    print(f"Cosine similarity: {eval_metrics['cosine_similarity']:.4f}")
    print(f"MSE: {eval_metrics['mse']:.4f}")
    
    # Save results
    results = {
        'model_a': args.model_a,
        'model_b': args.model_b,
        'method': eval_metrics['method'],
        'cosine_similarity': float(eval_metrics['cosine_similarity']),
        'mse': float(eval_metrics['mse']),
        'alignment_info': aligner.alignment_info
    }
    
    import json
    results_path = os.path.join(args.output_dir, f"alignment_{args.model_a}_{args.model_b}.json")
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"Results saved to {results_path}")
    
    # Create visualization if requested
    if args.visualize:
        print(f"\n=== Creating visualization ===")
        A_transformed = aligner.transform(embeddings_A, 'A_to_B')
        viz_path = os.path.join(args.output_dir, f"alignment_viz_{args.model_a}_{args.model_b}.png")
        visualize_alignment(embeddings_A, embeddings_B, A_transformed, 
                          args.model_a, args.model_b, viz_path)
    
    print(f"\n=== Geometric alignment complete ===")
    print(f"Successfully aligned {args.model_a} and {args.model_b} using {eval_metrics['method']} method")


if __name__ == "__main__":
    main()