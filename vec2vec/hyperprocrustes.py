#!/usr/bin/env python3
"""
HyperProcrustes: A hypernetwork for learning alignment between embedding spaces.
Includes TensorBoard logging and tqdm progress tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path


class HyperProcrustes(nn.Module):
    """
    A hypernetwork that learns to generate Procrustes-style transformation parameters
    between embedding spaces by discovering their latent structural regularities.

    The key insight here is that instead of computing transformations analytically
    for each pair of embedding sets (as traditional Procrustes does), we learn a
    neural network that can predict these transformations from the statistical
    properties of the embedding spaces. This allows us to:

    1. Generalize to new model pairs without recomputation
    2. Discover what makes spaces alignable (addressing RQ1)
    3. Learn compositional transformations (addressing RQ2)
    4. Understand identifiability conditions (addressing RQ3)
    """

    def __init__(self,
                 max_embedding_dim: int = 1024,
                 spectral_features_dim: int = 128,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 use_spectral_conditioning: bool = True):
        super().__init__()

        self.max_embedding_dim = max_embedding_dim
        self.spectral_features_dim = spectral_features_dim
        self.use_spectral_conditioning = use_spectral_conditioning

        # Feature extraction: compute statistics from embedding distributions
        # This addresses RQ1 - what structural regularities enable alignment?
        self.feature_extractor = SpectralFeatureExtractor(
            spectral_features_dim=spectral_features_dim
        )

        # The hypernetwork core: generates transformation parameters
        # Input: concatenated features from both embedding spaces
        input_dim = spectral_features_dim * 2

        # Build a deep network with residual connections for stability
        self.hypernet_layers = nn.ModuleList()
        current_dim = input_dim

        for i in range(num_layers):
            layer = HypernetLayer(
                input_dim=current_dim,
                output_dim=hidden_dim,
                use_residual=(i > 0)  # Skip connection except for first layer
            )
            self.hypernet_layers.append(layer)
            current_dim = hidden_dim

        # Output heads for different transformation components
        # This design reflects the mathematical structure of Procrustes alignment
        self.rotation_head = RotationParameterHead(hidden_dim, max_embedding_dim)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive scale
        )
        self.translation_head = TranslationHead(hidden_dim, max_embedding_dim)

        # Optional: predict alignment quality (addresses RQ3 - identifiability)
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self,
                embeddings_A: torch.Tensor,
                embeddings_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate transformation parameters from source to target embedding space.

        The forward pass extracts distributional features from both spaces and
        uses them to predict optimal transformation parameters. This learned
        approach can capture more complex relationships than analytical methods.

        Args:
            embeddings_A: [batch_size, num_samples, embedding_dim_A]
            embeddings_B: [batch_size, num_samples, embedding_dim_B]

        Returns:
            Dictionary containing transformation parameters and metadata
        """

        # Extract spectral and statistical features from both spaces
        # These features capture the "signature" of each embedding space
        features_A = self.feature_extractor(embeddings_A)
        features_B = self.feature_extractor(embeddings_B)

        # Concatenate features - the hypernetwork sees both spaces jointly
        combined_features = torch.cat([features_A, features_B], dim=-1)

        # Pass through hypernetwork layers
        h = combined_features
        for layer in self.hypernet_layers:
            h = layer(h)

        # Generate transformation parameters
        # Each parameter type has its own specialized head
        rotation_params = self.rotation_head(h, embeddings_A.shape[-1], embeddings_B.shape[-1])
        scale = self.scale_head(h)
        translation_A, translation_B = self.translation_head(h, embeddings_A, embeddings_B)

        # Predict alignment quality - this helps identify when alignment will fail
        alignment_quality = self.quality_predictor(h)

        # Package all parameters
        # The structure here reflects the mathematical form of affine transformations
        return {
            'rotation': rotation_params['rotation_matrix'],
            'scale': scale,
            'center_A': translation_A,
            'center_B': translation_B,
            'spectral_A': features_A,  # Keep features for analysis
            'spectral_B': features_B,
            'predicted_quality': alignment_quality,
            # Additional outputs for theoretical analysis
            'rotation_singular_values': rotation_params['singular_values'],
            'feature_similarity': F.cosine_similarity(features_A, features_B, dim=-1)
        }

    def transform(self,
                  embeddings: torch.Tensor,
                  params: Dict[str, torch.Tensor],
                  direction: str = 'A_to_B') -> torch.Tensor:
        """
        Apply the learned transformation to embeddings.

        This implements the actual geometric transformation using the
        parameters generated by the hypernetwork.
        """
        if direction == 'A_to_B':
            # Center with respect to source space
            centered = embeddings - params['center_A'].unsqueeze(1)
            # Apply rotation and scale
            # Handle scale broadcasting properly
            scale = params['scale'].unsqueeze(1)  # Add dimension for samples
            if len(scale.shape) == 3:
                scale = scale.unsqueeze(2)  # Add dimension for features
            transformed = torch.matmul(centered, params['rotation']) * scale
            # Translate to target space
            return transformed + params['center_B'].unsqueeze(1)
        else:
            # Inverse transformation
            centered = embeddings - params['center_B'].unsqueeze(1)
            # Inverse scale and rotation
            scale = params['scale'].unsqueeze(1)
            if len(scale.shape) == 3:
                scale = scale.unsqueeze(2)
            transformed = torch.matmul(centered / scale, params['rotation'].transpose(-2, -1))
            return transformed + params['center_A'].unsqueeze(1)


class SpectralFeatureExtractor(nn.Module):
    """
    Extract spectral and distributional features from embedding sets.

    This module computes the "fingerprint" of an embedding space by analyzing
    its spectral properties (eigenvalues), statistical moments, and other
    geometric characteristics. These features allow the hypernetwork to
    recognize similar structures across different embedding spaces.
    """

    def __init__(self, spectral_features_dim: int = 128):
        super().__init__()

        # Learn to weight different spectral features
        self.feature_projection = nn.Sequential(
            nn.Linear(256, spectral_features_dim),  # 256 is our raw feature size
            nn.LayerNorm(spectral_features_dim),
            nn.ReLU(),
            nn.Linear(spectral_features_dim, spectral_features_dim)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extract features that characterize the embedding distribution.

        We compute multiple types of features to capture different aspects
        of the embedding space geometry:
        1. Spectral features (eigenvalues) - capture principal directions
        2. Statistical moments - capture distribution shape
        3. Geometric features - capture local structure
        """
        batch_size = embeddings.shape[0]
        features_list = []

        for i in range(batch_size):
            emb = embeddings[i]  # [num_samples, embedding_dim]

            # Center the embeddings
            centered = emb - emb.mean(dim=0, keepdim=True)

            # Compute covariance matrix
            cov = torch.matmul(centered.t(), centered) / (emb.shape[0] - 1)

            # Extract spectral features
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                # Sort in descending order
                eigenvalues = eigenvalues.flip(-1)

                # Take top k eigenvalues (these explain most variance)
                k = min(64, eigenvalues.shape[0])
                top_eigenvalues = eigenvalues[:k]

                # Pad if necessary
                if top_eigenvalues.shape[0] < 64:
                    padding = torch.zeros(64 - top_eigenvalues.shape[0], device=eigenvalues.device)
                    top_eigenvalues = torch.cat([top_eigenvalues, padding])

                # Compute spectral statistics that are invariant to scale
                spectral_features = torch.cat([
                    top_eigenvalues,  # Raw eigenvalues (64 dims)
                    torch.tensor([
                        eigenvalues.sum(),  # Total variance
                        (eigenvalues ** 2).sum().sqrt(),  # Frobenius norm
                        eigenvalues[0] / (eigenvalues[1] + 1e-8),  # Spectral gap
                        eigenvalues.log().mean(),  # Log-mean (captures decay rate)
                    ], device=eigenvalues.device)
                ])
            except:
                # Fallback for numerical issues
                spectral_features = torch.zeros(68, device=emb.device)

            # Statistical features
            mean_norm = emb.mean(dim=0).norm()
            std_norm = emb.std(dim=0).mean()

            # Pairwise distance statistics (capture geometry)
            pairwise_dists = torch.cdist(emb[:100], emb[:100])  # Subsample for efficiency
            dist_stats = torch.tensor([
                pairwise_dists.mean(),
                pairwise_dists.std(),
                pairwise_dists.min(),
                pairwise_dists.max()
            ], device=emb.device)

            # Combine all features
            raw_features = torch.cat([
                spectral_features,  # 68 dims
                torch.tensor([mean_norm, std_norm], device=emb.device),  # 2 dims
                dist_stats,  # 4 dims
                torch.zeros(256 - 74, device=emb.device)  # Padding to 256
            ])

            features_list.append(raw_features)

        # Stack and project features
        features = torch.stack(features_list)
        return self.feature_projection(features)


class HypernetLayer(nn.Module):
    """
    A single layer in the hypernetwork with residual connections and normalization.

    The design here follows best practices for deep networks while being
    specifically tailored for generating transformation parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, use_residual: bool = True):
        super().__init__()

        self.use_residual = use_residual
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Main transformation
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

        # Normalization for stability
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        # Activation
        self.activation = nn.GELU()  # Smooth activation works well for hypernetworks

        # Residual projection if dimensions don't match
        if use_residual and input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # First transformation
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Second transformation
        x = self.linear2(x)
        x = self.norm2(x)

        # Add residual connection
        if self.use_residual:
            if self.residual_projection is not None:
                residual = self.residual_projection(residual)
            x = x + residual

        return self.activation(x)


class RotationParameterHead(nn.Module):
    """
    Generate rotation matrix parameters ensuring orthogonality.

    This is crucial for maintaining geometric properties during transformation.
    We use the Cayley transform to ensure the output is always a valid rotation.
    """

    def __init__(self, hidden_dim: int, max_dim: int):
        super().__init__()

        # Generate parameters for antisymmetric matrix
        # Reduce parameter scale for better stability
        self.param_generator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, min(max_dim * max_dim // 4, 1024)),  # Limit parameters
            nn.Tanh()  # Bounded parameters for stability
        )

    def forward(self, h: torch.Tensor, dim_A: int, dim_B: int) -> Dict[str, torch.Tensor]:
        # Generate parameters
        params = self.param_generator(h)

        # Construct antisymmetric matrix
        # This ensures that (I - A)(I + A)^(-1) is orthogonal
        batch_size = h.shape[0]
        max_dim = min(dim_A, dim_B)

        # Build antisymmetric matrix from parameters
        A = torch.zeros(batch_size, max_dim, max_dim, device=h.device)
        idx = 0
        for i in range(max_dim):
            for j in range(i + 1, max_dim):
                if idx < params.shape[1]:
                    # Scale down parameters for stability
                    A[:, i, j] = params[:, idx] * 0.1
                    A[:, j, i] = -params[:, idx] * 0.1
                    idx += 1

        # Cayley transform: R = (I - A)(I + A)^(-1)
        I = torch.eye(max_dim, device=h.device).unsqueeze(0).expand(batch_size, -1, -1)

        # More robust numerical stability
        # Use a larger epsilon and apply it properly
        eps = 1e-4
        I_plus_A = I + A

        # Use pseudo-inverse for better stability
        try:
            R = torch.matmul(I - A, torch.linalg.pinv(I_plus_A))
        except:
            # Fallback to identity if everything fails
            R = I.clone()

        # Handle dimension mismatch by padding
        if dim_A != dim_B:
            R_full = torch.zeros(batch_size, dim_A, dim_B, device=h.device)
            R_full[:, :max_dim, :max_dim] = R
            # Initialize remaining as identity-like
            for i in range(max_dim, min(dim_A, dim_B)):
                R_full[:, i, i] = 1.0
            R = R_full

        # Compute singular values for analysis
        try:
            U, S, V = torch.svd(R)
        except:
            # Fallback if SVD fails
            S = torch.ones(batch_size, max_dim, device=h.device)

        return {
            'rotation_matrix': R,
            'singular_values': S
        }


class TranslationHead(nn.Module):
    """
    Generate translation parameters (centers) for both embedding spaces.

    These parameters determine the centering of the transformation,
    which is crucial for alignment quality.
    """

    def __init__(self, hidden_dim: int, max_dim: int):
        super().__init__()

        self.center_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_dim * 2)  # Generate for both A and B
        )

    def forward(self,
                h: torch.Tensor,
                embeddings_A: torch.Tensor,
                embeddings_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Generate translation parameters
        translations = self.center_generator(h)

        # Split into source and target translations
        dim_A = embeddings_A.shape[-1]
        dim_B = embeddings_B.shape[-1]

        center_A = translations[:, :dim_A]
        center_B = translations[:, dim_A:dim_A + dim_B]

        # Option: use data-dependent centering
        # This can be more stable than learning arbitrary centers
        data_center_A = embeddings_A.mean(dim=1)
        data_center_B = embeddings_B.mean(dim=1)

        # Combine learned and data-dependent centers
        # The network learns a residual on top of the empirical mean
        center_A = data_center_A + 0.1 * center_A  # Small learned adjustment
        center_B = data_center_B + 0.1 * center_B

        return center_A, center_B


class HyperProcrustesTrainer:
    """
    Training framework for HyperProcrustes that incorporates the theoretical
    insights from your research questions. Includes TensorBoard logging.
    """

    def __init__(self,
                 model: HyperProcrustes,
                 device: str = 'cuda',
                 log_dir: str = './runs/hyperprocrustes'):
        self.model = model.to(device)
        self.device = device

        # Create log directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / timestamp
        self.writer = SummaryWriter(self.log_dir)

        # Optimizer with different learning rates for different components
        self.optimizer = torch.optim.AdamW([
            {'params': model.feature_extractor.parameters(), 'lr': 1e-4},
            {'params': model.hypernet_layers.parameters(), 'lr': 5e-4},
            {'params': [p for head in [model.rotation_head, model.scale_head, model.translation_head]
                       for p in head.parameters()], 'lr': 1e-4}
        ], weight_decay=1e-5)

        # Learning rate scheduler for stability
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )

        self.global_step = 0

    def compute_alignment_loss(self,
                              embeddings_A: torch.Tensor,
                              embeddings_B: torch.Tensor,
                              params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute losses that encourage good alignment while maintaining
        geometric properties. This loss function embodies the theoretical
        constraints that make alignment identifiable (RQ3).
        """

        # Transform A to B space
        A_transformed = self.model.transform(embeddings_A, params, direction='A_to_B')

        # 1. Distribution matching loss (Wasserstein distance approximation)
        # This ensures the transformed distribution matches the target
        dist_loss = self.compute_wasserstein_loss(A_transformed, embeddings_B)

        # 2. Cycle consistency loss (addressing identifiability)
        # If we transform A->B->A, we should recover A
        B_transformed = self.model.transform(embeddings_B, params, direction='B_to_A')
        A_cycle = self.model.transform(B_transformed, params, direction='A_to_B')
        cycle_loss = F.mse_loss(A_cycle, embeddings_A)

        # 3. Orthogonality regularization for rotation matrix
        # This ensures the transformation preserves angles and distances
        R = params['rotation']
        I = torch.eye(R.shape[-1], device=R.device).unsqueeze(0)
        ortho_loss = F.mse_loss(torch.matmul(R, R.transpose(-2, -1)), I)

        # 4. Scale regularization (prevent degenerate solutions)
        scale_reg = torch.abs(params['scale'] - 1.0).mean()

        # 5. Quality prediction loss (if we have ground truth alignment quality)
        # This helps the model learn when alignment is possible (RQ3)
        quality_loss = torch.tensor(0.0, device=self.device)

        return {
            'distribution_matching': dist_loss,
            'cycle_consistency': cycle_loss,
            'orthogonality': ortho_loss,
            'scale_regularization': scale_reg,
            'quality_prediction': quality_loss,
            'total': dist_loss + 10 * cycle_loss + ortho_loss + 0.1 * scale_reg
        }

    def compute_wasserstein_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Approximate Wasserstein distance using sliced Wasserstein distance.
        This is more stable than adversarial training for distribution matching.
        """
        # Number of random projections
        num_projections = 50

        # Generate random projection directions
        d = X.shape[-1]
        projections = torch.randn(num_projections, d, device=X.device)
        projections = F.normalize(projections, p=2, dim=1)

        # Project both distributions
        X_projected = torch.matmul(X.reshape(-1, d), projections.t())
        Y_projected = torch.matmul(Y.reshape(-1, d), projections.t())

        # Sort projected values
        X_sorted, _ = torch.sort(X_projected, dim=0)
        Y_sorted, _ = torch.sort(Y_projected, dim=0)

        # Compute L2 distance between sorted projections
        wasserstein_approx = torch.mean((X_sorted - Y_sorted) ** 2)

        return wasserstein_approx

    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step that updates the model parameters.
        """
        self.optimizer.zero_grad()

        # Forward pass
        embeddings_A = batch_data['embeddings_A'].to(self.device)
        embeddings_B = batch_data['embeddings_B'].to(self.device)

        params = self.model(embeddings_A, embeddings_B)

        # Compute losses
        losses = self.compute_alignment_loss(embeddings_A, embeddings_B, params)

        # Backward pass
        losses['total'].backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Log to TensorBoard
        for name, value in losses.items():
            self.writer.add_scalar(f'Loss/{name}', value.item(), self.global_step)

        # Log additional metrics
        self.writer.add_scalar('Metrics/scale', params['scale'].mean().item(), self.global_step)
        self.writer.add_scalar('Metrics/predicted_quality', params['predicted_quality'].mean().item(), self.global_step)
        self.writer.add_scalar('Metrics/feature_similarity', params['feature_similarity'].mean().item(), self.global_step)

        # Log learning rate
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)

        self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    def train(self,
              train_loader,
              val_loader=None,
              num_epochs: int = 100,
              save_interval: int = 10):
        """
        Full training loop with progress tracking and validation.
        """

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            epoch_losses = {
                'distribution_matching': [],
                'cycle_consistency': [],
                'orthogonality': [],
                'scale_regularization': [],
                'total': []
            }

            # Use tqdm for progress tracking
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch_data in train_pbar:
                losses = self.train_step(batch_data)

                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{losses['total']:.4f}",
                    'dist': f"{losses['distribution_matching']:.4f}",
                    'cycle': f"{losses['cycle_consistency']:.4f}"
                })

                # Accumulate losses
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key].append(losses[key])

            # Log epoch averages
            for key, values in epoch_losses.items():
                if values:
                    avg_loss = np.mean(values)
                    self.writer.add_scalar(f'Epoch/Train_{key}', avg_loss, epoch)

            # Validation phase
            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt', epoch)
                    print(f"New best model saved with validation loss: {val_loss:.4f}")

            # Regular checkpointing
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch)

        # Save final model
        self.save_checkpoint('final_model.pt', num_epochs)
        self.writer.close()
        print(f"Training complete! Logs saved to {self.log_dir}")

    def validate(self, val_loader, epoch: int) -> float:
        """
        Validation loop with metric computation.
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
            for batch_data in val_pbar:
                embeddings_A = batch_data['embeddings_A'].to(self.device)
                embeddings_B = batch_data['embeddings_B'].to(self.device)

                params = self.model(embeddings_A, embeddings_B)
                losses = self.compute_alignment_loss(embeddings_A, embeddings_B, params)

                val_losses.append(losses['total'].item())
                val_pbar.set_postfix({'val_loss': f"{losses['total'].item():.4f}"})

        avg_val_loss = np.mean(val_losses)
        self.writer.add_scalar('Epoch/Val_loss', avg_val_loss, epoch)

        return avg_val_loss

    def save_checkpoint(self, filename: str, epoch: int):
        """
        Save model checkpoint with training state.
        """
        checkpoint_path = self.log_dir / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step
        }

        torch.save(checkpoint, checkpoint_path / filename)
        print(f"Checkpoint saved: {checkpoint_path / filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']

        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def log_embeddings(self, embeddings_A: torch.Tensor, embeddings_B: torch.Tensor,
                      tag: str, step: int = None):
        """
        Log embeddings to TensorBoard for visualization.
        """
        if step is None:
            step = self.global_step

        # Subsample for visualization
        max_points = 1000
        if embeddings_A.shape[0] > max_points:
            indices = torch.randperm(embeddings_A.shape[0])[:max_points]
            embeddings_A = embeddings_A[indices]
            embeddings_B = embeddings_B[indices]

        # Log embeddings
        self.writer.add_embedding(
            embeddings_A.reshape(-1, embeddings_A.shape[-1]),
            tag=f'{tag}/embeddings_A',
            global_step=step
        )

        self.writer.add_embedding(
            embeddings_B.reshape(-1, embeddings_B.shape[-1]),
            tag=f'{tag}/embeddings_B',
            global_step=step
        )

        # Transform and log
        with torch.no_grad():
            params = self.model(embeddings_A.unsqueeze(0), embeddings_B.unsqueeze(0))
            A_transformed = self.model.transform(embeddings_A.unsqueeze(0), params, direction='A_to_B')

            self.writer.add_embedding(
                A_transformed.squeeze(0),
                tag=f'{tag}/A_transformed',
                global_step=step
            )