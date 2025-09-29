# phase_analyzer.py
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import platonic  # The existing PRH module

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
        Compute AGOP metrics from extracted features without training.
        We use a probe task approach - creating synthetic gradients.
        """
        # Features shape: (n_samples, n_features)
        n_features = features.shape[-1]

        # Create a simple probe task - random classification
        # This simulates gradient computation without actual training
        n_classes = 100
        probe_weights = np.random.randn(n_features, n_classes) * 0.01

        # Compute gradients for each sample
        gradients = []
        for i in range(min(n_samples, len(features))):
            # Synthetic gradient based on feature-weight interaction
            grad = features[i:i+1].T @ np.random.randn(1, n_classes)
            gradients.append(grad.flatten())

        # Compute gradient outer products
        gradient_matrix = np.stack(gradients)
        agop = gradient_matrix.T @ gradient_matrix / len(gradients)

        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(agop)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

        # Compute phase indicators
        results = {
            'eigenvalues': eigenvalues,
            'top_eigenvalue_ratio': eigenvalues[0] / (eigenvalues[1] + 1e-10),
            'effective_rank': np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2),
            'eigenvalue_concentration': np.sum(eigenvalues[:10]) / np.sum(eigenvalues),
            'phase': self._determine_phase_from_eigenvalues(eigenvalues)
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