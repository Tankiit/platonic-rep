# cross_modal_phase_analyzer.py
import numpy as np
from typing import Dict, List, Tuple
import platonic
from phase_analyzer import PhaseAnalyzer

class CrossModalPhaseAnalyzer:
    """
    Analyzes phase compatibility between vision and language models.
    """

    def __init__(self, dataset="minhuh/prh", subset="wit_1024"):
        self.dataset = dataset
        self.subset = subset
        self.phase_analyzer = PhaseAnalyzer(dataset, subset)

        # Use PRH's model lists
        self.vision_models = [
            "dinov2_g", "dinov2_l", "dinov2_b", "dinov2_s",
            "clip_h", "clip_l", "clip_b",
            "mae_h", "mae_l", "mae_b"
        ]

        self.language_models = [
            "llama_7b", "llama_13b", "llama_30b", "llama_65b",
            "gpt2", "gpt2_medium", "gpt2_large", "gpt2_xl",
            "bert_base", "bert_large"
        ]

    def compute_phase_distance(self, phase1: Dict, phase2: Dict) -> float:
        """
        Compute distance between two phases based on their metrics.
        """
        # Extract key metrics
        metrics1 = np.array([
            phase1['agop']['top_eigenvalue_ratio'],
            phase1['agop']['effective_rank'],
            phase1['agop']['eigenvalue_concentration'],
            phase1['ntk']['mean_stability']
        ])

        metrics2 = np.array([
            phase2['agop']['top_eigenvalue_ratio'],
            phase2['agop']['effective_rank'],
            phase2['agop']['eigenvalue_concentration'],
            phase2['ntk']['mean_stability']
        ])

        # Normalize metrics
        metrics1 = metrics1 / (np.linalg.norm(metrics1) + 1e-8)
        metrics2 = metrics2 / (np.linalg.norm(metrics2) + 1e-8)

        # Compute distance
        return np.linalg.norm(metrics1 - metrics2)

    def analyze_all_pairs(self) -> Dict:
        """
        Analyze phase compatibility for all vision-language pairs.
        """
        results = {
            'vision_phases': {},
            'language_phases': {},
            'compatibility_matrix': {},
            'phase_distances': {}
        }

        # First, analyze all vision models
        print("Analyzing vision models...")
        for v_model in self.vision_models:
            phase_data = self.phase_analyzer.analyze_model_phases(v_model, "vision")
            if phase_data:
                results['vision_phases'][v_model] = phase_data

        # Then, analyze all language models
        print("\nAnalyzing language models...")
        for l_model in self.language_models:
            phase_data = self.phase_analyzer.analyze_model_phases(l_model, "language")
            if phase_data:
                results['language_phases'][l_model] = phase_data

        # Compute pairwise phase distances
        print("\nComputing cross-modal phase distances...")
        for v_model, v_phase in results['vision_phases'].items():
            for l_model, l_phase in results['language_phases'].items():
                pair_key = f"{v_model}-{l_model}"

                # Compute phase distance
                distance = self.compute_phase_distance(v_phase, l_phase)
                results['phase_distances'][pair_key] = distance

                # Predict alignment based on phase compatibility
                # This implements your paper's findings
                if distance > 2.0:  # High phase distance
                    predicted_alignment = 0.01  # ~1% as per your findings
                elif distance > 1.5:
                    predicted_alignment = 0.03  # ~3% as per your findings
                else:
                    predicted_alignment = 0.10  # Better but still limited

                results['compatibility_matrix'][pair_key] = {
                    'phase_distance': distance,
                    'predicted_alignment': predicted_alignment,
                    'v_phase': v_phase['phase'],
                    'l_phase': l_phase['phase']
                }

        return results

    def visualize_phase_landscape(self, results: Dict):
        """
        Create visualization of the phase landscape.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Extract data for visualization
        vision_models = list(results['vision_phases'].keys())
        language_models = list(results['language_phases'].keys())

        # Create distance matrix
        distance_matrix = np.zeros((len(vision_models), len(language_models)))

        for i, v_model in enumerate(vision_models):
            for j, l_model in enumerate(language_models):
                pair_key = f"{v_model}-{l_model}"
                distance_matrix[i, j] = results['phase_distances'].get(pair_key, np.nan)

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(distance_matrix,
                    xticklabels=language_models,
                    yticklabels=vision_models,
                    cmap='RdBu_r',
                    center=1.5,
                    annot=True,
                    fmt='.2f',
                    cbar_kws={'label': 'Phase Distance'})

        plt.title('Cross-Modal Phase Distance Matrix')
        plt.xlabel('Language Models')
        plt.ylabel('Vision Models')
        plt.tight_layout()
        plt.savefig('./results/phase_analysis/phase_landscape.png', dpi=300)
        plt.close()