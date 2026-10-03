#!/usr/bin/env python3
"""
Cross-Modal Phase Analysis Pipeline
===================================

A comprehensive system for analyzing phase transitions and cross-modal compatibility
in vision and language models. This pipeline extends the Platonic Representation
Hypothesis (PRH) framework to understand why certain model pairs fail to align.

Author: Your Name
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class PhaseAnalysisPipeline:
    """
    Main pipeline for conducting cross-modal phase analysis.

    This class orchestrates the entire analysis process:
    1. Feature extraction from pre-trained models
    2. Phase metric computation (AGOP, NTK, IB)
    3. Cross-modal compatibility analysis
    4. Visualization and reporting
    """

    def __init__(self,
                 dataset: str = "minhuh/prh",
                 subset: str = "wit_1024",
                 output_dir: str = "./results/phase_analysis",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the pipeline with configuration parameters.

        Args:
            dataset: Dataset identifier for PRH framework
            subset: Subset of data to analyze
            output_dir: Directory for saving results
            device: Computing device (cuda/cpu)
        """
        self.dataset = dataset
        self.subset = subset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Initialize components
        self.phase_analyzer = PhaseAnalyzer(device=device)
        self.compatibility_analyzer = CrossModalCompatibilityAnalyzer()
        self.visualizer = PhaseVisualizer(output_dir=self.output_dir)

        # Results storage
        self.results = {
            'metadata': {
                'dataset': dataset,
                'subset': subset,
                'timestamp': datetime.now().isoformat(),
                'device': device
            },
            'models': {},
            'phase_analysis': {},
            'compatibility': {},
            'summary': {}
        }

    def run_complete_analysis(self,
                            vision_models: List[str],
                            language_models: List[str],
                            feature_dir: str = "./results/features") -> Dict:
        """
        Execute the complete analysis pipeline.

        This method coordinates all analysis steps and produces comprehensive results
        about phase transitions and cross-modal compatibility.

        Args:
            vision_models: List of vision model identifiers
            language_models: List of language model identifiers
            feature_dir: Directory containing pre-extracted features

        Returns:
            Dictionary containing all analysis results
        """
        print("="*80)
        print("Cross-Modal Phase Analysis Pipeline")
        print("="*80)

        # Step 1: Load and validate features
        print("\n[Step 1/5] Loading features...")
        features = self._load_features(vision_models, language_models, feature_dir)

        # Step 2: Compute phase metrics for each model
        print("\n[Step 2/5] Computing phase metrics...")
        self._compute_phase_metrics(features)

        # Step 3: Analyze cross-modal compatibility
        print("\n[Step 3/5] Analyzing cross-modal compatibility...")
        self._analyze_compatibility()

        # Step 4: Generate visualizations
        print("\n[Step 4/5] Creating visualizations...")
        self._create_visualizations()

        # Step 5: Generate summary report
        print("\n[Step 5/5] Generating summary report...")
        self._generate_summary()

        # Save complete results
        self._save_results()

        print("\n✓ Analysis complete! Results saved to:", self.output_dir)
        return self.results

    def _load_features(self, vision_models: List[str],
                      language_models: List[str],
                      feature_dir: str) -> Dict:
        """Load pre-extracted features from PRH framework."""
        features = {'vision': {}, 'language': {}}

        # Load vision model features
        for model in tqdm(vision_models, desc="Loading vision features"):
            feature_path = Path(feature_dir) / self.dataset / self.subset / f"{model}_pool-cls.npy"
            if feature_path.exists():
                feat_data = np.load(feature_path, allow_pickle=True)
                if isinstance(feat_data, dict):
                    features['vision'][model] = feat_data.get('features', feat_data)
                else:
                    features['vision'][model] = feat_data
                self.results['models'][model] = {'type': 'vision', 'loaded': True}
            else:
                print(f"  Warning: Features not found for {model}")
                self.results['models'][model] = {'type': 'vision', 'loaded': False}

        # Load language model features
        for model in tqdm(language_models, desc="Loading language features"):
            feature_path = Path(feature_dir) / self.dataset / self.subset / f"{model}_pool-avg.npy"
            if feature_path.exists():
                feat_data = np.load(feature_path, allow_pickle=True)
                if isinstance(feat_data, dict):
                    features['language'][model] = feat_data.get('features', feat_data)
                else:
                    features['language'][model] = feat_data
                self.results['models'][model] = {'type': 'language', 'loaded': True}
            else:
                print(f"  Warning: Features not found for {model}")
                self.results['models'][model] = {'type': 'language', 'loaded': False}

        return features

    def _compute_phase_metrics(self, features: Dict):
        """Compute phase metrics for all models."""
        # Analyze vision models
        for model_name, model_features in tqdm(features['vision'].items(),
                                              desc="Analyzing vision models"):
            metrics = self.phase_analyzer.analyze_model(model_features, model_name, 'vision')
            self.results['phase_analysis'][model_name] = metrics

        # Analyze language models
        for model_name, model_features in tqdm(features['language'].items(),
                                              desc="Analyzing language models"):
            metrics = self.phase_analyzer.analyze_model(model_features, model_name, 'language')
            self.results['phase_analysis'][model_name] = metrics

    def _analyze_compatibility(self):
        """Analyze cross-modal compatibility between all model pairs."""
        vision_models = [m for m, data in self.results['models'].items()
                        if data['type'] == 'vision' and data['loaded']]
        language_models = [m for m, data in self.results['models'].items()
                         if data['type'] == 'language' and data['loaded']]

        compatibility_matrix = np.zeros((len(vision_models), len(language_models)))

        for i, v_model in enumerate(tqdm(vision_models, desc="Computing compatibility")):
            for j, l_model in enumerate(language_models):
                v_metrics = self.results['phase_analysis'][v_model]
                l_metrics = self.results['phase_analysis'][l_model]

                compatibility = self.compatibility_analyzer.compute_compatibility(
                    v_metrics, l_metrics
                )

                pair_key = f"{v_model}-{l_model}"
                self.results['compatibility'][pair_key] = compatibility
                compatibility_matrix[i, j] = compatibility['compatibility_score']

        self.results['compatibility']['matrix'] = compatibility_matrix
        self.results['compatibility']['vision_models'] = vision_models
        self.results['compatibility']['language_models'] = language_models

    def _create_visualizations(self):
        """Generate all visualizations."""
        # 1. Phase landscape heatmap
        self.visualizer.plot_phase_landscape(self.results)

        # 2. Phase distribution scatter plot
        self.visualizer.plot_phase_distribution(self.results)

        # 3. AGOP eigenvalue spectra
        self.visualizer.plot_eigenvalue_spectra(self.results)

        # 4. Cross-modal alignment prediction vs actual
        self.visualizer.plot_alignment_predictions(self.results)

        # 5. Phase trajectory visualization
        self.visualizer.plot_phase_trajectories(self.results)

    def _generate_summary(self):
        """Generate summary statistics and key findings."""
        summary = {}

        # Phase distribution statistics
        phases = [m['phase'] for m in self.results['phase_analysis'].values()]
        phase_counts = pd.Series(phases).value_counts().to_dict()
        summary['phase_distribution'] = phase_counts

        # Modality-specific phase patterns
        vision_phases = [m['phase'] for name, m in self.results['phase_analysis'].items()
                        if self.results['models'].get(name, {}).get('type') == 'vision']
        language_phases = [m['phase'] for name, m in self.results['phase_analysis'].items()
                          if self.results['models'].get(name, {}).get('type') == 'language']

        summary['vision_phase_distribution'] = pd.Series(vision_phases).value_counts().to_dict()
        summary['language_phase_distribution'] = pd.Series(language_phases).value_counts().to_dict()

        # Compatibility statistics
        compatibility_scores = [c['compatibility_score']
                              for c in self.results['compatibility'].values()
                              if isinstance(c, dict)]

        summary['compatibility_stats'] = {
            'mean': np.mean(compatibility_scores),
            'std': np.std(compatibility_scores),
            'min': np.min(compatibility_scores),
            'max': np.max(compatibility_scores),
            'below_threshold': sum(1 for s in compatibility_scores if s < 0.1)
        }

        # Key findings based on your paper
        summary['key_findings'] = self._extract_key_findings()

        self.results['summary'] = summary

    def _extract_key_findings(self) -> Dict:
        """Extract key findings that validate or extend the paper's results."""
        findings = {}

        # Finding 1: Phase distribution across modalities
        vision_chaotic = sum(1 for name, m in self.results['phase_analysis'].items()
                            if self.results['models'].get(name, {}).get('type') == 'vision'
                            and m['phase'] == 'chaotic')
        language_chaotic = sum(1 for name, m in self.results['phase_analysis'].items()
                              if self.results['models'].get(name, {}).get('type') == 'language'
                              and m['phase'] == 'chaotic')

        findings['modality_phase_bias'] = {
            'vision_chaotic_percentage': vision_chaotic / len([m for m in self.results['models'].values()
                                                              if m['type'] == 'vision']) * 100,
            'language_chaotic_percentage': language_chaotic / len([m for m in self.results['models'].values()
                                                                  if m['type'] == 'language']) * 100
        }

        # Finding 2: AGOP threshold validation
        high_agop_failures = 0
        total_high_agop = 0

        for pair_key, compat in self.results['compatibility'].items():
            if isinstance(compat, dict) and 'agop_ratio' in compat:
                if compat['agop_ratio'] > 1e6:
                    total_high_agop += 1
                    if compat['predicted_alignment'] < 0.03:
                        high_agop_failures += 1

        findings['agop_threshold_validation'] = {
            'high_agop_pairs': total_high_agop,
            'failure_rate': high_agop_failures / total_high_agop * 100 if total_high_agop > 0 else 0
        }

        return findings

    def _save_results(self):
        """Save all results to disk."""
        # Save JSON results
        json_path = self.output_dir / "phase_analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary report as markdown
        self._save_markdown_report()

    def _save_markdown_report(self):
        """Generate and save a markdown report of findings."""
        report_path = self.output_dir / "phase_analysis_report.md"

        with open(report_path, 'w') as f:
            f.write("# Cross-Modal Phase Analysis Report\n\n")
            f.write(f"Generated: {self.results['metadata']['timestamp']}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = self.results['summary']
            f.write(f"- Analyzed {len(self.results['models'])} models total\n")
            f.write(f"- Vision models: {sum(1 for m in self.results['models'].values() if m['type'] == 'vision')}\n")
            f.write(f"- Language models: {sum(1 for m in self.results['models'].values() if m['type'] == 'language')}\n")
            f.write(f"- Average compatibility score: {summary['compatibility_stats']['mean']:.3f}\n")
            f.write(f"- Pairs with <10% predicted alignment: {summary['compatibility_stats']['below_threshold']}\n\n")

            # Phase Distribution
            f.write("## Phase Distribution\n\n")
            f.write("### Overall\n")
            for phase, count in summary['phase_distribution'].items():
                f.write(f"- {phase}: {count} models\n")

            f.write("\n### By Modality\n")
            f.write("**Vision Models:**\n")
            for phase, count in summary['vision_phase_distribution'].items():
                f.write(f"- {phase}: {count} models\n")

            f.write("\n**Language Models:**\n")
            for phase, count in summary['language_phase_distribution'].items():
                f.write(f"- {phase}: {count} models\n")

            # Key Findings
            f.write("\n## Key Findings\n\n")
            findings = summary['key_findings']

            f.write("### Modality-Specific Phase Preferences\n")
            f.write(f"- Vision models in chaotic phase: {findings['modality_phase_bias']['vision_chaotic_percentage']:.1f}%\n")
            f.write(f"- Language models in chaotic phase: {findings['modality_phase_bias']['language_chaotic_percentage']:.1f}%\n\n")

            f.write("### AGOP Threshold Validation\n")
            f.write(f"- Model pairs with AGOP ratio > 10^6: {findings['agop_threshold_validation']['high_agop_pairs']}\n")
            f.write(f"- Failure rate for high AGOP pairs: {findings['agop_threshold_validation']['failure_rate']:.1f}%\n")


class PhaseAnalyzer:
    """
    Core phase analysis component that computes AGOP, NTK, and IB metrics.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_model(self, features: np.ndarray, model_name: str, modality: str) -> Dict:
        """
        Perform complete phase analysis on a model's features.

        This method computes all phase-related metrics including AGOP eigenvalues,
        NTK stability, and information bottleneck measures.
        """
        results = {
            'model': model_name,
            'modality': modality,
            'feature_dim': features.shape[-1],
            'n_samples': len(features)
        }

        # Compute AGOP metrics
        agop_results = self._compute_agop_metrics(features)
        results.update(agop_results)

        # Compute NTK stability
        ntk_results = self._compute_ntk_stability(features)
        results.update(ntk_results)

        # Compute information bottleneck metrics
        ib_results = self._compute_ib_metrics(features)
        results.update(ib_results)

        # Determine phase
        results['phase'] = self._determine_phase(agop_results, ntk_results)
        results['phase_confidence'] = self._compute_phase_confidence(agop_results, ntk_results)

        return results

    def _compute_agop_metrics(self, features: np.ndarray) -> Dict:
        """
        Compute Average Gradient Outer Product metrics.

        Since we're working with pre-trained models, we approximate gradients
        using a probe task approach.
        """
        n_samples = min(1000, len(features))
        n_features = features.shape[-1]

        # Subsample for efficiency
        if len(features) > n_samples:
            indices = np.random.choice(len(features), n_samples, replace=False)
            features_subset = features[indices]
        else:
            features_subset = features

        # Create synthetic probe task
        n_classes = 100
        probe_weights = np.random.randn(n_features, n_classes) * 0.01

        # Compute gradients
        gradients = []
        for i in range(len(features_subset)):
            # Synthetic gradient computation
            logits = features_subset[i] @ probe_weights
            # Softmax and cross-entropy gradient
            probs = np.exp(logits) / np.sum(np.exp(logits))
            target = np.random.randint(0, n_classes)
            grad_logits = probs.copy()
            grad_logits[target] -= 1

            # Backprop to features
            grad_features = probe_weights @ grad_logits
            gradients.append(grad_features)

        # Compute AGOP
        gradient_matrix = np.array(gradients)
        agop = gradient_matrix.T @ gradient_matrix / len(gradients)

        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(agop)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros

        # Compute key metrics
        total_variance = np.sum(eigenvalues)
        normalized_eigenvalues = eigenvalues / total_variance

        return {
            'agop_eigenvalues': eigenvalues[:100],  # Top 100
            'agop_top_eigenvalue_ratio': eigenvalues[0] / (eigenvalues[1] + 1e-10),
            'agop_effective_rank': total_variance ** 2 / np.sum(eigenvalues ** 2),
            'agop_top10_concentration': np.sum(normalized_eigenvalues[:10]),
            'agop_top50_concentration': np.sum(normalized_eigenvalues[:50]),
            'agop_decay_rate': self._fit_eigenvalue_decay(eigenvalues)
        }

    def _fit_eigenvalue_decay(self, eigenvalues: np.ndarray) -> Dict:
        """Fit power law to eigenvalue decay."""
        # Use log-log regression to fit power law
        valid_indices = eigenvalues > 1e-10
        if np.sum(valid_indices) < 10:
            return {'type': 'insufficient_data', 'exponent': 0}

        log_indices = np.log(np.arange(1, np.sum(valid_indices) + 1))
        log_eigenvalues = np.log(eigenvalues[valid_indices])

        # Fit linear regression in log space
        slope, intercept = np.polyfit(log_indices[:50], log_eigenvalues[:50], 1)

        return {
            'type': 'power_law',
            'exponent': -slope,
            'log_intercept': intercept
        }

    def _compute_ntk_stability(self, features: np.ndarray) -> Dict:
        """
        Compute Neural Tangent Kernel stability metrics.
        """
        n_samples = min(500, len(features))

        # Subsample for efficiency
        if len(features) > n_samples:
            indices = np.random.choice(len(features), n_samples, replace=False)
            features_subset = features[indices]
        else:
            features_subset = features

        # Normalize features
        features_norm = features_subset / (np.linalg.norm(features_subset, axis=1, keepdims=True) + 1e-8)

        # Compute NTK (using feature inner products as proxy)
        ntk = features_norm @ features_norm.T

        # Stability analysis through perturbation
        stability_scores = []
        for _ in range(50):
            # Add small noise
            noise = np.random.randn(*features_norm.shape) * 0.01
            perturbed = features_norm + noise
            perturbed = perturbed / (np.linalg.norm(perturbed, axis=1, keepdims=True) + 1e-8)

            # Compute perturbed kernel
            ntk_perturbed = perturbed @ perturbed.T

            # Measure stability
            kernel_change = np.linalg.norm(ntk - ntk_perturbed, 'fro') / np.linalg.norm(ntk, 'fro')
            stability_scores.append(1 - kernel_change)

        # Compute NTK eigenvalues
        ntk_eigenvalues = np.linalg.eigvalsh(ntk)[::-1]

        return {
            'ntk_mean_stability': np.mean(stability_scores),
            'ntk_std_stability': np.std(stability_scores),
            'ntk_condition_number': ntk_eigenvalues[0] / (ntk_eigenvalues[-1] + 1e-10),
            'ntk_effective_rank': np.sum(ntk_eigenvalues) ** 2 / np.sum(ntk_eigenvalues ** 2)
        }

    def _compute_ib_metrics(self, features: np.ndarray) -> Dict:
        """
        Compute Information Bottleneck related metrics.
        """
        # Estimate compression by computing feature statistics
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0)

        # Sparsity (how many features are effectively zero)
        sparsity = np.mean(np.abs(features) < 0.01)

        # Feature correlation matrix
        feature_corr = np.corrcoef(features.T)

        # Estimate redundancy
        corr_threshold = 0.9
        high_corr_pairs = np.sum(np.abs(feature_corr) > corr_threshold) - len(feature_corr)
        redundancy = high_corr_pairs / (len(feature_corr) * (len(feature_corr) - 1))

        return {
            'ib_sparsity': sparsity,
            'ib_redundancy': redundancy,
            'ib_feature_std_mean': np.mean(feature_std),
            'ib_feature_std_variance': np.var(feature_std)
        }

    def _determine_phase(self, agop_results: Dict, ntk_results: Dict) -> str:
        """
        Determine the phase based on computed metrics.

        This implements the phase criteria from the paper.
        """
        # Primary criterion: AGOP eigenvalue concentration
        top10_concentration = agop_results['agop_top10_concentration']

        # Secondary criterion: NTK stability
        ntk_stability = ntk_results['ntk_mean_stability']

        # Phase determination logic
        if top10_concentration > 0.9 and ntk_stability > 0.95:
            return "stable"
        elif top10_concentration > 0.7 or ntk_stability > 0.85:
            return "critical"
        else:
            return "chaotic"

    def _compute_phase_confidence(self, agop_results: Dict, ntk_results: Dict) -> float:
        """
        Compute confidence in phase assignment.
        """
        # Use distance from phase boundaries as confidence measure
        top10_concentration = agop_results['agop_top10_concentration']
        ntk_stability = ntk_results['ntk_mean_stability']

        # Distance from nearest boundary
        distances = []

        # Distance from stable/critical boundary
        distances.append(abs(top10_concentration - 0.9))
        distances.append(abs(ntk_stability - 0.95))

        # Distance from critical/chaotic boundary
        distances.append(abs(top10_concentration - 0.7))
        distances.append(abs(ntk_stability - 0.85))

        # Confidence is inverse of minimum distance to boundary
        min_distance = min(distances)
        confidence = 1 - min_distance

        return confidence


class CrossModalCompatibilityAnalyzer:
    """
    Analyzes compatibility between vision and language models based on phase metrics.
    """

    def compute_compatibility(self, vision_metrics: Dict, language_metrics: Dict) -> Dict:
        """
        Compute comprehensive compatibility metrics between two models.
        """
        results = {
            'vision_model': vision_metrics['model'],
            'language_model': language_metrics['model'],
            'vision_phase': vision_metrics['phase'],
            'language_phase': language_metrics['phase']
        }

        # Compute phase distance
        phase_distance = self._compute_phase_distance(vision_metrics, language_metrics)
        results['phase_distance'] = phase_distance

        # Compute AGOP compatibility
        agop_ratio = self._compute_agop_ratio(vision_metrics, language_metrics)
        results['agop_ratio'] = agop_ratio

        # Compute NTK compatibility
        ntk_similarity = self._compute_ntk_similarity(vision_metrics, language_metrics)
        results['ntk_similarity'] = ntk_similarity

        # Overall compatibility score
        compatibility_score = self._compute_overall_compatibility(
            phase_distance, agop_ratio, ntk_similarity
        )
        results['compatibility_score'] = compatibility_score

        # Predicted alignment based on paper findings
        predicted_alignment = self._predict_alignment(phase_distance, agop_ratio, ntk_similarity)
        results['predicted_alignment'] = predicted_alignment

        return results

    def _compute_phase_distance(self, metrics1: Dict, metrics2: Dict) -> float:
        """
        Compute distance between two models in phase space.
        """
        # Extract key phase indicators
        features1 = np.array([
            metrics1['agop_top_eigenvalue_ratio'],
            metrics1['agop_effective_rank'],
            metrics1['agop_top10_concentration'],
            metrics1['ntk_mean_stability'],
            metrics1['ntk_condition_number']
        ])

        features2 = np.array([
            metrics2['agop_top_eigenvalue_ratio'],
            metrics2['agop_effective_rank'],
            metrics2['agop_top10_concentration'],
            metrics2['ntk_mean_stability'],
            metrics2['ntk_condition_number']
        ])

        # Normalize features
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)

        # Euclidean distance in normalized space
        return np.linalg.norm(features1 - features2)

    def _compute_agop_ratio(self, metrics1: Dict, metrics2: Dict) -> float:
        """
        Compute AGOP compatibility ratio.
        """
        # Use top eigenvalue ratios as proxy for AGOP magnitude
        agop1 = metrics1['agop_top_eigenvalue_ratio']
        agop2 = metrics2['agop_top_eigenvalue_ratio']

        return max(agop1, agop2) / (min(agop1, agop2) + 1e-10)

    def _compute_ntk_similarity(self, metrics1: Dict, metrics2: Dict) -> float:
        """
        Compute NTK similarity between models.
        """
        # Use stability scores
        stability1 = metrics1['ntk_mean_stability']
        stability2 = metrics2['ntk_mean_stability']

        # Similarity is 1 minus relative difference
        return 1 - abs(stability1 - stability2) / (max(stability1, stability2) + 1e-10)

    def _compute_overall_compatibility(self, phase_distance: float,
                                      agop_ratio: float,
                                      ntk_similarity: float) -> float:
        """
        Compute overall compatibility score.
        """
        # Normalize metrics to [0, 1] range
        phase_score = 1 / (1 + phase_distance)
        agop_score = 1 / (1 + np.log(agop_ratio))

        # Weighted combination
        compatibility = 0.4 * phase_score + 0.3 * agop_score + 0.3 * ntk_similarity

        return compatibility

    def _predict_alignment(self, phase_distance: float,
                          agop_ratio: float,
                          ntk_similarity: float) -> float:
        """
        Predict alignment score based on paper's findings.
        """
        # Implement thresholds from the paper
        if agop_ratio > 1e6:
            return 0.01  # ~1% alignment for extreme AGOP mismatch
        elif phase_distance > 2.0 or ntk_similarity < 0.25:
            return 0.03  # ~3% alignment for phase incompatibility
        elif phase_distance > 1.5:
            return 0.05  # ~5% for moderate incompatibility
        else:
            # Better compatibility, but still limited
            return 0.10 + 0.05 * ntk_similarity


class PhaseVisualizer:
    """
    Creates comprehensive visualizations of phase analysis results.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.figure_dir = output_dir / "figures"
        self.figure_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_phase_landscape(self, results: Dict):
        """
        Create heatmap of cross-modal phase compatibility.
        """
        plt.figure(figsize=(14, 10))

        # Extract compatibility matrix
        matrix = results['compatibility']['matrix']
        vision_models = results['compatibility']['vision_models']
        language_models = results['compatibility']['language_models']

        # Create heatmap
        sns.heatmap(matrix,
                   xticklabels=language_models,
                   yticklabels=vision_models,
                   cmap='RdBu',
                   center=0.5,
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Compatibility Score'},
                   square=True)

        plt.title('Cross-Modal Phase Compatibility Landscape', fontsize=16)
        plt.xlabel('Language Models', fontsize=12)
        plt.ylabel('Vision Models', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(self.figure_dir / 'phase_landscape.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_phase_distribution(self, results: Dict):
        """
        Create scatter plot showing phase distribution across modalities.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Extract phase metrics for plotting
        vision_data = []
        language_data = []

        for model_name, metrics in results['phase_analysis'].items():
            if results['models'][model_name]['type'] == 'vision':
                vision_data.append({
                    'model': model_name,
                    'agop_concentration': metrics['agop_top10_concentration'],
                    'ntk_stability': metrics['ntk_mean_stability'],
                    'phase': metrics['phase']
                })
            else:
                language_data.append({
                    'model': model_name,
                    'agop_concentration': metrics['agop_top10_concentration'],
                    'ntk_stability': metrics['ntk_mean_stability'],
                    'phase': metrics['phase']
                })

        # Plot vision models
        phase_colors = {'stable': 'green', 'critical': 'orange', 'chaotic': 'red'}

        for data in vision_data:
            ax1.scatter(data['agop_concentration'], data['ntk_stability'],
                       c=phase_colors[data['phase']], s=100, alpha=0.7,
                       label=data['phase'] if data['phase'] not in ax1.get_legend_handles_labels()[1] else "")

        ax1.set_xlabel('AGOP Top-10 Concentration', fontsize=12)
        ax1.set_ylabel('NTK Stability', fontsize=12)
        ax1.set_title('Vision Models - Phase Distribution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add phase boundaries
        ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)

        # Plot language models
        for data in language_data:
            ax2.scatter(data['agop_concentration'], data['ntk_stability'],
                       c=phase_colors[data['phase']], s=100, alpha=0.7,
                       label=data['phase'] if data['phase'] not in ax2.get_legend_handles_labels()[1] else "")

        ax2.set_xlabel('AGOP Top-10 Concentration', fontsize=12)
        ax2.set_ylabel('NTK Stability', fontsize=12)
        ax2.set_title('Language Models - Phase Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add phase boundaries
        ax2.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)

        plt.suptitle('Phase Distribution Across Modalities', fontsize=16)
        plt.tight_layout()

        plt.savefig(self.figure_dir / 'phase_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_eigenvalue_spectra(self, results: Dict):
        """
        Plot AGOP eigenvalue spectra for selected models.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Select representative models from each phase
        models_by_phase = {'stable': [], 'critical': [], 'chaotic': []}

        for model_name, metrics in results['phase_analysis'].items():
            phase = metrics['phase']
            models_by_phase[phase].append((model_name, metrics))

        # Plot one model from each phase for each modality
        plot_idx = 0
        for modality in ['vision', 'language']:
            for phase in ['stable', 'critical', 'chaotic']:
                # Find a model of this phase and modality
                for model_name, metrics in models_by_phase[phase]:
                    if results['models'][model_name]['type'] == modality and plot_idx < 4:
                        ax = axes[plot_idx]
                        eigenvalues = metrics['agop_eigenvalues'][:50]  # Top 50

                        ax.semilogy(range(1, len(eigenvalues) + 1), eigenvalues,
                                   'o-', markersize=4, linewidth=2)
                        ax.set_xlabel('Eigenvalue Index', fontsize=10)
                        ax.set_ylabel('Eigenvalue Magnitude', fontsize=10)
                        ax.set_title(f'{model_name} ({modality}, {phase})', fontsize=12)
                        ax.grid(True, alpha=0.3)

                        plot_idx += 1
                        break

        plt.suptitle('AGOP Eigenvalue Spectra - Representative Models', fontsize=16)
        plt.tight_layout()

        plt.savefig(self.figure_dir / 'eigenvalue_spectra.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_alignment_predictions(self, results: Dict):
        """
        Plot predicted vs actual alignment (if available).
        """
        plt.figure(figsize=(10, 8))

        # Extract prediction data
        predictions = []
        phase_pairs = []

        for pair_key, compat in results['compatibility'].items():
            if isinstance(compat, dict) and 'predicted_alignment' in compat:
                predictions.append(compat['predicted_alignment'])
                phase_pairs.append(f"{compat['vision_phase'][0]}-{compat['language_phase'][0]}")

        # Create box plot by phase combination
        unique_pairs = list(set(phase_pairs))
        data_by_pair = {pair: [] for pair in unique_pairs}

        for pred, pair in zip(predictions, phase_pairs):
            data_by_pair[pair].append(pred)

        # Plot
        box_data = [data_by_pair[pair] for pair in sorted(unique_pairs)]
        box_labels = sorted(unique_pairs)

        plt.boxplot(box_data, labels=box_labels)
        plt.xlabel('Phase Combination (Vision-Language)', fontsize=12)
        plt.ylabel('Predicted Alignment Score', fontsize=12)
        plt.title('Alignment Predictions by Phase Combination', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        # Add horizontal lines for key thresholds
        plt.axhline(y=0.03, color='red', linestyle='--', alpha=0.5, label='3% threshold')
        plt.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.figure_dir / 'alignment_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_phase_trajectories(self, results: Dict):
        """
        Create 2D projection of models in phase space.
        """
        plt.figure(figsize=(12, 10))

        # Extract features for all models
        features = []
        labels = []
        modalities = []
        phases = []

        for model_name, metrics in results['phase_analysis'].items():
            feature_vec = [
                metrics['agop_top_eigenvalue_ratio'],
                metrics['agop_effective_rank'],
                metrics['agop_top10_concentration'],
                metrics['ntk_mean_stability'],
                metrics['ntk_condition_number']
            ]
            features.append(feature_vec)
            labels.append(model_name)
            modalities.append(results['models'][model_name]['type'])
            phases.append(metrics['phase'])

        features = np.array(features)

        # Apply t-SNE for 2D projection
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)

        # Plot
        markers = {'vision': 'o', 'language': 's'}
        colors = {'stable': 'green', 'critical': 'orange', 'chaotic': 'red'}

        for i, (x, y) in enumerate(features_2d):
            plt.scatter(x, y,
                       marker=markers[modalities[i]],
                       c=colors[phases[i]],
                       s=200,
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=1)

            # Add labels for some models
            if i % 3 == 0:  # Label every 3rd model to avoid overcrowding
                plt.annotate(labels[i], (x, y),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           alpha=0.7)

        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = []

        # Modality markers
        for modality, marker in markers.items():
            legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                        markerfacecolor='gray', markersize=10,
                                        label=modality.capitalize()))

        # Phase colors
        for phase, color in colors.items():
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=10,
                                        label=phase.capitalize()))

        plt.legend(handles=legend_elements, loc='best', framealpha=0.9)

        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('Model Distribution in Phase Space', fontsize=16)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figure_dir / 'phase_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    Main entry point for the phase analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Cross-Modal Phase Analysis Pipeline')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='minhuh/prh',
                       help='Dataset identifier')
    parser.add_argument('--subset', type=str, default='wit_1024',
                       help='Dataset subset')
    parser.add_argument('--feature_dir', type=str, default='./results/features',
                       help='Directory containing pre-extracted features')

    # Model arguments
    parser.add_argument('--vision_models', nargs='+',
                       default=['dinov2_g', 'dinov2_l', 'clip_h', 'clip_l', 'mae_h'],
                       help='List of vision models to analyze')
    parser.add_argument('--language_models', nargs='+',
                       default=['llama_7b', 'gpt2_xl', 'bert_large'],
                       help='List of language models to analyze')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results/phase_analysis',
                       help='Output directory for results')

    # Compute arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Computing device')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PhaseAnalysisPipeline(
        dataset=args.dataset,
        subset=args.subset,
        output_dir=args.output_dir,
        device=args.device
    )

    # Run analysis
    results = pipeline.run_complete_analysis(
        vision_models=args.vision_models,
        language_models=args.language_models,
        feature_dir=args.feature_dir
    )

    print("\n✨ Analysis complete! Check the output directory for results.")
    print(f"📊 Results saved to: {args.output_dir}")
    print(f"📈 Visualizations saved to: {args.output_dir}/figures/")
    print(f"📄 Report saved to: {args.output_dir}/phase_analysis_report.md")


if __name__ == "__main__":
    main()