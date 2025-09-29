#!/usr/bin/env python3
"""
Enhanced Multiscale Analysis Pipeline
Extends the existing multiscale analysis to handle multiple datasets and comprehensive cross-modal comparisons
"""

import torch
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler

# Import existing analysis components
from multi_scale import MultiscaleInformationAnalysis, MultiscaleIntegration, NumpyEncoder
from multi_dataset_config import MultiDatasetConfig, DatasetConfig, DatasetType
from medium_scale_cross_modal_analysis import analyze_architecture_breakdown
from platonic.alignment import Alignment


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    dataset_name: str
    model_name: str
    dataset_type: str
    analysis_type: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str


class EnhancedMultiscaleAnalyzer:
    """
    Enhanced analyzer that handles multiple datasets and provides comprehensive
    cross-modal analysis with multiscale perspectives
    """

    def __init__(self,
                 features_dir: str = "./results/features_exhaustive",
                 output_dir: str = "./results/enhanced_multiscale_analysis",
                 config_dir: str = "./config"):

        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "individual_analyses").mkdir(exist_ok=True)
        (self.output_dir / "cross_dataset_analyses").mkdir(exist_ok=True)
        (self.output_dir / "cross_modal_analyses").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)

        # Initialize configuration
        self.config = MultiDatasetConfig(config_dir)

        # Initialize existing analyzers
        self.multiscale_analyzer = MultiscaleInformationAnalysis()
        self.multiscale_integrator = MultiscaleIntegration()

        # Storage for all analysis results
        self.all_results = []

        # Analysis statistics
        self.stats = {
            "total_datasets": 0,
            "total_models": 0,
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "cross_modal_comparisons": 0,
            "start_time": None,
            "end_time": None
        }

    def discover_available_features(self) -> Dict[str, Dict[str, List[str]]]:
        """Discover all available feature files organized by dataset and model type"""

        available_features = {}

        print("Discovering available feature files...")

        for dataset_dir in self.features_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            available_features[dataset_name] = {"llm": [], "lvm": [], "paths": {}}

            # Check for subset directories
            for potential_feature_dir in [dataset_dir, *dataset_dir.glob("*")]:
                if not potential_feature_dir.is_dir():
                    continue

                for feature_file in potential_feature_dir.glob("*.pt"):
                    model_identifier = feature_file.stem

                    # Remove pooling suffix (_avg, _cls, etc.)
                    for suffix in ["_avg", "_cls", "_last", "_max"]:
                        if model_identifier.endswith(suffix):
                            model_identifier = model_identifier[:-len(suffix)]
                            break

                    # Determine model type based on name patterns
                    if ("/" in model_identifier and not any(x in model_identifier.lower()
                                                           for x in ["vit", "deit", "clip"])) or \
                       any(family in model_identifier.lower()
                           for family in ["bloom", "llama", "gpt", "pythia", "opt", "gemma", "mistral"]):
                        model_type = "llm"
                    else:
                        model_type = "lvm"

                    available_features[dataset_name][model_type].append(model_identifier)
                    available_features[dataset_name]["paths"][model_identifier] = str(feature_file)

        # Print discovery summary
        total_models = 0
        for dataset_name, models in available_features.items():
            llm_count = len(models["llm"])
            lvm_count = len(models["lvm"])
            total = llm_count + lvm_count
            total_models += total

            if total > 0:
                print(f"  {dataset_name}: {llm_count} language models, {lvm_count} vision models ({total} total)")

        print(f"Total: {len(available_features)} datasets, {total_models} model-dataset combinations")
        self.stats["total_datasets"] = len(available_features)
        self.stats["total_models"] = total_models

        return available_features

    def run_individual_multiscale_analysis(self,
                                         available_features: Dict[str, Dict[str, List[str]]],
                                         max_models_per_dataset: int = 10) -> List[AnalysisResult]:
        """Run multiscale analysis on individual model-dataset combinations"""

        print(f"\n{'='*80}")
        print("RUNNING INDIVIDUAL MULTISCALE ANALYSES")
        print(f"{'='*80}")

        individual_results = []

        for dataset_name, models in tqdm(available_features.items(), desc="Processing datasets"):
            if not any(models[model_type] for model_type in ["llm", "lvm"]):
                continue

            print(f"\nProcessing dataset: {dataset_name}")

            # Process language models
            llm_models = models["llm"][:max_models_per_dataset]
            for model_name in llm_models:
                try:
                    result = self._analyze_single_model(dataset_name, model_name, "llm",
                                                      models["paths"][model_name])
                    if result:
                        individual_results.append(result)
                        self.stats["successful_analyses"] += 1
                except Exception as e:
                    print(f"  Failed to analyze {model_name}: {e}")
                    self.stats["failed_analyses"] += 1

                self.stats["total_analyses"] += 1

            # Process vision models
            lvm_models = models["lvm"][:max_models_per_dataset]
            for model_name in lvm_models:
                try:
                    result = self._analyze_single_model(dataset_name, model_name, "lvm",
                                                      models["paths"][model_name])
                    if result:
                        individual_results.append(result)
                        self.stats["successful_analyses"] += 1
                except Exception as e:
                    print(f"  Failed to analyze {model_name}: {e}")
                    self.stats["failed_analyses"] += 1

                self.stats["total_analyses"] += 1

        print(f"Completed {len(individual_results)} individual analyses")
        return individual_results

    def _analyze_single_model(self, dataset_name: str, model_name: str,
                            model_type: str, feature_path: str) -> Optional[AnalysisResult]:
        """Analyze a single model-dataset combination"""

        try:
            # Load features
            feature_data = torch.load(feature_path, map_location='cpu')
            features = feature_data['feats']

            # Ensure proper shape [N, L, D]
            if len(features.shape) == 2:
                features = features.unsqueeze(1)

            # Run multiscale analysis
            results = {}

            # Microscopic analysis (individual features)
            results['microscopic'] = self._run_microscopic_analysis(features)

            # Mesoscopic analysis (layer-level patterns)
            results['mesoscopic'] = self._run_mesoscopic_analysis(features)

            # Macroscopic analysis (network-level information flow)
            results['macroscopic'] = self._run_macroscopic_analysis(features, dataset_name)

            # Cross-scale connections
            results['cross_scale'] = self._analyze_cross_scale_connections(
                results['microscopic'], results['mesoscopic'], results['macroscopic']
            )

            # Additional metadata
            metadata = {
                "feature_shape": list(features.shape),
                "num_params": feature_data.get('num_params', 0),
                "dataset_info": feature_data.get('dataset_name', dataset_name),
                "pooling_strategy": feature_data.get('pooling_strategy', 'unknown')
            }

            return AnalysisResult(
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_type=model_type,
                analysis_type="individual_multiscale",
                results=results,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"    Error analyzing {model_name} on {dataset_name}: {e}")
            return None

    def _run_microscopic_analysis(self, features: torch.Tensor) -> Dict[str, Any]:
        """Run microscopic (neuron-level) analysis"""

        num_layers = features.shape[1]
        microscopic_results = {}

        for layer_idx in range(num_layers):
            layer_features = features[:, layer_idx, :].numpy()

            # Basic activation statistics
            activation_stats = {
                "mean_activation": float(np.mean(layer_features)),
                "std_activation": float(np.std(layer_features)),
                "sparsity": float(np.mean(np.abs(layer_features) < 0.01)),
                "dead_neurons": float(np.mean(np.std(layer_features, axis=0) < 1e-6))
            }

            # Feature selectivity
            feature_selectivity = {
                "kurtosis": float(np.mean([self._safe_kurtosis(layer_features[:, i])
                                         for i in range(min(100, layer_features.shape[1]))])),
                "max_activation_ratio": float(np.mean(np.max(layer_features, axis=0) /
                                                    (np.std(layer_features, axis=0) + 1e-8)))
            }

            # Population dynamics
            population_dynamics = {
                "effective_rank": self._compute_effective_rank(layer_features),
                "participation_ratio": self._compute_participation_ratio(layer_features)
            }

            microscopic_results[f'layer_{layer_idx}'] = {
                **activation_stats,
                **feature_selectivity,
                **population_dynamics
            }

        return microscopic_results

    def _run_mesoscopic_analysis(self, features: torch.Tensor) -> Dict[str, Any]:
        """Run mesoscopic (layer-level) analysis"""

        num_layers = features.shape[1]
        mesoscopic_results = {}

        for layer_idx in range(num_layers):
            layer_features = features[:, layer_idx, :].numpy()

            # Dimensionality measures
            dimensionality = {
                "intrinsic_dim": self._compute_intrinsic_dimensionality(layer_features),
                "effective_dim": self._compute_effective_dimensionality(layer_features)
            }

            # Layer coherence
            coherence = {
                "layer_coherence": self._compute_layer_coherence(layer_features),
                "representational_stability": self._compute_representational_stability(layer_features)
            }

            # Information measures
            information = {
                "entropy_estimate": self._estimate_layer_entropy(layer_features),
                "mutual_information_estimate": self._estimate_layer_mutual_information(layer_features)
            }

            mesoscopic_results[f'layer_{layer_idx}'] = {
                **dimensionality,
                **coherence,
                **information
            }

        return mesoscopic_results

    def _run_macroscopic_analysis(self, features: torch.Tensor, dataset_name: str) -> Dict[str, Any]:
        """Run macroscopic (network-level) analysis"""

        # Information flow analysis
        information_flow = self._analyze_information_flow(features)

        # Layer progression analysis
        layer_progression = self._analyze_layer_progression(features)

        # Network-level properties
        network_properties = {
            "total_layers": features.shape[1],
            "feature_dimension": features.shape[2],
            "network_depth_utilization": self._compute_depth_utilization(features),
            "representational_geometry": self._analyze_representational_geometry(features)
        }

        return {
            "information_flow": information_flow,
            "layer_progression": layer_progression,
            "network_properties": network_properties
        }

    def _analyze_cross_scale_connections(self, micro: Dict, meso: Dict, macro: Dict) -> Dict[str, Any]:
        """Analyze connections between microscopic, mesoscopic, and macroscopic scales"""

        connections = {}

        # Extract layer-wise metrics for correlation analysis
        layers = sorted([k for k in micro.keys() if k.startswith('layer_')],
                       key=lambda x: int(x.split('_')[1]))

        if len(layers) > 1:
            # Micro-meso correlations
            micro_sparsity = [micro[l]['sparsity'] for l in layers]
            meso_intrinsic_dim = [meso[l]['intrinsic_dim'] for l in layers]

            connections['micro_meso_correlation'] = float(pearsonr(micro_sparsity, meso_intrinsic_dim)[0])

            # Meso-macro correlations
            meso_coherence = [meso[l]['layer_coherence'] for l in layers]
            macro_flow = macro['layer_progression']['information_flow_per_layer']

            if len(macro_flow) == len(meso_coherence):
                connections['meso_macro_correlation'] = float(pearsonr(meso_coherence, macro_flow)[0])

        # Emergent properties
        connections['emergent_properties'] = self._identify_emergent_properties(micro, meso, macro)

        return connections

    def run_cross_dataset_analysis(self, individual_results: List[AnalysisResult]) -> Dict[str, Any]:
        """Compare results across different datasets"""

        print(f"\n{'='*80}")
        print("RUNNING CROSS-DATASET ANALYSIS")
        print(f"{'='*80}")

        # Organize results by dataset and model type
        results_by_dataset = {}
        for result in individual_results:
            dataset = result.dataset_name
            if dataset not in results_by_dataset:
                results_by_dataset[dataset] = {"llm": [], "lvm": []}
            results_by_dataset[dataset][result.dataset_type].append(result)

        cross_dataset_analysis = {}

        # 1. Dataset characteristics comparison
        cross_dataset_analysis['dataset_characteristics'] = self._compare_dataset_characteristics(results_by_dataset)

        # 2. Model performance across datasets
        cross_dataset_analysis['model_transferability'] = self._analyze_model_transferability(results_by_dataset)

        # 3. Universal patterns vs dataset-specific patterns
        cross_dataset_analysis['universal_patterns'] = self._identify_universal_patterns(results_by_dataset)

        # 4. Dataset difficulty ranking
        cross_dataset_analysis['dataset_difficulty'] = self._rank_dataset_difficulty(results_by_dataset)

        return cross_dataset_analysis

    def run_cross_modal_analysis(self, individual_results: List[AnalysisResult]) -> Dict[str, Any]:
        """Compare language and vision models across datasets"""

        print(f"\n{'='*80}")
        print("RUNNING CROSS-MODAL ANALYSIS")
        print(f"{'='*80}")

        # Separate language and vision results
        llm_results = [r for r in individual_results if r.dataset_type == "llm"]
        lvm_results = [r for r in individual_results if r.dataset_type == "lvm"]

        cross_modal_analysis = {}

        # 1. Compare representation properties across modalities
        cross_modal_analysis['representation_comparison'] = self._compare_modality_representations(llm_results, lvm_results)

        # 2. Alignment potential analysis
        cross_modal_analysis['alignment_analysis'] = self._analyze_cross_modal_alignment_potential(llm_results, lvm_results)

        # 3. Scaling behavior comparison
        cross_modal_analysis['scaling_comparison'] = self._compare_scaling_behaviors(llm_results, lvm_results)

        # 4. Architecture impact analysis
        cross_modal_analysis['architecture_impact'] = self._analyze_architecture_impact(llm_results, lvm_results)

        self.stats["cross_modal_comparisons"] = len(llm_results) * len(lvm_results)

        return cross_modal_analysis

    def generate_comprehensive_report(self,
                                    individual_results: List[AnalysisResult],
                                    cross_dataset_analysis: Dict[str, Any],
                                    cross_modal_analysis: Dict[str, Any]):
        """Generate comprehensive analysis report"""

        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*80}")

        # Create comprehensive report
        report = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_individual_analyses": len(individual_results),
                "total_datasets": self.stats["total_datasets"],
                "total_models": self.stats["total_models"],
                "statistics": self.stats
            },
            "executive_summary": self._generate_executive_summary(individual_results, cross_dataset_analysis, cross_modal_analysis),
            "individual_analyses_summary": self._summarize_individual_analyses(individual_results),
            "cross_dataset_findings": cross_dataset_analysis,
            "cross_modal_findings": cross_modal_analysis,
            "key_insights": self._extract_key_insights(individual_results, cross_dataset_analysis, cross_modal_analysis),
            "recommendations": self._generate_recommendations(individual_results, cross_dataset_analysis, cross_modal_analysis)
        }

        # Save comprehensive report
        report_path = self.output_dir / "comprehensive_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        # Generate markdown summary
        self._generate_markdown_report(report)

        # Generate visualizations
        self._generate_comprehensive_visualizations(individual_results, cross_dataset_analysis, cross_modal_analysis)

        print(f"Comprehensive report saved to: {report_path}")
        return report

    # Utility methods for analyses
    def _safe_kurtosis(self, data):
        """Safely compute kurtosis"""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            return 0.0

    def _compute_effective_rank(self, features):
        """Compute effective rank of feature matrix"""
        U, s, V = np.linalg.svd(features, full_matrices=False)
        s_normalized = s / np.sum(s)
        entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-12))
        return float(np.exp(entropy))

    def _compute_participation_ratio(self, features):
        """Compute participation ratio"""
        mean_features = np.mean(features, axis=0)
        sum_of_squares = np.sum(mean_features**2)
        sum_of_fourth_powers = np.sum(mean_features**4)
        return float(sum_of_squares**2 / (len(mean_features) * sum_of_fourth_powers + 1e-12))

    def _compute_intrinsic_dimensionality(self, features):
        """Estimate intrinsic dimensionality using PCA"""
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(features)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.argmax(explained_variance >= 0.95) + 1
        return int(intrinsic_dim)

    def _compute_effective_dimensionality(self, features):
        """Compute effective dimensionality using eigenvalue analysis"""
        cov_matrix = np.cov(features.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = eigenvals[eigenvals > 1e-10]
        normalized_eigenvals = eigenvals / np.sum(eigenvals)
        entropy = -np.sum(normalized_eigenvals * np.log(normalized_eigenvals + 1e-12))
        return float(np.exp(entropy))

    def _compute_layer_coherence(self, features):
        """Compute how coherent the layer representations are"""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, features.shape[1]))
        reduced = pca.fit_transform(features)
        reconstructed = pca.inverse_transform(reduced)
        reconstruction_error = np.mean((features - reconstructed) ** 2)
        return float(1 / (1 + reconstruction_error))

    def _compute_representational_stability(self, features):
        """Compute representational stability across samples"""
        # Split into two halves and compute similarity
        n_samples = features.shape[0]
        if n_samples < 10:
            return 0.5

        half1 = features[:n_samples//2]
        half2 = features[n_samples//2:n_samples//2*2]

        mean1 = np.mean(half1, axis=0)
        mean2 = np.mean(half2, axis=0)

        correlation = np.corrcoef(mean1, mean2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def _estimate_layer_entropy(self, features):
        """Estimate entropy of layer activations"""
        # Use histogram-based entropy estimation
        flat_features = features.flatten()
        hist, _ = np.histogram(flat_features, bins=50)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zero probabilities
        entropy = -np.sum(prob * np.log2(prob))
        return float(entropy)

    def _estimate_layer_mutual_information(self, features):
        """Estimate mutual information within layer"""
        # Simplified MI estimation between first few features
        if features.shape[1] < 2:
            return 0.0

        # Use mutual information between pairs of features
        mi_scores = []
        for i in range(min(10, features.shape[1])):
            for j in range(i+1, min(10, features.shape[1])):
                try:
                    mi = mutual_info_score(
                        np.digitize(features[:, i], bins=np.linspace(features[:, i].min(), features[:, i].max(), 10)),
                        np.digitize(features[:, j], bins=np.linspace(features[:, j].min(), features[:, j].max(), 10))
                    )
                    mi_scores.append(mi)
                except:
                    continue

        return float(np.mean(mi_scores)) if mi_scores else 0.0

    def _analyze_information_flow(self, features):
        """Analyze information flow through layers"""
        num_layers = features.shape[1]

        # Compute layer-wise information content
        layer_entropies = []
        for layer_idx in range(num_layers):
            layer_features = features[:, layer_idx, :].numpy()
            entropy = self._estimate_layer_entropy(layer_features)
            layer_entropies.append(entropy)

        # Information flow metrics
        information_gain = np.diff(layer_entropies)
        total_information_change = layer_entropies[-1] - layer_entropies[0]

        return {
            "layer_entropies": [float(h) for h in layer_entropies],
            "information_gain_per_layer": [float(g) for g in information_gain],
            "total_information_change": float(total_information_change),
            "information_processing_efficiency": float(np.std(information_gain))
        }

    def _analyze_layer_progression(self, features):
        """Analyze how representations progress through layers"""
        num_layers = features.shape[1]

        # Layer-wise feature magnitude
        layer_magnitudes = []
        for layer_idx in range(num_layers):
            magnitude = torch.norm(features[:, layer_idx, :], dim=-1).mean().item()
            layer_magnitudes.append(magnitude)

        # Layer-wise sparsity
        layer_sparsities = []
        for layer_idx in range(num_layers):
            sparsity = (torch.abs(features[:, layer_idx, :]) < 0.01).float().mean().item()
            layer_sparsities.append(sparsity)

        return {
            "layer_magnitudes": layer_magnitudes,
            "layer_sparsities": layer_sparsities,
            "magnitude_progression": float(np.mean(np.diff(layer_magnitudes))),
            "sparsity_progression": float(np.mean(np.diff(layer_sparsities))),
            "information_flow_per_layer": layer_magnitudes  # Use magnitudes as proxy for information flow
        }

    def _compute_depth_utilization(self, features):
        """Compute how well the network utilizes its depth"""
        num_layers = features.shape[1]

        # Compute pairwise layer similarities
        layer_similarities = []
        for i in range(num_layers - 1):
            layer1 = features[:, i, :].flatten()
            layer2 = features[:, i+1, :].flatten()
            correlation = np.corrcoef(layer1, layer2)[0, 1]
            layer_similarities.append(correlation if not np.isnan(correlation) else 0.0)

        # High similarity = low utilization, low similarity = high utilization
        depth_utilization = 1.0 - np.mean(layer_similarities)
        return float(depth_utilization)

    def _analyze_representational_geometry(self, features):
        """Analyze the geometry of representations"""
        # Use final layer for geometry analysis
        final_layer = features[:, -1, :].numpy()

        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(final_layer)

        geometry_metrics = {
            "mean_pairwise_distance": float(np.mean(distances)),
            "distance_variance": float(np.var(distances)),
            "representational_radius": float(np.max(distances)),
            "representational_density": float(1.0 / (np.mean(distances) + 1e-8))
        }

        return geometry_metrics

    def _identify_emergent_properties(self, micro, meso, macro):
        """Identify properties that emerge at higher scales"""
        emergent = {}

        # Check for compression (information reduction)
        info_flow = macro.get('information_flow', {})
        if 'total_information_change' in info_flow:
            if info_flow['total_information_change'] < -0.1:  # Significant compression
                emergent['information_compression'] = True
                emergent['compression_strength'] = float(abs(info_flow['total_information_change']))

        # Check for specialization (increasing selectivity)
        layers = sorted([k for k in micro.keys() if k.startswith('layer_')])
        if len(layers) > 2:
            first_layer_sparsity = micro[layers[0]]['sparsity']
            last_layer_sparsity = micro[layers[-1]]['sparsity']
            if last_layer_sparsity > first_layer_sparsity + 0.1:
                emergent['progressive_specialization'] = True
                emergent['specialization_strength'] = float(last_layer_sparsity - first_layer_sparsity)

        return emergent

    # Analysis comparison methods
    def _compare_dataset_characteristics(self, results_by_dataset):
        """Compare characteristics across datasets"""
        dataset_chars = {}

        for dataset_name, results in results_by_dataset.items():
            all_results = results["llm"] + results["lvm"]
            if not all_results:
                continue

            # Extract key metrics
            sparsity_values = []
            coherence_values = []
            info_flow_values = []

            for result in all_results:
                # Extract sparsity from microscopic analysis
                micro = result.results.get('microscopic', {})
                for layer_key in micro:
                    if 'sparsity' in micro[layer_key]:
                        sparsity_values.append(micro[layer_key]['sparsity'])

                # Extract coherence from mesoscopic analysis
                meso = result.results.get('mesoscopic', {})
                for layer_key in meso:
                    if 'layer_coherence' in meso[layer_key]:
                        coherence_values.append(meso[layer_key]['layer_coherence'])

                # Extract information flow from macroscopic analysis
                macro = result.results.get('macroscopic', {})
                if 'information_flow' in macro and 'total_information_change' in macro['information_flow']:
                    info_flow_values.append(macro['information_flow']['total_information_change'])

            dataset_chars[dataset_name] = {
                'avg_sparsity': float(np.mean(sparsity_values)) if sparsity_values else 0.0,
                'avg_coherence': float(np.mean(coherence_values)) if coherence_values else 0.0,
                'avg_info_flow': float(np.mean(info_flow_values)) if info_flow_values else 0.0,
                'num_models': len(all_results)
            }

        return dataset_chars

    def _analyze_model_transferability(self, results_by_dataset):
        """Analyze how well models transfer across datasets"""
        transferability = {}

        # For each model that appears in multiple datasets
        model_appearances = {}
        for dataset_name, results in results_by_dataset.items():
            all_results = results["llm"] + results["lvm"]
            for result in all_results:
                model_name = result.model_name
                if model_name not in model_appearances:
                    model_appearances[model_name] = []
                model_appearances[model_name].append((dataset_name, result))

        # Analyze models that appear in multiple datasets
        for model_name, appearances in model_appearances.items():
            if len(appearances) > 1:
                # Compare performance across datasets
                performance_scores = []
                for dataset_name, result in appearances:
                    # Use coherence as a proxy for performance
                    meso = result.results.get('mesoscopic', {})
                    coherences = [meso[k].get('layer_coherence', 0) for k in meso if 'layer_coherence' in meso[k]]
                    avg_coherence = np.mean(coherences) if coherences else 0.0
                    performance_scores.append(avg_coherence)

                transferability[model_name] = {
                    'datasets': [app[0] for app in appearances],
                    'performance_variance': float(np.var(performance_scores)),
                    'mean_performance': float(np.mean(performance_scores)),
                    'transferability_score': float(1.0 / (1.0 + np.var(performance_scores)))
                }

        return transferability

    def _identify_universal_patterns(self, results_by_dataset):
        """Identify patterns that are universal vs dataset-specific"""
        universal_patterns = {}

        # Collect patterns across all datasets
        all_patterns = {
            'sparsity_progression': [],
            'coherence_progression': [],
            'information_flow_patterns': []
        }

        for dataset_name, results in results_by_dataset.items():
            all_results = results["llm"] + results["lvm"]

            for result in all_results:
                # Sparsity progression pattern
                micro = result.results.get('microscopic', {})
                layers = sorted([k for k in micro.keys() if k.startswith('layer_')],
                              key=lambda x: int(x.split('_')[1]))
                if len(layers) > 1:
                    sparsity_values = [micro[l].get('sparsity', 0) for l in layers]
                    sparsity_trend = np.polyfit(range(len(sparsity_values)), sparsity_values, 1)[0]
                    all_patterns['sparsity_progression'].append(sparsity_trend)

                # Similar for coherence and information flow
                meso = result.results.get('mesoscopic', {})
                if len(layers) > 1:
                    coherence_values = [meso[l].get('layer_coherence', 0) for l in layers if l in meso]
                    if len(coherence_values) > 1:
                        coherence_trend = np.polyfit(range(len(coherence_values)), coherence_values, 1)[0]
                        all_patterns['coherence_progression'].append(coherence_trend)

        # Determine if patterns are universal (low variance) or diverse (high variance)
        for pattern_name, values in all_patterns.items():
            if values:
                universal_patterns[pattern_name] = {
                    'mean': float(np.mean(values)),
                    'variance': float(np.var(values)),
                    'is_universal': float(np.var(values)) < 0.1,  # Low variance = universal
                    'strength': float(abs(np.mean(values)))
                }

        return universal_patterns

    def _rank_dataset_difficulty(self, results_by_dataset):
        """Rank datasets by difficulty based on model performance"""
        difficulty_ranking = {}

        for dataset_name, results in results_by_dataset.items():
            all_results = results["llm"] + results["lvm"]
            if not all_results:
                continue

            # Use multiple metrics to assess difficulty
            difficulty_scores = []

            for result in all_results:
                # Lower coherence = higher difficulty
                meso = result.results.get('mesoscopic', {})
                coherences = [meso[k].get('layer_coherence', 0) for k in meso if 'layer_coherence' in meso[k]]
                avg_coherence = np.mean(coherences) if coherences else 0.0

                # Higher sparsity = higher difficulty (harder to represent)
                micro = result.results.get('microscopic', {})
                sparsities = [micro[k].get('sparsity', 0) for k in micro if 'sparsity' in micro[k]]
                avg_sparsity = np.mean(sparsities) if sparsities else 0.0

                # Combine into difficulty score (higher = more difficult)
                difficulty_score = avg_sparsity / (avg_coherence + 1e-6)
                difficulty_scores.append(difficulty_score)

            difficulty_ranking[dataset_name] = {
                'difficulty_score': float(np.mean(difficulty_scores)),
                'difficulty_variance': float(np.var(difficulty_scores)),
                'num_models': len(all_results)
            }

        # Sort by difficulty
        sorted_difficulty = sorted(difficulty_ranking.items(), key=lambda x: x[1]['difficulty_score'], reverse=True)

        return {
            'rankings': sorted_difficulty,
            'easiest_dataset': sorted_difficulty[-1][0] if sorted_difficulty else None,
            'hardest_dataset': sorted_difficulty[0][0] if sorted_difficulty else None
        }

    def _compare_modality_representations(self, llm_results, lvm_results):
        """Compare representation properties between language and vision models"""
        comparison = {}

        # Extract key metrics for each modality
        for modality, results in [("language", llm_results), ("vision", lvm_results)]:
            metrics = {
                'sparsity': [],
                'coherence': [],
                'intrinsic_dim': [],
                'information_flow': []
            }

            for result in results:
                # Sparsity
                micro = result.results.get('microscopic', {})
                for layer_key in micro:
                    if 'sparsity' in micro[layer_key]:
                        metrics['sparsity'].append(micro[layer_key]['sparsity'])

                # Coherence
                meso = result.results.get('mesoscopic', {})
                for layer_key in meso:
                    if 'layer_coherence' in meso[layer_key]:
                        metrics['coherence'].append(meso[layer_key]['layer_coherence'])
                    if 'intrinsic_dim' in meso[layer_key]:
                        metrics['intrinsic_dim'].append(meso[layer_key]['intrinsic_dim'])

                # Information flow
                macro = result.results.get('macroscopic', {})
                if 'information_flow' in macro and 'total_information_change' in macro['information_flow']:
                    metrics['information_flow'].append(macro['information_flow']['total_information_change'])

            comparison[modality] = {
                metric: {
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                    'count': len(values)
                }
                for metric, values in metrics.items()
            }

        # Statistical comparisons
        if llm_results and lvm_results:
            comparison['statistical_differences'] = {}
            for metric in ['sparsity', 'coherence', 'intrinsic_dim']:
                if (comparison['language'][metric]['count'] > 0 and
                    comparison['vision'][metric]['count'] > 0):

                    lang_mean = comparison['language'][metric]['mean']
                    vis_mean = comparison['vision'][metric]['mean']

                    comparison['statistical_differences'][metric] = {
                        'language_higher': lang_mean > vis_mean,
                        'difference_magnitude': float(abs(lang_mean - vis_mean)),
                        'relative_difference': float(abs(lang_mean - vis_mean) / (max(lang_mean, vis_mean) + 1e-6))
                    }

        return comparison

    def _analyze_cross_modal_alignment_potential(self, llm_results, lvm_results):
        """Analyze potential for cross-modal alignment"""
        alignment_analysis = {}

        # Find models that could potentially align well
        potential_alignments = []

        for llm_result in llm_results:
            for lvm_result in lvm_results:
                # Check if they're from the same dataset
                if llm_result.dataset_name == lvm_result.dataset_name:
                    # Compute alignment potential based on representational similarity
                    llm_coherence = np.mean([
                        llm_result.results.get('mesoscopic', {}).get(k, {}).get('layer_coherence', 0)
                        for k in llm_result.results.get('mesoscopic', {})
                    ])

                    lvm_coherence = np.mean([
                        lvm_result.results.get('mesoscopic', {}).get(k, {}).get('layer_coherence', 0)
                        for k in lvm_result.results.get('mesoscopic', {})
                    ])

                    # Similar coherence suggests better alignment potential
                    coherence_similarity = 1.0 - abs(llm_coherence - lvm_coherence)

                    potential_alignments.append({
                        'llm_model': llm_result.model_name,
                        'lvm_model': lvm_result.model_name,
                        'dataset': llm_result.dataset_name,
                        'alignment_potential': coherence_similarity,
                        'llm_coherence': float(llm_coherence),
                        'lvm_coherence': float(lvm_coherence)
                    })

        # Sort by alignment potential
        potential_alignments.sort(key=lambda x: x['alignment_potential'], reverse=True)

        alignment_analysis = {
            'potential_alignments': potential_alignments[:20],  # Top 20
            'best_alignment_potential': potential_alignments[0]['alignment_potential'] if potential_alignments else 0.0,
            'average_alignment_potential': float(np.mean([a['alignment_potential'] for a in potential_alignments])) if potential_alignments else 0.0
        }

        return alignment_analysis

    def _compare_scaling_behaviors(self, llm_results, lvm_results):
        """Compare how language and vision models scale"""
        scaling_comparison = {}

        # Analyze scaling patterns for each modality
        for modality, results in [("language", llm_results), ("vision", lvm_results)]:
            model_sizes = []
            performance_metrics = []

            for result in results:
                # Get model size
                size = result.metadata.get('num_params', 0)
                if size > 0:
                    model_sizes.append(size)

                    # Use coherence as performance metric
                    meso = result.results.get('mesoscopic', {})
                    coherences = [meso[k].get('layer_coherence', 0) for k in meso if 'layer_coherence' in meso[k]]
                    avg_coherence = np.mean(coherences) if coherences else 0.0
                    performance_metrics.append(avg_coherence)

            if len(model_sizes) > 1:
                # Fit scaling law
                log_sizes = np.log(model_sizes)
                correlation = np.corrcoef(log_sizes, performance_metrics)[0, 1]

                scaling_comparison[modality] = {
                    'size_range': [float(min(model_sizes)), float(max(model_sizes))],
                    'performance_range': [float(min(performance_metrics)), float(max(performance_metrics))],
                    'scaling_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                    'num_models': len(model_sizes)
                }

        return scaling_comparison

    def _analyze_architecture_impact(self, llm_results, lvm_results):
        """Analyze impact of different architectures"""
        architecture_impact = {}

        # Group by architecture families
        for modality, results in [("language", llm_results), ("vision", lvm_results)]:
            arch_groups = {}

            for result in results:
                model_name = result.model_name.lower()

                # Determine architecture family
                if modality == "language":
                    if "bloom" in model_name:
                        arch_family = "bloom"
                    elif "llama" in model_name:
                        arch_family = "llama"
                    elif "gpt" in model_name:
                        arch_family = "gpt"
                    elif "pythia" in model_name:
                        arch_family = "pythia"
                    else:
                        arch_family = "other"
                else:  # vision
                    if "vit" in model_name:
                        arch_family = "vit"
                    elif "deit" in model_name:
                        arch_family = "deit"
                    elif "clip" in model_name:
                        arch_family = "clip"
                    else:
                        arch_family = "other"

                if arch_family not in arch_groups:
                    arch_groups[arch_family] = []
                arch_groups[arch_family].append(result)

            # Analyze each architecture family
            arch_analysis = {}
            for arch_family, arch_results in arch_groups.items():
                if len(arch_results) > 0:
                    # Compute average metrics
                    coherences = []
                    sparsities = []

                    for result in arch_results:
                        # Coherence
                        meso = result.results.get('mesoscopic', {})
                        for layer_key in meso:
                            if 'layer_coherence' in meso[layer_key]:
                                coherences.append(meso[layer_key]['layer_coherence'])

                        # Sparsity
                        micro = result.results.get('microscopic', {})
                        for layer_key in micro:
                            if 'sparsity' in micro[layer_key]:
                                sparsities.append(micro[layer_key]['sparsity'])

                    arch_analysis[arch_family] = {
                        'num_models': len(arch_results),
                        'avg_coherence': float(np.mean(coherences)) if coherences else 0.0,
                        'avg_sparsity': float(np.mean(sparsities)) if sparsities else 0.0,
                        'coherence_std': float(np.std(coherences)) if coherences else 0.0,
                        'sparsity_std': float(np.std(sparsities)) if sparsities else 0.0
                    }

            architecture_impact[modality] = arch_analysis

        return architecture_impact

    def _generate_executive_summary(self, individual_results, cross_dataset_analysis, cross_modal_analysis):
        """Generate executive summary of findings"""
        summary = {}

        # Key statistics
        summary['key_statistics'] = {
            'total_analyses': len(individual_results),
            'unique_datasets': len(set(r.dataset_name for r in individual_results)),
            'unique_models': len(set(r.model_name for r in individual_results)),
            'language_models': len([r for r in individual_results if r.dataset_type == "llm"]),
            'vision_models': len([r for r in individual_results if r.dataset_type == "lvm"])
        }

        # Top-level findings
        summary['top_findings'] = []

        # Dataset difficulty
        if 'dataset_difficulty' in cross_dataset_analysis:
            difficulty = cross_dataset_analysis['dataset_difficulty']
            if 'hardest_dataset' in difficulty and difficulty['hardest_dataset']:
                summary['top_findings'].append(f"Most challenging dataset: {difficulty['hardest_dataset']}")
            if 'easiest_dataset' in difficulty and difficulty['easiest_dataset']:
                summary['top_findings'].append(f"Most accessible dataset: {difficulty['easiest_dataset']}")

        # Cross-modal differences
        if 'representation_comparison' in cross_modal_analysis:
            rep_comp = cross_modal_analysis['representation_comparison']
            if 'statistical_differences' in rep_comp:
                for metric, diff in rep_comp['statistical_differences'].items():
                    if diff['relative_difference'] > 0.2:  # Significant difference
                        modality = "Language" if diff['language_higher'] else "Vision"
                        summary['top_findings'].append(f"{modality} models show higher {metric}")

        # Alignment potential
        if 'alignment_analysis' in cross_modal_analysis:
            alignment = cross_modal_analysis['alignment_analysis']
            best_potential = alignment.get('best_alignment_potential', 0)
            if best_potential > 0.8:
                summary['top_findings'].append("High cross-modal alignment potential detected")
            elif best_potential < 0.5:
                summary['top_findings'].append("Limited cross-modal alignment potential")

        return summary

    def _summarize_individual_analyses(self, individual_results):
        """Summarize individual analysis results"""
        summary = {}

        # Group by dataset type
        by_type = {"llm": [], "lvm": []}
        for result in individual_results:
            by_type[result.dataset_type].append(result)

        for model_type, results in by_type.items():
            if not results:
                continue

            # Extract key metrics
            sparsity_values = []
            coherence_values = []
            intrinsic_dims = []

            for result in results:
                # Sparsity
                micro = result.results.get('microscopic', {})
                for layer_key in micro:
                    if 'sparsity' in micro[layer_key]:
                        sparsity_values.append(micro[layer_key]['sparsity'])

                # Coherence
                meso = result.results.get('mesoscopic', {})
                for layer_key in meso:
                    if 'layer_coherence' in meso[layer_key]:
                        coherence_values.append(meso[layer_key]['layer_coherence'])
                    if 'intrinsic_dim' in meso[layer_key]:
                        intrinsic_dims.append(meso[layer_key]['intrinsic_dim'])

            summary[model_type] = {
                'count': len(results),
                'avg_sparsity': float(np.mean(sparsity_values)) if sparsity_values else 0.0,
                'avg_coherence': float(np.mean(coherence_values)) if coherence_values else 0.0,
                'avg_intrinsic_dim': float(np.mean(intrinsic_dims)) if intrinsic_dims else 0.0,
                'sparsity_range': [float(min(sparsity_values)), float(max(sparsity_values))] if sparsity_values else [0.0, 0.0],
                'coherence_range': [float(min(coherence_values)), float(max(coherence_values))] if coherence_values else [0.0, 0.0]
            }

        return summary

    def _extract_key_insights(self, individual_results, cross_dataset_analysis, cross_modal_analysis):
        """Extract key insights from all analyses"""
        insights = []

        # Universal patterns
        if 'universal_patterns' in cross_dataset_analysis:
            patterns = cross_dataset_analysis['universal_patterns']
            for pattern_name, pattern_data in patterns.items():
                if pattern_data.get('is_universal', False):
                    insights.append(f"Universal pattern detected: {pattern_name} shows consistent behavior across datasets")

        # Modality differences
        if 'representation_comparison' in cross_modal_analysis:
            rep_comp = cross_modal_analysis['representation_comparison']
            if 'statistical_differences' in rep_comp:
                for metric, diff in rep_comp['statistical_differences'].items():
                    if diff['relative_difference'] > 0.3:
                        insights.append(f"Significant modality difference in {metric}: {diff['relative_difference']:.1%} relative difference")

        # Architecture effects
        if 'architecture_impact' in cross_modal_analysis:
            arch_impact = cross_modal_analysis['architecture_impact']
            for modality, arch_data in arch_impact.items():
                best_arch = max(arch_data.items(), key=lambda x: x[1].get('avg_coherence', 0))
                if best_arch[1]['num_models'] > 1:
                    insights.append(f"Best performing {modality} architecture: {best_arch[0]} (coherence: {best_arch[1]['avg_coherence']:.3f})")

        # Dataset complexity
        if 'dataset_difficulty' in cross_dataset_analysis:
            difficulty = cross_dataset_analysis['dataset_difficulty']
            if 'rankings' in difficulty and len(difficulty['rankings']) > 2:
                hardest = difficulty['rankings'][0]
                easiest = difficulty['rankings'][-1]
                ratio = hardest[1]['difficulty_score'] / (easiest[1]['difficulty_score'] + 1e-6)
                if ratio > 2:
                    insights.append(f"Large complexity gap: {hardest[0]} is {ratio:.1f}x more difficult than {easiest[0]}")

        return insights

    def _generate_recommendations(self, individual_results, cross_dataset_analysis, cross_modal_analysis):
        """Generate actionable recommendations"""
        recommendations = []

        # Model selection recommendations
        if 'alignment_analysis' in cross_modal_analysis:
            alignment = cross_modal_analysis['alignment_analysis']
            best_alignments = alignment.get('potential_alignments', [])[:3]
            for alignment_pair in best_alignments:
                recommendations.append(
                    f"Recommended model pair for {alignment_pair['dataset']}: "
                    f"{alignment_pair['llm_model']} + {alignment_pair['lvm_model']} "
                    f"(potential: {alignment_pair['alignment_potential']:.3f})"
                )

        # Dataset recommendations
        if 'dataset_difficulty' in cross_dataset_analysis:
            difficulty = cross_dataset_analysis['dataset_difficulty']
            if 'easiest_dataset' in difficulty:
                recommendations.append(f"For initial experiments: Start with {difficulty['easiest_dataset']} (most accessible)")
            if 'hardest_dataset' in difficulty:
                recommendations.append(f"For challenging evaluation: Use {difficulty['hardest_dataset']} (most difficult)")

        # Architecture recommendations
        if 'architecture_impact' in cross_modal_analysis:
            arch_impact = cross_modal_analysis['architecture_impact']
            for modality, arch_data in arch_impact.items():
                if arch_data:
                    best_arch = max(arch_data.items(), key=lambda x: x[1].get('avg_coherence', 0))
                    recommendations.append(f"For {modality} tasks: Consider {best_arch[0]} architecture")

        # Transferability recommendations
        if 'model_transferability' in cross_dataset_analysis:
            transfer = cross_dataset_analysis['model_transferability']
            best_transfer = max(transfer.items(), key=lambda x: x[1]['transferability_score']) if transfer else None
            if best_transfer:
                recommendations.append(f"Most transferable model: {best_transfer[0]} (score: {best_transfer[1]['transferability_score']:.3f})")

        return recommendations

    def _generate_markdown_report(self, report):
        """Generate markdown summary report"""
        md_content = f"""# Enhanced Multiscale Analysis Report

Generated: {report['metadata']['analysis_timestamp']}

## Executive Summary

### Key Statistics
- Total Analyses: {report['metadata']['total_individual_analyses']}
- Datasets: {report['metadata']['total_datasets']}
- Models: {report['metadata']['total_models']}
- Success Rate: {report['metadata']['statistics']['successful_analyses']}/{report['metadata']['statistics']['total_analyses']} ({100*report['metadata']['statistics']['successful_analyses']/max(1,report['metadata']['statistics']['total_analyses']):.1f}%)

### Top Findings
"""

        for finding in report['executive_summary'].get('top_findings', []):
            md_content += f"- {finding}\n"

        md_content += f"""

## Key Insights

"""
        for insight in report.get('key_insights', []):
            md_content += f"- {insight}\n"

        md_content += f"""

## Recommendations

"""
        for rec in report.get('recommendations', []):
            md_content += f"- {rec}\n"

        # Save markdown report
        md_path = self.output_dir / "analysis_summary.md"
        with open(md_path, 'w') as f:
            f.write(md_content)

        print(f"Markdown report saved to: {md_path}")

    def _generate_comprehensive_visualizations(self, individual_results, cross_dataset_analysis, cross_modal_analysis):
        """Generate comprehensive visualizations"""

        print("Generating visualizations...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Dataset comparison visualization
        self._plot_dataset_comparison(cross_dataset_analysis)

        # 2. Cross-modal comparison
        self._plot_cross_modal_comparison(cross_modal_analysis)

        # 3. Individual results overview
        self._plot_individual_results_overview(individual_results)

        # 4. Architecture comparison
        self._plot_architecture_comparison(cross_modal_analysis)

        print(f"Visualizations saved to: {self.output_dir / 'visualizations'}")

    def _plot_dataset_comparison(self, cross_dataset_analysis):
        """Plot dataset comparison"""
        if 'dataset_characteristics' not in cross_dataset_analysis:
            return

        chars = cross_dataset_analysis['dataset_characteristics']
        datasets = list(chars.keys())

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Sparsity comparison
        sparsity_values = [chars[d]['avg_sparsity'] for d in datasets]
        axes[0].bar(datasets, sparsity_values, color='skyblue')
        axes[0].set_title('Dataset Sparsity Comparison')
        axes[0].set_ylabel('Average Sparsity')
        axes[0].tick_params(axis='x', rotation=45)

        # Coherence comparison
        coherence_values = [chars[d]['avg_coherence'] for d in datasets]
        axes[1].bar(datasets, coherence_values, color='lightgreen')
        axes[1].set_title('Dataset Coherence Comparison')
        axes[1].set_ylabel('Average Coherence')
        axes[1].tick_params(axis='x', rotation=45)

        # Information flow comparison
        info_flow_values = [chars[d]['avg_info_flow'] for d in datasets]
        axes[2].bar(datasets, info_flow_values, color='lightcoral')
        axes[2].set_title('Dataset Information Flow Comparison')
        axes[2].set_ylabel('Average Information Flow')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cross_modal_comparison(self, cross_modal_analysis):
        """Plot cross-modal comparison"""
        if 'representation_comparison' not in cross_modal_analysis:
            return

        rep_comp = cross_modal_analysis['representation_comparison']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics = ['sparsity', 'coherence', 'intrinsic_dim']
        modalities = ['language', 'vision']

        # Comparison bars for each metric
        for i, metric in enumerate(metrics):
            if i >= 3:
                break

            row = i // 2
            col = i % 2

            lang_mean = rep_comp['language'][metric]['mean']
            vis_mean = rep_comp['vision'][metric]['mean']
            lang_std = rep_comp['language'][metric]['std']
            vis_std = rep_comp['vision'][metric]['std']

            x = np.arange(2)
            means = [lang_mean, vis_mean]
            stds = [lang_std, vis_std]

            axes[row, col].bar(x, means, yerr=stds, capsize=5,
                              color=['blue', 'red'], alpha=0.7)
            axes[row, col].set_title(f'{metric.capitalize()} Comparison')
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(modalities)
            axes[row, col].set_ylabel(metric.capitalize())

        # Alignment potential in last subplot
        if 'alignment_analysis' in cross_modal_analysis:
            alignment = cross_modal_analysis['alignment_analysis']
            potential_alignments = alignment.get('potential_alignments', [])[:10]

            if potential_alignments:
                pairs = [f"{a['llm_model'][:10]}+{a['lvm_model'][:10]}" for a in potential_alignments]
                potentials = [a['alignment_potential'] for a in potential_alignments]

                axes[1, 1].barh(range(len(pairs)), potentials, color='green', alpha=0.7)
                axes[1, 1].set_title('Top Alignment Potentials')
                axes[1, 1].set_yticks(range(len(pairs)))
                axes[1, 1].set_yticklabels(pairs, fontsize=8)
                axes[1, 1].set_xlabel('Alignment Potential')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'cross_modal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_individual_results_overview(self, individual_results):
        """Plot overview of individual results"""
        # Create scatter plot of key metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract data for plotting
        llm_results = [r for r in individual_results if r.dataset_type == "llm"]
        lvm_results = [r for r in individual_results if r.dataset_type == "lvm"]

        # Sparsity vs Coherence
        for results, label, color in [(llm_results, "Language", "blue"), (lvm_results, "Vision", "red")]:
            sparsity_vals = []
            coherence_vals = []

            for result in results:
                # Extract average sparsity and coherence
                micro = result.results.get('microscopic', {})
                meso = result.results.get('mesoscopic', {})

                sparsities = [micro[k].get('sparsity', 0) for k in micro if 'sparsity' in micro[k]]
                coherences = [meso[k].get('layer_coherence', 0) for k in meso if 'layer_coherence' in meso[k]]

                if sparsities and coherences:
                    sparsity_vals.append(np.mean(sparsities))
                    coherence_vals.append(np.mean(coherences))

            if sparsity_vals and coherence_vals:
                axes[0, 0].scatter(sparsity_vals, coherence_vals, label=label, color=color, alpha=0.6)

        axes[0, 0].set_xlabel('Sparsity')
        axes[0, 0].set_ylabel('Coherence')
        axes[0, 0].set_title('Sparsity vs Coherence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Model size vs performance (if available)
        for results, label, color in [(llm_results, "Language", "blue"), (lvm_results, "Vision", "red")]:
            sizes = []
            performances = []

            for result in results:
                size = result.metadata.get('num_params', 0)
                if size > 0:
                    meso = result.results.get('mesoscopic', {})
                    coherences = [meso[k].get('layer_coherence', 0) for k in meso if 'layer_coherence' in meso[k]]
                    if coherences:
                        sizes.append(size / 1e6)  # Convert to millions
                        performances.append(np.mean(coherences))

            if sizes and performances:
                axes[0, 1].scatter(sizes, performances, label=label, color=color, alpha=0.6)

        axes[0, 1].set_xlabel('Model Size (M parameters)')
        axes[0, 1].set_ylabel('Average Coherence')
        axes[0, 1].set_title('Model Size vs Performance')
        axes[0, 1].set_xscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Dataset distribution
        dataset_counts = {}
        for result in individual_results:
            dataset = result.dataset_name
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        if dataset_counts:
            datasets = list(dataset_counts.keys())
            counts = list(dataset_counts.values())
            axes[1, 0].bar(datasets, counts, color='green', alpha=0.7)
            axes[1, 0].set_title('Models per Dataset')
            axes[1, 0].set_ylabel('Number of Models')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # Model type distribution
        type_counts = {"Language": len(llm_results), "Vision": len(lvm_results)}
        axes[1, 1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
                      colors=['blue', 'red'], alpha=0.7)
        axes[1, 1].set_title('Model Type Distribution')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'individual_results_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_architecture_comparison(self, cross_modal_analysis):
        """Plot architecture comparison"""
        if 'architecture_impact' not in cross_modal_analysis:
            return

        arch_impact = cross_modal_analysis['architecture_impact']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, (modality, arch_data) in enumerate(arch_impact.items()):
            if not arch_data:
                continue

            architectures = list(arch_data.keys())
            coherences = [arch_data[arch]['avg_coherence'] for arch in architectures]
            sparsities = [arch_data[arch]['avg_sparsity'] for arch in architectures]

            # Coherence comparison
            x = np.arange(len(architectures))
            bars = axes[i].bar(x, coherences, color='lightblue', alpha=0.7)

            # Add sparsity as error bars or secondary axis
            ax2 = axes[i].twinx()
            ax2.plot(x, sparsities, 'ro-', alpha=0.7)
            ax2.set_ylabel('Sparsity', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            axes[i].set_title(f'{modality.capitalize()} Architecture Comparison')
            axes[i].set_xlabel('Architecture')
            axes[i].set_ylabel('Coherence', color='blue')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(architectures, rotation=45)
            axes[i].tick_params(axis='y', labelcolor='blue')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_complete_analysis(self, max_models_per_dataset: int = 10) -> Dict[str, Any]:
        """Run complete enhanced multiscale analysis"""

        self.stats["start_time"] = datetime.now()

        print(f"\n{'='*100}")
        print("ENHANCED MULTISCALE ANALYSIS PIPELINE")
        print(f"Start time: {self.stats['start_time']}")
        print(f"{'='*100}")

        # 1. Discover available features
        available_features = self.discover_available_features()

        # 2. Run individual multiscale analyses
        individual_results = self.run_individual_multiscale_analysis(
            available_features, max_models_per_dataset
        )

        # 3. Run cross-dataset analysis
        cross_dataset_analysis = self.run_cross_dataset_analysis(individual_results)

        # 4. Run cross-modal analysis
        cross_modal_analysis = self.run_cross_modal_analysis(individual_results)

        # 5. Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report(
            individual_results, cross_dataset_analysis, cross_modal_analysis
        )

        self.stats["end_time"] = datetime.now()
        self.stats["total_time"] = self.stats["end_time"] - self.stats["start_time"]

        print(f"\n{'='*100}")
        print("ENHANCED MULTISCALE ANALYSIS COMPLETED")
        print(f"End time: {self.stats['end_time']}")
        print(f"Total time: {self.stats['total_time']}")
        print(f"Success rate: {self.stats['successful_analyses']}/{self.stats['total_analyses']} ({100*self.stats['successful_analyses']/max(1,self.stats['total_analyses']):.1f}%)")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*100}")

        return comprehensive_report


def main():
    """Main function for enhanced multiscale analysis"""

    parser = argparse.ArgumentParser(
        description="Enhanced Multiscale Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis with default settings
  python enhanced_multiscale_analysis.py

  # Run with custom directories and limits
  python enhanced_multiscale_analysis.py --features_dir ./results/features_exhaustive --max_models_per_dataset 5

  # Run with specific output directory
  python enhanced_multiscale_analysis.py --output_dir ./results/my_analysis
        """)

    parser.add_argument("--features_dir", type=str, default="./results/features_exhaustive",
                       help="Directory containing extracted features")
    parser.add_argument("--output_dir", type=str, default="./results/enhanced_multiscale_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--config_dir", type=str, default="./config",
                       help="Configuration directory")
    parser.add_argument("--max_models_per_dataset", type=int, default=10,
                       help="Maximum models to analyze per dataset")

    args = parser.parse_args()

    # Initialize and run analyzer
    analyzer = EnhancedMultiscaleAnalyzer(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        config_dir=args.config_dir
    )

    # Run complete analysis
    report = analyzer.run_complete_analysis(
        max_models_per_dataset=args.max_models_per_dataset
    )

    return report


if __name__ == "__main__":
    main()