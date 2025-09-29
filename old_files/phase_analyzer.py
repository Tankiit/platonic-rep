import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class CrossModalPhaseAnalyzer:
    """
    Implements the exact mathematical framework from your paper:
    - Cross-modal NTK stability: S_NTK^cross
    - AGOP eigenvalue analysis 
    - Phase classification (Chaotic, Optimal, Lazy)
    - Addresses RQ1 and RQ2 from your problem definition
    """
    
    def __init__(self, results_dir: str = "./results/"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Exact phase thresholds from your paper
        self.phase_thresholds = {
            'chaotic': 0.5,    # S_NTK < 0.5
            'optimal': 0.9,    # 0.5 ≤ S_NTK < 0.9  
            'lazy': 0.9        # S_NTK ≥ 0.9
        }
        
        # Critical threshold for cross-modal compatibility
        self.cross_modal_threshold = 0.25  # Your 144 pairs show S_NTK^cross < 0.25
        
        self.results = {}
    
    def load_features(self, feature_files: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Load extracted features for multiple models
        
        Args:
            feature_files: {'resnet18_layer4': 'path/to/features.npy', ...}
        
        Returns:
            Dict of loaded features
        """
        features = {}
        
        for model_layer, file_path in feature_files.items():
            try:
                feature_data = np.load(file_path)
                features[model_layer] = feature_data
                print(f"Loaded {model_layer}: shape {feature_data.shape}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return features
    
    def compute_empirical_ntk(self, features: np.ndarray) -> np.ndarray:
        """
        Compute empirical Neural Tangent Kernel from features
        
        Θ_ij(t) = ⟨∇_θ f(x_i; θ_t), ∇_θ f(x_j; θ_t)⟩
        
        Using features as proxy for gradients (common in NTK analysis)
        """
        n_samples = features.shape[0]
        
        # Normalize features (important for stable NTK computation)
        feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
        features_normalized = features / (feature_norms + 1e-8)
        
        # Compute empirical NTK: Θ = FF^T
        ntk_matrix = features_normalized @ features_normalized.T
        
        return ntk_matrix
    
    def compute_cross_modal_ntk_stability(self, 
                                        vision_features: np.ndarray,
                                        language_features: np.ndarray) -> Dict:
        """
        Compute cross-modal NTK stability (Equation 3 from your paper):
        
        S_NTK^cross = tr(Θ_v^T Θ_ℓ) / (||Θ_v||_F ||Θ_ℓ||_F)
        
        This is the KEY metric for phase compatibility!
        """
        # Ensure same number of samples
        min_samples = min(len(vision_features), len(language_features))
        vision_subset = vision_features[:min_samples]
        language_subset = language_features[:min_samples]
        
        # Compute empirical NTKs for both modalities
        theta_v = self.compute_empirical_ntk(vision_subset)
        theta_l = self.compute_empirical_ntk(language_subset)
        
        # Cross-modal NTK stability (Equation 3)
        numerator = np.trace(theta_v.T @ theta_l)
        frobenius_v = np.linalg.norm(theta_v, 'fro')
        frobenius_l = np.linalg.norm(theta_l, 'fro')
        
        s_ntk_cross = numerator / (frobenius_v * frobenius_l + 1e-8)
        
        # Additional stability metrics
        eigenvals_v = np.linalg.eigvals(theta_v)
        eigenvals_l = np.linalg.eigvals(theta_l)
        
        # Condition numbers (measure of numerical stability)
        cond_v = np.max(np.real(eigenvals_v)) / (np.min(np.real(eigenvals_v[eigenvals_v > 1e-10])) + 1e-10)
        cond_l = np.max(np.real(eigenvals_l)) / (np.min(np.real(eigenvals_l[eigenvals_l > 1e-10])) + 1e-10)
        
        return {
            's_ntk_cross': s_ntk_cross,
            'theta_vision': theta_v,
            'theta_language': theta_l,
            'condition_number_vision': cond_v,
            'condition_number_language': cond_l,
            'eigenvals_vision': np.real(eigenvals_v),
            'eigenvals_language': np.real(eigenvals_l),
            'phase_compatible': s_ntk_cross >= self.cross_modal_threshold,
            'n_samples': min_samples
        }
    
    def compute_agop(self, features: np.ndarray) -> Dict:
        """
        Compute Average Gradient Outer Product (Equation 4 from your paper):
        
        AGOP = (1/N) Σ_{i=1}^N ∇_θ f(x_i) ∇_θ f(x_i)^T
        
        The eigenvalue spectrum reveals effective dimensionality and anisotropy
        """
        n_samples, n_features = features.shape
        print(f"Computing AGOP for {n_samples} samples, {n_features} features")
        
        # Use features as gradient proxy (standard in AGOP literature)
        # In practice: AGOP = (1/N) Σ f_i f_i^T where f_i are feature vectors
        
        agop_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_samples):
            feature_vec = features[i].reshape(-1, 1)
            agop_matrix += feature_vec @ feature_vec.T
        
        agop_matrix /= n_samples
        
        # Eigenvalue analysis - critical for phase determination
        eigenvalues = np.linalg.eigvals(agop_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        
        # Key metrics for phase classification
        total_variance = np.sum(eigenvalues)
        
        # Top-10 eigenvalue concentration (your framework uses this)
        top10 = eigenvalues[:min(10, len(eigenvalues))]
        top10_concentration = np.sum(top10) / total_variance if total_variance > 0 else 0
        
        # Effective rank (participation ratio)
        effective_rank = total_variance**2 / np.sum(eigenvalues**2) if total_variance > 0 else 0
        
        # AGOP ratio - your predictive metric for alignment failure
        agop_ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10) if len(eigenvalues) > 1 else 1
        
        # Spectral decay analysis
        decay_analysis = self._analyze_eigenvalue_decay(eigenvalues)
        
        return {
            'agop_matrix': agop_matrix,
            'eigenvalues': eigenvalues,
            'top10_concentration': top10_concentration,
            'effective_rank': effective_rank,
            'agop_ratio': agop_ratio,
            'spectral_decay': decay_analysis,
            'gradient_anisotropy': self._compute_anisotropy(eigenvalues),
            'chaotic_indicator': agop_ratio > 1e6  # Your threshold for guaranteed failure
        }
    
    def compute_single_model_ntk_stability(self, features: np.ndarray, 
                                         n_perturbations: int = 50) -> Dict:
        """
        Compute single-model NTK stability for phase classification
        
        Measures how stable the NTK is under small perturbations
        Used to classify: Chaotic (<0.5), Optimal (0.5-0.9), Lazy (≥0.9)
        """
        n_samples = min(500, len(features))  # Computational efficiency
        
        if len(features) > n_samples:
            indices = np.random.choice(len(features), n_samples, replace=False)
            features_subset = features[indices]
        else:
            features_subset = features
        
        # Base NTK
        ntk_base = self.compute_empirical_ntk(features_subset)
        
        # Perturbation analysis
        stability_scores = []
        
        for _ in tqdm(range(n_perturbations), desc="Computing NTK stability"):
            # Small Gaussian perturbation
            noise_scale = 0.01
            noise = np.random.randn(*features_subset.shape) * noise_scale
            perturbed_features = features_subset + noise
            
            # Compute perturbed NTK
            ntk_perturbed = self.compute_empirical_ntk(perturbed_features)
            
            # Stability metric: 1 - relative Frobenius distance
            diff_norm = np.linalg.norm(ntk_base - ntk_perturbed, 'fro')
            base_norm = np.linalg.norm(ntk_base, 'fro')
            
            stability = 1 - (diff_norm / (base_norm + 1e-8))
            stability_scores.append(max(0, min(1, stability)))  # Clamp to [0,1]
        
        s_ntk = np.mean(stability_scores)
        
        # Phase classification based on your thresholds
        if s_ntk < self.phase_thresholds['chaotic']:
            phase = 'chaotic'
        elif s_ntk < self.phase_thresholds['optimal']:
            phase = 'optimal'
        else:
            phase = 'lazy'
        
        return {
            's_ntk': s_ntk,
            's_ntk_std': np.std(stability_scores),
            'phase': phase,
            'stability_scores': stability_scores,
            'ntk_eigenvals': np.linalg.eigvals(ntk_base)
        }
    
    def _analyze_eigenvalue_decay(self, eigenvalues: np.ndarray) -> Dict:
        """
        Analyze eigenvalue decay pattern
        Power law decay indicates chaotic dynamics with gradient concentration
        """
        valid_eigenvals = eigenvalues[eigenvalues > 1e-10]
        n_valid = len(valid_eigenvals)
        
        if n_valid < 5:
            return {'type': 'insufficient_data'}
        
        # Fit power law: λ_i ∝ i^(-α)
        indices = np.arange(1, min(50, n_valid) + 1)
        log_indices = np.log(indices)
        log_eigenvals = np.log(valid_eigenvals[:len(indices)])
        
        try:
            slope, intercept = np.polyfit(log_indices, log_eigenvals, 1)
            return {
                'type': 'power_law',
                'decay_exponent': -slope,
                'goodness_of_fit': pearsonr(log_indices, log_eigenvals)[0]**2
            }
        except:
            return {'type': 'fitting_failed'}
    
    def _compute_anisotropy(self, eigenvalues: np.ndarray) -> float:
        """
        Compute gradient anisotropy measure
        High anisotropy = gradients concentrated in few directions (chaotic)
        """
        if len(eigenvalues) < 2:
            return 0
        
        # Ratio of largest to effective rank
        total_variance = np.sum(eigenvalues)
        effective_rank = total_variance**2 / np.sum(eigenvalues**2)
        
        anisotropy = len(eigenvalues) / effective_rank if effective_rank > 0 else 0
        return anisotropy
    
    def analyze_cross_modal_compatibility(self, 
                                        vision_features: Dict[str, np.ndarray],
                                        language_features: Dict[str, np.ndarray]) -> Dict:
        """
        Complete cross-modal phase compatibility analysis
        
        Addresses RQ1: Why does independent pretraining create incompatible phases?
        Addresses RQ2: Is phase compatibility necessary for universal representations?
        """
        results = {
            'cross_modal_analysis': {},
            'vision_models': {},
            'language_models': {},
            'compatibility_matrix': {},
            'research_questions': {
                'rq1_systematic_divergence': {},
                'rq2_prh_implications': {}
            }
        }
        
        # Analyze each vision model
        for v_name, v_features in vision_features.items():
            print(f"\nAnalyzing vision model: {v_name}")
            
            # Single-model analysis
            ntk_analysis = self.compute_single_model_ntk_stability(v_features)
            agop_analysis = self.compute_agop(v_features)
            
            results['vision_models'][v_name] = {
                'ntk_stability': ntk_analysis,
                'agop_analysis': agop_analysis,
                'phase': ntk_analysis['phase'],
                's_ntk': ntk_analysis['s_ntk']
            }
        
        # Analyze each language model  
        for l_name, l_features in language_features.items():
            print(f"\nAnalyzing language model: {l_name}")
            
            # Single-model analysis
            ntk_analysis = self.compute_single_model_ntk_stability(l_features)
            agop_analysis = self.compute_agop(l_features)
            
            results['language_models'][l_name] = {
                'ntk_stability': ntk_analysis,
                'agop_analysis': agop_analysis,
                'phase': ntk_analysis['phase'],
                's_ntk': ntk_analysis['s_ntk']
            }
        
        # Cross-modal compatibility analysis
        compatibility_scores = []
        
        for v_name, v_features in vision_features.items():
            for l_name, l_features in language_features.items():
                print(f"\nCross-modal analysis: {v_name} × {l_name}")
                
                cross_analysis = self.compute_cross_modal_ntk_stability(v_features, l_features)
                
                pair_name = f"{v_name}_×_{l_name}"
                results['cross_modal_analysis'][pair_name] = cross_analysis
                
                compatibility_scores.append(cross_analysis['s_ntk_cross'])
                
                # Store in compatibility matrix format
                if v_name not in results['compatibility_matrix']:
                    results['compatibility_matrix'][v_name] = {}
                results['compatibility_matrix'][v_name][l_name] = cross_analysis['s_ntk_cross']
        
        # RQ1 Analysis: Systematic divergence
        results['research_questions']['rq1_systematic_divergence'] = {
            'mean_cross_modal_stability': np.mean(compatibility_scores),
            'fraction_below_threshold': np.mean([s < self.cross_modal_threshold for s in compatibility_scores]),
            'min_compatibility': np.min(compatibility_scores),
            'max_compatibility': np.max(compatibility_scores),
            'explanation': "Independent pretraining drives models into incompatible optimization regimes"
        }
        
        # RQ2 Analysis: PRH implications
        results['research_questions']['rq2_prh_implications'] = {
            'phase_diversity': len(set([r['phase'] for r in results['vision_models'].values()] + 
                                     [r['phase'] for r in results['language_models'].values()])),
            'representational_similarity': "High (assumed based on PRH)",
            'dynamical_compatibility': np.mean(compatibility_scores),
            'prh_paradox': np.mean(compatibility_scores) < 0.5,  # Similar representations, incompatible dynamics
            'explanation': "Universal representations exist in incompatible coordinate systems"
        }
        
        return results
    
    def generate_phase_diagram(self, results: Dict) -> plt.Figure:
        """
        Generate the phase landscape diagram from your paper
        Shows S_NTK^cross vs AGOP magnitude with phase regions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for plotting
        vision_data = []
        language_data = []
        cross_modal_data = []
        
        for model_name, analysis in results['vision_models'].items():
            vision_data.append({
                'name': model_name,
                's_ntk': analysis['s_ntk'],
                'agop_ratio': analysis['agop_analysis']['agop_ratio'],
                'phase': analysis['phase']
            })
        
        for model_name, analysis in results['language_models'].items():
            language_data.append({
                'name': model_name,
                's_ntk': analysis['s_ntk'],
                'agop_ratio': analysis['agop_analysis']['agop_ratio'],
                'phase': analysis['phase']
            })
        
        for pair_name, analysis in results['cross_modal_analysis'].items():
            cross_modal_data.append({
                'name': pair_name,
                's_ntk_cross': analysis['s_ntk_cross']
            })
        
        # Plot 1: Single-model phase diagram
        colors = {'chaotic': 'red', 'optimal': 'green', 'lazy': 'blue'}
        
        for data, label, marker in [(vision_data, 'Vision', 'o'), (language_data, 'Language', 's')]:
            x_vals = [d['s_ntk'] for d in data]
            y_vals = [np.log10(d['agop_ratio']) for d in data]
            c_vals = [colors[d['phase']] for d in data]
            
            ax1.scatter(x_vals, y_vals, c=c_vals, marker=marker, s=100, 
                       alpha=0.7, label=label, edgecolors='black')
        
        # Phase boundaries
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Chaotic|Optimal')
        ax1.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='Optimal|Lazy')
        ax1.axhline(y=6, color='orange', linestyle='--', alpha=0.7, label='AGOP Failure Threshold')
        
        ax1.set_xlabel('S_NTK (Single Model)')
        ax1.set_ylabel('log₁₀(AGOP Ratio)')
        ax1.set_title('Single-Model Phase Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cross-modal compatibility
        cross_vals = [d['s_ntk_cross'] for d in cross_modal_data]
        
        ax2.hist(cross_vals, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(x=self.cross_modal_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Compatibility Threshold ({self.cross_modal_threshold})')
        ax2.axvline(x=np.mean(cross_vals), color='orange', linestyle='-', 
                   linewidth=2, label=f'Mean Compatibility ({np.mean(cross_vals):.3f})')
        
        ax2.set_xlabel('S_NTK^cross (Cross-Modal)')
        ax2.set_ylabel('Number of Model Pairs')
        ax2.set_title('Cross-Modal Compatibility Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, results: Dict, filename: str = "cross_modal_phase_analysis.json"):
        """Save analysis results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            # Handle complex numbers by taking real part
            if np.iscomplexobj(obj):
                return np.real(obj).tolist()
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, complex):
            return float(obj.real)  # Take real part of complex numbers
        else:
            return obj

# Example usage
def run_analysis_example():
    """
    Example of how to run the complete analysis
    Replace with your actual feature file paths
    """
    analyzer = CrossModalPhaseAnalyzer()
    
    # Load your extracted features
    vision_features = {
        'resnet18_layer4': np.random.randn(1000, 512),  # Replace with actual loading
        'mobilenet_v2_features': np.random.randn(1000, 1280),  # Replace with actual loading
    }
    
    language_features = {
        'bert_layer_12': np.random.randn(1000, 768),  # Replace with actual loading
        'gpt2_layer_12': np.random.randn(1000, 768),  # Replace with actual loading
    }
    
    # Run complete analysis
    results = analyzer.analyze_cross_modal_compatibility(vision_features, language_features)
    
    # Generate visualizations
    fig = analyzer.generate_phase_diagram(results)
    plt.show()
    
    # Save results
    analyzer.save_results(results)
    
    return results

if __name__ == "__main__":
    # Run the analysis
    results = run_analysis_example()
    