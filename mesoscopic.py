# analyze_mesoscopic_dynamics.py
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MesoscopicAnalysis:
    """
    Analyzes feature evolution and kernel dynamics (mesoscopic scale)
    """
    
    def __init__(self, feature_dir="./results/features/minhuh/prh/wit_1024/"):
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path("./results/mesoscopic_analysis/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_all_models(self):
        """Run mesoscopic analysis on all models"""
        for feature_file in tqdm(list(self.feature_dir.glob("*.pt"))):
            print(f"\nAnalyzing {feature_file.name}...")
            self.analyze_model(feature_file)
            
    def analyze_model(self, feature_path):
        """Complete mesoscopic analysis for one model"""
        # Load features
        data = torch.load(feature_path, map_location='cpu')
        features = data['feats']  # [N, L, D]
        
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
            
        model_name = feature_path.stem
        
        results = {
            'model': model_name,
            'layers': {},
            'evolution': {}
        }
        
        # 1. Compute empirical NTK for each layer
        print("  Computing empirical NTK...")
        ntk_analysis = self.compute_ntk_spectrum(features)
        results['ntk'] = ntk_analysis
        
        # 2. Analyze feature evolution across layers
        print("  Analyzing feature evolution...")
        evolution_analysis = self.analyze_feature_evolution(features)
        results['evolution'] = evolution_analysis
        
        # 3. Compute feature dynamics metrics
        print("  Computing feature dynamics...")
        dynamics_analysis = self.compute_feature_dynamics(features)
        results['dynamics'] = dynamics_analysis
        
        # 4. Analyze representational change
        print("  Analyzing representational change...")
        repr_change = self.analyze_representational_change(features)
        results['representational_change'] = repr_change
        
        # Save results
        output_path = self.output_dir / f"{model_name}_mesoscopic.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
            
        # Generate visualizations
        self.visualize_mesoscopic_analysis(results, model_name)
        
    def compute_ntk_spectrum(self, features):
        """Compute empirical NTK and analyze its spectrum"""
        ntk_results = {'layers': {}}
        
        for layer_idx in range(features.shape[1]):
            layer_features = features[:, layer_idx, :]
            
            # Compute empirical NTK (Gram matrix of features)
            # K(x_i, x_j) = <f(x_i), f(x_j)>
            ntk = self.compute_empirical_ntk(layer_features)
            
            # Analyze spectrum
            from scipy.linalg import eigvalsh
            eigenvalues = eigvalsh(ntk)
            eigenvalues = eigenvalues[::-1]  # Descending order
            
            # Compute spectral metrics
            spectral_metrics = {
                'top_eigenvalue': float(eigenvalues[0]),
                'effective_rank': float(self.compute_effective_rank(eigenvalues)),
                'spectral_decay_rate': float(self.compute_spectral_decay(eigenvalues)),
                'kernel_alignment': float(self.compute_kernel_target_alignment(ntk)),
                'eigenvalue_gap': float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]) if len(eigenvalues) > 1 else 0
            }
            
            ntk_results['layers'][f'layer_{layer_idx}'] = spectral_metrics
            
        # Analyze evolution across layers
        ntk_results['evolution'] = self.analyze_ntk_evolution(ntk_results['layers'])
        
        return ntk_results
    
    def compute_empirical_ntk(self, features, kernel='linear'):
        """Compute empirical NTK matrix"""
        if kernel == 'linear':
            # Linear kernel (standard NTK in infinite width limit)
            ntk = features @ features.T
        elif kernel == 'rbf':
            # RBF kernel for non-linear feature similarity
            ntk = rbf_kernel(features.numpy())
            ntk = torch.from_numpy(ntk)
        
        # Normalize
        ntk = ntk / features.shape[1]
        
        return ntk.numpy()
    
    def compute_effective_rank(self, eigenvalues):
        """Compute effective rank using participation ratio"""
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 0
        
        # Participation ratio
        pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        return pr
    
    def compute_spectral_decay(self, eigenvalues):
        """Compute rate of spectral decay"""
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) < 10:
            return 0
        
        # Fit power law to eigenvalues
        log_indices = np.log(np.arange(1, min(100, len(eigenvalues)) + 1))
        log_eigenvals = np.log(eigenvalues[:min(100, len(eigenvalues))])
        
        # Linear fit in log-log space
        decay_rate = -np.polyfit(log_indices, log_eigenvals, 1)[0]
        
        return decay_rate
    
    def compute_kernel_target_alignment(self, ntk, labels=None):
        """Compute kernel-target alignment (if labels available)"""
        # For now, compute kernel self-alignment (coherence)
        # In practice, you'd use actual labels
        n = ntk.shape[0]
        
        # Frobenius norm alignment
        alignment = np.trace(ntk @ ntk) / (np.linalg.norm(ntk, 'fro') ** 2)
        
        return alignment
    
    def analyze_feature_evolution(self, features):
        """Analyze how features evolve across layers"""
        evolution_metrics = {
            'layer_similarity': [],
            'feature_drift': [],
            'representation_speed': [],
            'convergence_metrics': {}
        }
        
        num_layers = features.shape[1]
        
        # Compute layer-to-layer similarity
        for i in range(num_layers - 1):
            curr_features = features[:, i, :]
            next_features = features[:, i + 1, :]
            
            # CKA similarity
            similarity = self.compute_cka(curr_features, next_features)
            evolution_metrics['layer_similarity'].append(float(similarity))
            
            # Feature drift (average displacement)
            drift = torch.norm(next_features - curr_features, dim=1).mean()
            evolution_metrics['feature_drift'].append(float(drift))
            
            # Representation speed (change rate)
            speed = drift  # Could normalize by layer depth
            evolution_metrics['representation_speed'].append(float(speed))
        
        # Analyze convergence
        if num_layers > 3:
            # Check if representations are stabilizing
            late_similarities = evolution_metrics['layer_similarity'][-3:]
            convergence_rate = np.mean(late_similarities)
            is_converged = convergence_rate > 0.95
            
            evolution_metrics['convergence_metrics'] = {
                'convergence_rate': float(convergence_rate),
                'is_converged': bool(is_converged),
                'stable_from_layer': int(self.find_stability_point(evolution_metrics['layer_similarity']))
            }
        
        return evolution_metrics
    
    def compute_cka(self, X, Y):
        """Compute Centered Kernel Alignment"""
        # Center the features
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute kernels
        K_X = X @ X.T
        K_Y = Y @ Y.T
        
        # CKA
        hsic_xy = torch.trace(K_X @ K_Y)
        hsic_xx = torch.trace(K_X @ K_X)
        hsic_yy = torch.trace(K_Y @ K_Y)
        
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        
        return cka.item()
    
    def find_stability_point(self, similarities, threshold=0.95):
        """Find layer where representation stabilizes"""
        for i, sim in enumerate(similarities):
            if sim > threshold:
                return i
        return len(similarities)
    
    def compute_feature_dynamics(self, features):
        """Compute feature dynamics metrics"""
        dynamics = {
            'intrinsic_dimension': [],
            'feature_complexity': [],
            'manifold_capacity': []
        }
        
        for layer_idx in range(features.shape[1]):
            layer_features = features[:, layer_idx, :].numpy()
            
            # Intrinsic dimension via PCA
            _, s, _ = np.linalg.svd(layer_features - layer_features.mean(axis=0), full_matrices=False)
            s = s[s > 1e-10]
            
            # Participation ratio as intrinsic dimension
            intrinsic_dim = (np.sum(s) ** 2) / np.sum(s ** 2)
            dynamics['intrinsic_dimension'].append(float(intrinsic_dim))
            
            # Feature complexity (entropy of singular values)
            s_norm = s / s.sum()
            complexity = -np.sum(s_norm * np.log(s_norm + 1e-10))
            dynamics['feature_complexity'].append(float(complexity))
            
            # Manifold capacity (uses local dimension estimates)
            capacity = self.estimate_manifold_capacity(layer_features)
            dynamics['manifold_capacity'].append(float(capacity))
        
        return dynamics
    
    def estimate_manifold_capacity(self, features, k=10):
        """Estimate manifold capacity using local dimension estimates"""
        n_samples = min(1000, features.shape[0])
        indices = np.random.choice(features.shape[0], n_samples, replace=False)
        
        local_dims = []
        
        for idx in indices:
            # Find k nearest neighbors
            distances = np.linalg.norm(features - features[idx], axis=1)
            k_nearest = np.sort(distances)[1:k+1]  # Exclude self
            
            # Maximum likelihood estimate of local dimension
            if k_nearest[-1] > 1e-10:
                # Add small epsilon to avoid log(0) and division by 0
                log_ratios = np.log(np.maximum(k_nearest[-1] / k_nearest[:-1], 1e-10))
                log_sum = np.sum(log_ratios)
                if abs(log_sum) > 1e-10:
                    local_dim = (k - 1) / log_sum
                    if np.isfinite(local_dim) and local_dim > 0:
                        local_dims.append(local_dim)
        
        return np.median(local_dims) if local_dims else 0
    
    def analyze_representational_change(self, features):
        """Analyze how representations change across layers"""
        change_metrics = {
            'total_displacement': [],
            'angular_change': [],
            'topology_change': []
        }
        
        num_layers = features.shape[1]
        
        # Reference: first layer features
        reference_features = features[:, 0, :]
        
        for layer_idx in range(1, num_layers):
            curr_features = features[:, layer_idx, :]
            
            # Total displacement from first layer
            displacement = torch.norm(curr_features - reference_features, dim=1).mean()
            change_metrics['total_displacement'].append(float(displacement))
            
            # Angular change (average cosine distance)
            cos_sim = torch.nn.functional.cosine_similarity(
                reference_features, curr_features, dim=1
            ).mean()
            angular_change = torch.acos(torch.clamp(cos_sim, -1, 1))
            change_metrics['angular_change'].append(float(angular_change))
            
            # Topology change (correlation of distance matrices)
            ref_dists = pdist(reference_features.numpy())
            curr_dists = pdist(curr_features.numpy())
            topology_correlation = np.corrcoef(ref_dists, curr_dists)[0, 1]
            change_metrics['topology_change'].append(float(1 - topology_correlation))
        
        return change_metrics
    
    def analyze_ntk_evolution(self, layer_results):
        """Analyze how NTK properties evolve across layers"""
        layers = sorted(layer_results.keys(), key=lambda x: int(x.split('_')[1]))
        
        # Extract metrics
        effective_ranks = [layer_results[l]['effective_rank'] for l in layers]
        top_eigenvals = [layer_results[l]['top_eigenvalue'] for l in layers]
        
        return {
            'rank_compression': float((effective_ranks[0] - effective_ranks[-1]) / effective_ranks[0]) if effective_ranks[0] > 0 else 0,
            'eigenvalue_concentration': float(top_eigenvals[-1] / sum(top_eigenvals)),
            'spectral_sharpening': float(top_eigenvals[-1] / top_eigenvals[0]) if top_eigenvals[0] > 0 else 1
        }
    
    def visualize_mesoscopic_analysis(self, results, model_name):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. NTK spectrum evolution
        ax = axes[0, 0]
        layers = sorted([l for l in results['ntk']['layers'].keys()], 
                       key=lambda x: int(x.split('_')[1]))
        effective_ranks = [results['ntk']['layers'][l]['effective_rank'] for l in layers]
        ax.plot(range(len(layers)), effective_ranks, 'bo-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('NTK Effective Rank')
        ax.set_title('Kernel Complexity Evolution')
        ax.grid(True, alpha=0.3)
        
        # 2. Feature evolution
        ax = axes[0, 1]
        similarities = results['evolution']['layer_similarity']
        ax.plot(range(len(similarities)), similarities, 'go-')
        ax.set_xlabel('Layer Transition')
        ax.set_ylabel('CKA Similarity')
        ax.set_title('Layer-to-Layer Feature Similarity')
        ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # 3. Feature dynamics
        ax = axes[0, 2]
        intrinsic_dims = results['dynamics']['intrinsic_dimension']
        ax.plot(range(len(intrinsic_dims)), intrinsic_dims, 'ro-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Intrinsic Dimension')
        ax.set_title('Feature Space Dimensionality')
        ax.grid(True, alpha=0.3)
        
        # 4. Representational change
        ax = axes[1, 0]
        displacements = results['representational_change']['total_displacement']
        ax.plot(range(len(displacements)), displacements, 'mo-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Distance from Layer 0')
        ax.set_title('Cumulative Feature Drift')
        ax.grid(True, alpha=0.3)
        
        # 5. Spectral decay rates
        ax = axes[1, 1]
        decay_rates = [results['ntk']['layers'][l]['spectral_decay_rate'] for l in layers]
        ax.plot(range(len(layers)), decay_rates, 'co-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spectral Decay Rate')
        ax.set_title('NTK Eigenvalue Decay')
        ax.grid(True, alpha=0.3)
        
        # 6. Feature complexity
        ax = axes[1, 2]
        complexity = results['dynamics']['feature_complexity']
        ax.plot(range(len(complexity)), complexity, 'yo-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Feature Complexity')
        ax.set_title('Representation Entropy')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Mesoscopic Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_mesoscopic_analysis.png', dpi=150)
        plt.close()

    def load_results(self, results_dir="./results/mesoscopic_analysis/"):
        """Load all mesoscopic analysis results"""
        results_dir = Path(results_dir)
        if not results_dir.exists():
            print(f"Results directory {results_dir} does not exist")
            return {}
            
        all_results = {}
        for json_file in results_dir.glob("*_mesoscopic.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    model_name = data['model']
                    all_results[model_name] = data
                    print(f"Loaded results for {model_name}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return all_results

    def compare_models(self, results_dict):
        """Compare mesoscopic properties across models"""
        if not results_dict:
            print("No results to compare")
            return
            
        print("=== Mesoscopic Analysis Comparison ===")
        
        comparison = {
            'models': list(results_dict.keys()),
            'ntk_properties': {},
            'evolution_patterns': {},
            'dynamics_summary': {}
        }
        
        for model_name, results in results_dict.items():
            print(f"\n--- {model_name} ---")
            
            # NTK analysis summary
            if 'ntk' in results and 'evolution' in results['ntk']:
                ntk_evo = results['ntk']['evolution']
                comparison['ntk_properties'][model_name] = {
                    'rank_compression': ntk_evo.get('rank_compression', 0),
                    'eigenvalue_concentration': ntk_evo.get('eigenvalue_concentration', 0),
                    'spectral_sharpening': ntk_evo.get('spectral_sharpening', 1)
                }
                print(f"  NTK Rank Compression: {ntk_evo.get('rank_compression', 0):.3f}")
                print(f"  Eigenvalue Concentration: {ntk_evo.get('eigenvalue_concentration', 0):.3f}")
                print(f"  Spectral Sharpening: {ntk_evo.get('spectral_sharpening', 1):.3f}")
            
            # Evolution analysis summary  
            if 'evolution' in results and 'convergence_metrics' in results['evolution']:
                conv = results['evolution']['convergence_metrics']
                comparison['evolution_patterns'][model_name] = {
                    'convergence_rate': conv.get('convergence_rate', 0),
                    'is_converged': conv.get('is_converged', False),
                    'stable_from_layer': conv.get('stable_from_layer', -1)
                }
                print(f"  Convergence Rate: {conv.get('convergence_rate', 0):.3f}")
                print(f"  Is Converged: {conv.get('is_converged', False)}")
                print(f"  Stable From Layer: {conv.get('stable_from_layer', -1)}")
                
            # Dynamics summary
            if 'dynamics' in results:
                dynamics = results['dynamics']
                avg_intrinsic_dim = np.mean(dynamics.get('intrinsic_dimension', [0]))
                avg_complexity = np.mean(dynamics.get('feature_complexity', [0]))
                avg_capacity = np.mean(dynamics.get('manifold_capacity', [0]))
                
                comparison['dynamics_summary'][model_name] = {
                    'avg_intrinsic_dimension': avg_intrinsic_dim,
                    'avg_feature_complexity': avg_complexity, 
                    'avg_manifold_capacity': avg_capacity
                }
                print(f"  Avg Intrinsic Dimension: {avg_intrinsic_dim:.3f}")
                print(f"  Avg Feature Complexity: {avg_complexity:.3f}")
                print(f"  Avg Manifold Capacity: {avg_capacity:.3f}")
        
        # Generate comparison visualizations
        self.visualize_model_comparison(comparison)
        
        return comparison

    def visualize_model_comparison(self, comparison):
        """Create comparative visualizations"""
        models = comparison['models']
        if len(models) < 2:
            print("Need at least 2 models for comparison")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. NTK Properties Comparison
        ax = axes[0, 0]
        ntk_props = comparison['ntk_properties']
        
        if ntk_props:
            metrics = ['rank_compression', 'eigenvalue_concentration', 'spectral_sharpening']
            x = np.arange(len(models))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [ntk_props[model].get(metric, 0) for model in models]
                ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
                
            ax.set_xlabel('Models')
            ax.set_ylabel('Value')
            ax.set_title('NTK Properties Comparison')
            ax.set_xticks(x + width)
            ax.set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in models], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Evolution Patterns
        ax = axes[0, 1]
        evo_props = comparison['evolution_patterns']
        
        if evo_props:
            conv_rates = [evo_props[model].get('convergence_rate', 0) for model in models]
            stable_layers = [evo_props[model].get('stable_from_layer', -1) for model in models]
            
            ax.scatter(conv_rates, stable_layers, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model[:10], (conv_rates[i], stable_layers[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                           
            ax.set_xlabel('Convergence Rate')
            ax.set_ylabel('Stable From Layer')
            ax.set_title('Evolution Patterns')
            ax.grid(True, alpha=0.3)
        
        # 3. Dynamics Summary
        ax = axes[1, 0]
        dyn_props = comparison['dynamics_summary']
        
        if dyn_props:
            intrinsic_dims = [dyn_props[model].get('avg_intrinsic_dimension', 0) for model in models]
            complexities = [dyn_props[model].get('avg_feature_complexity', 0) for model in models]
            
            ax.scatter(intrinsic_dims, complexities, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model[:10], (intrinsic_dims[i], complexities[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                           
            ax.set_xlabel('Avg Intrinsic Dimension')
            ax.set_ylabel('Avg Feature Complexity')
            ax.set_title('Feature Dynamics')
            ax.grid(True, alpha=0.3)
        
        # 4. Summary Statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary text
        summary_text = "Model Summary:\n\n"
        for model in models:
            summary_text += f"{model}:\n"
            if model in ntk_props:
                summary_text += f"  NTK Rank Compression: {ntk_props[model].get('rank_compression', 0):.3f}\n"
            if model in evo_props:
                summary_text += f"  Convergence Rate: {evo_props[model].get('convergence_rate', 0):.3f}\n"
            if model in dyn_props:
                summary_text += f"  Avg Intrinsic Dim: {dyn_props[model].get('avg_intrinsic_dimension', 0):.3f}\n"
            summary_text += "\n"
            
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Mesoscopic Analysis Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mesoscopic_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main analysis function"""
    analyzer = MesoscopicAnalysis()
    
    # Option 1: Run fresh analysis
    # analyzer.analyze_all_models()
    
    # Option 2: Load and analyze existing results
    results = analyzer.load_results()
    if results:
        comparison = analyzer.compare_models(results)
        return results, comparison
    else:
        print("No existing results found. Run analyzer.analyze_all_models() first.")
        return None, None

if __name__ == "__main__":
    main()