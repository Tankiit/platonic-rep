import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
import h5py
import re

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
    
    def __init__(self, feature_dir="./embeddings_comprehensive/"):
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path("./results/mesoscopic_analysis/")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset for labels and inputs
        from datasets import load_dataset
        self.dataset = load_dataset("minhuh/prh", revision="wit_1024", split='train')

    def _pad_tensors_to_match(self, tensor_a, tensor_b):
        """Pads the smaller tensor with zeros to match the larger one's last dimension."""
        shape_a = tensor_a.shape
        shape_b = tensor_b.shape
        if shape_a[-1] == shape_b[-1]:
            return tensor_a, tensor_b

        if shape_a[-1] < shape_b[-1]:
            padding_size = shape_b[-1] - shape_a[-1]
            padding = torch.zeros(*shape_a[:-1], padding_size, device=tensor_a.device, dtype=tensor_a.dtype)
            tensor_a_padded = torch.cat([tensor_a, padding], dim=-1)
            return tensor_a_padded, tensor_b
        else:  # shape_b[-1] < shape_a[-1]
            padding_size = shape_a[-1] - shape_b[-1]
            padding = torch.zeros(*shape_b[:-1], padding_size, device=tensor_b.device, dtype=tensor_b.dtype)
            tensor_b_padded = torch.cat([tensor_b, padding], dim=-1)
            return tensor_a, tensor_b_padded
        
    def analyze_all_models(self):
        """Run mesoscopic analysis on all models"""
        for feature_file in tqdm(list(self.feature_dir.glob("*.h5"))):
            print(f"\nAnalyzing {feature_file.name}...")
            self.analyze_model(feature_file)
            
    def analyze_model(self, feature_path):
        """Complete mesoscopic analysis for one model"""
        with h5py.File(feature_path, 'r') as f:
            layer_keys = [k for k in f.keys() if k.startswith('layer')]
            if not layer_keys:
                layer_keys = [k for k in f.keys() if k.startswith('features')]
            
            if not layer_keys:
                print(f"    Warning: No layer or features keys found in {feature_path.name}. Skipping file.")
                return

            layer_keys = sorted(layer_keys, key=lambda x: int(re.search(r'\d+', x).group()))
            try:
                layer_features_list = [torch.from_numpy(f[key][:]) for key in layer_keys]
            except Exception as e:
                print(f"    Warning: Error loading features from {feature_path.name}. Skipping file. Error: {e}")
                return

        if not layer_features_list:
            print(f"    Warning: No features loaded from {feature_path.name}. Skipping file.")
            return

        model_name = feature_path.stem
        
        results = {
            'model': model_name,
            'layers': {},
            'evolution': {}
        }
        
        # 1. Compute empirical NTK for each layer
        print("  Computing empirical NTK...")
        ntk_analysis = self.compute_ntk_spectrum(layer_features_list)
        results['ntk'] = ntk_analysis
        
        # 2. Analyze feature evolution across layers
        print("  Analyzing feature evolution...")
        evolution_analysis = self.analyze_feature_evolution(layer_features_list)
        results['evolution'] = evolution_analysis
        
        # 3. Compute feature dynamics metrics
        print("  Computing feature dynamics...")
        dynamics_analysis = self.compute_feature_dynamics(layer_features_list)
        results['dynamics'] = dynamics_analysis
        
        # 4. Analyze representational change
        print("  Analyzing representational change...")
        repr_change = self.analyze_representational_change(layer_features_list)
        results['representational_change'] = repr_change
        
        # Save results
        output_path = self.output_dir / f"{model_name}_mesoscopic.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
            
        # Generate visualizations
        self.visualize_mesoscopic_analysis(results, model_name)
        
        return results
        
    def compute_ntk_spectrum(self, layer_features_list):
        """Compute empirical NTK and analyze its spectrum"""
        ntk_results = {'layers': {}}
        
        for layer_idx, layer_features in enumerate(layer_features_list):
            # Compute empirical NTK (Gram matrix of features)
            ntk = self.compute_empirical_ntk(layer_features)
            
            # Analyze spectrum
            from scipy.linalg import eigvalsh
            eigenvalues = eigvalsh(ntk)
            eigenvalues = eigenvalues[::-1]  # Descending order
            
            # Compute spectral metrics
            spectral_metrics = {
                'top_eigenvalue': float(eigenvalues[0]) if len(eigenvalues) > 0 else 0,
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
            ntk = features @ features.T
        elif kernel == 'rbf':
            ntk = rbf_kernel(features.numpy())
            ntk = torch.from_numpy(ntk)
        
        ntk = ntk / features.shape[1]
        return ntk.numpy()
    
    def compute_effective_rank(self, eigenvalues):
        """Compute effective rank using participation ratio"""
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 0
        pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        return pr
    
    def compute_spectral_decay(self, eigenvalues):
        """Compute rate of spectral decay"""
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) < 10:
            return 0
        log_indices = np.log(np.arange(1, min(100, len(eigenvalues)) + 1))
        log_eigenvals = np.log(eigenvalues[:min(100, len(eigenvalues))])
        decay_rate = -np.polyfit(log_indices, log_eigenvals, 1)[0]
        return decay_rate
    
    def compute_kernel_target_alignment(self, ntk, labels=None):
        """Compute kernel-target alignment (if labels available)"""
        alignment = np.trace(ntk @ ntk) / (np.linalg.norm(ntk, 'fro') ** 2)
        return alignment
    
    def analyze_feature_evolution(self, layer_features_list):
        """Analyze how features evolve across layers"""
        evolution_metrics = {
            'layer_similarity': [],
            'feature_drift': [],
            'representation_speed': [],
            'convergence_metrics': {}
        }
        
        num_layers = len(layer_features_list)
        
        for i in range(num_layers - 1):
            curr_features = layer_features_list[i]
            next_features = layer_features_list[i+1]
            
            similarity = self.compute_cka(curr_features, next_features)
            evolution_metrics['layer_similarity'].append(float(similarity))
            
            # Pad for drift calculation
            curr_padded, next_padded = self._pad_tensors_to_match(curr_features, next_features)
            drift = torch.norm(next_padded - curr_padded, dim=1).mean()
            evolution_metrics['feature_drift'].append(float(drift))
            
            speed = drift
            evolution_metrics['representation_speed'].append(float(speed))
        
        if num_layers > 3:
            late_similarities = evolution_metrics['layer_similarity'][-3:]
            convergence_rate = np.mean(late_similarities) if late_similarities else 0
            is_converged = convergence_rate > 0.95
            
            evolution_metrics['convergence_metrics'] = {
                'convergence_rate': float(convergence_rate),
                'is_converged': bool(is_converged),
                'stable_from_layer': int(self.find_stability_point(evolution_metrics['layer_similarity']))
            }
        
        return evolution_metrics
    
    def compute_cka(self, X, Y):
        """Compute Centered Kernel Alignment"""
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        K_X = X @ X.T
        K_Y = Y @ Y.T
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
    
    def compute_feature_dynamics(self, layer_features_list):
        """Compute feature dynamics metrics"""
        dynamics = {
            'intrinsic_dimension': [],
            'feature_complexity': [],
            'manifold_capacity': []
        }
        
        for layer_features in layer_features_list:
            layer_features_np = layer_features.numpy()
            
            _, s, _ = np.linalg.svd(layer_features_np - layer_features_np.mean(axis=0), full_matrices=False)
            s = s[s > 1e-10]
            
            intrinsic_dim = (np.sum(s) ** 2) / np.sum(s ** 2) if np.sum(s**2) > 0 else 0
            dynamics['intrinsic_dimension'].append(float(intrinsic_dim))
            
            s_norm = s / s.sum() if s.sum() > 0 else s
            complexity = -np.sum(s_norm * np.log(s_norm + 1e-10))
            dynamics['feature_complexity'].append(float(complexity))
            
            try:
                capacity = self.estimate_manifold_capacity_fast(layer_features_np)
            except:
                capacity = intrinsic_dim
            dynamics['manifold_capacity'].append(float(capacity))
        
        return dynamics
    
    def estimate_manifold_capacity_fast(self, features):
        """Fast estimate of manifold capacity using PCA"""
        _, s, _ = np.linalg.svd(features - features.mean(axis=0), full_matrices=False)
        s = s[s > 1e-10]
        if len(s) == 0:
            return 0
        capacity = (np.sum(s) ** 2) / np.sum(s ** 2)
        return capacity
    
    def analyze_representational_change(self, layer_features_list):
        """Analyze how representations change across layers"""
        change_metrics = {
            'total_displacement': [],
            'angular_change': [],
            'topology_change': []
        }
        
        if not layer_features_list:
            return change_metrics

        reference_features = layer_features_list[0]
        
        for layer_idx in range(1, len(layer_features_list)):
            curr_features = layer_features_list[layer_idx]
            
            ref_padded, curr_padded = self._pad_tensors_to_match(reference_features, curr_features)
            
            displacement = torch.norm(curr_padded - ref_padded, dim=1).mean()
            change_metrics['total_displacement'].append(float(displacement))
            
            cos_sim = torch.nn.functional.cosine_similarity(ref_padded, curr_padded, dim=1).mean()
            angular_change = torch.acos(torch.clamp(cos_sim, -1, 1))
            change_metrics['angular_change'].append(float(angular_change))
            
            ref_dists = pdist(reference_features.numpy())
            curr_dists = pdist(curr_features.numpy())
            if len(ref_dists) > 1 and len(curr_dists) > 1 and np.std(ref_dists) > 0 and np.std(curr_dists) > 0:
                topology_correlation = np.corrcoef(ref_dists, curr_dists)[0, 1]
                change_metrics['topology_change'].append(float(1 - topology_correlation))
            else:
                change_metrics['topology_change'].append(0.0)

        return change_metrics
    
    def analyze_ntk_evolution(self, layer_results):
        """Analyze how NTK properties evolve across layers"""
        if not layer_results:
            return {}
        layers = sorted(layer_results.keys(), key=lambda x: int(x.split('_')[1]))
        
        effective_ranks = [layer_results[l]['effective_rank'] for l in layers]
        top_eigenvals = [layer_results[l]['top_eigenvalue'] for l in layers]
        
        return {
            'rank_compression': float((effective_ranks[0] - effective_ranks[-1]) / effective_ranks[0]) if effective_ranks and effective_ranks[0] > 0 else 0,
            'eigenvalue_concentration': float(top_eigenvals[-1] / sum(top_eigenvals)) if top_eigenvals and sum(top_eigenvals) > 0 else 0,
            'spectral_sharpening': float(top_eigenvals[-1] / top_eigenvals[0]) if top_eigenvals and top_eigenvals[0] > 0 else 1
        }
    
    def visualize_mesoscopic_analysis(self, results, model_name):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. NTK spectrum evolution
        ax = axes[0, 0]
        if results.get('ntk', {}).get('layers'):
            layers = sorted([l for l in results['ntk']['layers'].keys()], key=lambda x: int(x.split('_')[1]))
            effective_ranks = [results['ntk']['layers'][l]['effective_rank'] for l in layers]
            ax.plot(range(len(layers)), effective_ranks, 'bo-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('NTK Effective Rank')
        ax.set_title('Kernel Complexity Evolution')
        ax.grid(True, alpha=0.3)
        
        # 2. Feature evolution
        ax = axes[0, 1]
        if results.get('evolution', {}).get('layer_similarity'):
            similarities = results['evolution']['layer_similarity']
            ax.plot(range(len(similarities)), similarities, 'go-')
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer Transition')
        ax.set_ylabel('CKA Similarity')
        ax.set_title('Layer-to-Layer Feature Similarity')
        ax.grid(True, alpha=0.3)
        
        # 3. Feature dynamics
        ax = axes[0, 2]
        if results.get('dynamics', {}).get('intrinsic_dimension'):
            intrinsic_dims = results['dynamics']['intrinsic_dimension']
            ax.plot(range(len(intrinsic_dims)), intrinsic_dims, 'ro-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Intrinsic Dimension')
        ax.set_title('Feature Space Dimensionality')
        ax.grid(True, alpha=0.3)
        
        # 4. Representational change
        ax = axes[1, 0]
        if results.get('representational_change', {}).get('total_displacement'):
            displacements = results['representational_change']['total_displacement']
            ax.plot(range(len(displacements)), displacements, 'mo-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Distance from Layer 0')
        ax.set_title('Cumulative Feature Drift')
        ax.grid(True, alpha=0.3)
        
        # 5. Spectral decay rates
        ax = axes[1, 1]
        if results.get('ntk', {}).get('layers'):
            layers = sorted([l for l in results['ntk']['layers'].keys()], key=lambda x: int(x.split('_')[1]))
            decay_rates = [results['ntk']['layers'][l]['spectral_decay_rate'] for l in layers]
            ax.plot(range(len(layers)), decay_rates, 'co-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spectral Decay Rate')
        ax.set_title('NTK Eigenvalue Decay')
        ax.grid(True, alpha=0.3)
        
        # 6. Feature complexity
        ax = axes[1, 2]
        if results.get('dynamics', {}).get('feature_complexity'):
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
            
            if 'ntk' in results and 'evolution' in results['ntk']:
                ntk_evo = results['ntk']['evolution']
                comparison['ntk_properties'][model_name] = ntk_evo
                print(f"  NTK Rank Compression: {ntk_evo.get('rank_compression', 0):.3f}")
                print(f"  Eigenvalue Concentration: {ntk_evo.get('eigenvalue_concentration', 0):.3f}")
                print(f"  Spectral Sharpening: {ntk_evo.get('spectral_sharpening', 1):.3f}")
            
            if 'evolution' in results and 'convergence_metrics' in results['evolution']:
                conv = results['evolution']['convergence_metrics']
                comparison['evolution_patterns'][model_name] = conv
                print(f"  Convergence Rate: {conv.get('convergence_rate', 0):.3f}")
                print(f"  Is Converged: {conv.get('is_converged', False)}")
                print(f"  Stable From Layer: {conv.get('stable_from_layer', -1)}")
                
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
        
        self.visualize_model_comparison(comparison)
        return comparison

    def visualize_model_comparison(self, comparison):
        """Create comparative visualizations"""
        models = comparison.get('models', [])
        if len(models) < 2:
            print("Need at least 2 models for comparison")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. NTK Properties Comparison
        ax = axes[0, 0]
        ntk_props = comparison.get('ntk_properties', {})
        if ntk_props:
            metrics = ['rank_compression', 'eigenvalue_concentration', 'spectral_sharpening']
            x = np.arange(len(models))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [ntk_props.get(model, {}).get(metric, 0) for model in models]
                ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
                
            ax.set_xlabel('Models')
            ax.set_ylabel('Value')
            ax.set_title('NTK Properties Comparison')
            ax.set_xticks(x + width)
            ax.set_xticklabels([m[:15] for m in models], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Evolution Patterns
        ax = axes[0, 1]
        evo_props = comparison.get('evolution_patterns', {})
        if evo_props:
            conv_rates = [evo_props.get(model, {}).get('convergence_rate', 0) for model in models]
            stable_layers = [evo_props.get(model, {}).get('stable_from_layer', -1) for model in models]
            
            ax.scatter(conv_rates, stable_layers, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model[:10], (conv_rates[i], stable_layers[i]), xytext=(5, 5), textcoords='offset points')
                           
            ax.set_xlabel('Convergence Rate')
            ax.set_ylabel('Stable From Layer')
            ax.set_title('Evolution Patterns')
            ax.grid(True, alpha=0.3)
        
        # 3. Dynamics Summary
        ax = axes[1, 0]
        dyn_props = comparison.get('dynamics_summary', {})
        if dyn_props:
            intrinsic_dims = [dyn_props.get(model, {}).get('avg_intrinsic_dimension', 0) for model in models]
            complexities = [dyn_props.get(model, {}).get('avg_feature_complexity', 0) for model in models]
            
            ax.scatter(intrinsic_dims, complexities, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model[:10], (intrinsic_dims[i], complexities[i]), xytext=(5, 5), textcoords='offset points')
                           
            ax.set_xlabel('Avg Intrinsic Dimension')
            ax.set_ylabel('Avg Feature Complexity')
            ax.set_title('Feature Dynamics')
            ax.grid(True, alpha=0.3)
        
        # 4. Summary Statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = "Model Summary:\n\n"
        for model in models:
            summary_text += f"{model[:20]}:\n"
            if model in ntk_props:
                summary_text += f"  Rank Compression: {ntk_props.get(model, {}).get('rank_compression', 0):.3f}\n"
            if model in evo_props:
                summary_text += f"  Convergence Rate: {evo_props.get(model, {}).get('convergence_rate', 0):.3f}\n"
            if model in dyn_props:
                summary_text += f"  Avg Intrinsic Dim: {dyn_props.get(model, {}).get('avg_intrinsic_dimension', 0):.3f}\n"
            summary_text += "\n"
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Mesoscopic Analysis Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mesoscopic_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main analysis function"""
    analyzer = MesoscopicAnalysis()
    
    # Option 1: Run fresh analysis
    analyzer.analyze_all_models()
    
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