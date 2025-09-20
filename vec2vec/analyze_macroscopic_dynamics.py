import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
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

class MacroscopicAnalysis:
    """
    Analyzes information flow and phase transitions (macroscopic scale)
    """
    
    def __init__(self, feature_dir="./embeddings_comprehensive/"):
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path("./results/macroscopic_analysis/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        """Run macroscopic analysis on all models"""
        for feature_file in tqdm(list(self.feature_dir.glob("*.h5"))):
            print(f"\nAnalyzing {feature_file.name}...")
            self.analyze_model(feature_file)
            
    def analyze_model(self, feature_path):
        """Complete macroscopic analysis for one model"""
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
            'information_flow': {},
            'phase_analysis': {},
            'critical_transitions': {}
        }
        
        print("  Computing information bottleneck trajectories...")
        ib_analysis = self._compute_simplified_ib_trajectory(layer_features_list)
        results['information_flow'] = ib_analysis
        
        print("  Detecting phase transitions...")
        phase_analysis = self.detect_phase_transitions(ib_analysis)
        results['phase_analysis'] = phase_analysis
        
        print("  Identifying critical layers...")
        critical_layers = self.identify_critical_layers(layer_features_list, ib_analysis)
        results['critical_transitions'] = critical_layers
        
        print("  Analyzing information dynamics...")
        info_dynamics = self.analyze_information_dynamics(layer_features_list, ib_analysis)
        results['information_dynamics'] = info_dynamics
        
        output_path = self.output_dir / f"{model_name}_macroscopic.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
            
        self.visualize_macroscopic_analysis(results, model_name)
        
        return results
        
    def _compute_simplified_ib_trajectory(self, layer_features_list):
        """Compute simplified IB trajectory for feature-based analysis"""
        num_samples = layer_features_list[0].shape[0]
        
        input_dim = 100
        inputs = np.random.randn(num_samples, input_dim)
        labels = np.random.randint(0, 10, num_samples)
        
        trajectory = {
            'layers': {},
            'summary': {}
        }
        
        for layer_idx, layer_features in enumerate(layer_features_list):
            layer_features_np = layer_features.numpy()
            
            min_samples = min(inputs.shape[0], layer_features_np.shape[0])
            inputs_truncated = inputs[:min_samples]
            features_truncated = layer_features_np[:min_samples]
            
            try:
                i_xt = self._compute_simple_correlation(inputs_truncated, features_truncated)
                i_yt = self._compute_simple_correlation(features_truncated, labels[:min_samples].reshape(-1, 1))
                h_t = self._compute_simple_entropy(features_truncated)
                
                efficiency = i_yt / (i_xt + 1e-6)
                compression = 1 - (i_xt / (self._compute_simple_entropy(inputs_truncated) + 1e-6))
                
            except Exception as e:
                print(f"    Warning: Error computing metrics for layer {layer_idx}: {e}")
                i_xt, i_yt, h_t, efficiency, compression = 0.0, 0.0, 0.0, 0.0, 0.0
            
            trajectory['layers'][f'layer_{layer_idx}'] = {
                'I_X_T': float(i_xt),
                'I_Y_T': float(i_yt),
                'H_T': float(h_t),
                'efficiency': float(efficiency),
                'compression': float(compression),
                'layer_idx': layer_idx
            }
        
        trajectory['summary'] = self.summarize_trajectory(trajectory['layers'])
        return trajectory
    
    def _compute_simple_correlation(self, X, Y):
        """Compute simple correlation-based similarity"""
        try:
            X_mean = X - X.mean(axis=0)
            Y_mean = Y - Y.mean(axis=0)
            
            # Pad if necessary
            if X.shape[1] != Y.shape[1]:
                if X.shape[1] < Y.shape[1]:
                    pad_width = Y.shape[1] - X.shape[1]
                    X_mean = np.pad(X_mean, ((0,0), (0, pad_width)))
                else:
                    pad_width = X.shape[1] - Y.shape[1]
                    Y_mean = np.pad(Y_mean, ((0,0), (0, pad_width)))

            cov = (X_mean.T @ Y_mean) / (X_mean.shape[0] - 1)
            std_x = np.std(X, axis=0)
            std_y = np.std(Y, axis=0)
            corr = cov / (np.outer(std_x, std_y) + 1e-6)
            return np.mean(np.abs(corr))
        except:
            return 0.1
    
    def _compute_simple_entropy(self, X):
        """Compute simple entropy estimate"""
        try:
            return np.log(np.linalg.det(np.cov(X.T)) + 1e-6)
        except:
            return 1.0
        
    def detect_phase_transitions(self, ib_analysis):
        """Detect phase transitions in information plane"""
        if not ib_analysis.get('layers'):
            return {}
        layers = sorted(ib_analysis['layers'].keys(), key=lambda x: int(x.split('_')[1]))
        
        i_xt_traj = [ib_analysis['layers'][l]['I_X_T'] for l in layers]
        i_yt_traj = [ib_analysis['layers'][l]['I_Y_T'] for l in layers]
        
        compression_start = None
        for i in range(1, len(i_xt_traj)):
            if i_xt_traj[i] < i_xt_traj[i-1] * 0.95:
                compression_start = i
                break
                
        phase_transition = None
        if len(i_yt_traj) > 3:
            growth_rates = np.diff(i_yt_traj)
            for i in range(2, len(growth_rates)):
                if growth_rates[i] < 0.1 * growth_rates[0]:
                    phase_transition = i
                    break
                    
        return {
            'compression_start_layer': compression_start,
            'phase_transition_layer': phase_transition,
            'has_compression_phase': compression_start is not None,
            'phases': self.identify_phases(i_xt_traj, i_yt_traj)
        }
    
    def identify_phases(self, i_xt, i_yt):
        """Identify different phases of learning"""
        phases = []
        fitting_end = 0
        for i in range(1, len(i_xt)):
            if i_xt[i] <= i_xt[i-1] or i_yt[i] <= i_yt[i-1] * 1.01:
                fitting_end = i
                break
        if fitting_end > 0:
            phases.append({'name': 'fitting', 'start': 0, 'end': fitting_end})
            
        compression_start = fitting_end
        for i in range(fitting_end + 1, len(i_xt)):
            if i_xt[i] < i_xt[compression_start] * 0.9:
                phases.append({'name': 'compression', 'start': compression_start, 'end': i})
                break
        return phases
    
    def identify_critical_layers(self, layer_features_list, ib_analysis):
        """Identify critical layers for information processing"""
        if not ib_analysis.get('layers'):
            return []
        critical_layers = []
        layers = sorted(ib_analysis['layers'].keys(), key=lambda x: int(x.split('_')[1]))
        
        i_yt_values = [ib_analysis['layers'][l]['I_Y_T'] for l in layers]
        if i_yt_values:
            max_iyt_layer = np.argmax(i_yt_values)
            critical_layers.append({'layer': int(max_iyt_layer), 'type': 'max_task_information', 'value': float(i_yt_values[max_iyt_layer])})
        
        efficiency_values = [ib_analysis['layers'][l]['efficiency'] for l in layers]
        if efficiency_values:
            max_eff_layer = np.argmax(efficiency_values)
            critical_layers.append({'layer': int(max_eff_layer), 'type': 'max_efficiency', 'value': float(efficiency_values[max_eff_layer])})
        
        phase_analysis = self.detect_phase_transitions(ib_analysis)
        if phase_analysis.get('phase_transition_layer') is not None:
            critical_layers.append({'layer': phase_analysis['phase_transition_layer'], 'type': 'phase_transition', 'value': 0})
            
        return critical_layers
    
    def analyze_information_dynamics(self, layer_features_list, ib_analysis):
        """Analyze dynamics of information flow"""
        if not ib_analysis.get('layers'):
            return {}
        layers = sorted(ib_analysis['layers'].keys(), key=lambda x: int(x.split('_')[1]))
        
        i_xt_traj = [ib_analysis['layers'][l]['I_X_T'] for l in layers]
        i_yt_traj = [ib_analysis['layers'][l]['I_Y_T'] for l in layers]
        
        velocity_x = np.diff(i_xt_traj)
        velocity_y = np.diff(i_yt_traj)
        accel_x = np.diff(velocity_x)
        accel_y = np.diff(velocity_y)
        
        path_length = np.sum(np.sqrt(velocity_x**2 + velocity_y**2))
        straightness = path_length / (np.sqrt((i_xt_traj[-1] - i_xt_traj[0])**2 + (i_yt_traj[-1] - i_yt_traj[0])**2) + 1e-6)
            
        return {
            'velocity': {'I_X_T': velocity_x.tolist(), 'I_Y_T': velocity_y.tolist(), 'mean_speed': float(np.mean(np.sqrt(velocity_x**2 + velocity_y**2))) if len(velocity_x) > 0 else 0},
            'acceleration': {'I_X_T': accel_x.tolist(), 'I_Y_T': accel_y.tolist()},
            'path_length': float(path_length),
            'straightness': float(straightness)
        }
    
    def summarize_trajectory(self, layer_results):
        """Summarize the information trajectory"""
        if not layer_results:
            return {}
        layers = sorted(layer_results.keys(), key=lambda x: int(x.split('_')[1]))
        i_xt_vals = [layer_results[l]['I_X_T'] for l in layers]
        i_yt_vals = [layer_results[l]['I_Y_T'] for l in layers]
        
        return {
            'initial_state': {'I_X_T': float(i_xt_vals[0]), 'I_Y_T': float(i_yt_vals[0])},
            'final_state': {'I_X_T': float(i_xt_vals[-1]), 'I_Y_T': float(i_yt_vals[-1])},
            'total_compression': float((i_xt_vals[0] - i_xt_vals[-1]) / i_xt_vals[0]) if i_xt_vals[0] > 0 else 0,
            'total_task_info_gain': float(i_yt_vals[-1] - i_yt_vals[0]),
            'peak_task_info': float(max(i_yt_vals)),
            'peak_task_info_layer': int(np.argmax(i_yt_vals))
        }
    
    def visualize_macroscopic_analysis(self, results, model_name):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Information plane trajectory
        ax = axes[0, 0]
        if results.get('information_flow', {}).get('layers'):
            layers = sorted(results['information_flow']['layers'].keys(), key=lambda x: int(x.split('_')[1]))
            i_xt = [results['information_flow']['layers'][l]['I_X_T'] for l in layers]
            i_yt = [results['information_flow']['layers'][l]['I_Y_T'] for l in layers]
            ax.plot(i_xt, i_yt, 'bo-', markersize=8, linewidth=2)
            for i, (x, y) in enumerate(zip(i_xt, i_yt)):
                ax.annotate(f'L{i}', (x, y), xytext=(5, 5), textcoords='offset points')
            if results.get('phase_analysis', {}).get('phase_transition_layer') is not None:
                pt_idx = results['phase_analysis']['phase_transition_layer']
                if pt_idx < len(i_xt):
                    ax.scatter(i_xt[pt_idx], i_yt[pt_idx], color='red', s=200, marker='*', label='Phase Transition')
        ax.set_xlabel('I(X;T)')
        ax.set_ylabel('I(Y;T)')
        ax.set_title('Information Plane Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Information dynamics
        ax = axes[0, 1]
        if results.get('information_flow', {}).get('layers'):
            i_xt = [results['information_flow']['layers'][l]['I_X_T'] for l in layers]
            i_yt = [results['information_flow']['layers'][l]['I_Y_T'] for l in layers]
            ax.plot(range(len(i_xt)), i_xt, 'g-', label='I(X;T)')
            ax.plot(range(len(i_yt)), i_yt, 'r-', label='I(Y;T)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Information (bits)')
        ax.set_title('Layer-wise Information')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Information efficiency
        ax = axes[1, 0]
        if results.get('information_flow', {}).get('layers'):
            efficiency = [results['information_flow']['layers'][l]['efficiency'] for l in layers]
            ax.plot(range(len(efficiency)), efficiency, 'mo-')
        ax.set_xlabel('Layer')
        ax.set_ylabel('I(Y;T) / I(X;T)')
        ax.set_title('Information Efficiency')
        ax.grid(True, alpha=0.3)
        
        # 4. Phase diagram
        ax = axes[1, 1]
        if results.get('phase_analysis', {}).get('phases'):
            for phase in results['phase_analysis']['phases']:
                ax.axvspan(phase['start'], phase['end'], alpha=0.3, label=phase['name'])
        if results.get('information_dynamics', {}).get('velocity', {}).get('mean_speed') is not None:
            velocity = results['information_dynamics']['velocity']['mean_speed']
            ax.text(0.5, 0.5, f'Mean Info Velocity: {velocity:.3f}', transform=ax.transAxes, ha='center')
        ax.set_xlabel('Layer')
        ax.set_title('Learning Phases')
        ax.legend()
        
        plt.suptitle(f'Macroscopic Information Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_macroscopic_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def load_results(self, results_dir="./results/macroscopic_analysis/"):
        """Load all macroscopic analysis results"""
        results_dir = Path(results_dir)
        if not results_dir.exists():
            return {}
        all_results = {}
        for json_file in results_dir.glob("*_macroscopic.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    model_name = data['model']
                    all_results[model_name] = data
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        return all_results

    def compare_models(self, results_dict):
        """Compare macroscopic properties across models"""
        if not results_dict:
            return
        comparison = {
            'models': list(results_dict.keys()),
            'information_flow': {},
            'phase_transitions': {},
            'critical_layers': {},
            'information_dynamics': {}
        }
        for model_name, results in results_dict.items():
            if 'information_flow' in results and 'summary' in results['information_flow']:
                comparison['information_flow'][model_name] = results['information_flow']['summary']
            if 'phase_analysis' in results:
                comparison['phase_transitions'][model_name] = results['phase_analysis']
            if 'critical_transitions' in results:
                comparison['critical_layers'][model_name] = {info['type']: info for info in results['critical_transitions']}
            if 'information_dynamics' in results:
                comparison['information_dynamics'][model_name] = results['information_dynamics']
        self.visualize_model_comparison(comparison)
        return comparison

    def visualize_model_comparison(self, comparison):
        """Create comparative visualizations"""
        models = comparison.get('models', [])
        if len(models) < 2:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Information Flow Properties
        ax = axes[0, 0]
        info_flow = comparison.get('information_flow', {})
        if info_flow:
            compressions = [info_flow.get(m, {}).get('total_compression', 0) for m in models]
            task_gains = [info_flow.get(m, {}).get('total_task_info_gain', 0) for m in models]
            ax.scatter(compressions, task_gains, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model[:10], (compressions[i], task_gains[i]), xytext=(5, 5), textcoords='offset points')
            ax.set_xlabel('Total Compression')
            ax.set_ylabel('Task Info Gain')
            ax.set_title('Information Flow Trade-off')
            ax.grid(True, alpha=0.3)
        
        # 2. Phase Transitions
        ax = axes[0, 1]
        phase_data = comparison.get('phase_transitions', {})
        if phase_data:
            has_compression = [1 if phase_data.get(m, {}).get('has_compression_phase') else 0 for m in models]
            starts = [phase_data.get(m, {}).get('compression_start_layer', -1) for m in models]
            colors = ['red' if x == 1 else 'blue' for x in has_compression]
            ax.scatter(range(len(models)), starts, c=colors, s=100, alpha=0.7)
            ax.set_xlabel('Model Index')
            ax.set_ylabel('Compression Start Layer')
            ax.set_title('Phase Transitions (Red=Has Compression)')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m[:10] for m in models], rotation=45, ha="right")
            ax.grid(True, alpha=0.3)
        
        # 3. Critical Layers Distribution
        ax = axes[1, 0]
        critical_data = comparison.get('critical_layers', {})
        if critical_data:
            types = sorted(list(set(k for v in critical_data.values() for k in v.keys())))
            x = np.arange(len(models))
            width = 0.8 / len(types)
            for i, type in enumerate(types):
                layers = [critical_data.get(m, {}).get(type, {}).get('layer', -1) for m in models]
                ax.bar(x + i * width, layers, width, label=type.replace('_', ' ').title())
            ax.set_xlabel('Models')
            ax.set_ylabel('Layer Index')
            ax.set_title('Critical Layers by Type')
            ax.set_xticks(x + width * (len(types) - 1) / 2)
            ax.set_xticklabels([m[:10] for m in models], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Information Dynamics
        ax = axes[1, 1]
        dynamics_data = comparison.get('information_dynamics', {})
        if dynamics_data:
            speeds = [dynamics_data.get(m, {}).get('mean_speed', 0) for m in models]
            straightness = [dynamics_data.get(m, {}).get('straightness', 1) for m in models]
            ax.scatter(speeds, straightness, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model[:10], (speeds[i], straightness[i]), xytext=(5, 5), textcoords='offset points')
            ax.set_xlabel('Mean Information Speed')
            ax.set_ylabel('Path Straightness')
            ax.set_title('Information Processing Efficiency')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Macroscopic Analysis Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'macroscopic_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main analysis function"""
    analyzer = MacroscopicAnalysis()
    
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