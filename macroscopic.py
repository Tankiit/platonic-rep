# analyze_macroscopic_dynamics.py
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

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
    
    def __init__(self, feature_dir="./results/features/minhuh/prh/wit_1024/"):
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path("./results/macroscopic_analysis/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset for labels and inputs
        from datasets import load_dataset
        self.dataset = load_dataset("minhuh/prh", revision="wit_1024", split='train')
        
    def analyze_all_models(self):
        """Run macroscopic analysis on all models"""
        for feature_file in tqdm(list(self.feature_dir.glob("*.pt"))):
            print(f"\nAnalyzing {feature_file.name}...")
            self.analyze_model(feature_file)
            
    def analyze_model(self, feature_path):
        """Complete macroscopic analysis for one model"""
        # Load features
        data = torch.load(feature_path, map_location='cpu')
        features = data['feats']  # [N, L, D]
        
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
            
        model_name = feature_path.stem
        
        results = {
            'model': model_name,
            'information_flow': {},
            'phase_analysis': {},
            'critical_transitions': {}
        }
        
        # 1. Information bottleneck analysis
        print("  Computing information bottleneck trajectories...")
        ib_analysis = self.compute_information_bottleneck_trajectory(features)
        results['information_flow'] = ib_analysis
        
        # 2. Phase transition detection
        print("  Detecting phase transitions...")
        phase_analysis = self.detect_phase_transitions(ib_analysis)
        results['phase_analysis'] = phase_analysis
        
        # 3. Critical layer identification
        print("  Identifying critical layers...")
        critical_layers = self.identify_critical_layers(features, ib_analysis)
        results['critical_transitions'] = critical_layers
        
        # 4. Information dynamics analysis
        print("  Analyzing information dynamics...")
        info_dynamics = self.analyze_information_dynamics(features, ib_analysis)
        results['information_dynamics'] = info_dynamics
        
        # Save results
        output_path = self.output_dir / f"{model_name}_macroscopic.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
            
        # Generate visualizations
        self.visualize_macroscopic_analysis(results, model_name)
        
    def compute_information_bottleneck_trajectory(self, features):
        """Compute full IB trajectory across layers"""
        num_layers = features.shape[1]
        
        # Get inputs and labels
        inputs = self.get_input_representation()
        labels = self.get_task_labels()
        
        trajectory = {
            'layers': {},
            'summary': {}
        }
        
        for layer_idx in range(num_layers):
            layer_features = features[:, layer_idx, :]
            
            # Compute I(X;T)
            i_xt = self.compute_mutual_information_hd(inputs, layer_features)
            
            # Compute I(Y;T)
            i_yt = self.compute_mutual_information_ld(layer_features, labels)
            
            # Additional information measures
            h_t = self.compute_entropy_hd(layer_features)
            
            # Information efficiency
            efficiency = i_yt / (i_xt + 1e-6)
            
            # Compression ratio
            compression = 1 - (i_xt / self.compute_entropy_hd(inputs))
            
            trajectory['layers'][f'layer_{layer_idx}'] = {
                'I_X_T': float(i_xt),
                'I_Y_T': float(i_yt),
                'H_T': float(h_t),
                'efficiency': float(efficiency),
                'compression': float(compression),
                'layer_idx': layer_idx
            }
        
        # Compute trajectory summary
        trajectory['summary'] = self.summarize_trajectory(trajectory['layers'])
        
        return trajectory
    
    def get_input_representation(self):
        """Get input representation for I(X;T) computation"""
        # For images: use downsampled pixels
        # For text: use bag-of-words
        
        if 'image' in str(self.dataset[0].keys()):
            # Vision models
            inputs = []
            for item in self.dataset:
                img = item['image']
                img_small = img.resize((32, 32))
                img_array = np.array(img_small).flatten()
                inputs.append(img_array)
            return np.array(inputs)
        else:
            # Language models
            texts = [item['text'][0] for item in self.dataset]
            # Simple BoW representation
            vocab = set()
            for text in texts:
                vocab.update(text.lower().split())
            vocab = list(vocab)[:1000]
            
            vocab_to_idx = {w: i for i, w in enumerate(vocab)}
            
            inputs = []
            for text in texts:
                vec = np.zeros(len(vocab))
                for word in text.lower().split():
                    if word in vocab_to_idx:
                        vec[vocab_to_idx[word]] += 1
                inputs.append(vec)
            
            return np.array(inputs)
    
    def get_task_labels(self):
        """Get task labels for I(Y;T) computation"""
        # Create pseudo-labels from text clustering
        texts = [item['text'][0] for item in self.dataset]
        
        # Simple clustering based on text length
        lengths = [len(text.split()) for text in texts]
        
        # Discretize into 20 bins
        from sklearn.preprocessing import KBinsDiscretizer
        kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
        labels = kbd.fit_transform(np.array(lengths).reshape(-1, 1)).flatten()
        
        return labels.astype(int)
    
    def compute_mutual_information_hd(self, X, Y, method='kde_approximation'):
        """Compute MI for high-dimensional data"""
        # Reduce dimensionality first
        from sklearn.decomposition import PCA
        
        # Reduce X
        if X.shape[1] > 50:
            pca_x = PCA(n_components=50, random_state=42)
            X_reduced = pca_x.fit_transform(X)
        else:
            X_reduced = X
            
        # Reduce Y
        if isinstance(Y, torch.Tensor):
            Y = Y.numpy()
            
        if Y.shape[1] > 50:
            pca_y = PCA(n_components=50, random_state=42)
            Y_reduced = pca_y.fit_transform(Y)
        else:
            Y_reduced = Y
        
        # Use nearest neighbor MI estimator
        from sklearn.feature_selection import mutual_info_regression
        
        # Estimate MI by averaging over dimensions
        mi_scores = []
        for i in range(min(10, Y_reduced.shape[1])):
            mi = np.mean(mutual_info_regression(X_reduced, Y_reduced[:, i], 
                                              random_state=42))
            mi_scores.append(mi)
            
        return np.mean(mi_scores)
    
    def compute_mutual_information_ld(self, X, Y):
        """Compute MI between high-dim X and low-dim Y (labels)"""
        if isinstance(X, torch.Tensor):
            X = X.numpy()
            
        # Use first few principal components
        from sklearn.decomposition import PCA
        if X.shape[1] > 10:
            pca = PCA(n_components=10, random_state=42)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
            
        # Average MI across components
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_reduced, Y, random_state=42)
        
        return np.mean(mi_scores)
    
    def compute_entropy_hd(self, X):
        """Compute entropy for high-dimensional data"""
        if isinstance(X, torch.Tensor):
            X = X.numpy()
            
        # Use covariance-based entropy estimate
        # H(X) ≈ 0.5 * log(det(2πe * Cov(X)))
        
        # Add small regularization for numerical stability
        cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
        
        # Use eigenvalues for stable computation
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvals = eigenvals[eigenvals > 1e-10]
        
        # Differential entropy
        entropy_est = 0.5 * np.sum(np.log(2 * np.pi * np.e * eigenvals))
        
        return entropy_est / X.shape[1]  # Normalize by dimension
    
    def detect_phase_transitions(self, ib_analysis):
        """Detect phase transitions in information plane"""
        layers = sorted(ib_analysis['layers'].keys(), 
                       key=lambda x: int(x.split('_')[1]))
        
        # Extract trajectories
        i_xt_traj = [ib_analysis['layers'][l]['I_X_T'] for l in layers]
        i_yt_traj = [ib_analysis['layers'][l]['I_Y_T'] for l in layers]
        
        # Detect fitting phase → compression phase transition
        phase_transition = None
        compression_start = None
        
        # Method 1: Detect when I(X;T) starts decreasing
        for i in range(1, len(i_xt_traj)):
            if i_xt_traj[i] < i_xt_traj[i-1] * 0.95:  # 5% decrease
                compression_start = i
                break
                
        # Method 2: Detect when I(Y;T) growth slows
        if len(i_yt_traj) > 3:
            growth_rates = np.diff(i_yt_traj)
            for i in range(2, len(growth_rates)):
                if growth_rates[i] < 0.1 * growth_rates[0]:  # Growth slows to 10%
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
        
        # Phase 1: Fitting (both I(X;T) and I(Y;T) increase)
        fitting_end = 0
        for i in range(1, len(i_xt)):
            if i_xt[i] <= i_xt[i-1] or i_yt[i] <= i_yt[i-1] * 1.01:
                fitting_end = i
                break
                
        if fitting_end > 0:
            phases.append({
                'name': 'fitting',
                'start': 0,
                'end': fitting_end,
                'description': 'Both I(X;T) and I(Y;T) increase'
            })
            
        # Phase 2: Compression (I(X;T) decreases, I(Y;T) stable)
        compression_start = fitting_end
        for i in range(fitting_end + 1, len(i_xt)):
            if i_xt[i] < i_xt[compression_start] * 0.9:
                phases.append({
                    'name': 'compression',
                    'start': compression_start,
                    'end': i,
                    'description': 'I(X;T) decreases while I(Y;T) stabilizes'
                })
                break
                
        return phases
    
    def identify_critical_layers(self, features, ib_analysis):
        """Identify critical layers for information processing"""
        critical_layers = []
        
        layers = sorted(ib_analysis['layers'].keys(), 
                       key=lambda x: int(x.split('_')[1]))
        
        # Critical layer 1: Maximum I(Y;T)
        i_yt_values = [ib_analysis['layers'][l]['I_Y_T'] for l in layers]
        max_iyt_layer = np.argmax(i_yt_values)
        critical_layers.append({
            'layer': max_iyt_layer,
            'type': 'max_task_information',
            'value': float(i_yt_values[max_iyt_layer])
        })
        
        # Critical layer 2: Maximum efficiency
        efficiency_values = [ib_analysis['layers'][l]['efficiency'] for l in layers]
        max_eff_layer = np.argmax(efficiency_values)
        critical_layers.append({
            'layer': max_eff_layer,
            'type': 'max_efficiency',
            'value': float(efficiency_values[max_eff_layer])
        })
        
        # Critical layer 3: Phase transition
        phase_analysis = self.detect_phase_transitions(ib_analysis)
        if phase_analysis['phase_transition_layer'] is not None:
            critical_layers.append({
                'layer': phase_analysis['phase_transition_layer'],
                'type': 'phase_transition',
                'value': 0
            })
            
        return critical_layers
    
    def analyze_information_dynamics(self, features, ib_analysis):
        """Analyze dynamics of information flow"""
        layers = sorted(ib_analysis['layers'].keys(), 
                       key=lambda x: int(x.split('_')[1]))
        
        # Information velocity (rate of change)
        i_xt_traj = [ib_analysis['layers'][l]['I_X_T'] for l in layers]
        i_yt_traj = [ib_analysis['layers'][l]['I_Y_T'] for l in layers]
        
        velocity_x = np.diff(i_xt_traj)
        velocity_y = np.diff(i_yt_traj)
        
        # Information acceleration
        accel_x = np.diff(velocity_x)
        accel_y = np.diff(velocity_y)
        
        # Path length in information plane
        path_length = 0
        for i in range(len(i_xt_traj) - 1):
            dx = i_xt_traj[i+1] - i_xt_traj[i]
            dy = i_yt_traj[i+1] - i_yt_traj[i]
            path_length += np.sqrt(dx**2 + dy**2)
            
        return {
            'velocity': {
                'I_X_T': velocity_x.tolist(),
                'I_Y_T': velocity_y.tolist(),
                'mean_speed': float(np.mean(np.sqrt(velocity_x**2 + velocity_y**2)))
            },
            'acceleration': {
                'I_X_T': accel_x.tolist(),
                'I_Y_T': accel_y.tolist()
            },
            'path_length': float(path_length),
            'straightness': float(path_length / np.sqrt(
                (i_xt_traj[-1] - i_xt_traj[0])**2 + 
                (i_yt_traj[-1] - i_yt_traj[0])**2
            )) if path_length > 0 else 1
        }
    
    def summarize_trajectory(self, layer_results):
        """Summarize the information trajectory"""
        layers = sorted(layer_results.keys(), key=lambda x: int(x.split('_')[1]))
        
        i_xt_vals = [layer_results[l]['I_X_T'] for l in layers]
        i_yt_vals = [layer_results[l]['I_Y_T'] for l in layers]
        
        return {
            'initial_state': {
                'I_X_T': float(i_xt_vals[0]),
                'I_Y_T': float(i_yt_vals[0])
            },
            'final_state': {
                'I_X_T': float(i_xt_vals[-1]),
                'I_Y_T': float(i_yt_vals[-1])
            },
            'total_compression': float((i_xt_vals[0] - i_xt_vals[-1]) / i_xt_vals[0]) 
                                if i_xt_vals[0] > 0 else 0,
            'total_task_info_gain': float(i_yt_vals[-1] - i_yt_vals[0]),
            'peak_task_info': float(max(i_yt_vals)),
            'peak_task_info_layer': int(np.argmax(i_yt_vals))
        }
    
    def visualize_macroscopic_analysis(self, results, model_name):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Information plane trajectory
        ax = axes[0, 0]
        layers = sorted(results['information_flow']['layers'].keys(), 
                       key=lambda x: int(x.split('_')[1]))
        
        i_xt = [results['information_flow']['layers'][l]['I_X_T'] for l in layers]
        i_yt = [results['information_flow']['layers'][l]['I_Y_T'] for l in layers]
        
        # Plot trajectory
        ax.plot(i_xt, i_yt, 'bo-', markersize=8, linewidth=2)
        
        # Annotate layers
        for i, (x, y) in enumerate(zip(i_xt, i_yt)):
            ax.annotate(f'L{i}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
            
        # Mark phase transition
        if results['phase_analysis']['phase_transition_layer'] is not None:
            pt_idx = results['phase_analysis']['phase_transition_layer']
            ax.scatter(i_xt[pt_idx], i_yt[pt_idx], color='red', s=200, 
                      marker='*', label='Phase Transition')
            
        ax.set_xlabel('I(X;T)', fontsize=12)
        ax.set_ylabel('I(Y;T)', fontsize=12)
        ax.set_title('Information Plane Trajectory', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Information dynamics
        ax = axes[0, 1]
        ax.plot(range(len(i_xt)), i_xt, 'g-', label='I(X;T)', linewidth=2)
        ax.plot(range(len(i_yt)), i_yt, 'r-', label='I(Y;T)', linewidth=2)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Information (bits)', fontsize=12)
        ax.set_title('Layer-wise Information', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Information efficiency
        ax = axes[1, 0]
        efficiency = [results['information_flow']['layers'][l]['efficiency'] 
                     for l in layers]
        ax.plot(range(len(efficiency)), efficiency, 'mo-', linewidth=2)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('I(Y;T) / I(X;T)', fontsize=12)
        ax.set_title('Information Efficiency', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 4. Phase diagram
        ax = axes[1, 1]
        if results['phase_analysis']['phases']:
            for phase in results['phase_analysis']['phases']:
                ax.axvspan(phase['start'], phase['end'], 
                          alpha=0.3, label=phase['name'])
                
        # Plot information velocity
        velocity = results['information_dynamics']['velocity']['mean_speed']
        ax.text(0.5, 0.5, f'Mean Information Velocity: {velocity:.3f}', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_title('Learning Phases', fontsize=14)
        ax.legend()
        
        plt.suptitle(f'Macroscopic Information Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_macroscopic_analysis.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def load_results(self, results_dir="./results/macroscopic_analysis/"):
        """Load all macroscopic analysis results"""
        results_dir = Path(results_dir)
        if not results_dir.exists():
            print(f"Results directory {results_dir} does not exist")
            return {}
            
        all_results = {}
        for json_file in results_dir.glob("*_macroscopic.json"):
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
        """Compare macroscopic properties across models"""
        if not results_dict:
            print("No results to compare")
            return
            
        print("=== Macroscopic Analysis Comparison ===")
        
        comparison = {
            'models': list(results_dict.keys()),
            'information_flow': {},
            'phase_transitions': {},
            'critical_layers': {},
            'information_dynamics': {}
        }
        
        for model_name, results in results_dict.items():
            print(f"\n--- {model_name} ---")
            
            # Information flow summary
            if 'information_flow' in results and 'summary' in results['information_flow']:
                summary = results['information_flow']['summary']
                comparison['information_flow'][model_name] = {
                    'total_compression': summary.get('total_compression', 0),
                    'total_task_info_gain': summary.get('total_task_info_gain', 0),
                    'peak_task_info': summary.get('peak_task_info', 0),
                    'peak_task_info_layer': summary.get('peak_task_info_layer', -1)
                }
                print(f"  Total Compression: {summary.get('total_compression', 0):.3f}")
                print(f"  Task Info Gain: {summary.get('total_task_info_gain', 0):.3f}")
                print(f"  Peak Task Info: {summary.get('peak_task_info', 0):.3f}")
                print(f"  Peak at Layer: {summary.get('peak_task_info_layer', -1)}")
            
            # Phase analysis summary
            if 'phase_analysis' in results:
                phase = results['phase_analysis']
                comparison['phase_transitions'][model_name] = {
                    'has_compression_phase': phase.get('has_compression_phase', False),
                    'compression_start_layer': phase.get('compression_start_layer', -1),
                    'phase_transition_layer': phase.get('phase_transition_layer', -1)
                }
                print(f"  Has Compression Phase: {phase.get('has_compression_phase', False)}")
                print(f"  Compression Start: {phase.get('compression_start_layer', -1)}")
                print(f"  Phase Transition: {phase.get('phase_transition_layer', -1)}")
            
            # Critical layers
            if 'critical_transitions' in results:
                critical = results['critical_transitions']
                critical_summary = {}
                for layer_info in critical:
                    layer_type = layer_info.get('type', 'unknown')
                    critical_summary[layer_type] = {
                        'layer': layer_info.get('layer', -1),
                        'value': layer_info.get('value', 0)
                    }
                comparison['critical_layers'][model_name] = critical_summary
                
                for layer_type, info in critical_summary.items():
                    print(f"  {layer_type.replace('_', ' ').title()}: Layer {info['layer']} (value: {info['value']:.3f})")
            
            # Information dynamics
            if 'information_dynamics' in results:
                dynamics = results['information_dynamics']
                comparison['information_dynamics'][model_name] = {
                    'mean_speed': dynamics.get('velocity', {}).get('mean_speed', 0),
                    'path_length': dynamics.get('path_length', 0),
                    'straightness': dynamics.get('straightness', 1)
                }
                print(f"  Mean Info Speed: {dynamics.get('velocity', {}).get('mean_speed', 0):.3f}")
                print(f"  Path Length: {dynamics.get('path_length', 0):.3f}")
                print(f"  Path Straightness: {dynamics.get('straightness', 1):.3f}")
        
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
        
        # 1. Information Flow Properties
        ax = axes[0, 0]
        info_flow = comparison['information_flow']
        
        if info_flow:
            compressions = [info_flow[model].get('total_compression', 0) for model in models]
            task_gains = [info_flow[model].get('total_task_info_gain', 0) for model in models]
            
            ax.scatter(compressions, task_gains, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model[:10], (compressions[i], task_gains[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                           
            ax.set_xlabel('Total Compression')
            ax.set_ylabel('Task Info Gain')
            ax.set_title('Information Flow Trade-off')
            ax.grid(True, alpha=0.3)
        
        # 2. Phase Transitions
        ax = axes[0, 1]
        phase_data = comparison['phase_transitions']
        
        if phase_data:
            has_compression = [1 if phase_data[model].get('has_compression_phase', False) else 0 
                              for model in models]
            compression_starts = [phase_data[model].get('compression_start_layer', -1) 
                                 for model in models]
            
            colors = ['red' if x == 1 else 'blue' for x in has_compression]
            ax.scatter(range(len(models)), compression_starts, c=colors, s=100, alpha=0.7)
            
            for i, model in enumerate(models):
                ax.annotate(model[:10], (i, compression_starts[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                           
            ax.set_xlabel('Model Index')
            ax.set_ylabel('Compression Start Layer')
            ax.set_title('Phase Transitions (Red=Has Compression)')
            ax.set_xticks(range(len(models)))
            ax.grid(True, alpha=0.3)
        
        # 3. Critical Layers Distribution
        ax = axes[1, 0]
        critical_data = comparison['critical_layers']
        
        if critical_data:
            layer_types = set()
            for model_data in critical_data.values():
                layer_types.update(model_data.keys())
            layer_types = list(layer_types)
            
            x = np.arange(len(models))
            width = 0.2
            
            for i, layer_type in enumerate(layer_types):
                layers = [critical_data[model].get(layer_type, {}).get('layer', -1) 
                         for model in models]
                ax.bar(x + i * width, layers, width, label=layer_type.replace('_', ' ').title())
                
            ax.set_xlabel('Models')
            ax.set_ylabel('Layer Index')
            ax.set_title('Critical Layers by Type')
            ax.set_xticks(x + width)
            ax.set_xticklabels([m[:10] + '...' if len(m) > 10 else m for m in models], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Information Dynamics
        ax = axes[1, 1]
        dynamics_data = comparison['information_dynamics']
        
        if dynamics_data:
            speeds = [dynamics_data[model].get('mean_speed', 0) for model in models]
            straightness = [dynamics_data[model].get('straightness', 1) for model in models]
            
            ax.scatter(speeds, straightness, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model[:10], (speeds[i], straightness[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                           
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