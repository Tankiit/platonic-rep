# analyze_multiscale_dynamics.py
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.cluster import KMeans
from datasets import load_dataset

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

class MultiscaleInformationAnalysis:
    """
    Analyzes neural representations at multiple scales:
    - Microscopic: AGOP structure
    - Mesoscopic: NTK/feature evolution  
    - Macroscopic: Information bottleneck
    """
    
    def __init__(self, feature_dir="./results/features/minhuh/prh/wit_1024/"):
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path("./results/multiscale_analysis/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset for labels and inputs
        self.dataset = load_dataset("minhuh/prh", revision="wit_1024", split='train')
        
    def analyze_all_models(self):
        """Run multiscale analysis on all models"""
        for feature_file in tqdm(list(self.feature_dir.glob("*.pt"))):
            print(f"\nAnalyzing {feature_file.name}...")
            self.analyze_model(feature_file)
            
    def analyze_model(self, feature_path):
        """Complete multiscale analysis for one model"""
        # Load features
        data = torch.load(feature_path, map_location='cpu')
        features = data['feats']  # [N, L, D]
        
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
            
        model_name = feature_path.stem
        
        results = {
            'model': model_name,
            'microscopic': {},
            'mesoscopic': {},
            'macroscopic': {},
            'cross_scale_connections': {}
        }
        
        # 1. Microscopic analysis (individual neuron/feature level)
        print("  Performing microscopic analysis...")
        micro_analysis = self.analyze_microscopic(features)
        results['microscopic'] = micro_analysis
        
        # 2. Mesoscopic analysis (layer/group level)
        print("  Performing mesoscopic analysis...")
        meso_analysis = self.analyze_mesoscopic(features)
        results['mesoscopic'] = meso_analysis
        
        # 3. Macroscopic analysis (network level)
        print("  Performing macroscopic analysis...")
        macro_analysis = self.analyze_macroscopic(features)
        results['macroscopic'] = macro_analysis
        
        # 4. Cross-scale connections
        print("  Analyzing cross-scale connections...")
        cross_scale = self.analyze_cross_scale_connections(micro_analysis, meso_analysis, macro_analysis)
        results['cross_scale_connections'] = cross_scale
        
        # Save results
        output_path = self.output_dir / f"{model_name}_multiscale.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
            
        # Generate visualizations
        self.visualize_multiscale_analysis(results, model_name)
        
    def analyze_microscopic(self, features):
        """Analyze at the microscopic (neuron/feature) level"""
        num_layers = features.shape[1]
        microscopic_results = {}
        
        for layer_idx in range(num_layers):
            layer_features = features[:, layer_idx, :]
            
            # Individual feature statistics
            feature_means = np.mean(layer_features.numpy(), axis=0)
            feature_vars = np.var(layer_features.numpy(), axis=0)
            
            # Sparsity analysis
            activation_sparsity = self.compute_sparsity(layer_features)
            
            # Feature selectivity
            selectivity = self.compute_selectivity(layer_features)
            
            microscopic_results[f'layer_{layer_idx}'] = {
                'mean_activation': float(np.mean(feature_means)),
                'variance_activation': float(np.mean(feature_vars)),
                'sparsity': activation_sparsity,
                'selectivity': selectivity,
                'feature_entropy': self.compute_feature_entropy(layer_features)
            }
            
        return microscopic_results
    
    def analyze_mesoscopic(self, features):
        """Analyze at the mesoscopic (layer/group) level"""
        num_layers = features.shape[1]
        mesoscopic_results = {}
        
        for layer_idx in range(num_layers):
            layer_features = features[:, layer_idx, :]
            
            # Dimensionality analysis
            intrinsic_dim = self.compute_intrinsic_dimensionality(layer_features)
            
            # Feature correlations
            correlation_structure = self.analyze_correlation_structure(layer_features)
            
            # Clustering analysis
            cluster_metrics = self.analyze_clustering(layer_features)
            
            mesoscopic_results[f'layer_{layer_idx}'] = {
                'intrinsic_dimensionality': intrinsic_dim,
                'correlation_structure': correlation_structure,
                'clustering_metrics': cluster_metrics,
                'layer_coherence': self.compute_layer_coherence(layer_features)
            }
            
        return mesoscopic_results
    
    def analyze_macroscopic(self, features):
        """Analyze at the macroscopic (network) level"""
        # Get inputs and labels
        inputs = self.get_input_representation()
        labels = self.get_task_labels()
        
        num_layers = features.shape[1]
        macroscopic_results = {
            'information_bottleneck': {},
            'phase_transitions': {},
            'critical_points': {}
        }
        
        # Information bottleneck trajectory
        ib_trajectory = self.compute_information_bottleneck_trajectory(features, inputs, labels)
        macroscopic_results['information_bottleneck'] = ib_trajectory
        
        # Phase transition detection
        phase_transitions = self.detect_phase_transitions(ib_trajectory)
        macroscopic_results['phase_transitions'] = phase_transitions
        
        # Critical point identification
        critical_points = self.identify_critical_points(features, ib_trajectory)
        macroscopic_results['critical_points'] = critical_points
        
        return macroscopic_results
    
    def analyze_cross_scale_connections(self, micro, meso, macro):
        """Analyze connections between different scales"""
        connections = {
            'micro_meso_correlations': {},
            'meso_macro_correlations': {},
            'emergent_properties': {}
        }
        
        # Correlations between microscopic and mesoscopic properties
        layers = sorted(micro.keys(), key=lambda x: int(x.split('_')[1]))
        
        micro_sparsity = [micro[l]['sparsity'] for l in layers]
        meso_intrinsic_dim = [meso[l]['intrinsic_dimensionality'] for l in layers]
        
        if len(micro_sparsity) > 1:
            sparsity_dim_corr = spearmanr(micro_sparsity, meso_intrinsic_dim)[0]
            connections['micro_meso_correlations']['sparsity_vs_intrinsic_dim'] = float(sparsity_dim_corr)
        
        # Correlations between mesoscopic and macroscopic properties
        macro_i_xt = [macro['information_bottleneck']['layers'][l]['I_X_T'] for l in layers]
        meso_coherence = [meso[l]['layer_coherence'] for l in layers]
        
        if len(macro_i_xt) > 1:
            coherence_info_corr = spearmanr(meso_coherence, macro_i_xt)[0]
            connections['meso_macro_correlations']['coherence_vs_I_X_T'] = float(coherence_info_corr)
        
        # Emergent properties (properties that appear at higher scales)
        connections['emergent_properties'] = self.identify_emergent_properties(micro, meso, macro)
        
        return connections
    
    def compute_sparsity(self, features):
        """Compute sparsity of activations"""
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        # Percentage of activations close to zero
        threshold = 0.1 * np.max(np.abs(features))
        sparsity = np.mean(np.abs(features) < threshold)
        
        return float(sparsity)
    
    def compute_selectivity(self, features):
        """Compute selectivity of features"""
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        # For each feature, measure how selective it is across samples
        selectivities = []
        for i in range(features.shape[1]):
            feature_vals = features[:, i]
            # Normalize
            feature_vals = (feature_vals - np.min(feature_vals)) / (np.max(feature_vals) - np.min(feature_vals) + 1e-10)
            # Selectivity as kurtosis (peakedness)
            selectivity = np.mean((feature_vals - np.mean(feature_vals))**4) / (np.var(feature_vals)**2 + 1e-10)
            selectivities.append(selectivity)
            
        return float(np.mean(selectivities))
    
    def compute_feature_entropy(self, features):
        """Compute entropy of feature activations"""
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        # Discretize and compute entropy
        entropies = []
        for i in range(min(100, features.shape[1])):  # Limit to 100 features for efficiency
            feature_vals = features[:, i]
            # Discretize into 10 bins
            discretized = np.digitize(feature_vals, bins=np.linspace(np.min(feature_vals), np.max(feature_vals), 10))
            # Compute entropy
            counts = np.bincount(discretized)
            probs = counts / np.sum(counts)
            entropies.append(-np.sum(probs * np.log2(probs + 1e-10)))
            
        return float(np.mean(entropies))
    
    def compute_intrinsic_dimensionality(self, features):
        """Compute intrinsic dimensionality of representations"""
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        # Use PCA to estimate intrinsic dimensionality
        pca = PCA()
        pca.fit(features)
        
        # Find number of components needed for 95% variance
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.argmax(explained_variance >= 0.95) + 1
        
        return int(intrinsic_dim)
    
    def analyze_correlation_structure(self, features):
        """Analyze correlation structure of features"""
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        # Compute correlation matrix (on a subset for efficiency)
        n_samples = min(1000, features.shape[0])
        subset = features[:n_samples, :100]  # Use first 100 features
        
        corr_matrix = np.corrcoef(subset.T)
        
        # Analyze correlation distribution
        corr_vals = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        return {
            'mean_correlation': float(np.mean(corr_vals)),
            'correlation_entropy': float(entropy(np.histogram(corr_vals, bins=20)[0])),
            'high_correlation_fraction': float(np.mean(np.abs(corr_vals) > 0.7))
        }
    
    def analyze_clustering(self, features):
        """Analyze clustering structure of representations"""
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        # Use K-means to evaluate clustering structure
        n_samples = min(1000, features.shape[0])
        subset = features[:n_samples]
        
        # Reduce dimensionality for efficiency
        if subset.shape[1] > 50:
            pca = PCA(n_components=50)
            subset = pca.fit_transform(subset)
            
        # Try different numbers of clusters
        silhouette_scores = []
        for n_clusters in [2, 5, 10]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(subset)
            
            # Compute silhouette score
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) > 1:
                score = silhouette_score(subset, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
                
        return {
            'silhouette_scores': [float(s) for s in silhouette_scores],
            'optimal_clusters': int([2, 5, 10][np.argmax(silhouette_scores)])
        }
    
    def compute_layer_coherence(self, features):
        """Compute how coherent the layer representations are"""
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        # Measure how well features can be reconstructed from a low-dimensional subspace
        pca = PCA(n_components=10)
        reduced = pca.fit_transform(features)
        reconstructed = pca.inverse_transform(reduced)
        
        # Reconstruction error
        reconstruction_error = np.mean((features - reconstructed) ** 2)
        
        return float(1 / (1 + reconstruction_error))  # Coherence score
    
    def compute_information_bottleneck_trajectory(self, features, inputs, labels):
        """Compute information bottleneck trajectory across layers"""
        num_layers = features.shape[1]
        
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
        kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
        labels = kbd.fit_transform(np.array(lengths).reshape(-1, 1)).flatten()
        
        return labels.astype(int)
    
    def compute_mutual_information_hd(self, X, Y, method='kde_approximation'):
        """Compute MI for high-dimensional data"""
        # Reduce dimensionality first
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
        if X.shape[1] > 10:
            pca = PCA(n_components=10, random_state=42)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
            
        # Average MI across components
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
    
    def identify_critical_points(self, features, ib_analysis):
        """Identify critical points for information processing"""
        critical_points = []
        
        layers = sorted(ib_analysis['layers'].keys(), 
                       key=lambda x: int(x.split('_')[1]))
        
        # Critical layer 1: Maximum I(Y;T)
        i_yt_values = [ib_analysis['layers'][l]['I_Y_T'] for l in layers]
        max_iyt_layer = np.argmax(i_yt_values)
        critical_points.append({
            'layer': max_iyt_layer,
            'type': 'max_task_information',
            'value': float(i_yt_values[max_iyt_layer])
        })
        
        # Critical layer 2: Maximum efficiency
        efficiency_values = [ib_analysis['layers'][l]['efficiency'] for l in layers]
        max_eff_layer = np.argmax(efficiency_values)
        critical_points.append({
            'layer': max_eff_layer,
            'type': 'max_efficiency',
            'value': float(efficiency_values[max_eff_layer])
        })
        
        # Critical layer 3: Phase transition
        phase_analysis = self.detect_phase_transitions(ib_analysis)
        if phase_analysis['phase_transition_layer'] is not None:
            critical_points.append({
                'layer': phase_analysis['phase_transition_layer'],
                'type': 'phase_transition',
                'value': 0
            })
            
        return critical_points
    
    def identify_emergent_properties(self, micro, meso, macro):
        """Identify properties that emerge at higher scales"""
        emergent = {}
        
        layers = sorted(micro.keys(), key=lambda x: int(x.split('_')[1]))
        
        # Check if information compression emerges
        compression_values = [macro['information_bottleneck']['layers'][l]['compression'] for l in layers]
        if np.max(compression_values) > 0.5:  # Significant compression
            emergent['information_compression'] = {
                'emerges': True,
                'strength': float(np.max(compression_values)),
                'emergence_layer': int(np.argmax(compression_values))
            }
        
        # Check if efficient coding emerges
        efficiency_values = [macro['information_bottleneck']['layers'][l]['efficiency'] for l in layers]
        if np.max(efficiency_values) > np.mean(efficiency_values) * 1.5:  # Significant peak
            emergent['efficient_coding'] = {
                'emerges': True,
                'peak_efficiency': float(np.max(efficiency_values)),
                'peak_layer': int(np.argmax(efficiency_values))
            }
            
        return emergent
    
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
    
    def visualize_multiscale_analysis(self, results, model_name):
        """Create comprehensive visualization of multiscale analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        layers = sorted(results['microscopic'].keys(), key=lambda x: int(x.split('_')[1]))
        
        # 1. Microscopic vs Mesoscopic
        ax = axes[0, 0]
        micro_sparsity = [results['microscopic'][l]['sparsity'] for l in layers]
        meso_intrinsic_dim = [results['mesoscopic'][l]['intrinsic_dimensionality'] for l in layers]
        
        ax.plot(range(len(micro_sparsity)), micro_sparsity, 'b-', label='Sparsity', linewidth=2)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Sparsity', color='b', fontsize=12)
        ax.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax.twinx()
        ax2.plot(range(len(meso_intrinsic_dim)), meso_intrinsic_dim, 'r-', label='Intrinsic Dim', linewidth=2)
        ax2.set_ylabel('Intrinsic Dimensionality', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('Microscopic vs Mesoscopic Properties', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 2. Mesoscopic vs Macroscopic
        ax = axes[0, 1]
        meso_coherence = [results['mesoscopic'][l]['layer_coherence'] for l in layers]
        macro_i_xt = [results['macroscopic']['information_bottleneck']['layers'][l]['I_X_T'] for l in layers]
        
        ax.plot(range(len(meso_coherence)), meso_coherence, 'g-', label='Coherence', linewidth=2)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Coherence', color='g', fontsize=12)
        ax.tick_params(axis='y', labelcolor='g')
        
        ax2 = ax.twinx()
        ax2.plot(range(len(macro_i_xt)), macro_i_xt, 'm-', label='I(X;T)', linewidth=2)
        ax2.set_ylabel('I(X;T)', color='m', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='m')
        
        ax.set_title('Mesoscopic vs Macroscopic Properties', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 3. Information plane trajectory
        ax = axes[1, 0]
        i_xt = [results['macroscopic']['information_bottleneck']['layers'][l]['I_X_T'] for l in layers]
        i_yt = [results['macroscopic']['information_bottleneck']['layers'][l]['I_Y_T'] for l in layers]
        
        ax.plot(i_xt, i_yt, 'bo-', markersize=8, linewidth=2)
        for i, (x, y) in enumerate(zip(i_xt, i_yt)):
            ax.annotate(f'L{i}', (x, y), xytext=(5, 5), textcoords='offset points')
            
        # Mark phase transition if exists
        if results['macroscopic']['phase_transitions']['phase_transition_layer'] is not None:
            pt_idx = results['macroscopic']['phase_transitions']['phase_transition_layer']
            ax.scatter(i_xt[pt_idx], i_yt[pt_idx], color='red', s=200, 
                      marker='*', label='Phase Transition')
            
        ax.set_xlabel('I(X;T)', fontsize=12)
        ax.set_ylabel('I(Y;T)', fontsize=12)
        ax.set_title('Information Plane Trajectory', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Cross-scale correlations
        ax = axes[1, 1]
        cross_scale = results['cross_scale_connections']
        
        # Create a summary table
        ax.text(0.1, 0.9, 'Cross-Scale Correlations', transform=ax.transAxes, 
               fontsize=14, fontweight='bold')
        
        y_pos = 0.8
        if 'micro_meso_correlations' in cross_scale:
            for key, value in cross_scale['micro_meso_correlations'].items():
                ax.text(0.1, y_pos, f"{key}: {value:.3f}", transform=ax.transAxes, fontsize=12)
                y_pos -= 0.1
                
        if 'meso_macro_correlations' in cross_scale:
            for key, value in cross_scale['meso_macro_correlations'].items():
                ax.text(0.1, y_pos, f"{key}: {value:.3f}", transform=ax.transAxes, fontsize=12)
                y_pos -= 0.1
                
        ax.axis('off')
        
        plt.suptitle(f'Multiscale Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_multiscale_analysis.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()


class MultiscaleIntegration:
    """
    Integrates mesoscopic and macroscopic analyses
    """
    
    def __init__(self):
        self.dirs = {
            'features': Path("./results/features/minhuh/prh/wit_1024/"),
            'multiscale': Path("./results/multiscale_analysis/"),
            'perceptual': Path("./results/perceptual_scores/")
        }
        self.output_dir = Path("./results/multiscale_integration/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_integration(self):
        """Run complete multiscale integration"""
        print("Running Multiscale Integration Analysis")
        print("=" * 50)
        
        # First run individual analyses
        print("\n1. Running multiscale analysis...")
        analyzer = MultiscaleInformationAnalysis()
        analyzer.analyze_all_models()
        
        # Now integrate results
        print("\n2. Integrating multiscale results...")
        integrated_results = self.integrate_all_scales()
        
        # Test specific hypotheses
        print("\n3. Testing multiscale hypotheses...")
        hypothesis_results = self.test_multiscale_hypotheses(integrated_results)
        
        # Generate final report
        print("\n4. Generating integrated report...")
        self.generate_integrated_report(integrated_results, hypothesis_results)
        
        return integrated_results, hypothesis_results
    
    def integrate_all_scales(self):
        """Integrate results across all scales"""
        integrated = {}
        
        # Get all analyzed models
        models = set()
        for multiscale_file in self.dirs['multiscale'].glob("*_multiscale.json"):
            model_name = multiscale_file.stem.replace("_multiscale", "")
            models.add(model_name)
            
        for model in models:
            print(f"  Integrating {model}...")
            
            # Load results from different scales
            multiscale_path = self.dirs['multiscale'] / f"{model}_multiscale.json"
            percept_path = self.dirs['perceptual'] / f"{model}_perceptual_scores.json"
            
            if not multiscale_path.exists():
                continue
                
            with open(multiscale_path, 'r') as f:
                multiscale_data = json.load(f)
                
            # Optional: load perceptual data if available
            percept_data = None
            if percept_path.exists():
                with open(percept_path, 'r') as f:
                    percept_data = json.load(f)
                    
            # Integrate across scales
            integrated[model] = self.integrate_model_results(
                multiscale_data, percept_data
            )
            
        return integrated
    
    def integrate_model_results(self, multiscale, percept=None):
        """Integrate results for a single model"""
        integrated = {
            'layer_analysis': {},
            'scale_connections': {},
            'critical_points': {}
        }
        
        # Get layer keys
        layers = sorted([k for k in multiscale['microscopic'].keys()], 
                      key=lambda x: int(x.split('_')[1]))
        
        # Integrate layer-by-layer
        for i, layer in enumerate(layers):
            layer_data = {
                'layer_idx': i,
                # Microscopic features
                'sparsity': multiscale['microscopic'][layer]['sparsity'],
                'selectivity': multiscale['microscopic'][layer]['selectivity'],
                # Mesoscopic features
                'intrinsic_dim': multiscale['mesoscopic'][layer]['intrinsic_dimensionality'],
                'coherence': multiscale['mesoscopic'][layer]['layer_coherence'],
                # Macroscopic features
                'I_X_T': multiscale['macroscopic']['information_bottleneck']['layers'][layer]['I_X_T'],
                'I_Y_T': multiscale['macroscopic']['information_bottleneck']['layers'][layer]['I_Y_T'],
                'info_efficiency': multiscale['macroscopic']['information_bottleneck']['layers'][layer]['efficiency'],
                # Perceptual features (if available)
                'perceptual_alignment': None
            }
            
            if percept and layer in percept.get('layers', {}):
                layer_data['perceptual_alignment'] = percept['layers'][layer]['rsa_score']['correlation']
                
            integrated['layer_analysis'][layer] = layer_data
            
        # Find scale connections
        integrated['scale_connections'] = multiscale['cross_scale_connections']
        
        # Identify critical points across scales
        integrated['critical_points'] = self.identify_multiscale_critical_points(
            multiscale, integrated['layer_analysis']
        )
        
        return integrated
    
    def identify_multiscale_critical_points(self, multiscale, layer_analysis):
        """Identify critical points that appear across scales"""
        critical_points = []
        
        # 1. Phase transition points
        if multiscale['macroscopic']['phase_transitions']['phase_transition_layer'] is not None:
            pt_layer = multiscale['macroscopic']['phase_transitions']['phase_transition_layer']
            
            critical_points.append({
                'layer': pt_layer,
                'type': 'phase_transition',
                'microscopic_signature': {
                    'sparsity': layer_analysis[f'layer_{pt_layer}']['sparsity'],
                    'selectivity': layer_analysis[f'layer_{pt_layer}']['selectivity']
                },
                'mesoscopic_signature': {
                    'intrinsic_dim': layer_analysis[f'layer_{pt_layer}']['intrinsic_dim'],
                    'coherence': layer_analysis[f'layer_{pt_layer}']['coherence']
                },
                'macroscopic_signature': {
                    'I_X_T': layer_analysis[f'layer_{pt_layer}']['I_X_T'],
                    'I_Y_T': layer_analysis[f'layer_{pt_layer}']['I_Y_T']
                }
            })
            
        # 2. Maximum efficiency point
        layers = sorted(layer_analysis.keys(), key=lambda x: int(x.split('_')[1]))
        efficiencies = [layer_analysis[l]['info_efficiency'] for l in layers]
        max_eff_layer = np.argmax(efficiencies)
        
        critical_points.append({
            'layer': max_eff_layer,
            'type': 'max_efficiency',
            'value': float(efficiencies[max_eff_layer])
        })
        
        return critical_points
    
    def test_multiscale_hypotheses(self, integrated_results):
        """Test specific multiscale hypotheses"""
        hypotheses = {}
        
        # Hypothesis 1: Coherence increase precedes information compression
        print("  Testing H1: Coherence increase → Information compression...")
        h1_results = self.test_coherence_compression_hypothesis(integrated_results)
        hypotheses['H1_coherence_compression'] = h1_results
        
        # Hypothesis 2: Information efficiency correlates with perceptual alignment
        print("  Testing H2: Information efficiency ↔ Perceptual alignment...")
        h2_results = self.test_efficiency_perception_hypothesis(integrated_results)
        hypotheses['H2_efficiency_perception'] = h2_results
        
        # Hypothesis 3: Phase transitions involve changes at all scales
        print("  Testing H3: Phase transitions involve multiscale changes...")
        h3_results = self.test_multiscale_phase_hypothesis(integrated_results)
        hypotheses['H3_multiscale_phases'] = h3_results
        
        return hypotheses
    
    def test_coherence_compression_hypothesis(self, integrated):
        """Test if coherence increase precedes compression"""
        evidence = []
        
        for model, data in integrated.items():
            layers = sorted(data['layer_analysis'].keys(), 
                          key=lambda x: int(x.split('_')[1]))
            
            # Get coherence and I(X;T)
            coherence = [data['layer_analysis'][l]['coherence'] for l in layers]
            i_xt = [data['layer_analysis'][l]['I_X_T'] for l in layers]
            
            # Find where coherence increases
            coherence_jump_layer = None
            for i in range(1, len(coherence)):
                if coherence[i] > 1.2 * coherence[i-1]:  # 20% increase
                    coherence_jump_layer = i
                    break
                    
            # Find where compression starts
            compression_layer = None
            for i in range(1, len(i_xt)):
                if i_xt[i] < 0.95 * i_xt[i-1]:  # 5% drop
                    compression_layer = i
                    break
                    
            if coherence_jump_layer is not None and compression_layer is not None:
                evidence.append({
                    'model': model,
                    'coherence_jump_layer': coherence_jump_layer,
                    'compression_layer': compression_layer,
                    'coherence_precedes': coherence_jump_layer <= compression_layer
                })
                
        # Summarize
        if evidence:
            precedes_count = sum(1 for e in evidence if e['coherence_precedes'])
            proportion = precedes_count / len(evidence)
        else:
            proportion = 0
            
        return {
            'hypothesis': 'Coherence increase precedes information compression',
            'evidence': evidence,
            'support_ratio': proportion,
            'supported': proportion > 0.7
        }
    
    def test_efficiency_perception_hypothesis(self, integrated):
        """Test correlation between information efficiency and perception"""
        all_efficiencies = []
        all_perceptual = []
        
        for model, data in integrated.items():
            for layer, layer_data in data['layer_analysis'].items():
                if layer_data['perceptual_alignment'] is not None:
                    all_efficiencies.append(layer_data['info_efficiency'])
                    all_perceptual.append(layer_data['perceptual_alignment'])
                    
        if len(all_efficiencies) > 10:
            corr, p_value = spearmanr(all_efficiencies, all_perceptual)
        else:
            corr, p_value = 0, 1
            
        return {
            'hypothesis': 'Information efficiency correlates with perceptual alignment',
            'correlation': float(corr),
            'p_value': float(p_value),
            'n_samples': len(all_efficiencies),
            'supported': p_value < 0.05 and corr > 0.5
        }
    
    def test_multiscale_phase_hypothesis(self, integrated):
        """Test if phase transitions involve changes at all scales"""
        evidence = []
        
        for model, data in integrated.items():
            # Check if phase transitions show changes at all scales
            if 'phase_transition' in str(data.get('critical_points', [])):
                # Look for phase transition layer
                pt_layer = None
                for cp in data.get('critical_points', []):
                    if cp['type'] == 'phase_transition':
                        pt_layer = cp['layer']
                        break
                
                if pt_layer is not None:
                    # Check for changes at all scales around this layer
                    layers = sorted(data['layer_analysis'].keys(), 
                                  key=lambda x: int(x.split('_')[1]))
                    
                    # Get metrics around phase transition
                    if pt_layer > 0 and pt_layer < len(layers) - 1:
                        # Check for changes in microscopic properties
                        sparsity_before = data['layer_analysis'][layers[pt_layer-1]]['sparsity']
                        sparsity_after = data['layer_analysis'][layers[pt_layer+1]]['sparsity']
                        sparsity_change = abs(sparsity_after - sparsity_before)
                        
                        # Check for changes in mesoscopic properties
                        coherence_before = data['layer_analysis'][layers[pt_layer-1]]['coherence']
                        coherence_after = data['layer_analysis'][layers[pt_layer+1]]['coherence']
                        coherence_change = abs(coherence_after - coherence_before)
                        
                        # Check for changes in macroscopic properties
                        i_xt_before = data['layer_analysis'][layers[pt_layer-1]]['I_X_T']
                        i_xt_after = data['layer_analysis'][layers[pt_layer+1]]['I_X_T']
                        i_xt_change = abs(i_xt_after - i_xt_before)
                        
                        # Consider it multiscale if all show significant changes
                        multiscale_change = (sparsity_change > 0.1 and 
                                           coherence_change > 0.1 and 
                                           i_xt_change > 0.1)
                        
                        evidence.append({
                            'model': model,
                            'phase_transition_layer': pt_layer,
                            'multiscale_changes': multiscale_change,
                            'sparsity_change': sparsity_change,
                            'coherence_change': coherence_change,
                            'information_change': i_xt_change
                        })
                
        return {
            'hypothesis': 'Phase transitions involve changes at all scales',
            'evidence': evidence,
            'supported': len(evidence) > 0 and all(e['multiscale_changes'] for e in evidence)
        }
    
    def generate_integrated_report(self, integrated, hypotheses):
        """Generate comprehensive multiscale report"""
        report = f"""# Multiscale Analysis Report

## Executive Summary

This analysis integrates microscopic (individual features), mesoscopic (layer properties), 
and macroscopic (information flow) perspectives to understand neural network representations.

## Key Findings

### Scale Connections
"""
        
        # Add model-specific findings
        for model, data in integrated.items():
            if 'scale_connections' in data:
                conn = data['scale_connections']
                report += f"\n**{model}**:\n"
                if 'micro_meso_correlations' in conn:
                    for key, value in conn['micro_meso_correlations'].items():
                        report += f"- {key}: {value:.3f}\n"
                if 'meso_macro_correlations' in conn:
                    for key, value in conn['meso_macro_correlations'].items():
                        report += f"- {key}: {value:.3f}\n"
                
        report += "\n### Hypothesis Tests\n\n"
        
        # Add hypothesis results
        for h_name, h_results in hypotheses.items():
            report += f"**{h_results['hypothesis']}**\n"
            report += f"- Supported: {'Yes' if h_results['supported'] else 'No'}\n"
            
            if 'correlation' in h_results:
                report += f"- Correlation: {h_results['correlation']:.3f} (p={h_results['p_value']:.4f})\n"
            if 'support_ratio' in h_results:
                report += f"- Support ratio: {h_results['support_ratio']:.2%}\n"
                
            report += "\n"
            
        # Save report
        with open(self.output_dir / 'multiscale_report.md', 'w') as f:
            f.write(report)
            
        # Create summary visualization
        self.create_summary_visualization(integrated, hypotheses)
        
    def create_summary_visualization(self, integrated, hypotheses):
        """Create summary visualization of multiscale analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Coherence vs Information for all models
        ax = axes[0, 0]
        for model, data in integrated.items():
            if 'layer_analysis' not in data:
                continue
                
            layers = sorted(data['layer_analysis'].keys(), 
                          key=lambda x: int(x.split('_')[1]))
            
            coherence = [data['layer_analysis'][l]['coherence'] for l in layers]
            i_xt = [data['layer_analysis'][l]['I_X_T'] for l in layers]
            
            ax.scatter(coherence, i_xt, label=model[:20], alpha=0.6)
            
        ax.set_xlabel('Coherence')
        ax.set_ylabel('I(X;T)')
        ax.set_title('Mesoscopic-Macroscopic Connection')
        if len(integrated) < 5:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Efficiency vs Perceptual alignment
        ax = axes[0, 1]
        all_eff = []
        all_percept = []
        
        for model, data in integrated.items():
            for layer, layer_data in data.get('layer_analysis', {}).items():
                if layer_data.get('perceptual_alignment') is not None:
                    all_eff.append(layer_data['info_efficiency'])
                    all_percept.append(layer_data['perceptual_alignment'])
                    
        if all_eff:
            ax.scatter(all_eff, all_percept, alpha=0.6)
            # Add trend line
            z = np.polyfit(all_eff, all_percept, 1)
            p = np.poly1d(z)
            ax.plot(sorted(all_eff), p(sorted(all_eff)), "r--", alpha=0.8)
            
        ax.set_xlabel('Information Efficiency')
        ax.set_ylabel('Perceptual Alignment')
        ax.set_title('Efficiency-Perception Relationship')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Phase diagram
        ax = axes[1, 0]
        # Aggregate phase information
        ax.text(0.1, 0.9, 'Multiscale Phase Analysis', transform=ax.transAxes, 
               fontsize=14, fontweight='bold')
        
        y_pos = 0.7
        for i, (h_name, h_results) in enumerate(hypotheses.items()):
            status = "✓" if h_results['supported'] else "✗"
            ax.text(0.1, y_pos - i*0.15, f"{status} {h_results['hypothesis'][:50]}...", 
                   transform=ax.transAxes, fontsize=11)
                   
        ax.axis('off')
        
        # Plot 4: Critical points across scales
        ax = axes[1, 1]
        critical_layers = []
        critical_types = []
        
        for model, data in integrated.items():
            for cp in data.get('critical_points', []):
                critical_layers.append(cp['layer'])
                critical_types.append(cp['type'])
                
        if critical_layers:
            from collections import Counter
            layer_counts = Counter(critical_layers)
            
            layers = sorted(layer_counts.keys())
            counts = [layer_counts[l] for l in layers]
            
            ax.bar(layers, counts)
            ax.set_xlabel('Layer')
            ax.set_ylabel('# Models with Critical Point')
            ax.set_title('Critical Layers Across Models')
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.suptitle('Multiscale Analysis Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'multiscale_summary.png', dpi=150)
        plt.close()


def main():
    """Main function to run the complete multiscale analysis"""
    integrator = MultiscaleIntegration()
    integrated_results, hypothesis_results = integrator.run_integration()
    
    print("\n✓ Multiscale analysis complete!")
    print(f"Results saved in: {integrator.output_dir}")


if __name__ == "__main__":
    main()