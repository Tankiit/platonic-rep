#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.interpolate import griddata
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import alignment computation from platonic
try:
    from measure_alignment import compute_score
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False
    print("Warning: Alignment computation not available. Install platonic package.")

class CrossModalPhaseAnalyzer:
    """
    Cross-modal phase diagram analysis for multi-model representations
    Extends NeuREPs phase diagram to cross-modal settings
    """
    
    def __init__(self, output_dir="./results/cross_modal_phase_analysis/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        self.device = self._auto_detect_device()
        print(f"Cross-modal phase analyzer using device: {self.device}")
        
        # Phase diagram parameters
        self.ntk_thresholds = {
            'lazy': 0.9,
            'optimal': 0.7,
            'chaotic': 0.5
        }
        
        # Supported modality pairs
        self.supported_modalities = {
            'vision': ['resnet18', 'resnet34', 'resnet50', 'vit_base_patch16_224', 'convnext_tiny'],
            'text': ['bert_base', 'roberta_base', 'distilbert_base', 'gpt2_medium'],
            'multimodal': ['clip_base', 'clip_large', 'dinov2_base', 'dinov2_large']
        }
        
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("MPS (Apple Silicon) available")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        
        return device
    
    def compute_cross_modal_ntk_stability(self, vision_features: torch.Tensor, 
                                        text_features: torch.Tensor) -> float:
        """
        Compute cross-modal NTK stability metric
        
        Args:
            vision_features: Vision model features [N, D_v]
            text_features: Text model features [N, D_t]
            
        Returns:
            NTK stability score
        """
        # Ensure same number of samples
        n = min(vision_features.shape[0], text_features.shape[0])
        v_feat = vision_features[:n]
        t_feat = text_features[:n]
        
        # Handle different feature dimensions by projecting to common space
        # Use PCA or simple linear projection to align dimensions
        min_dim = min(v_feat.shape[1], t_feat.shape[1])
        
        # Project both features to the same dimension
        if v_feat.shape[1] > min_dim:
            # Use first min_dim dimensions for vision features
            v_feat = v_feat[:, :min_dim]
        elif t_feat.shape[1] > min_dim:
            # Use first min_dim dimensions for text features
            t_feat = t_feat[:, :min_dim]
        
        # Compute cross-modal NTK
        ntk_cross = v_feat @ t_feat.T
        
        # Compute within-modal NTKs
        ntk_vv = v_feat @ v_feat.T
        ntk_tt = t_feat @ t_feat.T
        
        # NTK stability metric (how much cross-modal kernel preserves structure)
        ntk_stability = torch.norm(ntk_cross) / (torch.norm(ntk_vv) * torch.norm(ntk_tt))**0.5
        
        return ntk_stability.item()
    
    def compute_cross_modal_agop_proxy(self, vision_features: torch.Tensor, 
                                      text_features: torch.Tensor) -> float:
        """
        Compute cross-modal AGOP proxy using feature gradients approximation
        
        Args:
            vision_features: Vision model features [N, D_v]
            text_features: Text model features [N, D_t]
            
        Returns:
            AGOP magnitude score
        """
        # Ensure same number of samples
        n = min(vision_features.shape[0], text_features.shape[0])
        v_feat = vision_features[:n]
        t_feat = text_features[:n]
        
        # Handle different feature dimensions by projecting to common space
        min_dim = min(v_feat.shape[1], t_feat.shape[1])
        
        # Project both features to the same dimension
        if v_feat.shape[1] > min_dim:
            v_feat = v_feat[:, :min_dim]
        elif t_feat.shape[1] > min_dim:
            t_feat = t_feat[:, :min_dim]
        
        # Approximate gradients using finite differences
        if n > 1:
            v_grad_proxy = v_feat[1:] - v_feat[:-1]  # Approximate gradients
            t_grad_proxy = t_feat[1:] - t_feat[:-1]
            
            # Cross-modal AGOP
            agop_cross = torch.norm(v_grad_proxy.T @ t_grad_proxy) / len(v_grad_proxy)
        else:
            agop_cross = torch.tensor(0.0)
        
        return agop_cross.item()
    
    def compute_cross_modal_alignment(self, vision_features: torch.Tensor, 
                                     text_features: torch.Tensor, 
                                     metric: str = "mutual_knn", 
                                     topk: int = 10) -> float:
        """
        Compute cross-modal alignment using platonic metrics
        
        Args:
            vision_features: Vision model features [N, D_v]
            text_features: Text model features [N, D_t]
            metric: Alignment metric to use
            topk: Number of nearest neighbors for KNN metrics
            
        Returns:
            Alignment score
        """
        if not ALIGNMENT_AVAILABLE:
            print("Warning: Alignment computation not available. Using cosine similarity.")
            return self._compute_cosine_similarity(vision_features, text_features)
        
        try:
            # Ensure same number of samples
            n = min(vision_features.shape[0], text_features.shape[0])
            v_feat = vision_features[:n]
            t_feat = text_features[:n]
            
            # Handle different feature dimensions by projecting to common space
            min_dim = min(v_feat.shape[1], t_feat.shape[1])
            
            # Project both features to the same dimension
            if v_feat.shape[1] > min_dim:
                v_feat = v_feat[:, :min_dim]
            elif t_feat.shape[1] > min_dim:
                t_feat = t_feat[:, :min_dim]
            
            # Normalize features
            v_feat_norm = torch.nn.functional.normalize(v_feat, p=2, dim=-1)
            t_feat_norm = torch.nn.functional.normalize(t_feat, p=2, dim=-1)
            
            # Compute alignment score
            alignment_score, _ = compute_score([v_feat_norm], [t_feat_norm], 
                                             metric=metric, topk=topk, normalize=True)
            
            return alignment_score
            
        except Exception as e:
            print(f"Error computing alignment: {e}. Falling back to cosine similarity.")
            return self._compute_cosine_similarity(vision_features, text_features)
    
    def _compute_cosine_similarity(self, features1: torch.Tensor, 
                                  features2: torch.Tensor) -> float:
        """Fallback cosine similarity computation"""
        # Ensure same number of samples
        n = min(features1.shape[0], features2.shape[0])
        f1 = features1[:n]
        f2 = features2[:n]
        
        # Handle different feature dimensions by projecting to common space
        min_dim = min(f1.shape[1], f2.shape[1])
        
        # Project both features to the same dimension
        if f1.shape[1] > min_dim:
            f1 = f1[:, :min_dim]
        elif f2.shape[1] > min_dim:
            f2 = f2[:, :min_dim]
        
        # Normalize
        f1_norm = torch.nn.functional.normalize(f1, p=2, dim=-1)
        f2_norm = torch.nn.functional.normalize(f2, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.mean(torch.sum(f1_norm * f2_norm, dim=-1))
        
        return similarity.item()
    
    def compute_cross_modal_phase_diagram(self, vision_features: Dict[str, torch.Tensor], 
                                         text_features: Dict[str, torch.Tensor], 
                                         model_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """
        Compute cross-modal phase diagram metrics for model pairs
        
        Args:
            vision_features: Dictionary of vision model features
            text_features: Dictionary of text model features
            model_pairs: List of (vision_model, text_model) pairs to analyze
            
        Returns:
            List of phase diagram results
        """
        results = []
        
        print(f"Computing cross-modal phase diagram for {len(model_pairs)} model pairs...")
        
        for v_model, t_model in tqdm(model_pairs, desc="Analyzing model pairs"):
            if v_model not in vision_features or t_model not in text_features:
                print(f"Warning: Missing features for {v_model} or {t_model}")
                continue
            
            v_feat = vision_features[v_model]
            t_feat = text_features[t_model]
            
            # Compute phase diagram metrics
            ntk_stability = self.compute_cross_modal_ntk_stability(v_feat, t_feat)
            agop_magnitude = self.compute_cross_modal_agop_proxy(v_feat, t_feat)
            alignment = self.compute_cross_modal_alignment(v_feat, t_feat)
            
            # Determine phase region
            phase_region = self._determine_phase_region(ntk_stability, agop_magnitude)
            
            results.append({
                'v_model': v_model,
                't_model': t_model,
                'ntk_stability': ntk_stability,
                'agop_magnitude': agop_magnitude,
                'alignment': alignment,
                'phase_region': phase_region,
                'metadata': {
                    'v_feature_dim': v_feat.shape[1],
                    't_feature_dim': t_feat.shape[1],
                    'num_samples': min(v_feat.shape[0], t_feat.shape[0])
                }
            })
        
        return results
    
    def _determine_phase_region(self, ntk_stability: float, agop_magnitude: float) -> str:
        """Determine phase region based on NTK stability and AGOP magnitude"""
        if ntk_stability >= self.ntk_thresholds['lazy']:
            return 'lazy'
        elif ntk_stability >= self.ntk_thresholds['optimal']:
            return 'optimal'
        else:
            return 'chaotic'
    
    def plot_cross_modal_phase_diagram(self, results: List[Dict], 
                                      save_path: str = 'phase_diagram_crossmodal.png',
                                      title: str = "Cross-Modal Phase Diagram") -> plt.Figure:
        """
        Create comprehensive cross-modal phase diagram
        
        Args:
            results: Phase diagram results from compute_cross_modal_phase_diagram
            save_path: Path to save the plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not results:
            print("No results to plot")
            return None
        
        # Extract data
        ntk_vals = [r['ntk_stability'] for r in results]
        agop_vals = [r['agop_magnitude'] for r in results]
        align_vals = [r['alignment'] for r in results]
        phase_regions = [r['phase_region'] for r in results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Color mapping for phase regions
        phase_colors = {'lazy': 'red', 'optimal': 'green', 'chaotic': 'blue'}
        colors = [phase_colors.get(phase, 'gray') for phase in phase_regions]
        
        # (a) Alignment heatmap
        ax = axes[0, 0]
        self._plot_alignment_heatmap(ax, ntk_vals, agop_vals, align_vals, results)
        
        # (b) Phase space scatter
        ax = axes[0, 1]
        self._plot_phase_space_scatter(ax, ntk_vals, agop_vals, align_vals, colors, results)
        
        # (c) Alignment vs NTK stability
        ax = axes[0, 2]
        self._plot_alignment_vs_ntk(ax, ntk_vals, align_vals, colors)
        
        # (d) Phase distribution
        ax = axes[1, 0]
        self._plot_phase_distribution(ax, phase_regions)
        
        # (e) AGOP vs NTK relationship
        ax = axes[1, 1]
        self._plot_agop_vs_ntk(ax, ntk_vals, agop_vals, colors)
        
        # (f) Summary statistics
        ax = axes[1, 2]
        self._plot_summary_statistics(ax, results)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase diagram saved to {save_path}")
        
        return fig
    
    def _plot_alignment_heatmap(self, ax, ntk_vals, agop_vals, align_vals, results):
        """Plot alignment heatmap"""
        if len(results) < 3:
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(a) Cross-Modal Alignment Heatmap')
            return
        
        # Create grid for heatmap
        ntk_range = np.linspace(min(ntk_vals), max(ntk_vals), 50)
        agop_range = np.linspace(min(agop_vals), max(agop_vals), 50)
        
        # Interpolate alignment values on grid
        points = np.array([(r['ntk_stability'], r['agop_magnitude']) for r in results])
        grid_x, grid_y = np.meshgrid(ntk_range, agop_range)
        
        try:
            grid_z = griddata(points, align_vals, (grid_x, grid_y), method='cubic', fill_value=np.nan)
            
            # Plot heatmap
            im = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.7)
            ax.scatter(ntk_vals, agop_vals, c=align_vals, s=50, edgecolor='black', linewidth=1, cmap='viridis')
            
            # Add phase boundaries
            ax.axvline(x=self.ntk_thresholds['lazy'], color='red', linestyle='--', alpha=0.5, label='Lazy')
            ax.axvline(x=self.ntk_thresholds['optimal'], color='orange', linestyle='--', alpha=0.5, label='Optimal')
            
        except Exception as e:
            # Fallback to scatter plot
            ax.scatter(ntk_vals, agop_vals, c=align_vals, s=100, cmap='viridis')
        
        ax.set_xlabel('Cross-Modal NTK Stability')
        ax.set_ylabel('Cross-Modal AGOP Magnitude')
        ax.set_title('(a) Cross-Modal Alignment Heatmap')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        if 'im' in locals():
            plt.colorbar(im, ax=ax, label='Alignment Score')
    
    def _plot_phase_space_scatter(self, ax, ntk_vals, agop_vals, align_vals, colors, results):
        """Plot phase space scatter plot"""
        scatter = ax.scatter(ntk_vals, agop_vals, c=align_vals, s=100, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Cross-Modal NTK Stability')
        ax.set_ylabel('Cross-Modal AGOP Magnitude')
        ax.set_title('(b) Model Pairs in Phase Space')
        ax.grid(True, alpha=0.3)
        
        # Add model labels (top 5 for clarity)
        for i, r in enumerate(results[:5]):
            label = f"{r['v_model'][:4]}-{r['t_model'][:4]}"
            ax.annotate(label, (r['ntk_stability'], r['agop_magnitude']),
                       fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter, ax=ax, label='Alignment Score')
    
    def _plot_alignment_vs_ntk(self, ax, ntk_vals, align_vals, colors):
        """Plot alignment vs NTK stability"""
        ax.scatter(ntk_vals, align_vals, c=colors, s=50, alpha=0.7)
        ax.set_xlabel('Cross-Modal NTK Stability')
        ax.set_ylabel('Alignment Score')
        ax.set_title('(c) Optimal Alignment Region')
        ax.grid(True, alpha=0.3)
        
        # Highlight optimal region
        optimal_region = [(n, a) for n, a in zip(ntk_vals, align_vals) 
                         if self.ntk_thresholds['optimal'] < n < self.ntk_thresholds['lazy']]
        if optimal_region:
            opt_ntk, opt_align = zip(*optimal_region)
            ax.scatter(opt_ntk, opt_align, s=100, color='green', alpha=0.5, label='Optimal')
            ax.legend()
    
    def _plot_phase_distribution(self, ax, phase_regions):
        """Plot phase region distribution"""
        phase_counts = {}
        for phase in phase_regions:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        phases = list(phase_counts.keys())
        counts = list(phase_counts.values())
        colors = [phase_colors.get(phase, 'gray') for phase in phases]
        
        bars = ax.bar(phases, counts, color=colors, alpha=0.7)
        ax.set_xlabel('Phase Region')
        ax.set_ylabel('Number of Model Pairs')
        ax.set_title('(d) Phase Region Distribution')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
    
    def _plot_agop_vs_ntk(self, ax, ntk_vals, agop_vals, colors):
        """Plot AGOP vs NTK relationship"""
        ax.scatter(ntk_vals, agop_vals, c=colors, s=50, alpha=0.7)
        ax.set_xlabel('Cross-Modal NTK Stability')
        ax.set_ylabel('Cross-Modal AGOP Magnitude')
        ax.set_title('(e) AGOP vs NTK Relationship')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(ntk_vals) > 1:
            z = np.polyfit(ntk_vals, agop_vals, 1)
            p = np.poly1d(z)
            ax.plot(ntk_vals, p(ntk_vals), "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
            ax.legend()
    
    def _plot_summary_statistics(self, ax, results):
        """Plot summary statistics"""
        ax.axis('off')
        
        # Calculate statistics
        ntk_vals = [r['ntk_stability'] for r in results]
        agop_vals = [r['agop_magnitude'] for r in results]
        align_vals = [r['alignment'] for r in results]
        
        stats_text = f"Cross-Modal Phase Analysis Summary\n"
        stats_text += f"Total Model Pairs: {len(results)}\n\n"
        
        stats_text += f"NTK Stability:\n"
        stats_text += f"  Mean: {np.mean(ntk_vals):.3f}\n"
        stats_text += f"  Std: {np.std(ntk_vals):.3f}\n"
        stats_text += f"  Range: [{np.min(ntk_vals):.3f}, {np.max(ntk_vals):.3f}]\n\n"
        
        stats_text += f"AGOP Magnitude:\n"
        stats_text += f"  Mean: {np.mean(agop_vals):.3f}\n"
        stats_text += f"  Std: {np.std(agop_vals):.3f}\n"
        stats_text += f"  Range: [{np.min(agop_vals):.3f}, {np.max(agop_vals):.3f}]\n\n"
        
        stats_text += f"Alignment Score:\n"
        stats_text += f"  Mean: {np.mean(align_vals):.3f}\n"
        stats_text += f"  Std: {np.std(align_vals):.3f}\n"
        stats_text += f"  Range: [{np.min(align_vals):.3f}, {np.max(align_vals):.3f}]\n\n"
        
        # Phase distribution
        phase_counts = {}
        for r in results:
            phase = r['phase_region']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        stats_text += f"Phase Distribution:\n"
        for phase, count in phase_counts.items():
            percentage = (count / len(results)) * 100
            stats_text += f"  {phase.capitalize()}: {count} ({percentage:.1f}%)\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace', fontweight='bold')
    
    def analyze_cross_modal_representations(self, vision_features: Dict[str, torch.Tensor],
                                          text_features: Dict[str, torch.Tensor],
                                          model_pairs: List[Tuple[str, str]] = None,
                                          save_results: bool = True) -> Dict:
        """
        Complete cross-modal phase analysis pipeline
        
        Args:
            vision_features: Dictionary of vision model features
            text_features: Dictionary of text model features
            model_pairs: List of (vision_model, text_model) pairs to analyze
            save_results: Whether to save results to file
            
        Returns:
            Analysis results dictionary
        """
        # Generate model pairs if not provided
        if model_pairs is None:
            model_pairs = []
            for v_model in vision_features.keys():
                for t_model in text_features.keys():
                    model_pairs.append((v_model, t_model))
        
        print(f"Starting cross-modal phase analysis with {len(model_pairs)} model pairs...")
        
        # Compute phase diagram
        phase_results = self.compute_cross_modal_phase_diagram(
            vision_features, text_features, model_pairs
        )
        
        # Create visualizations
        fig = self.plot_cross_modal_phase_diagram(
            phase_results, 
            save_path=self.output_dir / "cross_modal_phase_diagram.png"
        )
        
        # Compile results
        analysis_results = {
            'phase_results': phase_results,
            'summary': {
                'total_pairs': len(phase_results),
                'vision_models': list(vision_features.keys()),
                'text_models': list(text_features.keys()),
                'phase_distribution': self._get_phase_distribution(phase_results),
                'average_metrics': self._get_average_metrics(phase_results)
            },
            'metadata': {
                'analysis_timestamp': str(Path.cwd()),
                'device': str(self.device),
                'ntk_thresholds': self.ntk_thresholds
            }
        }
        
        # Save results
        if save_results:
            results_path = self.output_dir / "cross_modal_phase_analysis.json"
            with open(results_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            print(f"Analysis results saved to {results_path}")
        
        return analysis_results
    
    def _get_phase_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Get distribution of phase regions"""
        distribution = {}
        for r in results:
            phase = r['phase_region']
            distribution[phase] = distribution.get(phase, 0) + 1
        return distribution
    
    def _get_average_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Get average metrics across all model pairs"""
        if not results:
            return {}
        
        ntk_vals = [r['ntk_stability'] for r in results]
        agop_vals = [r['agop_magnitude'] for r in results]
        align_vals = [r['alignment'] for r in results]
        
        return {
            'avg_ntk_stability': np.mean(ntk_vals),
            'avg_agop_magnitude': np.mean(agop_vals),
            'avg_alignment': np.mean(align_vals),
            'std_ntk_stability': np.std(ntk_vals),
            'std_agop_magnitude': np.std(agop_vals),
            'std_alignment': np.std(align_vals)
        }

# Global color mapping for phase regions
phase_colors = {'lazy': 'red', 'optimal': 'green', 'chaotic': 'blue'}

def main():
    """Example usage of cross-modal phase analysis"""
    # Initialize analyzer
    analyzer = CrossModalPhaseAnalyzer()
    
    # Example: Create dummy features for demonstration
    # In practice, these would come from your multi-model analysis
    vision_features = {
        'resnet18': torch.randn(100, 512),
        'vit_base': torch.randn(100, 768),
        'convnext_tiny': torch.randn(100, 768)
    }
    
    text_features = {
        'bert_base': torch.randn(100, 768),
        'roberta_base': torch.randn(100, 768),
        'gpt2_medium': torch.randn(100, 1024)
    }
    
    # Run analysis
    results = analyzer.analyze_cross_modal_representations(
        vision_features, text_features
    )
    
    print("Cross-modal phase analysis complete!")
    print(f"Results saved to: {analyzer.output_dir}")

if __name__ == "__main__":
    main()
