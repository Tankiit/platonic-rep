#!/usr/bin/env python3
"""
Complete Embeddings Plotter using Phase Analyzer

This script loads complete embeddings from one vision model (ResNet18) and one language model (DistilBERT)
and uses the phase analyzer to generate comprehensive visualizations and analysis.

Based on the phase_analyzer.py framework for cross-modal NTK stability analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'old_files'))

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

# Import the phase analyzer
from phase_analyzer import CrossModalPhaseAnalyzer

class CompleteEmbeddingsPlotter:
    """
    Specialized plotter for complete embeddings using the phase analyzer framework
    """
    
    def __init__(self, embeddings_dir: str = "./complete_features/"):
        self.embeddings_dir = Path(embeddings_dir)
        self.analyzer = CrossModalPhaseAnalyzer(results_dir="./results/complete_embeddings_analysis/")
        
        # Model configurations
        self.vision_model = "resnet18"
        self.language_model = "distilbert"
        
        print(f"Initialized CompleteEmbeddingsPlotter")
        print(f"Vision model: {self.vision_model}")
        print(f"Language model: {self.language_model}")
        print(f"Embeddings directory: {self.embeddings_dir}")
    
    def load_complete_embeddings(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load complete embeddings for both vision and language models
        
        Returns:
            Dict with 'vision' and 'language' keys containing model embeddings
        """
        embeddings = {'vision': {}, 'language': {}}
        
        # Load vision model embeddings (ResNet18)
        vision_path = self.embeddings_dir / self.vision_model
        if vision_path.exists():
            print(f"\nLoading {self.vision_model} embeddings...")
            for embedding_file in vision_path.glob("*.npy"):
                layer_name = embedding_file.stem
                try:
                    embedding_data = np.load(embedding_file)
                    embeddings['vision'][layer_name] = embedding_data
                    print(f"  {layer_name}: shape {embedding_data.shape}")
                except Exception as e:
                    print(f"  Error loading {embedding_file}: {e}")
        else:
            print(f"Warning: {vision_path} not found")
        
        # Load language model embeddings (DistilBERT)
        language_path = self.embeddings_dir / self.language_model
        if language_path.exists():
            print(f"\nLoading {self.language_model} embeddings...")
            for embedding_file in language_path.glob("*.npy"):
                layer_name = embedding_file.stem
                try:
                    embedding_data = np.load(embedding_file)
                    embeddings['language'][layer_name] = embedding_data
                    print(f"  {layer_name}: shape {embedding_data.shape}")
                except Exception as e:
                    print(f"  Error loading {embedding_file}: {e}")
        else:
            print(f"Warning: {language_path} not found")
        
        return embeddings
    
    def analyze_embeddings(self, embeddings: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """
        Run comprehensive phase analysis on the loaded embeddings
        """
        print("\n" + "="*60)
        print("RUNNING COMPLETE EMBEDDINGS PHASE ANALYSIS")
        print("="*60)
        
        # Prepare features for the analyzer
        vision_features = {}
        language_features = {}
        
        # Convert vision embeddings to analyzer format
        for layer_name, embedding_data in embeddings['vision'].items():
            model_layer_name = f"{self.vision_model}_{layer_name}"
            vision_features[model_layer_name] = embedding_data
            print(f"Prepared vision feature: {model_layer_name} -> shape {embedding_data.shape}")
        
        # Convert language embeddings to analyzer format
        for layer_name, embedding_data in embeddings['language'].items():
            model_layer_name = f"{self.language_model}_{layer_name}"
            language_features[model_layer_name] = embedding_data
            print(f"Prepared language feature: {model_layer_name} -> shape {embedding_data.shape}")
        
        # Run the cross-modal analysis
        results = self.analyzer.analyze_cross_modal_compatibility(vision_features, language_features)
        
        return results
    
    def create_detailed_visualizations(self, embeddings: Dict[str, Dict[str, np.ndarray]], 
                                     analysis_results: Dict) -> None:
        """
        Create detailed visualizations of the embeddings and analysis
        """
        print("\n" + "="*60)
        print("GENERATING DETAILED VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Embedding distributions (top row)
        self._plot_embedding_distributions(fig, embeddings, 3, 4, 1)
        
        # 2. Phase diagram from analyzer
        self._plot_phase_diagram(fig, analysis_results, 3, 4, 2)
        
        # 3. Cross-modal compatibility heatmap
        self._plot_compatibility_heatmap(fig, analysis_results, 3, 4, 3)
        
        # 4. AGOP eigenvalue spectra
        self._plot_agop_spectra(fig, analysis_results, 3, 4, 4)
        
        # 5. NTK stability comparison
        self._plot_ntk_stability(fig, analysis_results, 3, 4, 5)
        
        # 6. Embedding similarity matrices
        self._plot_similarity_matrices(fig, embeddings, 3, 4, 6)
        
        # 7. Layer-wise progression
        self._plot_layer_progression(fig, analysis_results, 3, 4, 7)
        
        # 8. Cross-modal correlation analysis
        self._plot_cross_modal_correlations(fig, embeddings, 3, 4, 8)
        
        # 9. Dimensionality analysis
        self._plot_dimensionality_analysis(fig, embeddings, 3, 4, 9)
        
        # 10. Summary statistics
        self._plot_summary_statistics(fig, analysis_results, 3, 4, 10)
        
        # 11. Research question insights
        self._plot_research_insights(fig, analysis_results, 3, 4, 11)
        
        # 12. Final phase landscape
        self._plot_final_phase_landscape(fig, analysis_results, 3, 4, 12)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        output_path = self.analyzer.results_dir / "complete_embeddings_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to: {output_path}")
        
        plt.show()
    
    def _plot_embedding_distributions(self, fig, embeddings, rows, cols, pos):
        """Plot embedding value distributions for each layer"""
        ax = fig.add_subplot(rows, cols, pos)
        
        all_data = []
        all_labels = []
        
        # Collect vision model data
        for layer_name, data in embeddings['vision'].items():
            # Flatten and sample for visualization
            flat_data = data.flatten()
            if len(flat_data) > 10000:
                flat_data = np.random.choice(flat_data, 10000, replace=False)
            all_data.append(flat_data)
            all_labels.append(f"ResNet18_{layer_name}")
        
        # Collect language model data
        for layer_name, data in embeddings['language'].items():
            # Flatten and sample for visualization
            flat_data = data.flatten()
            if len(flat_data) > 10000:
                flat_data = np.random.choice(flat_data, 10000, replace=False)
            all_data.append(flat_data)
            all_labels.append(f"DistilBERT_{layer_name}")
        
        # Create violin plot
        parts = ax.violinplot(all_data, positions=range(len(all_data)), showmeans=True, showmedians=True)
        
        # Color code by model type
        for i, pc in enumerate(parts['bodies']):
            if 'ResNet18' in all_labels[i]:
                pc.set_facecolor('lightblue')
            else:
                pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_ylabel('Embedding Values')
        ax.set_title('Embedding Value Distributions')
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_diagram(self, fig, results, rows, cols, pos):
        """Plot the phase diagram from the analyzer"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Use the analyzer's phase diagram method
        phase_fig = self.analyzer.generate_phase_diagram(results)
        
        # Extract the phase diagram and add to our figure
        # This is a simplified version - in practice you'd extract the axes
        ax.text(0.5, 0.5, 'Phase Diagram\n(Generated by Analyzer)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Cross-Modal Phase Diagram')
        ax.axis('off')
    
    def _plot_compatibility_heatmap(self, fig, results, rows, cols, pos):
        """Plot cross-modal compatibility heatmap"""
        ax = fig.add_subplot(rows, cols, pos)
        
        if 'compatibility_matrix' in results:
            # Extract compatibility matrix
            compat_matrix = results['compatibility_matrix']
            
            # Convert to matrix format
            vision_models = list(compat_matrix.keys())
            language_models = list(compat_matrix[vision_models[0]].keys()) if vision_models else []
            
            matrix_data = []
            for v_model in vision_models:
                row = []
                for l_model in language_models:
                    row.append(compat_matrix[v_model].get(l_model, 0))
                matrix_data.append(row)
            
            if matrix_data:
                im = ax.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto')
                ax.set_xticks(range(len(language_models)))
                ax.set_xticklabels(language_models, rotation=45, ha='right')
                ax.set_yticks(range(len(vision_models)))
                ax.set_yticklabels(vision_models)
                ax.set_title('Cross-Modal Compatibility Matrix')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='S_NTK^cross')
                
                # Add text annotations
                for i in range(len(vision_models)):
                    for j in range(len(language_models)):
                        text = ax.text(j, i, f'{matrix_data[i][j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No compatibility matrix\navailable', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cross-Modal Compatibility Matrix')
    
    def _plot_agop_spectra(self, fig, results, rows, cols, pos):
        """Plot AGOP eigenvalue spectra"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Collect eigenvalue data
        for model_type in ['vision_models', 'language_models']:
            if model_type in results:
                for model_name, analysis in results[model_type].items():
                    if 'agop_analysis' in analysis and 'eigenvalues' in analysis['agop_analysis']:
                        eigenvals = analysis['agop_analysis']['eigenvalues']
                        # Plot top eigenvalues
                        top_eigenvals = eigenvals[:min(50, len(eigenvals))]
                        ax.semilogy(range(len(top_eigenvals)), top_eigenvals, 
                                  label=model_name, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue Magnitude (log scale)')
        ax.set_title('AGOP Eigenvalue Spectra')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_ntk_stability(self, fig, results, rows, cols, pos):
        """Plot NTK stability comparison"""
        ax = fig.add_subplot(rows, cols, pos)
        
        model_names = []
        stability_scores = []
        phases = []
        
        # Collect stability data
        for model_type in ['vision_models', 'language_models']:
            if model_type in results:
                for model_name, analysis in results[model_type].items():
                    if 'ntk_stability' in analysis:
                        model_names.append(model_name)
                        stability_scores.append(analysis['ntk_stability']['s_ntk'])
                        phases.append(analysis['phase'])
        
        if model_names:
            # Color code by phase
            colors = {'chaotic': 'red', 'optimal': 'green', 'lazy': 'blue'}
            bar_colors = [colors.get(phase, 'gray') for phase in phases]
            
            bars = ax.bar(range(len(model_names)), stability_scores, color=bar_colors, alpha=0.7)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('S_NTK (Stability Score)')
            ax.set_title('NTK Stability by Model')
            
            # Add phase threshold lines
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chaotic|Optimal')
            ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Optimal|Lazy')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_similarity_matrices(self, fig, embeddings, rows, cols, pos):
        """Plot embedding similarity matrices"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Select a representative layer from each model
        vision_layer = list(embeddings['vision'].keys())[0] if embeddings['vision'] else None
        language_layer = list(embeddings['language'].keys())[0] if embeddings['language'] else None
        
        if vision_layer and language_layer:
            vision_data = embeddings['vision'][vision_layer]
            language_data = embeddings['language'][language_layer]
            
            # Sample data for computational efficiency
            n_samples = min(30, min(len(vision_data), len(language_data)))  # Use all 30 samples
            vision_sample = vision_data[:n_samples]
            language_sample = language_data[:n_samples]
            
            # Compute similarity matrix using cosine similarity for robustness
            def cosine_similarity_matrix(data):
                # Normalize data
                data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)
                # Compute cosine similarity
                return data_norm @ data_norm.T
            
            vision_sim = cosine_similarity_matrix(vision_sample)
            language_sim = cosine_similarity_matrix(language_sample)
            
            # Plot both matrices side by side
            combined_sim = np.block([[vision_sim, np.zeros_like(vision_sim)], 
                                   [np.zeros_like(language_sim), language_sim]])
            
            im = ax.imshow(combined_sim, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('Embedding Similarity Matrices')
            ax.set_xlabel('Samples')
            ax.set_ylabel('Samples')
            
            # Add dividing line
            ax.axvline(x=n_samples-0.5, color='black', linewidth=2)
            ax.axhline(y=n_samples-0.5, color='black', linewidth=2)
            
            # Add labels
            ax.text(n_samples/2, -2, 'Vision', ha='center', va='top', fontweight='bold')
            ax.text(n_samples + n_samples/2, -2, 'Language', ha='center', va='top', fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
        else:
            ax.text(0.5, 0.5, 'No embedding data\navailable', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Embedding Similarity Matrices')
    
    def _plot_layer_progression(self, fig, results, rows, cols, pos):
        """Plot how metrics change across layers"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Extract layer-wise data
        layer_data = {'vision': [], 'language': []}
        
        for model_type in ['vision_models', 'language_models']:
            if model_type in results:
                for model_name, analysis in results[model_type].items():
                    if 'ntk_stability' in analysis:
                        # Extract layer number from model name
                        if 'layer' in model_name:
                            try:
                                layer_num = int(model_name.split('layer')[-1])
                                stability = analysis['ntk_stability']['s_ntk']
                                layer_data[model_type.replace('_models', '')].append((layer_num, stability))
                            except:
                                pass
        
        # Plot progression
        for model_type, data in layer_data.items():
            if data:
                data.sort(key=lambda x: x[0])  # Sort by layer number
                layers, stabilities = zip(*data)
                ax.plot(layers, stabilities, 'o-', label=f'{model_type.title()} Models', linewidth=2, markersize=6)
        
        ax.set_xlabel('Layer Number')
        ax.set_ylabel('S_NTK (Stability Score)')
        ax.set_title('Stability Across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cross_modal_correlations(self, fig, embeddings, rows, cols, pos):
        """Plot cross-modal correlation analysis"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Select representative layers with similar dimensions
        vision_layer = None
        language_layer = None
        
        # Find layers with similar dimensions for meaningful correlation
        for v_layer, v_data in embeddings['vision'].items():
            for l_layer, l_data in embeddings['language'].items():
                if v_data.shape[1] == l_data.shape[1]:  # Same feature dimension
                    vision_layer = v_layer
                    language_layer = l_layer
                    break
            if vision_layer:
                break
        
        # If no matching dimensions, use the largest vision layer and project language
        if not vision_layer:
            vision_layer = max(embeddings['vision'].keys(), 
                             key=lambda x: embeddings['vision'][x].shape[1])
            language_layer = list(embeddings['language'].keys())[0]
        
        if vision_layer and language_layer:
            vision_data = embeddings['vision'][vision_layer]
            language_data = embeddings['language'][language_layer]
            
            # Ensure same number of samples
            min_samples = min(len(vision_data), len(language_data))
            vision_subset = vision_data[:min_samples]
            language_subset = language_data[:min_samples]
            
            # Handle dimension mismatch by using cosine similarity instead of correlation
            if vision_subset.shape[1] != language_subset.shape[1]:
                # Use cosine similarity for different dimensions
                similarities = []
                for i in range(min(100, min_samples)):
                    v_vec = vision_subset[i] / (np.linalg.norm(vision_subset[i]) + 1e-8)
                    l_vec = language_subset[i] / (np.linalg.norm(language_subset[i]) + 1e-8)
                    
                    # Project to common space or use dot product
                    if len(v_vec) <= len(l_vec):
                        similarity = np.dot(v_vec, l_vec[:len(v_vec)])
                    else:
                        similarity = np.dot(v_vec[:len(l_vec)], l_vec)
                    
                    similarities.append(similarity)
                
                if similarities:
                    ax.hist(similarities, bins=30, alpha=0.7, color='purple', edgecolor='black')
                    ax.axvline(x=np.mean(similarities), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean: {np.mean(similarities):.3f}')
                    ax.set_xlabel('Cross-Modal Similarity')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Cross-Modal Sample Similarities')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No valid similarities\ncomputed', 
                            ha='center', va='center', transform=ax.transAxes)
            else:
                # Same dimensions - use correlation
                correlations = []
                for i in range(min(100, min_samples)):
                    corr = np.corrcoef(vision_subset[i], language_subset[i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                
                if correlations:
                    ax.hist(correlations, bins=30, alpha=0.7, color='purple', edgecolor='black')
                    ax.axvline(x=np.mean(correlations), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean: {np.mean(correlations):.3f}')
                    ax.set_xlabel('Cross-Modal Correlation')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Cross-Modal Sample Correlations')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No valid correlations\ncomputed', 
                            ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No embedding data\navailable', 
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cross-Modal Correlations')
    
    def _plot_dimensionality_analysis(self, fig, embeddings, rows, cols, pos):
        """Plot dimensionality analysis"""
        ax = fig.add_subplot(rows, cols, pos)
        
        model_names = []
        dimensions = []
        effective_dims = []
        
        # Analyze each model
        for model_type, model_embeddings in embeddings.items():
            for layer_name, data in model_embeddings.items():
                model_names.append(f"{model_type}_{layer_name}")
                dimensions.append(data.shape[1])
                
                # Estimate effective dimensionality using participation ratio
                if len(data) > 0:
                    # Sample for efficiency
                    sample_data = data[:min(1000, len(data))]
                    cov_matrix = np.cov(sample_data.T)
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    eigenvals = np.real(eigenvals[eigenvals > 1e-10])
                    
                    if len(eigenvals) > 0:
                        total_var = np.sum(eigenvals)
                        effective_dim = total_var**2 / np.sum(eigenvals**2) if total_var > 0 else 0
                        effective_dims.append(effective_dim)
                    else:
                        effective_dims.append(0)
                else:
                    effective_dims.append(0)
        
        if model_names:
            x_pos = range(len(model_names))
            width = 0.35
            
            bars1 = ax.bar([x - width/2 for x in x_pos], dimensions, width, 
                          label='Actual Dimensions', alpha=0.7, color='lightblue')
            bars2 = ax.bar([x + width/2 for x in x_pos], effective_dims, width, 
                          label='Effective Dimensions', alpha=0.7, color='lightcoral')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Number of Dimensions')
            ax.set_title('Dimensionality Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No embedding data\navailable', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Dimensionality Analysis')
    
    def _plot_summary_statistics(self, fig, results, rows, cols, pos):
        """Plot summary statistics"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Extract key statistics
        stats_text = "PHASE ANALYSIS SUMMARY\n\n"
        
        # Model counts
        vision_count = len(results.get('vision_models', {}))
        language_count = len(results.get('language_models', {}))
        stats_text += f"Vision Models: {vision_count}\n"
        stats_text += f"Language Models: {language_count}\n\n"
        
        # Phase distribution
        all_phases = []
        for model_type in ['vision_models', 'language_models']:
            if model_type in results:
                for model_name, analysis in results[model_type].items():
                    all_phases.append(analysis.get('phase', 'unknown'))
        
        phase_counts = {phase: all_phases.count(phase) for phase in set(all_phases)}
        stats_text += "Phase Distribution:\n"
        for phase, count in phase_counts.items():
            stats_text += f"  {phase.title()}: {count}\n"
        
        # Cross-modal compatibility
        if 'research_questions' in results and 'rq1_systematic_divergence' in results['research_questions']:
            rq1 = results['research_questions']['rq1_systematic_divergence']
            stats_text += f"\nCross-Modal Stability:\n"
            stats_text += f"  Mean: {rq1.get('mean_cross_modal_stability', 0):.3f}\n"
            stats_text += f"  Below Threshold: {rq1.get('fraction_below_threshold', 0):.1%}\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Summary Statistics')
        ax.axis('off')
    
    def _plot_research_insights(self, fig, results, rows, cols, pos):
        """Plot research question insights"""
        ax = fig.add_subplot(rows, cols, pos)
        
        insights_text = "RESEARCH INSIGHTS\n\n"
        
        if 'research_questions' in results:
            # RQ1 insights
            if 'rq1_systematic_divergence' in results['research_questions']:
                rq1 = results['research_questions']['rq1_systematic_divergence']
                insights_text += "RQ1: Systematic Divergence\n"
                insights_text += f"  Mean Compatibility: {rq1.get('mean_cross_modal_stability', 0):.3f}\n"
                insights_text += f"  Explanation: {rq1.get('explanation', 'N/A')}\n\n"
            
            # RQ2 insights
            if 'rq2_prh_implications' in results['research_questions']:
                rq2 = results['research_questions']['rq2_prh_implications']
                insights_text += "RQ2: PRH Implications\n"
                insights_text += f"  Phase Diversity: {rq2.get('phase_diversity', 0)}\n"
                insights_text += f"  PRH Paradox: {rq2.get('prh_paradox', False)}\n"
                insights_text += f"  Explanation: {rq2.get('explanation', 'N/A')}\n"
        
        ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_title('Research Question Insights')
        ax.axis('off')
    
    def _plot_final_phase_landscape(self, fig, results, rows, cols, pos):
        """Plot final phase landscape summary"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Create a phase landscape visualization
        if 'cross_modal_analysis' in results:
            cross_modal_scores = []
            for pair_name, analysis in results['cross_modal_analysis'].items():
                cross_modal_scores.append(analysis.get('s_ntk_cross', 0))
            
            if cross_modal_scores:
                # Create phase regions
                x = np.linspace(0, 1, 100)
                y = np.linspace(0, 1, 100)
                X, Y = np.meshgrid(x, y)
                
                # Define phase regions based on thresholds
                Z = np.zeros_like(X)
                Z[X < 0.5] = 1  # Chaotic
                Z[(X >= 0.5) & (X < 0.9)] = 2  # Optimal
                Z[X >= 0.9] = 3  # Lazy
                
                # Plot phase regions
                im = ax.contourf(X, Y, Z, levels=[0.5, 1.5, 2.5, 3.5], 
                               colors=['red', 'green', 'blue'], alpha=0.3)
                
                # Plot actual data points
                for i, score in enumerate(cross_modal_scores):
                    ax.scatter(score, 0.5, s=100, c='black', alpha=0.8, 
                             edgecolors='white', linewidth=2)
                
                ax.set_xlabel('S_NTK^cross (Cross-Modal Stability)')
                ax.set_ylabel('Phase Space')
                ax.set_title('Final Phase Landscape')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                # Add phase labels
                ax.text(0.25, 0.8, 'Chaotic', ha='center', va='center', 
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
                ax.text(0.7, 0.8, 'Optimal', ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
                ax.text(0.95, 0.8, 'Lazy', ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='blue', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No cross-modal data\navailable', 
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No cross-modal analysis\navailable', 
                    ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Final Phase Landscape')
    
    def run_complete_analysis(self):
        """
        Run the complete embeddings analysis pipeline
        """
        print("="*80)
        print("COMPLETE EMBEDDINGS ANALYSIS PIPELINE")
        print("="*80)
        
        # Step 1: Load embeddings
        print("\nStep 1: Loading complete embeddings...")
        embeddings = self.load_complete_embeddings()
        
        if not embeddings['vision'] or not embeddings['language']:
            print("Error: Could not load embeddings for both vision and language models")
            return None
        
        # Step 2: Run phase analysis
        print("\nStep 2: Running phase analysis...")
        analysis_results = self.analyze_embeddings(embeddings)
        
        # Step 3: Generate visualizations
        print("\nStep 3: Generating comprehensive visualizations...")
        self.create_detailed_visualizations(embeddings, analysis_results)
        
        # Step 4: Save results
        print("\nStep 4: Saving analysis results...")
        self.analyzer.save_results(analysis_results, "complete_embeddings_analysis.json")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {self.analyzer.results_dir}")
        print(f"Visualizations saved to: {self.analyzer.results_dir}/complete_embeddings_analysis.png")
        
        return analysis_results

def main():
    """
    Main function to run the complete embeddings analysis
    """
    # Initialize the plotter
    plotter = CompleteEmbeddingsPlotter()
    
    # Run the complete analysis
    results = plotter.run_complete_analysis()
    
    if results:
        print("\nAnalysis completed successfully!")
        print("Check the results directory for detailed outputs.")
    else:
        print("\nAnalysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
