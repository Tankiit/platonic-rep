#!/usr/bin/env python3
"""
Phase Transition Architecture Insights and Optimal Model Combination Analysis
Advanced analysis focusing on phase transitions and optimal model combinations
"""

import torch
import numpy as np
from pathlib import Path
from multi_model_analysis import MultiModelAnalyzer
from cross_modal_phase_analysis import CrossModalPhaseAnalyzer
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

class PhaseTransitionAnalyzer:
    """Advanced analyzer for phase transitions and optimal combinations"""
    
    def __init__(self, output_dir="./results/phase_transition_analysis/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase transition thresholds
        self.phase_thresholds = {
            'lazy': 0.9,
            'optimal': 0.7,
            'chaotic': 0.5
        }
        
        # Architecture categories
        self.architecture_categories = {
            'cnn': ['resnet', 'convnext', 'efficientnet'],
            'transformer': ['vit', 'deit', 'swin'],
            'mlp': ['mixer'],
            'hybrid': ['convnext']  # Can be both CNN and modern
        }
        
    def analyze_phase_transitions(self, phase_results):
        """Analyze phase transitions across model pairs"""
        
        print("=== Phase Transition Analysis ===")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(phase_results)
        
        # Phase distribution analysis
        phase_dist = df['phase_region'].value_counts()
        print(f"Phase Distribution: {phase_dist.to_dict()}")
        
        # NTK stability analysis
        ntk_stats = {
            'mean': df['ntk_stability'].mean(),
            'std': df['ntk_stability'].std(),
            'min': df['ntk_stability'].min(),
            'max': df['ntk_stability'].max(),
            'median': df['ntk_stability'].median()
        }
        
        # AGOP magnitude analysis
        agop_stats = {
            'mean': df['agop_magnitude'].mean(),
            'std': df['agop_magnitude'].std(),
            'min': df['agop_magnitude'].min(),
            'max': df['agop_magnitude'].max(),
            'median': df['agop_magnitude'].median()
        }
        
        # Alignment analysis
        align_stats = {
            'mean': df['alignment'].mean(),
            'std': df['alignment'].std(),
            'min': df['alignment'].min(),
            'max': df['alignment'].max(),
            'median': df['alignment'].median()
        }
        
        # Phase transition patterns
        phase_transitions = self._analyze_phase_transition_patterns(df)
        
        # Architecture-specific phase analysis
        arch_phase_analysis = self._analyze_architecture_phase_patterns(df)
        
        # Optimal region identification
        optimal_regions = self._identify_optimal_regions(df)
        
        return {
            'phase_distribution': phase_dist.to_dict(),
            'ntk_statistics': ntk_stats,
            'agop_statistics': agop_stats,
            'alignment_statistics': align_stats,
            'phase_transitions': phase_transitions,
            'architecture_phase_analysis': arch_phase_analysis,
            'optimal_regions': optimal_regions
        }
    
    def _analyze_phase_transition_patterns(self, df):
        """Analyze patterns in phase transitions"""
        
        # Categorize by NTK stability ranges
        df['ntk_category'] = pd.cut(df['ntk_stability'], 
                                   bins=[0, 0.5, 0.7, 0.9, 1.0], 
                                   labels=['chaotic', 'transition', 'optimal', 'lazy'])
        
        # Analyze transitions
        transition_analysis = {
            'chaotic_to_optimal': len(df[(df['ntk_stability'] >= 0.5) & (df['ntk_stability'] < 0.7)]),
            'optimal_to_lazy': len(df[(df['ntk_stability'] >= 0.7) & (df['ntk_stability'] < 0.9)]),
            'stable_lazy': len(df[df['ntk_stability'] >= 0.9]),
            'unstable_chaotic': len(df[df['ntk_stability'] < 0.5])
        }
        
        # Correlation analysis
        correlations = {
            'ntk_agop_corr': df['ntk_stability'].corr(df['agop_magnitude']),
            'ntk_align_corr': df['ntk_stability'].corr(df['alignment']),
            'agop_align_corr': df['agop_magnitude'].corr(df['alignment'])
        }
        
        return {
            'transition_counts': transition_analysis,
            'correlations': correlations,
            'ntk_distribution': df['ntk_category'].value_counts().to_dict()
        }
    
    def _analyze_architecture_phase_patterns(self, df):
        """Analyze phase patterns by architecture type"""
        
        # Categorize vision models
        def categorize_vision_model(model_name):
            for arch_type, keywords in self.architecture_categories.items():
                if any(keyword in model_name.lower() for keyword in keywords):
                    return arch_type
            return 'unknown'
        
        df['vision_arch'] = df['v_model'].apply(categorize_vision_model)
        
        # Analyze by architecture
        arch_analysis = {}
        for arch in df['vision_arch'].unique():
            arch_data = df[df['vision_arch'] == arch]
            arch_analysis[arch] = {
                'count': len(arch_data),
                'avg_ntk': arch_data['ntk_stability'].mean(),
                'avg_agop': arch_data['agop_magnitude'].mean(),
                'avg_alignment': arch_data['alignment'].mean(),
                'phase_distribution': arch_data['phase_region'].value_counts().to_dict(),
                'ntk_std': arch_data['ntk_stability'].std(),
                'alignment_std': arch_data['alignment'].std()
            }
        
        return arch_analysis
    
    def _identify_optimal_regions(self, df):
        """Identify optimal regions in the phase space"""
        
        # Define optimal region (high alignment, balanced NTK and AGOP)
        optimal_mask = (
            (df['alignment'] >= df['alignment'].quantile(0.75)) &  # Top 25% alignment
            (df['ntk_stability'] >= 0.6) & (df['ntk_stability'] <= 0.9) &  # Balanced NTK
            (df['agop_magnitude'] >= df['agop_magnitude'].quantile(0.25))  # Reasonable AGOP
        )
        
        optimal_pairs = df[optimal_mask]
        
        # Cluster analysis for optimal regions
        if len(optimal_pairs) > 3:
            # Use PCA for dimensionality reduction
            features = optimal_pairs[['ntk_stability', 'agop_magnitude', 'alignment']].values
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=min(3, len(optimal_pairs)), random_state=42)
            clusters = kmeans.fit_predict(features_2d)
            
            optimal_pairs['cluster'] = clusters
        else:
            optimal_pairs['cluster'] = 0
        
        return {
            'optimal_pairs': optimal_pairs.to_dict('records'),
            'optimal_count': len(optimal_pairs),
            'optimal_percentage': len(optimal_pairs) / len(df) * 100,
            'optimal_characteristics': {
                'avg_ntk': optimal_pairs['ntk_stability'].mean(),
                'avg_agop': optimal_pairs['agop_magnitude'].mean(),
                'avg_alignment': optimal_pairs['alignment'].mean()
            }
        }

class OptimalModelCombinationAnalyzer:
    """Analyzer for optimal model combinations"""
    
    def __init__(self, output_dir="./results/optimal_combinations/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def find_optimal_combinations(self, phase_results, criteria='alignment'):
        """Find optimal model combinations based on different criteria"""
        
        print("=== Optimal Model Combination Analysis ===")
        
        df = pd.DataFrame(phase_results)
        
        # Different optimization criteria
        optimization_results = {}
        
        # 1. Highest alignment
        top_alignment = df.nlargest(10, 'alignment')
        optimization_results['highest_alignment'] = top_alignment.to_dict('records')
        
        # 2. Balanced performance (high alignment + good NTK stability)
        df['balanced_score'] = (df['alignment'] * 0.7 + df['ntk_stability'] * 0.3)
        top_balanced = df.nlargest(10, 'balanced_score')
        optimization_results['balanced_performance'] = top_balanced.to_dict('records')
        
        # 3. Phase-optimal combinations
        optimal_phase = df[df['phase_region'] == 'optimal'].nlargest(10, 'alignment')
        optimization_results['phase_optimal'] = optimal_phase.to_dict('records')
        
        # 4. Architecture-specific optimal combinations
        arch_optimal = self._find_architecture_optimal_combinations(df)
        optimization_results['architecture_optimal'] = arch_optimal
        
        # 5. Cross-architecture compatibility
        cross_arch_compatibility = self._analyze_cross_architecture_compatibility(df)
        optimization_results['cross_architecture'] = cross_arch_compatibility
        
        return optimization_results
    
    def _find_architecture_optimal_combinations(self, df):
        """Find optimal combinations within each architecture type"""
        
        # Categorize architectures
        def get_vision_arch(model_name):
            if 'resnet' in model_name.lower():
                return 'cnn'
            elif 'vit' in model_name.lower() or 'deit' in model_name.lower():
                return 'transformer'
            elif 'mixer' in model_name.lower():
                return 'mlp'
            elif 'convnext' in model_name.lower():
                return 'modern_cnn'
            else:
                return 'other'
        
        def get_text_arch(model_name):
            if 'bert' in model_name.lower() or 'roberta' in model_name.lower():
                return 'transformer'
            elif 'gpt' in model_name.lower():
                return 'language_model'
            elif 'albert' in model_name.lower() or 'xlnet' in model_name.lower():
                return 'transformer_variant'
            else:
                return 'other'
        
        df['vision_arch'] = df['v_model'].apply(get_vision_arch)
        df['text_arch'] = df['t_model'].apply(get_text_arch)
        
        # Find best combinations for each architecture pair
        arch_optimal = {}
        for v_arch in df['vision_arch'].unique():
            for t_arch in df['text_arch'].unique():
                subset = df[(df['vision_arch'] == v_arch) & (df['text_arch'] == t_arch)]
                if len(subset) > 0:
                    best_pair = subset.loc[subset['alignment'].idxmax()]
                    arch_optimal[f"{v_arch}_{t_arch}"] = best_pair.to_dict()
        
        return arch_optimal
    
    def _analyze_cross_architecture_compatibility(self, df):
        """Analyze compatibility across different architectures"""
        
        # Create compatibility matrix
        vision_archs = df['vision_arch'].unique()
        text_archs = df['text_arch'].unique()
        
        compatibility_matrix = {}
        for v_arch in vision_archs:
            compatibility_matrix[v_arch] = {}
            for t_arch in text_archs:
                subset = df[(df['vision_arch'] == v_arch) & (df['text_arch'] == t_arch)]
                if len(subset) > 0:
                    compatibility_matrix[v_arch][t_arch] = {
                        'avg_alignment': subset['alignment'].mean(),
                        'avg_ntk': subset['ntk_stability'].mean(),
                        'count': len(subset),
                        'best_pair': subset.loc[subset['alignment'].idxmax()].to_dict()
                    }
        
        return compatibility_matrix

def run_comprehensive_phase_analysis():
    """Run comprehensive phase transition and optimal combination analysis"""
    
    # Initialize analyzers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/comprehensive_phase_analysis_{timestamp}/"
    
    phase_analyzer = PhaseTransitionAnalyzer(output_dir + "phase_transitions/")
    optimal_analyzer = OptimalModelCombinationAnalyzer(output_dir + "optimal_combinations/")
    
    # Load existing results or run new analysis
    print("Loading existing analysis results...")
    
    # Try to load from recent medium-scale analysis
    results_dirs = list(Path("./results").glob("medium_scale_analysis_*"))
    if results_dirs:
        latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
        print(f"Using results from: {latest_dir}")
        
        # Load cross-modal results
        cross_modal_file = latest_dir / "cross_modal_analysis" / "cross_modal_phase_analysis.json"
        if cross_modal_file.exists():
            with open(cross_modal_file, 'r') as f:
                cross_modal_results = json.load(f)
            
            phase_results = cross_modal_results.get('phase_results', [])
            
            if phase_results:
                print(f"Loaded {len(phase_results)} phase results")
                
                # Run phase transition analysis
                print("\n1. Running Phase Transition Analysis...")
                phase_analysis = phase_analyzer.analyze_phase_transitions(phase_results)
                
                # Run optimal combination analysis
                print("\n2. Running Optimal Combination Analysis...")
                optimal_analysis = optimal_analyzer.find_optimal_combinations(phase_results)
                
                # Generate comprehensive insights
                print("\n3. Generating Comprehensive Insights...")
                insights = generate_comprehensive_insights(phase_analysis, optimal_analysis)
                
                # Save results
                save_comprehensive_results(phase_analysis, optimal_analysis, insights, output_dir)
                
                # Generate visualizations
                print("\n4. Generating Visualizations...")
                generate_phase_visualizations(phase_analysis, optimal_analysis, output_dir)
                
                return phase_analysis, optimal_analysis, insights
            else:
                print("No phase results found in loaded data")
        else:
            print("Cross-modal analysis file not found")
    else:
        print("No medium-scale analysis results found")
    
    return None, None, None

def generate_comprehensive_insights(phase_analysis, optimal_analysis):
    """Generate comprehensive insights from analysis results"""
    
    insights = {
        'key_findings': [],
        'recommendations': [],
        'phase_transition_patterns': {},
        'optimal_combinations': {},
        'architecture_insights': {}
    }
    
    # Key findings
    if 'phase_distribution' in phase_analysis:
        phase_dist = phase_analysis['phase_distribution']
        insights['key_findings'].append(f"Phase distribution: {phase_dist}")
        
        # Identify dominant phase
        dominant_phase = max(phase_dist, key=phase_dist.get)
        insights['key_findings'].append(f"Dominant phase: {dominant_phase}")
    
    # NTK stability insights
    if 'ntk_statistics' in phase_analysis:
        ntk_stats = phase_analysis['ntk_statistics']
        insights['key_findings'].append(f"NTK stability range: {ntk_stats['min']:.3f} - {ntk_stats['max']:.3f}")
        insights['key_findings'].append(f"NTK stability mean: {ntk_stats['mean']:.3f}")
    
    # Alignment insights
    if 'alignment_statistics' in phase_analysis:
        align_stats = phase_analysis['alignment_statistics']
        insights['key_findings'].append(f"Alignment range: {align_stats['min']:.3f} - {align_stats['max']:.3f}")
        insights['key_findings'].append(f"Alignment mean: {align_stats['mean']:.3f}")
    
    # Phase transition patterns
    if 'phase_transitions' in phase_analysis:
        transitions = phase_analysis['phase_transitions']
        insights['phase_transition_patterns'] = {
            'transition_counts': transitions.get('transition_counts', {}),
            'correlations': transitions.get('correlations', {})
        }
    
    # Optimal combinations
    if 'highest_alignment' in optimal_analysis:
        top_alignments = optimal_analysis['highest_alignment'][:5]
        insights['optimal_combinations']['top_alignment'] = [
            f"{pair['v_model']} + {pair['t_model']}: {pair['alignment']:.4f}"
            for pair in top_alignments
        ]
    
    if 'balanced_performance' in optimal_analysis:
        top_balanced = optimal_analysis['balanced_performance'][:5]
        insights['optimal_combinations']['top_balanced'] = [
            f"{pair['v_model']} + {pair['t_model']}: {pair['balanced_score']:.4f}"
            for pair in top_balanced
        ]
    
    # Architecture insights
    if 'architecture_phase_analysis' in phase_analysis:
        arch_analysis = phase_analysis['architecture_phase_analysis']
        insights['architecture_insights'] = {
            arch: {
                'avg_alignment': data['avg_alignment'],
                'phase_distribution': data['phase_distribution']
            }
            for arch, data in arch_analysis.items()
        }
    
    # Recommendations
    insights['recommendations'] = generate_recommendations(phase_analysis, optimal_analysis)
    
    return insights

def generate_recommendations(phase_analysis, optimal_analysis):
    """Generate actionable recommendations"""
    
    recommendations = []
    
    # Phase-based recommendations
    if 'phase_distribution' in phase_analysis:
        phase_dist = phase_analysis['phase_distribution']
        chaotic_pct = phase_dist.get('chaotic', 0) / sum(phase_dist.values()) * 100
        optimal_pct = phase_dist.get('optimal', 0) / sum(phase_dist.values()) * 100
        
        if chaotic_pct > 50:
            recommendations.append("High chaotic phase percentage suggests need for better model alignment")
        if optimal_pct < 20:
            recommendations.append("Low optimal phase percentage indicates room for improvement in model combinations")
    
    # NTK-based recommendations
    if 'ntk_statistics' in phase_analysis:
        ntk_mean = phase_analysis['ntk_statistics']['mean']
        if ntk_mean < 0.6:
            recommendations.append("Low average NTK stability suggests unstable representations")
        elif ntk_mean > 0.9:
            recommendations.append("Very high NTK stability may indicate over-rigid representations")
    
    # Optimal combination recommendations
    if 'highest_alignment' in optimal_analysis:
        top_alignment = optimal_analysis['highest_alignment'][0]
        recommendations.append(f"Best performing pair: {top_alignment['v_model']} + {top_alignment['t_model']}")
    
    # Architecture recommendations
    if 'architecture_phase_analysis' in phase_analysis:
        arch_analysis = phase_analysis['architecture_phase_analysis']
        best_arch = max(arch_analysis.items(), key=lambda x: x[1]['avg_alignment'])
        recommendations.append(f"Best performing architecture: {best_arch[0]} (avg alignment: {best_arch[1]['avg_alignment']:.4f})")
    
    return recommendations

def save_comprehensive_results(phase_analysis, optimal_analysis, insights, output_dir):
    """Save comprehensive analysis results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save phase analysis
    with open(output_path / "phase_analysis.json", 'w') as f:
        json.dump(phase_analysis, f, indent=2, default=str)
    
    # Save optimal analysis
    with open(output_path / "optimal_analysis.json", 'w') as f:
        json.dump(optimal_analysis, f, indent=2, default=str)
    
    # Save insights
    with open(output_path / "comprehensive_insights.json", 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")

def generate_phase_visualizations(phase_analysis, optimal_analysis, output_dir):
    """Generate comprehensive visualizations"""
    
    output_path = Path(output_dir)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Phase distribution
    if 'phase_distribution' in phase_analysis:
        ax = axes[0, 0]
        phase_dist = phase_analysis['phase_distribution']
        phases = list(phase_dist.keys())
        counts = list(phase_dist.values())
        
        bars = ax.bar(phases, counts, color=['red', 'green', 'blue'])
        ax.set_title('Phase Distribution')
        ax.set_ylabel('Number of Model Pairs')
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 2. NTK vs Alignment scatter
    if 'phase_transitions' in phase_analysis and 'correlations' in phase_analysis['phase_transitions']:
        ax = axes[0, 1]
        # This would need the original data, but we can show correlation
        corr = phase_analysis['phase_transitions']['correlations'].get('ntk_align_corr', 0)
        ax.text(0.5, 0.5, f'NTK-Alignment\nCorrelation: {corr:.3f}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('NTK vs Alignment Correlation')
        ax.axis('off')
    
    # 3. Architecture performance comparison
    if 'architecture_phase_analysis' in phase_analysis:
        ax = axes[0, 2]
        arch_analysis = phase_analysis['architecture_phase_analysis']
        
        archs = list(arch_analysis.keys())
        alignments = [arch_analysis[arch]['avg_alignment'] for arch in archs]
        
        bars = ax.bar(archs, alignments, color='lightblue')
        ax.set_title('Average Alignment by Architecture')
        ax.set_ylabel('Average Alignment Score')
        ax.set_xticklabels(archs, rotation=45, ha='right')
    
    # 4. Top optimal combinations
    if 'highest_alignment' in optimal_analysis:
        ax = axes[1, 0]
        top_alignments = optimal_analysis['highest_alignment'][:8]
        
        pairs = [f"{pair['v_model'][:8]}+{pair['t_model'][:8]}" for pair in top_alignments]
        alignments = [pair['alignment'] for pair in top_alignments]
        
        bars = ax.barh(range(len(pairs)), alignments, color='lightgreen')
        ax.set_title('Top Alignment Combinations')
        ax.set_xlabel('Alignment Score')
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=8)
    
    # 5. Phase transition patterns
    if 'phase_transitions' in phase_analysis:
        ax = axes[1, 1]
        transitions = phase_analysis['phase_transitions'].get('transition_counts', {})
        
        if transitions:
            transition_types = list(transitions.keys())
            counts = list(transitions.values())
            
            bars = ax.bar(transition_types, counts, color='orange')
            ax.set_title('Phase Transition Patterns')
            ax.set_ylabel('Number of Pairs')
            ax.set_xticklabels(transition_types, rotation=45, ha='right')
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Phase Transition Analysis Summary\n\n"
    
    if 'ntk_statistics' in phase_analysis:
        ntk_stats = phase_analysis['ntk_statistics']
        summary_text += f"NTK Stability:\n"
        summary_text += f"  Mean: {ntk_stats['mean']:.3f}\n"
        summary_text += f"  Std: {ntk_stats['std']:.3f}\n\n"
    
    if 'alignment_statistics' in phase_analysis:
        align_stats = phase_analysis['alignment_statistics']
        summary_text += f"Alignment:\n"
        summary_text += f"  Mean: {align_stats['mean']:.3f}\n"
        summary_text += f"  Std: {align_stats['std']:.3f}\n\n"
    
    if 'optimal_regions' in phase_analysis:
        optimal_regions = phase_analysis['optimal_regions']
        summary_text += f"Optimal Pairs: {optimal_regions['optimal_count']}\n"
        summary_text += f"Optimal %: {optimal_regions['optimal_percentage']:.1f}%\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace', fontweight='bold')
    
    plt.suptitle('Phase Transition Architecture Insights', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_path / "phase_transition_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    plt.close()

def main():
    """Main function for comprehensive phase analysis"""
    
    print("Starting Comprehensive Phase Transition Analysis")
    print("=" * 80)
    
    # Run comprehensive analysis
    phase_analysis, optimal_analysis, insights = run_comprehensive_phase_analysis()
    
    if phase_analysis and optimal_analysis:
        print(f"\n{'='*80}")
        print("COMPREHENSIVE PHASE ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        # Print key insights
        print("\nKey Findings:")
        for finding in insights.get('key_findings', []):
            print(f"  • {finding}")
        
        print("\nTop Optimal Combinations:")
        if 'optimal_combinations' in insights and 'top_alignment' in insights['optimal_combinations']:
            for i, combo in enumerate(insights['optimal_combinations']['top_alignment'][:5]):
                print(f"  {i+1}. {combo}")
        
        print("\nRecommendations:")
        for rec in insights.get('recommendations', []):
            print(f"  • {rec}")
        
        print(f"\nResults saved to: ./results/comprehensive_phase_analysis_*/")
        print("\nNext steps:")
        print("  1. Review phase transition patterns")
        print("  2. Analyze optimal model combinations")
        print("  3. Implement recommended architectures")
        print("  4. Validate findings with additional experiments")
    else:
        print("Analysis failed. Please run medium-scale analysis first.")

if __name__ == "__main__":
    main()
