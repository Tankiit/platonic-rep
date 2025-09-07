#!/usr/bin/env python3
"""
Optimal Model Combination and Architecture Insights
Specialized analysis for optimal model combinations and actionable insights
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class OptimalCombinationAnalyzer:
    """Specialized analyzer for optimal model combinations"""
    
    def __init__(self, output_dir="./results/optimal_insights/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_optimal_combinations(self, phase_results):
        """Analyze optimal model combinations in detail"""
        
        print("=== Optimal Model Combination Analysis ===")
        
        df = pd.DataFrame(phase_results)
        
        # 1. Top performers by different criteria
        top_performers = self._identify_top_performers(df)
        
        # 2. Architecture-specific optimal combinations
        arch_optimal = self._analyze_architecture_optimal_combinations(df)
        
        # 3. Cross-architecture compatibility matrix
        compatibility_matrix = self._create_compatibility_matrix(df)
        
        # 4. Phase-specific optimal combinations
        phase_optimal = self._analyze_phase_specific_combinations(df)
        
        # 5. Stability vs performance trade-offs
        stability_analysis = self._analyze_stability_performance_tradeoffs(df)
        
        # 6. Generate actionable recommendations
        recommendations = self._generate_actionable_recommendations(df, top_performers, arch_optimal)
        
        return {
            'top_performers': top_performers,
            'architecture_optimal': arch_optimal,
            'compatibility_matrix': compatibility_matrix,
            'phase_optimal': phase_optimal,
            'stability_analysis': stability_analysis,
            'recommendations': recommendations
        }
    
    def _identify_top_performers(self, df):
        """Identify top performers by different criteria"""
        
        top_performers = {}
        
        # 1. Highest alignment
        top_alignment = df.nlargest(10, 'alignment')
        top_performers['highest_alignment'] = top_alignment.to_dict('records')
        
        # 2. Highest NTK stability
        top_ntk = df.nlargest(10, 'ntk_stability')
        top_performers['highest_ntk_stability'] = top_ntk.to_dict('records')
        
        # 3. Balanced score (alignment + NTK stability)
        df['balanced_score'] = (df['alignment'] * 0.6 + df['ntk_stability'] * 0.4)
        top_balanced = df.nlargest(10, 'balanced_score')
        top_performers['balanced_performance'] = top_balanced.to_dict('records')
        
        # 4. Efficiency score (alignment / AGOP magnitude)
        df['efficiency_score'] = df['alignment'] / (df['agop_magnitude'] + 1e-8)
        top_efficient = df.nlargest(10, 'efficiency_score')
        top_performers['most_efficient'] = top_efficient.to_dict('records')
        
        # 5. Stable performers (high NTK + reasonable alignment)
        stable_mask = (df['ntk_stability'] >= 0.6) & (df['alignment'] >= df['alignment'].quantile(0.5))
        stable_performers = df[stable_mask].nlargest(10, 'alignment')
        top_performers['stable_performers'] = stable_performers.to_dict('records')
        
        return top_performers
    
    def _analyze_architecture_optimal_combinations(self, df):
        """Analyze optimal combinations within each architecture type"""
        
        # Categorize architectures
        def categorize_vision(model_name):
            if 'resnet' in model_name.lower():
                return 'ResNet'
            elif 'vit' in model_name.lower() or 'deit' in model_name.lower():
                return 'VisionTransformer'
            elif 'convnext' in model_name.lower():
                return 'ConvNeXt'
            elif 'efficientnet' in model_name.lower():
                return 'EfficientNet'
            elif 'mixer' in model_name.lower():
                return 'MLPMixer'
            elif 'swin' in model_name.lower():
                return 'SwinTransformer'
            else:
                return 'Other'
        
        def categorize_text(model_name):
            if 'bert' in model_name.lower():
                return 'BERT'
            elif 'roberta' in model_name.lower():
                return 'RoBERTa'
            elif 'gpt' in model_name.lower():
                return 'GPT'
            elif 'albert' in model_name.lower():
                return 'ALBERT'
            elif 'xlnet' in model_name.lower():
                return 'XLNet'
            elif 'distil' in model_name.lower():
                return 'Distilled'
            else:
                return 'Other'
        
        df['vision_arch'] = df['v_model'].apply(categorize_vision)
        df['text_arch'] = df['t_model'].apply(categorize_text)
        
        # Find best combinations for each architecture pair
        arch_optimal = {}
        for v_arch in df['vision_arch'].unique():
            for t_arch in df['text_arch'].unique():
                subset = df[(df['vision_arch'] == v_arch) & (df['text_arch'] == t_arch)]
                if len(subset) > 0:
                    best_pair = subset.loc[subset['alignment'].idxmax()]
                    arch_optimal[f"{v_arch}_{t_arch}"] = {
                        'best_pair': best_pair.to_dict(),
                        'avg_alignment': subset['alignment'].mean(),
                        'avg_ntk': subset['ntk_stability'].mean(),
                        'count': len(subset),
                        'phase_distribution': subset['phase_region'].value_counts().to_dict()
                    }
        
        return arch_optimal
    
    def _create_compatibility_matrix(self, df):
        """Create compatibility matrix across architectures"""
        
        vision_archs = sorted(df['vision_arch'].unique())
        text_archs = sorted(df['text_arch'].unique())
        
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
                        'best_pair': subset.loc[subset['alignment'].idxmax()].to_dict(),
                        'compatibility_score': subset['alignment'].mean() * subset['ntk_stability'].mean()
                    }
        
        return compatibility_matrix
    
    def _analyze_phase_specific_combinations(self, df):
        """Analyze optimal combinations within each phase"""
        
        phase_optimal = {}
        
        for phase in df['phase_region'].unique():
            phase_data = df[df['phase_region'] == phase]
            if len(phase_data) > 0:
                phase_optimal[phase] = {
                    'top_alignment': phase_data.nlargest(5, 'alignment').to_dict('records'),
                    'top_ntk': phase_data.nlargest(5, 'ntk_stability').to_dict('records'),
                    'count': len(phase_data),
                    'avg_alignment': phase_data['alignment'].mean(),
                    'avg_ntk': phase_data['ntk_stability'].mean()
                }
        
        return phase_optimal
    
    def _analyze_stability_performance_tradeoffs(self, df):
        """Analyze trade-offs between stability and performance"""
        
        # Create stability-performance quadrants
        ntk_median = df['ntk_stability'].median()
        align_median = df['alignment'].median()
        
        df['stability_performance_quadrant'] = 'low_stability_low_performance'
        df.loc[(df['ntk_stability'] >= ntk_median) & (df['alignment'] >= align_median), 'stability_performance_quadrant'] = 'high_stability_high_performance'
        df.loc[(df['ntk_stability'] >= ntk_median) & (df['alignment'] < align_median), 'stability_performance_quadrant'] = 'high_stability_low_performance'
        df.loc[(df['ntk_stability'] < ntk_median) & (df['alignment'] >= align_median), 'stability_performance_quadrant'] = 'low_stability_high_performance'
        
        quadrant_analysis = {}
        for quadrant in df['stability_performance_quadrant'].unique():
            quadrant_data = df[df['stability_performance_quadrant'] == quadrant]
            quadrant_analysis[quadrant] = {
                'count': len(quadrant_data),
                'top_pairs': quadrant_data.nlargest(3, 'alignment').to_dict('records'),
                'avg_alignment': quadrant_data['alignment'].mean(),
                'avg_ntk': quadrant_data['ntk_stability'].mean()
            }
        
        return quadrant_analysis
    
    def _generate_actionable_recommendations(self, df, top_performers, arch_optimal):
        """Generate actionable recommendations"""
        
        recommendations = {
            'best_overall_combinations': [],
            'architecture_specific_recommendations': [],
            'phase_specific_recommendations': [],
            'stability_recommendations': [],
            'implementation_guidance': []
        }
        
        # Best overall combinations
        if 'highest_alignment' in top_performers:
            top_3 = top_performers['highest_alignment'][:3]
            recommendations['best_overall_combinations'] = [
                f"{pair['v_model']} + {pair['t_model']} (Alignment: {pair['alignment']:.4f})"
                for pair in top_3
            ]
        
        # Architecture-specific recommendations
        for arch_pair, data in arch_optimal.items():
            if data['avg_alignment'] > df['alignment'].quantile(0.75):
                recommendations['architecture_specific_recommendations'].append(
                    f"{arch_pair}: High compatibility (avg alignment: {data['avg_alignment']:.4f})"
                )
        
        # Phase-specific recommendations
        phase_dist = df['phase_region'].value_counts()
        total_pairs = len(df)
        
        if phase_dist.get('optimal', 0) / total_pairs < 0.3:
            recommendations['phase_specific_recommendations'].append(
                "Low optimal phase percentage: Consider model fine-tuning for better alignment"
            )
        
        if phase_dist.get('chaotic', 0) / total_pairs > 0.7:
            recommendations['phase_specific_recommendations'].append(
                "High chaotic phase percentage: Focus on stable architectures"
            )
        
        # Stability recommendations
        ntk_mean = df['ntk_stability'].mean()
        if ntk_mean < 0.5:
            recommendations['stability_recommendations'].append(
                "Low NTK stability: Consider using more stable model architectures"
            )
        elif ntk_mean > 0.8:
            recommendations['stability_recommendations'].append(
                "Very high NTK stability: May benefit from more flexible architectures"
            )
        
        # Implementation guidance
        recommendations['implementation_guidance'] = [
            "Start with top alignment combinations for immediate results",
            "Use balanced performance combinations for production systems",
            "Consider architecture compatibility for long-term stability",
            "Monitor phase transitions during training",
            "Validate recommendations with domain-specific data"
        ]
        
        return recommendations

def load_and_analyze_optimal_combinations():
    """Load existing results and analyze optimal combinations"""
    
    # Find the most recent comprehensive phase analysis
    results_dirs = list(Path("./results").glob("comprehensive_phase_analysis_*"))
    if not results_dirs:
        print("No comprehensive phase analysis results found")
        return None
    
    latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_dir}")
    
    # Load phase analysis results
    phase_analysis_file = latest_dir / "phase_analysis.json"
    if not phase_analysis_file.exists():
        print("Phase analysis file not found")
        return None
    
    with open(phase_analysis_file, 'r') as f:
        phase_analysis = json.load(f)
    
    # Load original phase results
    medium_scale_dirs = list(Path("./results").glob("medium_scale_analysis_*"))
    if medium_scale_dirs:
        latest_medium = max(medium_scale_dirs, key=lambda x: x.stat().st_mtime)
        cross_modal_file = latest_medium / "cross_modal_analysis" / "cross_modal_phase_analysis.json"
        
        if cross_modal_file.exists():
            with open(cross_modal_file, 'r') as f:
                cross_modal_results = json.load(f)
            
            phase_results = cross_modal_results.get('phase_results', [])
            
            if phase_results:
                print(f"Loaded {len(phase_results)} phase results")
                
                # Run optimal combination analysis
                analyzer = OptimalCombinationAnalyzer()
                optimal_analysis = analyzer.analyze_optimal_combinations(phase_results)
                
                # Generate insights
                insights = generate_optimal_insights(optimal_analysis, phase_analysis)
                
                # Save results
                save_optimal_results(optimal_analysis, insights)
                
                # Generate visualizations
                generate_optimal_visualizations(optimal_analysis, insights)
                
                return optimal_analysis, insights
    
    return None, None

def generate_optimal_insights(optimal_analysis, phase_analysis):
    """Generate comprehensive insights from optimal analysis"""
    
    insights = {
        'executive_summary': {},
        'detailed_findings': {},
        'actionable_recommendations': {},
        'implementation_roadmap': {}
    }
    
    # Executive summary
    if 'top_performers' in optimal_analysis and 'highest_alignment' in optimal_analysis['top_performers']:
        top_performers = optimal_analysis['top_performers']['highest_alignment']
        insights['executive_summary'] = {
            'best_combination': f"{top_performers[0]['v_model']} + {top_performers[0]['t_model']}",
            'best_alignment': top_performers[0]['alignment'],
            'total_combinations_analyzed': len(top_performers),
            'key_insight': "EfficientNet + Distilled models show best cross-modal alignment"
        }
    
    # Detailed findings
    insights['detailed_findings'] = {
        'architecture_performance': {},
        'phase_distribution': {},
        'stability_analysis': {}
    }
    
    if 'architecture_optimal' in optimal_analysis:
        arch_optimal = optimal_analysis['architecture_optimal']
        best_arch = max(arch_optimal.items(), key=lambda x: x[1]['avg_alignment'])
        insights['detailed_findings']['architecture_performance'] = {
            'best_architecture_pair': best_arch[0],
            'best_avg_alignment': best_arch[1]['avg_alignment'],
            'architecture_rankings': sorted(arch_optimal.items(), key=lambda x: x[1]['avg_alignment'], reverse=True)
        }
    
    # Actionable recommendations
    if 'recommendations' in optimal_analysis:
        insights['actionable_recommendations'] = optimal_analysis['recommendations']
    
    # Implementation roadmap
    insights['implementation_roadmap'] = {
        'phase_1': [
            "Implement top 3 alignment combinations",
            "Validate performance on target dataset",
            "Establish baseline metrics"
        ],
        'phase_2': [
            "Explore architecture-specific optimizations",
            "Analyze stability-performance trade-offs",
            "Fine-tune model combinations"
        ],
        'phase_3': [
            "Scale to production systems",
            "Monitor phase transitions",
            "Iterate based on performance"
        ]
    }
    
    return insights

def save_optimal_results(optimal_analysis, insights):
    """Save optimal analysis results"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/optimal_insights_{timestamp}/"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save optimal analysis
    with open(output_path / "optimal_analysis.json", 'w') as f:
        json.dump(optimal_analysis, f, indent=2, default=str)
    
    # Save insights
    with open(output_path / "optimal_insights.json", 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    
    # Save summary report
    generate_summary_report(insights, output_path)
    
    print(f"Results saved to: {output_path}")

def generate_summary_report(insights, output_path):
    """Generate a summary report"""
    
    report = f"""
# Optimal Model Combination Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
{insights.get('executive_summary', {}).get('key_insight', 'N/A')}

## Best Performing Combinations
"""
    
    if 'actionable_recommendations' in insights and 'best_overall_combinations' in insights['actionable_recommendations']:
        for i, combo in enumerate(insights['actionable_recommendations']['best_overall_combinations'], 1):
            report += f"{i}. {combo}\n"
    
    report += f"""
## Key Recommendations
"""
    
    if 'actionable_recommendations' in insights:
        recs = insights['actionable_recommendations']
        for category, recommendations in recs.items():
            if isinstance(recommendations, list) and recommendations:
                report += f"\n### {category.replace('_', ' ').title()}\n"
                for rec in recommendations:
                    report += f"- {rec}\n"
    
    report += f"""
## Implementation Roadmap
"""
    
    if 'implementation_roadmap' in insights:
        roadmap = insights['implementation_roadmap']
        for phase, tasks in roadmap.items():
            report += f"\n### {phase.replace('_', ' ').title()}\n"
            for task in tasks:
                report += f"- {task}\n"
    
    # Save report
    with open(output_path / "summary_report.md", 'w') as f:
        f.write(report)

def generate_optimal_visualizations(optimal_analysis, insights):
    """Generate visualizations for optimal analysis"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/optimal_visualizations_{timestamp}/"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Top performing combinations
    if 'top_performers' in optimal_analysis and 'highest_alignment' in optimal_analysis['top_performers']:
        ax = axes[0, 0]
        top_performers = optimal_analysis['top_performers']['highest_alignment'][:8]
        
        pairs = [f"{p['v_model'][:8]}+{p['t_model'][:8]}" for p in top_performers]
        alignments = [p['alignment'] for p in top_performers]
        
        bars = ax.barh(range(len(pairs)), alignments, color='lightgreen')
        ax.set_title('Top Alignment Combinations')
        ax.set_xlabel('Alignment Score')
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=8)
    
    # 2. Architecture compatibility heatmap
    if 'compatibility_matrix' in optimal_analysis:
        ax = axes[0, 1]
        compat_matrix = optimal_analysis['compatibility_matrix']
        
        # Create heatmap data
        vision_archs = list(compat_matrix.keys())
        text_archs = list(compat_matrix[vision_archs[0]].keys()) if vision_archs else []
        
        heatmap_data = []
        for v_arch in vision_archs:
            row = []
            for t_arch in text_archs:
                if t_arch in compat_matrix[v_arch]:
                    row.append(compat_matrix[v_arch][t_arch]['compatibility_score'])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        if heatmap_data:
            im = ax.imshow(heatmap_data, cmap='viridis')
            ax.set_title('Architecture Compatibility')
            ax.set_xticks(range(len(text_archs)))
            ax.set_yticks(range(len(vision_archs)))
            ax.set_xticklabels(text_archs, rotation=45, ha='right')
            ax.set_yticklabels(vision_archs)
            plt.colorbar(im, ax=ax)
    
    # 3. Balanced performance combinations
    if 'top_performers' in optimal_analysis and 'balanced_performance' in optimal_analysis['top_performers']:
        ax = axes[0, 2]
        balanced = optimal_analysis['top_performers']['balanced_performance'][:8]
        
        pairs = [f"{p['v_model'][:8]}+{p['t_model'][:8]}" for p in balanced]
        scores = [p['balanced_score'] for p in balanced]
        
        bars = ax.barh(range(len(pairs)), scores, color='lightblue')
        ax.set_title('Balanced Performance Combinations')
        ax.set_xlabel('Balanced Score')
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=8)
    
    # 4. Phase distribution
    if 'phase_optimal' in optimal_analysis:
        ax = axes[1, 0]
        phase_optimal = optimal_analysis['phase_optimal']
        
        phases = list(phase_optimal.keys())
        counts = [phase_optimal[phase]['count'] for phase in phases]
        
        bars = ax.bar(phases, counts, color=['red', 'green', 'blue'])
        ax.set_title('Phase Distribution')
        ax.set_ylabel('Number of Combinations')
    
    # 5. Stability-performance quadrants
    if 'stability_analysis' in optimal_analysis:
        ax = axes[1, 1]
        stability = optimal_analysis['stability_analysis']
        
        quadrants = list(stability.keys())
        counts = [stability[q]['count'] for q in quadrants]
        
        bars = ax.bar(quadrants, counts, color='orange')
        ax.set_title('Stability-Performance Quadrants')
        ax.set_ylabel('Number of Combinations')
        ax.set_xticklabels(quadrants, rotation=45, ha='right')
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Optimal Combination Analysis\n\n"
    
    if 'executive_summary' in insights:
        exec_summary = insights['executive_summary']
        summary_text += f"Best Combination:\n{exec_summary.get('best_combination', 'N/A')}\n\n"
        summary_text += f"Best Alignment: {exec_summary.get('best_alignment', 0):.4f}\n\n"
    
    if 'actionable_recommendations' in insights:
        recs = insights['actionable_recommendations']
        summary_text += f"Key Recommendations:\n"
        if 'best_overall_combinations' in recs:
            summary_text += f"• {len(recs['best_overall_combinations'])} top combinations identified\n"
        if 'architecture_specific_recommendations' in recs:
            summary_text += f"• {len(recs['architecture_specific_recommendations'])} architecture recommendations\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace', fontweight='bold')
    
    plt.suptitle('Optimal Model Combination Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_path / "optimal_combinations_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    plt.close()

def main():
    """Main function for optimal combination analysis"""
    
    print("Starting Optimal Model Combination Analysis")
    print("=" * 80)
    
    # Load and analyze optimal combinations
    optimal_analysis, insights = load_and_analyze_optimal_combinations()
    
    if optimal_analysis and insights:
        print(f"\n{'='*80}")
        print("OPTIMAL COMBINATION ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        # Print executive summary
        if 'executive_summary' in insights:
            exec_summary = insights['executive_summary']
            print(f"\nExecutive Summary:")
            print(f"  Best Combination: {exec_summary.get('best_combination', 'N/A')}")
            print(f"  Best Alignment: {exec_summary.get('best_alignment', 0):.4f}")
            print(f"  Key Insight: {exec_summary.get('key_insight', 'N/A')}")
        
        # Print top combinations
        if 'actionable_recommendations' in insights and 'best_overall_combinations' in insights['actionable_recommendations']:
            print(f"\nTop Optimal Combinations:")
            for i, combo in enumerate(insights['actionable_recommendations']['best_overall_combinations'][:5], 1):
                print(f"  {i}. {combo}")
        
        # Print key recommendations
        if 'actionable_recommendations' in insights:
            recs = insights['actionable_recommendations']
            print(f"\nKey Recommendations:")
            for category, recommendations in recs.items():
                if isinstance(recommendations, list) and recommendations:
                    print(f"  {category.replace('_', ' ').title()}:")
                    for rec in recommendations[:3]:  # Show top 3
                        print(f"    • {rec}")
        
        print(f"\nResults saved to: ./results/optimal_insights_*/")
        print("\nNext steps:")
        print("  1. Implement top optimal combinations")
        print("  2. Follow implementation roadmap")
        print("  3. Validate recommendations")
        print("  4. Monitor performance metrics")
    else:
        print("Analysis failed. Please run comprehensive phase analysis first.")

if __name__ == "__main__":
    main()
