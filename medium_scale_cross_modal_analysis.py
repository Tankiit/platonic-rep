#!/usr/bin/env python3
"""
Medium-Scale Cross-Modal Phase Analysis
Comprehensive analysis across medium-sized vision and text architectures
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

def medium_scale_analysis():
    """Run comprehensive analysis on medium-sized architectures"""
    
    # Initialize analyzer with timestamped output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/medium_scale_analysis_{timestamp}/"
    analyzer = MultiModelAnalyzer(output_dir=output_dir)
    
    # Medium-scale vision models (balanced size and performance)
    vision_models = [
        # ResNet family (medium variants)
        'resnet34', 'resnet50',
        
        # Vision Transformers (medium variants)
        'vit_base_patch16_224', 'vit_small_patch16_224', 'deit_base_patch16_224',
        
        # ConvNeXt family (medium variants)
        'convnext_small', 'convnext_base',
        
        # EfficientNet (medium variants)
        'efficientnet_b1', 'efficientnet_b2',
        
        # MLP-Mixer (medium variants)
        'mixer_b16_224', 'mixer_b32_224',
        
        # Modern alternatives
        'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224'
    ]
    
    # Medium-scale text models
    text_models = [
        # BERT family (medium variants)
        'bert_base', 'distilbert_base',
        
        # RoBERTa family (medium variants)
        'roberta_base', 'roberta_large',
        
        # GPT family (medium variants)
        'gpt2_medium', 'gpt2_large',
        
        # Other transformers (medium variants)
        'albert_base_v2', 'albert_large_v2',
        'xlnet_base_cased', 'xlnet_large_cased',
        
        # Modern alternatives
        'distilroberta_base', 'microsoft/DialoGPT-medium'
    ]
    
    # Multiple datasets for comprehensive analysis
    datasets = ['cifar10', 'cifar100', 'svhn']
    
    print("=== Medium-Scale Cross-Modal Phase Analysis ===")
    print(f"Vision models: {len(vision_models)} models")
    print(f"Text models: {len(text_models)} models")
    print(f"Datasets: {datasets}")
    print(f"Total combinations: {len(vision_models) * len(text_models) * len(datasets)}")
    print(f"Output directory: {output_dir}")
    
    # Store comprehensive results
    all_results = {}
    
    # Run analysis for each dataset
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"ANALYZING DATASET: {dataset.upper()}")
        print(f"{'='*80}")
        
        dataset_results = {}
        
        # Run cross-modal analysis
        try:
            print(f"Running cross-modal phase analysis for {dataset}...")
            cross_modal_results = analyzer.run_cross_modal_phase_analysis(
                vision_models=vision_models,
                text_models=text_models,
                dataset=dataset,
                pretrained=True
            )
            
            dataset_results['cross_modal'] = cross_modal_results
            print(f"✓ Cross-modal analysis complete for {dataset}")
            
        except Exception as e:
            print(f"✗ Cross-modal analysis failed for {dataset}: {e}")
            dataset_results['cross_modal'] = None
        
        # Run individual model analysis for comparison
        print(f"\nRunning individual model analysis for {dataset}...")
        individual_results = {}
        
        # Analyze subset of vision models for efficiency
        vision_subset = vision_models[:8]  # First 8 models
        
        for model in vision_subset:
            try:
                print(f"  Analyzing {model}...")
                result = analyzer.run_analysis(model, dataset, pretrained=True, save_features=True)
                if result:
                    individual_results[model] = result
                    print(f"  ✓ {model} analysis complete")
                else:
                    print(f"  ✗ {model} analysis failed")
            except Exception as e:
                print(f"  ✗ {model} analysis error: {e}")
        
        dataset_results['individual'] = individual_results
        all_results[dataset] = dataset_results
    
    # Generate comprehensive analysis
    generate_medium_scale_summary(all_results, output_dir)
    
    return all_results

def generate_medium_scale_summary(all_results, output_dir):
    """Generate comprehensive summary for medium-scale analysis"""
    
    print(f"\n{'='*80}")
    print("GENERATING MEDIUM-SCALE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_datasets': len(all_results),
        'dataset_summaries': {},
        'cross_modal_summary': {},
        'architecture_insights': {},
        'phase_transitions': {},
        'model_performance': {}
    }
    
    # Analyze each dataset
    for dataset, results in all_results.items():
        print(f"\nAnalyzing {dataset} results...")
        
        dataset_summary = {
            'cross_modal_available': results['cross_modal'] is not None,
            'individual_models': len(results['individual']),
            'metrics': {},
            'architecture_breakdown': {}
        }
        
        # Cross-modal analysis summary
        if results['cross_modal']:
            cross_modal = results['cross_modal']
            if 'summary' in cross_modal:
                avg_metrics = cross_modal['summary'].get('average_metrics', {})
                phase_dist = cross_modal['summary'].get('phase_distribution', {})
                
                dataset_summary['metrics'] = {
                    'avg_ntk_stability': avg_metrics.get('avg_ntk_stability', 0),
                    'avg_agop_magnitude': avg_metrics.get('avg_agop_magnitude', 0),
                    'avg_alignment': avg_metrics.get('avg_alignment', 0),
                    'phase_distribution': phase_dist
                }
                
                # Architecture breakdown
                if 'phase_results' in cross_modal:
                    dataset_summary['architecture_breakdown'] = analyze_architecture_breakdown(
                        cross_modal['phase_results']
                    )
        
        # Individual model analysis summary
        if results['individual']:
            individual_metrics = {}
            for model, result in results['individual'].items():
                if 'macroscopic' in result and 'information_flow' in result['macroscopic']:
                    summary_info = result['macroscopic']['information_flow'].get('summary', {})
                    individual_metrics[model] = {
                        'total_compression': summary_info.get('total_compression', 0),
                        'total_task_info_gain': summary_info.get('total_task_info_gain', 0),
                        'peak_task_info': summary_info.get('peak_task_info', 0)
                    }
            
            dataset_summary['individual_metrics'] = individual_metrics
        
        summary['dataset_summaries'][dataset] = dataset_summary
    
    # Generate cross-dataset insights
    summary['cross_modal_summary'] = generate_cross_dataset_insights(all_results)
    summary['architecture_insights'] = generate_architecture_insights(all_results)
    
    # Save comprehensive summary
    summary_path = Path(output_dir) / "medium_scale_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print_medium_scale_summary(summary)
    
    # Generate visualizations
    generate_medium_scale_visualizations(summary, output_dir)
    
    return summary

def analyze_architecture_breakdown(phase_results):
    """Analyze results by architecture type"""
    
    breakdown = {
        'resnet': {'count': 0, 'avg_ntk': 0, 'avg_agop': 0, 'avg_align': 0},
        'vit': {'count': 0, 'avg_ntk': 0, 'avg_agop': 0, 'avg_align': 0},
        'convnext': {'count': 0, 'avg_ntk': 0, 'avg_agop': 0, 'avg_align': 0},
        'efficientnet': {'count': 0, 'avg_ntk': 0, 'avg_agop': 0, 'avg_align': 0},
        'mixer': {'count': 0, 'avg_ntk': 0, 'avg_agop': 0, 'avg_align': 0},
        'swin': {'count': 0, 'avg_ntk': 0, 'avg_agop': 0, 'avg_align': 0}
    }
    
    for result in phase_results:
        v_model = result['v_model']
        
        # Categorize vision model
        if 'resnet' in v_model:
            category = 'resnet'
        elif 'vit' in v_model or 'deit' in v_model:
            category = 'vit'
        elif 'convnext' in v_model:
            category = 'convnext'
        elif 'efficientnet' in v_model:
            category = 'efficientnet'
        elif 'mixer' in v_model:
            category = 'mixer'
        elif 'swin' in v_model:
            category = 'swin'
        else:
            continue
        
        # Update statistics
        breakdown[category]['count'] += 1
        breakdown[category]['avg_ntk'] += result['ntk_stability']
        breakdown[category]['avg_agop'] += result['agop_magnitude']
        breakdown[category]['avg_align'] += result['alignment']
    
    # Calculate averages
    for category in breakdown:
        if breakdown[category]['count'] > 0:
            count = breakdown[category]['count']
            breakdown[category]['avg_ntk'] /= count
            breakdown[category]['avg_agop'] /= count
            breakdown[category]['avg_align'] /= count
    
    return breakdown

def generate_cross_dataset_insights(all_results):
    """Generate insights across datasets"""
    
    insights = {
        'dataset_comparison': {},
        'consistency_analysis': {},
        'phase_stability': {}
    }
    
    # Compare metrics across datasets
    for dataset, results in all_results.items():
        if results['cross_modal'] and 'summary' in results['cross_modal']:
            metrics = results['cross_modal']['summary'].get('average_metrics', {})
            insights['dataset_comparison'][dataset] = {
                'ntk_stability': metrics.get('avg_ntk_stability', 0),
                'agop_magnitude': metrics.get('avg_agop_magnitude', 0),
                'alignment': metrics.get('avg_alignment', 0)
            }
    
    return insights

def generate_architecture_insights(all_results):
    """Generate insights about architecture performance"""
    
    insights = {
        'best_performing_pairs': [],
        'architecture_compatibility': {},
        'phase_distribution': {}
    }
    
    # Collect all phase results
    all_phase_results = []
    for dataset, results in all_results.items():
        if results['cross_modal'] and 'phase_results' in results['cross_modal']:
            all_phase_results.extend(results['cross_modal']['phase_results'])
    
    if all_phase_results:
        # Find best performing pairs
        sorted_results = sorted(all_phase_results, key=lambda x: x['alignment'], reverse=True)
        insights['best_performing_pairs'] = [
            {
                'pair': f"{r['v_model']} + {r['t_model']}",
                'alignment': r['alignment'],
                'ntk_stability': r['ntk_stability'],
                'phase': r['phase_region']
            }
            for r in sorted_results[:10]  # Top 10
        ]
        
        # Analyze architecture compatibility
        insights['architecture_compatibility'] = analyze_compatibility_patterns(all_phase_results)
    
    return insights

def analyze_compatibility_patterns(phase_results):
    """Analyze compatibility patterns between architectures"""
    
    compatibility = {
        'cnn_transformer': {'count': 0, 'avg_alignment': 0},
        'vit_transformer': {'count': 0, 'avg_alignment': 0},
        'modern_cnn_language': {'count': 0, 'avg_alignment': 0},
        'mlp_transformer': {'count': 0, 'avg_alignment': 0}
    }
    
    for result in phase_results:
        v_model = result['v_model']
        t_model = result['t_model']
        
        # Categorize pair
        if 'resnet' in v_model and ('bert' in t_model or 'roberta' in t_model):
            category = 'cnn_transformer'
        elif 'vit' in v_model and ('bert' in t_model or 'roberta' in t_model):
            category = 'vit_transformer'
        elif ('convnext' in v_model or 'efficientnet' in v_model) and ('gpt' in t_model or 'albert' in t_model):
            category = 'modern_cnn_language'
        elif 'mixer' in v_model:
            category = 'mlp_transformer'
        else:
            continue
        
        compatibility[category]['count'] += 1
        compatibility[category]['avg_alignment'] += result['alignment']
    
    # Calculate averages
    for category in compatibility:
        if compatibility[category]['count'] > 0:
            compatibility[category]['avg_alignment'] /= compatibility[category]['count']
    
    return compatibility

def print_medium_scale_summary(summary):
    """Print comprehensive medium-scale summary"""
    
    print(f"\n{'='*80}")
    print("MEDIUM-SCALE CROSS-MODAL PHASE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print(f"Analysis completed: {summary['analysis_timestamp']}")
    print(f"Total datasets analyzed: {summary['total_datasets']}")
    
    # Cross-modal summary
    if 'cross_modal_summary' in summary and 'dataset_comparison' in summary['cross_modal_summary']:
        print(f"\nCross-Dataset Comparison:")
        for dataset, metrics in summary['cross_modal_summary']['dataset_comparison'].items():
            print(f"  {dataset.upper()}:")
            print(f"    NTK Stability: {metrics['ntk_stability']:.4f}")
            print(f"    AGOP Magnitude: {metrics['agop_magnitude']:.4f}")
            print(f"    Alignment: {metrics['alignment']:.4f}")
    
    # Architecture insights
    if 'architecture_insights' in summary:
        insights = summary['architecture_insights']
        
        if 'best_performing_pairs' in insights:
            print(f"\nTop Performing Model Pairs:")
            for i, pair in enumerate(insights['best_performing_pairs'][:5]):
                print(f"  {i+1}. {pair['pair']}: {pair['alignment']:.4f} ({pair['phase']})")
        
        if 'architecture_compatibility' in insights:
            print(f"\nArchitecture Compatibility:")
            for arch_type, stats in insights['architecture_compatibility'].items():
                if stats['count'] > 0:
                    print(f"  {arch_type}: {stats['avg_alignment']:.4f} ({stats['count']} pairs)")
    
    # Dataset details
    print(f"\nDataset Details:")
    for dataset, dataset_summary in summary['dataset_summaries'].items():
        print(f"  {dataset.upper()}:")
        print(f"    Cross-modal available: {dataset_summary['cross_modal_available']}")
        print(f"    Individual models: {dataset_summary['individual_models']}")
        
        if 'metrics' in dataset_summary and dataset_summary['metrics']:
            metrics = dataset_summary['metrics']
            print(f"    NTK Stability: {metrics.get('avg_ntk_stability', 0):.4f}")
            print(f"    AGOP Magnitude: {metrics.get('avg_agop_magnitude', 0):.4f}")
            print(f"    Alignment Score: {metrics.get('avg_alignment', 0):.4f}")

def generate_medium_scale_visualizations(summary, output_dir):
    """Generate comprehensive visualizations for medium-scale analysis"""
    
    print(f"\nGenerating visualizations...")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Cross-dataset comparison
    if 'cross_modal_summary' in summary and 'dataset_comparison' in summary['cross_modal_summary']:
        ax = axes[0, 0]
        dataset_comparison = summary['cross_modal_summary']['dataset_comparison']
        
        datasets = list(dataset_comparison.keys())
        alignments = [dataset_comparison[d]['alignment'] for d in datasets]
        
        bars = ax.bar(datasets, alignments, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Cross-Dataset Alignment Comparison')
        ax.set_ylabel('Average Alignment Score')
        ax.set_ylim(0, max(alignments) * 1.2)
        
        # Add value labels
        for bar, value in zip(bars, alignments):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.4f}', ha='center', va='bottom')
    
    # 2. Architecture compatibility
    if 'architecture_insights' in summary and 'architecture_compatibility' in summary['architecture_insights']:
        ax = axes[0, 1]
        compatibility = summary['architecture_insights']['architecture_compatibility']
        
        arch_types = list(compatibility.keys())
        alignments = [compatibility[a]['avg_alignment'] for a in arch_types]
        counts = [compatibility[a]['count'] for a in arch_types]
        
        bars = ax.bar(arch_types, alignments, color='lightblue')
        ax.set_title('Architecture Compatibility')
        ax.set_ylabel('Average Alignment Score')
        ax.set_xticklabels(arch_types, rotation=45, ha='right')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # 3. Best performing pairs
    if 'architecture_insights' in summary and 'best_performing_pairs' in summary['architecture_insights']:
        ax = axes[0, 2]
        best_pairs = summary['architecture_insights']['best_performing_pairs'][:8]
        
        pairs = [pair['pair'][:20] + '...' if len(pair['pair']) > 20 else pair['pair'] 
                for pair in best_pairs]
        alignments = [pair['alignment'] for pair in best_pairs]
        
        bars = ax.barh(range(len(pairs)), alignments, color='lightgreen')
        ax.set_title('Top Performing Model Pairs')
        ax.set_xlabel('Alignment Score')
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=8)
    
    # 4. Phase distribution across datasets
    ax = axes[1, 0]
    phase_data = {}
    for dataset, dataset_summary in summary['dataset_summaries'].items():
        if 'metrics' in dataset_summary and 'phase_distribution' in dataset_summary['metrics']:
            phase_dist = dataset_summary['metrics']['phase_distribution']
            for phase, count in phase_dist.items():
                if phase not in phase_data:
                    phase_data[phase] = {}
                phase_data[phase][dataset] = count
    
    if phase_data:
        phases = list(phase_data.keys())
        datasets = list(phase_data[phases[0]].keys()) if phases else []
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, phase in enumerate(phases):
            counts = [phase_data[phase].get(d, 0) for d in datasets]
            ax.bar(x + i*width, counts, width, label=phase.capitalize())
        
        ax.set_title('Phase Distribution Across Datasets')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Number of Model Pairs')
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets)
        ax.legend()
    
    # 5. NTK vs AGOP scatter
    ax = axes[1, 1]
    ntk_vals = []
    agop_vals = []
    colors = []
    
    for dataset, dataset_summary in summary['dataset_summaries'].items():
        if 'metrics' in dataset_summary and dataset_summary['metrics']:
            metrics = dataset_summary['metrics']
            ntk_vals.append(metrics.get('avg_ntk_stability', 0))
            agop_vals.append(metrics.get('avg_agop_magnitude', 0))
            colors.append('blue')
    
    if ntk_vals:
        ax.scatter(ntk_vals, agop_vals, c=colors, s=100, alpha=0.7)
        ax.set_xlabel('NTK Stability')
        ax.set_ylabel('AGOP Magnitude')
        ax.set_title('NTK vs AGOP Relationship')
        ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    stats_text = "Medium-Scale Analysis Summary\n\n"
    stats_text += f"Total Datasets: {summary['total_datasets']}\n"
    stats_text += f"Analysis Time: {summary['analysis_timestamp']}\n\n"
    
    if 'cross_modal_summary' in summary and 'dataset_comparison' in summary['cross_modal_summary']:
        dataset_comparison = summary['cross_modal_summary']['dataset_comparison']
        avg_alignment = np.mean([d['alignment'] for d in dataset_comparison.values()])
        stats_text += f"Average Alignment: {avg_alignment:.4f}\n"
    
    if 'architecture_insights' in summary and 'best_performing_pairs' in summary['architecture_insights']:
        best_pairs = summary['architecture_insights']['best_performing_pairs']
        if best_pairs:
            best_alignment = best_pairs[0]['alignment']
            stats_text += f"Best Alignment: {best_alignment:.4f}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace', fontweight='bold')
    
    plt.suptitle('Medium-Scale Cross-Modal Phase Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    viz_path = Path(output_dir) / "medium_scale_analysis_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    plt.close()

def main():
    """Main function for medium-scale analysis"""
    
    print("Starting Medium-Scale Cross-Modal Phase Analysis")
    print("=" * 80)
    
    # Run medium-scale analysis
    print("\nRunning comprehensive medium-scale analysis...")
    results = medium_scale_analysis()
    
    print(f"\n{'='*80}")
    print("MEDIUM-SCALE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("Results saved to: ./results/medium_scale_analysis_*/")
    print("\nKey deliverables:")
    print("  1. Cross-modal phase diagrams for each dataset")
    print("  2. Architecture compatibility analysis")
    print("  3. Cross-dataset comparison")
    print("  4. Best performing model pairs")
    print("  5. Comprehensive visualizations")
    print("\nNext steps:")
    print("  1. Review phase transition patterns")
    print("  2. Analyze architecture-specific insights")
    print("  3. Compare with single-modal results")
    print("  4. Identify optimal model combinations")

if __name__ == "__main__":
    main()
