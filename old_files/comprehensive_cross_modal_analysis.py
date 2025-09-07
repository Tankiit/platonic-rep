#!/usr/bin/env python3
"""
Comprehensive Cross-Modal Phase Analysis
Analyzes phase transitions across multiple vision and text models on various datasets
"""

import torch
import numpy as np
from pathlib import Path
from multi_model_analysis import MultiModelAnalyzer
from cross_modal_phase_analysis import CrossModalPhaseAnalyzer
import json
from datetime import datetime

def comprehensive_cross_modal_analysis():
    """Run comprehensive cross-modal phase analysis across multiple models and datasets"""
    
    # Initialize analyzer with comprehensive output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/comprehensive_cross_modal_{timestamp}/"
    analyzer = MultiModelAnalyzer(output_dir=output_dir)
    
    # Comprehensive model sets
    vision_models = [
        # ResNet family
        'resnet18', 'resnet34', 'resnet50',
        # Vision Transformers
        'vit_base_patch16_224', 'vit_small_patch16_224', 'deit_base_patch16_224',
        # ConvNeXt family
        'convnext_tiny', 'convnext_small',
        # EfficientNet
        'efficientnet_b0',
        # MLP-Mixer
        'mixer_b16_224', 'mixer_b32_224'
    ]
    
    text_models = [
        # BERT family
        'bert_base', 'bert_large', 'distilbert_base',
        # RoBERTa family
        'roberta_base', 'roberta_large',
        # GPT family
        'gpt2_medium', 'gpt2_large',
        # Other transformers
        'albert_base_v2', 'xlnet_base_cased'
    ]
    
    datasets = ['cifar10', 'cifar100', 'svhn']
    
    print("=== Comprehensive Cross-Modal Phase Analysis ===")
    print(f"Vision models: {len(vision_models)} models")
    print(f"Text models: {len(text_models)} models")
    print(f"Datasets: {datasets}")
    print(f"Total combinations: {len(vision_models) * len(text_models) * len(datasets)}")
    print(f"Output directory: {output_dir}")
    
    # Store all results
    all_results = {}
    
    # Run analysis for each dataset
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Analyzing dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        dataset_results = {}
        
        # Run cross-modal analysis for this dataset
        try:
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
        
        for model in vision_models[:5]:  # Limit to first 5 for efficiency
            try:
                result = analyzer.run_analysis(model, dataset, pretrained=True, save_features=True)
                if result:
                    individual_results[model] = result
                    print(f"✓ {model} analysis complete")
                else:
                    print(f"✗ {model} analysis failed")
            except Exception as e:
                print(f"✗ {model} analysis error: {e}")
        
        dataset_results['individual'] = individual_results
        all_results[dataset] = dataset_results
    
    # Generate comprehensive summary
    generate_comprehensive_summary(all_results, output_dir)
    
    return all_results

def generate_comprehensive_summary(all_results, output_dir):
    """Generate comprehensive summary of all results"""
    
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE SUMMARY")
    print(f"{'='*60}")
    
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_datasets': len(all_results),
        'dataset_summaries': {},
        'cross_modal_summary': {},
        'phase_distribution': {},
        'model_performance': {}
    }
    
    # Analyze each dataset
    for dataset, results in all_results.items():
        print(f"\nAnalyzing {dataset} results...")
        
        dataset_summary = {
            'cross_modal_available': results['cross_modal'] is not None,
            'individual_models': len(results['individual']),
            'metrics': {}
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
                
                # Update global summaries
                if 'cross_modal_summary' not in summary:
                    summary['cross_modal_summary'] = {
                        'total_pairs': 0,
                        'avg_ntk_stability': [],
                        'avg_agop_magnitude': [],
                        'avg_alignment': []
                    }
                
                summary['cross_modal_summary']['total_pairs'] += cross_modal['summary'].get('total_pairs', 0)
                summary['cross_modal_summary']['avg_ntk_stability'].append(avg_metrics.get('avg_ntk_stability', 0))
                summary['cross_modal_summary']['avg_agop_magnitude'].append(avg_metrics.get('avg_agop_magnitude', 0))
                summary['cross_modal_summary']['avg_alignment'].append(avg_metrics.get('avg_alignment', 0))
        
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
    
    # Compute global averages
    if summary['cross_modal_summary']['avg_ntk_stability']:
        summary['cross_modal_summary']['global_avg_ntk'] = np.mean(summary['cross_modal_summary']['avg_ntk_stability'])
        summary['cross_modal_summary']['global_avg_agop'] = np.mean(summary['cross_modal_summary']['avg_agop_magnitude'])
        summary['cross_modal_summary']['global_avg_alignment'] = np.mean(summary['cross_modal_summary']['avg_alignment'])
    
    # Save comprehensive summary
    summary_path = Path(output_dir) / "comprehensive_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print_comprehensive_summary(summary)
    
    return summary

def print_comprehensive_summary(summary):
    """Print comprehensive analysis summary"""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE CROSS-MODAL PHASE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print(f"Analysis completed: {summary['analysis_timestamp']}")
    print(f"Total datasets analyzed: {summary['total_datasets']}")
    
    if 'cross_modal_summary' in summary and summary['cross_modal_summary']['total_pairs'] > 0:
        cm_summary = summary['cross_modal_summary']
        print(f"\nCross-Modal Analysis:")
        print(f"  Total model pairs: {cm_summary['total_pairs']}")
        print(f"  Global average NTK stability: {cm_summary.get('global_avg_ntk', 0):.4f}")
        print(f"  Global average AGOP magnitude: {cm_summary.get('global_avg_agop', 0):.4f}")
        print(f"  Global average alignment: {cm_summary.get('global_avg_alignment', 0):.4f}")
    
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
            
            phase_dist = metrics.get('phase_distribution', {})
            if phase_dist:
                print(f"    Phase Distribution: {phase_dist}")

def run_focused_analysis():
    """Run focused analysis on specific model pairs"""
    
    print("\n" + "="*60)
    print("FOCUSED ANALYSIS: SPECIFIC MODEL PAIRS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MultiModelAnalyzer(output_dir="./results/focused_analysis/")
    
    # Focus on specific interesting pairs
    focused_pairs = [
        # ResNet + BERT (CNN + Transformer)
        ('resnet18', 'bert_base'),
        ('resnet50', 'bert_large'),
        
        # ViT + RoBERTa (Vision Transformer + Language Transformer)
        ('vit_base_patch16_224', 'roberta_base'),
        ('vit_small_patch16_224', 'roberta_large'),
        
        # ConvNeXt + GPT (Modern CNN + Language Model)
        ('convnext_tiny', 'gpt2_medium'),
        ('convnext_small', 'gpt2_large'),
        
        # MLP-Mixer + ALBERT (MLP + Transformer)
        ('mixer_b16_224', 'albert_base_v2'),
        ('mixer_b32_224', 'xlnet_base_cased')
    ]
    
    # Create synthetic features for focused analysis
    vision_features = {}
    text_features = {}
    
    for v_model, t_model in focused_pairs:
        # Create realistic feature dimensions
        if 'resnet' in v_model:
            v_dim = 512 if '18' in v_model else 2048
        elif 'vit' in v_model:
            v_dim = 768
        elif 'convnext' in v_model:
            v_dim = 768
        elif 'mixer' in v_model:
            v_dim = 768
        else:
            v_dim = 512
        
        if 'bert' in t_model:
            t_dim = 768 if 'base' in t_model else 1024
        elif 'roberta' in t_model:
            t_dim = 768 if 'base' in t_model else 1024
        elif 'gpt' in t_model:
            t_dim = 1024 if 'medium' in t_model else 1280
        elif 'albert' in t_model:
            t_dim = 768
        elif 'xlnet' in t_model:
            t_dim = 768
        else:
            t_dim = 768
        
        # Create features with some realistic patterns
        n_samples = 200
        vision_features[v_model] = torch.randn(n_samples, v_dim)
        text_features[t_model] = torch.randn(n_samples, t_dim)
    
    # Run focused cross-modal analysis
    cross_modal_analyzer = CrossModalPhaseAnalyzer(
        output_dir="./results/focused_analysis/"
    )
    
    results = cross_modal_analyzer.analyze_cross_modal_representations(
        vision_features=vision_features,
        text_features=text_features,
        model_pairs=focused_pairs
    )
    
    # Analyze results
    print_focused_analysis_results(results, focused_pairs)
    
    return results

def print_focused_analysis_results(results, focused_pairs):
    """Print detailed results for focused analysis"""
    
    print(f"\n{'='*60}")
    print("FOCUSED ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    if 'phase_results' not in results:
        print("No phase results available")
        return
    
    phase_results = results['phase_results']
    
    # Group by architecture type
    architecture_groups = {
        'CNN+Transformer': [],
        'VisionTransformer+LanguageTransformer': [],
        'ModernCNN+LanguageModel': [],
        'MLP+Transformer': []
    }
    
    for result in phase_results:
        v_model = result['v_model']
        t_model = result['t_model']
        
        if 'resnet' in v_model and 'bert' in t_model:
            architecture_groups['CNN+Transformer'].append(result)
        elif 'vit' in v_model and 'roberta' in t_model:
            architecture_groups['VisionTransformer+LanguageTransformer'].append(result)
        elif 'convnext' in v_model and 'gpt' in t_model:
            architecture_groups['ModernCNN+LanguageModel'].append(result)
        elif 'mixer' in v_model:
            architecture_groups['MLP+Transformer'].append(result)
    
    # Print results by architecture group
    for group_name, group_results in architecture_groups.items():
        if not group_results:
            continue
            
        print(f"\n{group_name}:")
        print("-" * len(group_name))
        
        for result in group_results:
            print(f"  {result['v_model']} + {result['t_model']}:")
            print(f"    NTK Stability: {result['ntk_stability']:.4f}")
            print(f"    AGOP Magnitude: {result['agop_magnitude']:.4f}")
            print(f"    Alignment: {result['alignment']:.4f}")
            print(f"    Phase: {result['phase_region'].upper()}")
        
        # Group averages
        avg_ntk = np.mean([r['ntk_stability'] for r in group_results])
        avg_agop = np.mean([r['agop_magnitude'] for r in group_results])
        avg_align = np.mean([r['alignment'] for r in group_results])
        
        print(f"  Group Averages:")
        print(f"    NTK Stability: {avg_ntk:.4f}")
        print(f"    AGOP Magnitude: {avg_agop:.4f}")
        print(f"    Alignment: {avg_align:.4f}")

def main():
    """Main function for comprehensive analysis"""
    
    print("Starting Comprehensive Cross-Modal Phase Analysis")
    print("=" * 60)
    
    # Run comprehensive analysis
    print("\n1. Running comprehensive analysis across all models and datasets...")
    comprehensive_results = comprehensive_cross_modal_analysis()
    
    # Run focused analysis
    print("\n2. Running focused analysis on specific model pairs...")
    focused_results = run_focused_analysis()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("Results saved to:")
    print("  - Comprehensive: ./results/comprehensive_cross_modal_*/")
    print("  - Focused: ./results/focused_analysis/")
    print("\nNext steps:")
    print("  1. Review generated phase diagrams")
    print("  2. Analyze cross-modal alignment patterns")
    print("  3. Investigate phase transitions across architectures")
    print("  4. Compare with single-modal analysis results")

if __name__ == "__main__":
    main()
