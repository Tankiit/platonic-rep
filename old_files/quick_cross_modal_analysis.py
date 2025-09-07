#!/usr/bin/env python3
"""
Quick Cross-Modal Phase Analysis
Faster analysis with a subset of models for testing and development
"""

import torch
import numpy as np
from pathlib import Path
from multi_model_analysis import MultiModelAnalyzer
from cross_modal_phase_analysis import CrossModalPhaseAnalyzer
import json
from datetime import datetime

def quick_cross_modal_analysis():
    """Run quick cross-modal analysis with a subset of models"""
    
    # Initialize analyzer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/quick_cross_modal_{timestamp}/"
    analyzer = MultiModelAnalyzer(output_dir=output_dir)
    
    # Quick model sets (subset for faster analysis)
    vision_models = [
        'resnet18',           # CNN
        'vit_base_patch16_224',  # Vision Transformer
        'convnext_tiny',      # Modern CNN
        'mixer_b16_224'       # MLP-Mixer
    ]
    
    text_models = [
        'bert_base',         # BERT
        'roberta_base',      # RoBERTa
        'gpt2_medium',       # GPT-2
        'albert_base_v2'      # ALBERT
    ]
    
    datasets = ['cifar10']  # Single dataset for speed
    
    print("=== Quick Cross-Modal Phase Analysis ===")
    print(f"Vision models: {vision_models}")
    print(f"Text models: {text_models}")
    print(f"Dataset: {datasets[0]}")
    print(f"Total combinations: {len(vision_models) * len(text_models)}")
    print(f"Output directory: {output_dir}")
    
    # Run cross-modal analysis
    try:
        cross_modal_results = analyzer.run_cross_modal_phase_analysis(
            vision_models=vision_models,
            text_models=text_models,
            dataset=datasets[0],
            pretrained=True
        )
        
        # Analyze results
        analyze_quick_results(cross_modal_results, vision_models, text_models)
        
        return cross_modal_results
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None

def analyze_quick_results(results, vision_models, text_models):
    """Analyze quick cross-modal results"""
    
    if not results or 'phase_results' not in results:
        print("No results to analyze")
        return
    
    phase_results = results['phase_results']
    
    print(f"\n{'='*60}")
    print("QUICK ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # Overall statistics
    ntk_vals = [r['ntk_stability'] for r in phase_results]
    agop_vals = [r['agop_magnitude'] for r in phase_results]
    align_vals = [r['alignment'] for r in phase_results]
    
    print(f"Overall Statistics:")
    print(f"  Total pairs: {len(phase_results)}")
    print(f"  NTK Stability: {np.mean(ntk_vals):.4f} ± {np.std(ntk_vals):.4f}")
    print(f"  AGOP Magnitude: {np.mean(agop_vals):.4f} ± {np.std(agop_vals):.4f}")
    print(f"  Alignment: {np.mean(align_vals):.4f} ± {np.std(align_vals):.4f}")
    
    # Phase distribution
    phase_counts = {}
    for result in phase_results:
        phase = result['phase_region']
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    print(f"\nPhase Distribution:")
    for phase, count in phase_counts.items():
        percentage = (count / len(phase_results)) * 100
        print(f"  {phase.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Architecture-specific analysis
    print(f"\nArchitecture Analysis:")
    
    # Group by vision model
    vision_groups = {}
    for result in phase_results:
        v_model = result['v_model']
        if v_model not in vision_groups:
            vision_groups[v_model] = []
        vision_groups[v_model].append(result)
    
    for v_model, group_results in vision_groups.items():
        print(f"\n  {v_model}:")
        avg_ntk = np.mean([r['ntk_stability'] for r in group_results])
        avg_agop = np.mean([r['agop_magnitude'] for r in group_results])
        avg_align = np.mean([r['alignment'] for r in group_results])
        
        print(f"    NTK Stability: {avg_ntk:.4f}")
        print(f"    AGOP Magnitude: {avg_agop:.4f}")
        print(f"    Alignment: {avg_align:.4f}")
    
    # Group by text model
    text_groups = {}
    for result in phase_results:
        t_model = result['t_model']
        if t_model not in text_groups:
            text_groups[t_model] = []
        text_groups[t_model].append(result)
    
    for t_model, group_results in text_groups.items():
        print(f"\n  {t_model}:")
        avg_ntk = np.mean([r['ntk_stability'] for r in group_results])
        avg_agop = np.mean([r['agop_magnitude'] for r in group_results])
        avg_align = np.mean([r['alignment'] for r in group_results])
        
        print(f"    NTK Stability: {avg_ntk:.4f}")
        print(f"    AGOP Magnitude: {avg_agop:.4f}")
        print(f"    Alignment: {avg_align:.4f}")
    
    # Best and worst pairs
    print(f"\nBest Performing Pairs:")
    sorted_by_align = sorted(phase_results, key=lambda x: x['alignment'], reverse=True)
    for i, result in enumerate(sorted_by_align[:3]):
        print(f"  {i+1}. {result['v_model']} + {result['t_model']}: {result['alignment']:.4f}")
    
    print(f"\nWorst Performing Pairs:")
    for i, result in enumerate(sorted_by_align[-3:]):
        print(f"  {i+1}. {result['v_model']} + {result['t_model']}: {result['alignment']:.4f}")

def run_architecture_comparison():
    """Compare different architecture combinations"""
    
    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*60}")
    
    # Define architecture pairs
    architecture_pairs = [
        # CNN + Transformer
        ('resnet18', 'bert_base'),
        ('resnet18', 'roberta_base'),
        
        # Vision Transformer + Language Transformer
        ('vit_base_patch16_224', 'bert_base'),
        ('vit_base_patch16_224', 'roberta_base'),
        
        # Modern CNN + Language Model
        ('convnext_tiny', 'gpt2_medium'),
        ('convnext_tiny', 'albert_base_v2'),
        
        # MLP + Transformer
        ('mixer_b16_224', 'bert_base'),
        ('mixer_b16_224', 'gpt2_medium')
    ]
    
    # Create synthetic features for comparison
    vision_features = {}
    text_features = {}
    
    for v_model, t_model in architecture_pairs:
        # Create realistic feature dimensions
        if 'resnet' in v_model:
            v_dim = 512
        elif 'vit' in v_model:
            v_dim = 768
        elif 'convnext' in v_model:
            v_dim = 768
        elif 'mixer' in v_model:
            v_dim = 768
        else:
            v_dim = 512
        
        if 'bert' in t_model:
            t_dim = 768
        elif 'roberta' in t_model:
            t_dim = 768
        elif 'gpt' in t_model:
            t_dim = 1024
        elif 'albert' in t_model:
            t_dim = 768
        else:
            t_dim = 768
        
        # Create features with some realistic patterns
        n_samples = 150
        vision_features[v_model] = torch.randn(n_samples, v_dim)
        text_features[t_model] = torch.randn(n_samples, t_dim)
    
    # Run analysis
    cross_modal_analyzer = CrossModalPhaseAnalyzer(
        output_dir="./results/architecture_comparison/"
    )
    
    results = cross_modal_analyzer.analyze_cross_modal_representations(
        vision_features=vision_features,
        text_features=text_features,
        model_pairs=architecture_pairs
    )
    
    # Analyze architecture patterns
    analyze_architecture_patterns(results, architecture_pairs)
    
    return results

def analyze_architecture_patterns(results, architecture_pairs):
    """Analyze patterns across different architectures"""
    
    if not results or 'phase_results' not in results:
        return
    
    phase_results = results['phase_results']
    
    print(f"\nArchitecture Pattern Analysis:")
    
    # Group by architecture type
    arch_types = {
        'CNN+Transformer': [],
        'VisionTransformer+Transformer': [],
        'ModernCNN+LanguageModel': [],
        'MLP+Transformer': []
    }
    
    for result in phase_results:
        v_model = result['v_model']
        t_model = result['t_model']
        
        if 'resnet' in v_model and ('bert' in t_model or 'roberta' in t_model):
            arch_types['CNN+Transformer'].append(result)
        elif 'vit' in v_model and ('bert' in t_model or 'roberta' in t_model):
            arch_types['VisionTransformer+Transformer'].append(result)
        elif 'convnext' in v_model and ('gpt' in t_model or 'albert' in t_model):
            arch_types['ModernCNN+LanguageModel'].append(result)
        elif 'mixer' in v_model:
            arch_types['MLP+Transformer'].append(result)
    
    # Compare architecture types
    for arch_type, group_results in arch_types.items():
        if not group_results:
            continue
            
        print(f"\n  {arch_type}:")
        avg_ntk = np.mean([r['ntk_stability'] for r in group_results])
        avg_agop = np.mean([r['agop_magnitude'] for r in group_results])
        avg_align = np.mean([r['alignment'] for r in group_results])
        
        print(f"    NTK Stability: {avg_ntk:.4f}")
        print(f"    AGOP Magnitude: {avg_agop:.4f}")
        print(f"    Alignment: {avg_align:.4f}")
        print(f"    Sample pairs: {len(group_results)}")

def main():
    """Main function for quick analysis"""
    
    print("Starting Quick Cross-Modal Phase Analysis")
    print("=" * 60)
    
    # Run quick analysis
    print("\n1. Running quick cross-modal analysis...")
    quick_results = quick_cross_modal_analysis()
    
    # Run architecture comparison
    print("\n2. Running architecture comparison...")
    arch_results = run_architecture_comparison()
    
    print(f"\n{'='*60}")
    print("QUICK ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("Results saved to:")
    print("  - Quick: ./results/quick_cross_modal_*/")
    print("  - Architecture: ./results/architecture_comparison/")
    print("\nKey insights:")
    print("  - Check phase distribution across architectures")
    print("  - Compare alignment scores between model types")
    print("  - Identify optimal architecture combinations")

if __name__ == "__main__":
    main()
