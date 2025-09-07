#!/usr/bin/env python3
"""
Example script demonstrating cross-modal phase diagram analysis
"""

import torch
import numpy as np
from pathlib import Path
from multi_model_analysis import MultiModelAnalyzer
from cross_modal_phase_analysis import CrossModalPhaseAnalyzer

def main():
    """Example cross-modal phase analysis"""
    
    # Initialize the multi-model analyzer
    analyzer = MultiModelAnalyzer(output_dir="./results/example_cross_modal/")
    
    # Define vision and text models to analyze
    vision_models = ['resnet18', 'vit_base_patch16_224', 'convnext_tiny']
    text_models = ['bert_base', 'roberta_base', 'gpt2_medium']
    
    print("=== Cross-Modal Phase Analysis Example ===")
    print(f"Vision models: {vision_models}")
    print(f"Text models: {text_models}")
    
    # Option 1: Run cross-modal analysis using the integrated method
    print("\n1. Running integrated cross-modal analysis...")
    cross_modal_results = analyzer.run_cross_modal_phase_analysis(
        vision_models=vision_models,
        text_models=text_models,
        dataset='cifar10',
        pretrained=True
    )
    
    # Option 2: Run standalone cross-modal analysis
    print("\n2. Running standalone cross-modal analysis...")
    standalone_analyzer = CrossModalPhaseAnalyzer(
        output_dir="./results/example_cross_modal/standalone/"
    )
    
    # Create synthetic features for demonstration
    # In practice, these would come from actual model feature extraction
    vision_features = {
        'resnet18': torch.randn(100, 512),
        'vit_base_patch16_224': torch.randn(100, 768),
        'convnext_tiny': torch.randn(100, 768)
    }
    
    text_features = {
        'bert_base': torch.randn(100, 768),
        'roberta_base': torch.randn(100, 768),
        'gpt2_medium': torch.randn(100, 1024)
    }
    
    # Define specific model pairs to analyze
    model_pairs = [
        ('resnet18', 'bert_base'),
        ('vit_base_patch16_224', 'roberta_base'),
        ('convnext_tiny', 'gpt2_medium'),
        ('resnet18', 'gpt2_medium'),
        ('vit_base_patch16_224', 'bert_base')
    ]
    
    standalone_results = standalone_analyzer.analyze_cross_modal_representations(
        vision_features=vision_features,
        text_features=text_features,
        model_pairs=model_pairs
    )
    
    # Option 3: Manual phase diagram computation
    print("\n3. Manual phase diagram computation...")
    
    # Create analyzer for manual computation
    manual_analyzer = CrossModalPhaseAnalyzer()
    
    # Compute metrics for a single pair
    v_feat = vision_features['resnet18']
    t_feat = text_features['bert_base']
    
    ntk_stability = manual_analyzer.compute_cross_modal_ntk_stability(v_feat, t_feat)
    agop_magnitude = manual_analyzer.compute_cross_modal_agop_proxy(v_feat, t_feat)
    alignment = manual_analyzer.compute_cross_modal_alignment(v_feat, t_feat)
    
    print(f"ResNet18-BERT Base pair:")
    print(f"  NTK Stability: {ntk_stability:.4f}")
    print(f"  AGOP Magnitude: {agop_magnitude:.4f}")
    print(f"  Alignment Score: {alignment:.4f}")
    
    # Option 4: Command line usage example
    print("\n4. Command line usage:")
    print("python multi_model_analysis.py --cross_modal --vision_models resnet18 vit_base_patch16_224 --text_models bert_base roberta_base --datasets cifar10")
    
    # Print summary
    print(f"\n=== Analysis Complete ===")
    print(f"Integrated results: {analyzer.output_dir}/cross_modal_analysis/")
    print(f"Standalone results: {standalone_analyzer.output_dir}")
    
    # Show some key results
    if cross_modal_results and 'summary' in cross_modal_results:
        summary = cross_modal_results['summary']
        avg_metrics = summary.get('average_metrics', {})
        print(f"\nAverage metrics across all pairs:")
        print(f"  NTK Stability: {avg_metrics.get('avg_ntk_stability', 0):.4f}")
        print(f"  AGOP Magnitude: {avg_metrics.get('avg_agop_magnitude', 0):.4f}")
        print(f"  Alignment Score: {avg_metrics.get('avg_alignment', 0):.4f}")
        
        phase_dist = summary.get('phase_distribution', {})
        print(f"\nPhase distribution:")
        for phase, count in phase_dist.items():
            print(f"  {phase.capitalize()}: {count} pairs")

if __name__ == "__main__":
    main()
