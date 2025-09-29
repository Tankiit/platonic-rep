#!/usr/bin/env python3
"""
Demo Phase Analysis with Synthetic Data
========================================

This script demonstrates the phase analysis pipeline using synthetic features
that mimic the characteristics of real vision and language models.
"""

import numpy as np
import os
from pathlib import Path
from phase_analysis_pipeline import PhaseAnalysisPipeline

def generate_synthetic_features(model_type, phase_type, n_samples=1000, feature_dim=768):
    """
    Generate synthetic features that mimic different phase characteristics.

    Args:
        model_type: 'vision' or 'language'
        phase_type: 'stable', 'critical', or 'chaotic'
        n_samples: Number of samples
        feature_dim: Feature dimension

    Returns:
        Synthetic feature array
    """
    np.random.seed(hash((model_type, phase_type)) % 2**32)

    if phase_type == 'stable':
        # Stable phase: high concentration in few principal components
        # Generate low-rank features
        rank = 10
        U = np.random.randn(n_samples, rank)
        V = np.random.randn(rank, feature_dim)
        features = U @ V

        # Add small noise
        features += np.random.randn(n_samples, feature_dim) * 0.1

    elif phase_type == 'critical':
        # Critical phase: moderate concentration
        rank = 50
        U = np.random.randn(n_samples, rank)
        V = np.random.randn(rank, feature_dim)
        features = U @ V

        # Add moderate noise
        features += np.random.randn(n_samples, feature_dim) * 0.5

    else:  # chaotic
        # Chaotic phase: features spread across all dimensions
        features = np.random.randn(n_samples, feature_dim)

        # Add correlation structure for realism
        if model_type == 'vision':
            # Vision models tend to have local correlations
            for i in range(0, feature_dim, 16):
                features[:, i:min(i+16, feature_dim)] = features[:, i:i+1] * np.random.rand(1, min(16, feature_dim-i))
        else:
            # Language models have different correlation patterns
            features = features @ np.random.randn(feature_dim, feature_dim) * 0.3

    # Normalize features
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    return features

def create_synthetic_dataset(output_dir="./results/features/synthetic"):
    """
    Create a synthetic dataset mimicking the PRH framework structure.
    """
    # Define synthetic models with different phase characteristics
    vision_models = {
        # Small vision models
        'vit_tiny_stable': 'stable',
        'vit_small_stable': 'stable',
        'resnet18_critical': 'critical',
        'resnet34_critical': 'critical',
        'convnext_tiny_chaotic': 'chaotic',
        'efficientnet_b0_chaotic': 'chaotic',
        'mobilenet_v2_chaotic': 'chaotic',
        'deit_tiny_critical': 'critical',
    }

    language_models = {
        # Small language models
        'bert_tiny_stable': 'stable',
        'bert_mini_stable': 'stable',
        'distilbert_critical': 'critical',
        'albert_tiny_critical': 'critical',
        'gpt2_small_chaotic': 'chaotic',
        'opt_125m_chaotic': 'chaotic',
        'bloom_560m_critical': 'critical',
        'roberta_tiny_stable': 'stable',
    }

    # Create directory structure
    dataset = "synthetic"
    subset = "demo_1024"
    base_path = Path(output_dir) / dataset / subset
    base_path.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic features...")
    print("="*50)

    # Generate vision model features
    print("\nVision Models:")
    for model_name, phase in vision_models.items():
        features = generate_synthetic_features('vision', phase, n_samples=1024)
        output_path = base_path / f"{model_name}_pool-cls.npy"
        np.save(output_path, features)
        print(f"  ✓ {model_name} ({phase} phase) - shape: {features.shape}")

    # Generate language model features
    print("\nLanguage Models:")
    for model_name, phase in language_models.items():
        features = generate_synthetic_features('language', phase, n_samples=1024)
        output_path = base_path / f"{model_name}_pool-avg.npy"
        np.save(output_path, features)
        print(f"  ✓ {model_name} ({phase} phase) - shape: {features.shape}")

    print(f"\nSynthetic features saved to: {base_path}")

    return {
        'dataset': dataset,
        'subset': subset,
        'vision_models': list(vision_models.keys()),
        'language_models': list(language_models.keys()),
        'feature_dir': output_dir
    }

def main():
    """
    Run demo phase analysis with synthetic data.
    """
    print("="*70)
    print("PHASE ANALYSIS PIPELINE DEMO")
    print("Using Synthetic Data to Demonstrate Phase Transitions")
    print("="*70)

    # Generate synthetic dataset
    print("\n[Step 1] Creating synthetic dataset...")
    dataset_info = create_synthetic_dataset()

    # Initialize pipeline
    print("\n[Step 2] Initializing phase analysis pipeline...")
    pipeline = PhaseAnalysisPipeline(
        dataset=dataset_info['dataset'],
        subset=dataset_info['subset'],
        output_dir="./results/phase_analysis/demo_synthetic",
        device='cpu'  # Use CPU for demo
    )

    # Run analysis
    print("\n[Step 3] Running complete analysis...")
    results = pipeline.run_complete_analysis(
        vision_models=dataset_info['vision_models'],
        language_models=dataset_info['language_models'],
        feature_dir=dataset_info['feature_dir']
    )

    # Print key findings
    print("\n" + "="*70)
    print("DEMO RESULTS SUMMARY")
    print("="*70)

    summary = results['summary']

    print("\n📊 Phase Distribution:")
    for phase, count in summary['phase_distribution'].items():
        print(f"  - {phase}: {count} models")

    print("\n🔬 Modality-Specific Patterns:")
    print(f"  Vision models:")
    for phase, count in summary['vision_phase_distribution'].items():
        print(f"    - {phase}: {count} models")
    print(f"  Language models:")
    for phase, count in summary['language_phase_distribution'].items():
        print(f"    - {phase}: {count} models")

    print("\n🔗 Cross-Modal Compatibility:")
    compat_stats = summary['compatibility_stats']
    print(f"  - Mean compatibility score: {compat_stats['mean']:.3f}")
    print(f"  - Std deviation: {compat_stats['std']:.3f}")
    print(f"  - Min/Max: {compat_stats['min']:.3f} / {compat_stats['max']:.3f}")
    print(f"  - Pairs with <10% predicted alignment: {compat_stats['below_threshold']}")

    print("\n✅ Key Findings:")
    findings = summary['key_findings']
    modality_bias = findings['modality_phase_bias']
    print(f"  - Vision models in chaotic phase: {modality_bias['vision_chaotic_percentage']:.1f}%")
    print(f"  - Language models in chaotic phase: {modality_bias['language_chaotic_percentage']:.1f}%")

    agop_validation = findings['agop_threshold_validation']
    print(f"  - High AGOP ratio pairs: {agop_validation['high_agop_pairs']}")
    print(f"  - Failure rate for high AGOP: {agop_validation['failure_rate']:.1f}%")

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nThis demonstration used synthetic data to showcase the phase analysis")
    print("pipeline. With real model features, the analysis would reveal actual")
    print("phase transitions and cross-modal compatibility patterns.")
    print(f"\n📁 Full results saved to: ./results/phase_analysis/demo_synthetic/")
    print(f"📊 Visualizations: ./results/phase_analysis/demo_synthetic/figures/")
    print(f"📄 Report: ./results/phase_analysis/demo_synthetic/phase_analysis_report.md")

if __name__ == "__main__":
    main()