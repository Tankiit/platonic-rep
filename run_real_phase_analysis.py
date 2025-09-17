#!/usr/bin/env python3
"""
Run phase analysis on real extracted features
"""

from phase_analysis_pipeline import PhaseAnalysisPipeline
from pathlib import Path
import numpy as np

def main():
    """Run phase analysis on extracted features"""

    # Get list of available models from extracted features
    feature_dir = Path("./results/features/minhuh/prh/wit_1024")

    vision_models = []
    language_models = []

    # Find all extracted features
    for file_path in feature_dir.glob("*.npy"):
        if "_pool-cls.npy" in str(file_path):
            model_name = file_path.stem.replace("_pool-cls", "")
            vision_models.append(model_name)
        elif "_pool-avg.npy" in str(file_path):
            model_name = file_path.stem.replace("_pool-avg", "")
            language_models.append(model_name)

    print("="*70)
    print("PHASE ANALYSIS - REAL FEATURES")
    print("="*70)
    print(f"\nFound {len(vision_models)} vision models:")
    for model in vision_models:
        print(f"  - {model}")

    print(f"\nFound {len(language_models)} language models:")
    for model in language_models:
        print(f"  - {model}")

    if len(vision_models) == 0 or len(language_models) == 0:
        print("\nError: Need at least one vision and one language model")
        return

    # Run phase analysis pipeline
    print("\n" + "="*70)
    print("Running Phase Analysis Pipeline")
    print("="*70)

    pipeline = PhaseAnalysisPipeline(
        dataset="minhuh/prh",
        subset="wit_1024",
        output_dir="./results/phase_analysis/real_features",
        device="cpu"
    )

    results = pipeline.run_complete_analysis(
        vision_models=vision_models,
        language_models=language_models,
        feature_dir="./results/features"
    )

    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)

    summary = results['summary']

    print("\n📊 Phase Distribution:")
    for phase, count in summary.get('phase_distribution', {}).items():
        print(f"  - {phase}: {count} models")

    print("\n🔬 Modality-Specific Analysis:")
    print("Vision models:")
    for phase, count in summary.get('vision_phase_distribution', {}).items():
        print(f"  - {phase}: {count}")
    print("Language models:")
    for phase, count in summary.get('language_phase_distribution', {}).items():
        print(f"  - {phase}: {count}")

    print("\n🔗 Cross-Modal Compatibility:")
    compat_stats = summary.get('compatibility_stats', {})
    print(f"  Mean score: {compat_stats.get('mean', 0):.3f}")
    print(f"  Std deviation: {compat_stats.get('std', 0):.3f}")
    print(f"  Range: [{compat_stats.get('min', 0):.3f}, {compat_stats.get('max', 0):.3f}]")

    print("\n✅ Key Findings:")
    findings = summary.get('key_findings', {})
    modality_bias = findings.get('modality_phase_bias', {})
    print(f"  Vision models in chaotic phase: {modality_bias.get('vision_chaotic_percentage', 0):.1f}%")
    print(f"  Language models in chaotic phase: {modality_bias.get('language_chaotic_percentage', 0):.1f}%")

    # Check for interesting patterns
    print("\n🔍 Phase Compatibility Analysis:")
    n_pairs = len(vision_models) * len(language_models)
    print(f"  Total model pairs analyzed: {n_pairs}")

    low_compat_pairs = []
    for pair_key, compat in results['compatibility'].items():
        if isinstance(compat, dict) and compat.get('compatibility_score', 1) < 0.5:
            low_compat_pairs.append((pair_key, compat['compatibility_score']))

    if low_compat_pairs:
        print(f"  Low compatibility pairs (score < 0.5): {len(low_compat_pairs)}")
        for pair, score in sorted(low_compat_pairs, key=lambda x: x[1])[:3]:
            print(f"    - {pair}: {score:.3f}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📁 Results: ./results/phase_analysis/real_features/")
    print(f"📊 Figures: ./results/phase_analysis/real_features/figures/")
    print(f"📄 Report: ./results/phase_analysis/real_features/phase_analysis_report.md")

if __name__ == "__main__":
    main()