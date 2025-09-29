#!/usr/bin/env python3
"""
Runner script for Multi-Modal Embedding Comparison
Provides predefined configurations for different scales of comparison
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime
from multi_modal_embedding_comparison import MultiModalComparison


# Predefined model configurations
MODEL_CONFIGS = {
    'minimal': {
        'vision': ['resnet18', 'mobilenet_v2'],
        'language': ['distilbert-base', 'sentence-bert-base'],
        'description': 'Minimal configuration for quick testing (2 vision, 2 language models)'
    },

    'small': {
        'vision': ['resnet18', 'mobilenet_v2', 'efficientnet_b0'],
        'language': ['distilbert-base', 'albert-base-v2', 'sentence-bert-base'],
        'description': 'Small models only (3 vision, 3 language models)'
    },

    'medium': {
        'vision': ['resnet34', 'resnet50', 'efficientnet_b1', 'vit_small_patch16_224'],
        'language': ['bert-base', 'roberta-base', 'gpt2-medium', 't5-small'],
        'description': 'Medium-sized models (4 vision, 4 language models)'
    },

    'mixed': {
        'vision': ['resnet18', 'mobilenet_v2', 'resnet34', 'vit_tiny_patch16_224', 'vit_small_patch16_224'],
        'language': ['distilbert-base', 'albert-base-v2', 'bert-base', 'gpt2', 'sentence-bert-base'],
        'description': 'Mix of small and medium models (5 vision, 5 language models)'
    },

    'clip_focused': {
        'vision': ['resnet18', 'vit_tiny_patch16_224', 'clip_rn50', 'clip_vit_b32'],
        'language': ['bert-base', 'gpt2', 'sentence-bert-base', 'sentence-bert-large'],
        'description': 'Focus on CLIP and comparable models (4 vision, 4 language models)'
    },

    'comprehensive': {
        'vision': [
            'resnet18', 'resnet34', 'resnet50',
            'mobilenet_v2', 'efficientnet_b0', 'efficientnet_b1',
            'vit_tiny_patch16_224', 'vit_small_patch16_224',
            'clip_rn50', 'clip_vit_b32'
        ],
        'language': [
            'distilbert-base', 'albert-base-v2', 'bert-base', 'roberta-base',
            'gpt2', 'gpt2-medium', 't5-small',
            'sentence-bert-base', 'sentence-bert-large'
        ],
        'description': 'Comprehensive comparison (10 vision, 9 language models)'
    }
}


def print_configuration_info():
    """Print information about available configurations"""
    print("\n" + "="*70)
    print("Available Model Configurations")
    print("="*70)

    for config_name, config in MODEL_CONFIGS.items():
        print(f"\n{config_name.upper()}:")
        print(f"  {config['description']}")
        print(f"  Vision Models ({len(config['vision'])}): {', '.join(config['vision'][:3])}...")
        print(f"  Language Models ({len(config['language'])}): {', '.join(config['language'][:3])}...")
        print(f"  Total Comparisons: {len(config['vision']) * len(config['language'])}")


def estimate_runtime(config_name: str, num_samples: int) -> str:
    """Estimate runtime for a configuration"""
    config = MODEL_CONFIGS[config_name]
    n_vision = len(config['vision'])
    n_language = len(config['language'])

    # Rough estimates (seconds per model)
    time_per_vision = 30 if 'clip' in str(config['vision']) else 20
    time_per_language = 25

    total_seconds = (n_vision * time_per_vision + n_language * time_per_language) * (num_samples / 500)
    total_minutes = total_seconds / 60

    if total_minutes < 60:
        return f"{total_minutes:.1f} minutes"
    else:
        return f"{total_minutes/60:.1f} hours"


def check_gpu_availability():
    """Check and report GPU availability"""
    print("\n" + "="*50)
    print("System Information")
    print("="*50)

    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS (Apple Silicon) Available: Yes")
    else:
        print(f"GPU: Not available, using CPU")

    print(f"PyTorch Version: {torch.__version__}")


def main():
    """Main runner function"""

    parser = argparse.ArgumentParser(
        description='Run Multi-Modal Embedding Comparison with predefined configurations'
    )

    # Configuration selection
    parser.add_argument(
        '--config',
        choices=list(MODEL_CONFIGS.keys()),
        default='minimal',
        help='Predefined model configuration to use'
    )

    # Custom model selection (overrides config)
    parser.add_argument(
        '--vision-models',
        nargs='+',
        default=None,
        help='Custom list of vision models (overrides config)'
    )
    parser.add_argument(
        '--language-models',
        nargs='+',
        default=None,
        help='Custom list of language models (overrides config)'
    )

    # Dataset configuration
    parser.add_argument(
        '--dataset',
        default='synthetic',
        choices=['synthetic', 'mscoco', 'conceptual_captions'],
        help='Dataset to use for evaluation'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=500,
        help='Number of samples to process'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: ./results/multi_modal_comparison_<timestamp>)'
    )

    # Other options
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available configurations and exit'
    )
    parser.add_argument(
        '--check-system',
        action='store_true',
        help='Check system capabilities and exit'
    )

    args = parser.parse_args()

    # Handle info requests
    if args.list_configs:
        print_configuration_info()
        return

    if args.check_system:
        check_gpu_availability()
        return

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/multi_modal_comparison_{timestamp}"
    else:
        output_dir = args.output_dir

    # Get model lists
    if args.vision_models is not None and args.language_models is not None:
        # Use custom models
        vision_models = args.vision_models
        language_models = args.language_models
        config_name = "custom"
    else:
        # Use predefined configuration
        config = MODEL_CONFIGS[args.config]
        vision_models = config['vision']
        language_models = config['language']
        config_name = args.config

    # Print run information
    print("\n" + "="*70)
    print("Multi-Modal Embedding Comparison")
    print("="*70)
    print(f"Configuration: {config_name}")
    if config_name in MODEL_CONFIGS:
        print(f"Description: {MODEL_CONFIGS[config_name]['description']}")
    print(f"Vision Models: {len(vision_models)}")
    print(f"Language Models: {len(language_models)}")
    print(f"Total Comparisons: {len(vision_models) * len(language_models)}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Output Directory: {output_dir}")

    if config_name in MODEL_CONFIGS:
        estimated_time = estimate_runtime(config_name, args.num_samples)
        print(f"Estimated Runtime: ~{estimated_time}")

    # Check system
    check_gpu_availability()

    # Confirm before running comprehensive configs
    if config_name in ['comprehensive'] or len(vision_models) * len(language_models) > 50:
        print("\n" + "!"*50)
        print("WARNING: This configuration will take significant time and resources!")
        print("!"*50)
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Create and run comparison
    print("\n" + "="*70)
    print("Starting Analysis...")
    print("="*70)

    comparison = MultiModalComparison(output_dir=output_dir)

    try:
        results = comparison.run_comparison(
            vision_models=vision_models,
            language_models=language_models,
            dataset_name=args.dataset,
            num_samples=args.num_samples
        )

        print("\n" + "="*70)
        print("SUCCESS: Analysis Complete!")
        print("="*70)
        print(f"Results saved to: {output_dir}")
        print("\nKey outputs:")
        print(f"  - Alignment matrices: {output_dir}/alignment_matrix_*.png")
        print(f"  - Embedding statistics: {output_dir}/*_embedding_stats.png")
        print(f"  - RSM plots: {output_dir}/rsm_*.png")
        print(f"  - JSON results: {output_dir}/comparison_results.json")

    except Exception as e:
        print("\n" + "!"*50)
        print(f"ERROR: Analysis failed with error: {e}")
        print("!"*50)
        raise


if __name__ == "__main__":
    main()