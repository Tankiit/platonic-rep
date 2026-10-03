#!/usr/bin/env python3
"""
Example script for running the Phase 1 experiment.

This script shows how to run the controlled misalignment experiment
with different configurations.
"""

import sys
import os
from pathlib import Path

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase1_misalignment_experiment import ExperimentConfig, run_phase1_experiment

def create_small_config():
    """Create a configuration for quick testing using HuggingFace datasets."""
    return ExperimentConfig(
        # Dataset settings - using HuggingFace datasets
        dataset_name="JotDe/mscoco_50k",  # HuggingFace dataset
        coco_root="/path/to/coco2017",  # Kept for compatibility
        output_dir="./phase1_small_results",

        # Dataset limits (small for quick testing)
        max_train_samples=1000,  # Use only 1000 training samples
        max_val_samples=200,     # Use only 200 validation samples

        # Model settings
        vision_encoder="resnet18",
        text_encoder="distilbert",
        embedding_dim=256,

        # Training settings (small for quick testing)
        batch_size=32,  # Smaller batch size for testing
        num_epochs=3,   # Fewer epochs for quick testing
        learning_rate=1e-4,

        # Misalignment ratios to test
        misalignment_ratios=[0.0, 0.5, 1.0],  # Fewer conditions for quick testing

        # Device settings
        device="mps",  # Change to "cuda" for NVIDIA GPUs or "cpu"
        num_workers=0,  # Use single-threaded loading to avoid multiprocessing issues

        # Metric computation frequency (more frequent for testing)
        repr_metric_every_n_steps=50,   # More frequent for testing
        optim_metric_every_n_epochs=1,
        checkpoint_epochs=[1, 3],  # Checkpoint at end of training

        # Seed for reproducibility
        seed=42,
    )

def create_full_config():
    """Create a full configuration for the complete experiment using HuggingFace datasets."""
    return ExperimentConfig(
        # Dataset settings - using HuggingFace datasets
        dataset_name="JotDe/mscoco_50k",  # HuggingFace dataset
        coco_root="/path/to/coco2017",  # Kept for compatibility
        output_dir="./phase1_full_results",

        # Dataset limits (None = use all samples)
        max_train_samples=None,  # Use all available training samples
        max_val_samples=None,    # Use all available validation samples

        # Model settings
        vision_encoder="resnet18",  # or "resnet34", "vit_tiny"
        text_encoder="distilbert",  # or "bert_tiny"
        embedding_dim=256,

        # Training settings (full experiment)
        batch_size=64,  # Adjust based on your GPU memory
        num_epochs=10,   # Slightly reduced for efficiency
        learning_rate=1e-4,

        # All misalignment ratios
        misalignment_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],

        # Device settings
        device="mps",  # Change to "cuda" for NVIDIA GPUs
        num_workers=0,  # Use single-threaded loading to avoid multiprocessing issues

        # Standard metric computation frequency
        repr_metric_every_n_steps=500,
        optim_metric_every_n_epochs=1,
        checkpoint_epochs=[3, 6, 9, 10],  # Adjusted for 10 epochs

        # Seed for reproducibility
        seed=42,
    )

def main():
    """Run the Phase 1 experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 1 Misalignment Experiment")
    parser.add_argument(
        "--mode",
        choices=["small", "full"],
        default="small",
        help="Experiment size: small (quick test) or full (complete experiment)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="JotDe/mscoco_50k",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (None = all)"
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Maximum number of validation samples to use (None = all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides default)"
    )

    args = parser.parse_args()

    # Create configuration
    if args.mode == "small":
        config = create_small_config()
        print("Running SMALL experiment (quick test)")
    else:
        config = create_full_config()
        print("Running FULL experiment")

    # Override with command line arguments
    config.dataset_name = args.dataset
    config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.max_val_samples is not None:
        config.max_val_samples = args.max_val_samples

    print(f"Configuration:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max train samples: {config.max_train_samples}")
    print(f"  Max val samples: {config.max_val_samples}")
    print(f"  Misalignment ratios: {config.misalignment_ratios}")
    print(f"  Output directory: {config.output_dir}")

    print(f"\n📥 Using HuggingFace dataset: {config.dataset_name}")
    print("   The dataset will be downloaded automatically on first run.")

    # Set environment variable for MPS fallback
    if config.device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("✅ Enabled MPS fallback for eigenvalue operations")

    print(f"\n🚀 Starting Phase 1 experiment...")

    try:
        results = run_phase1_experiment(config)
        print(f"\n✅ Experiment completed successfully!")
        print(f"📊 Results saved to: {config.output_dir}")
        print(f"📈 Check the figures directory for visualization plots")

    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()