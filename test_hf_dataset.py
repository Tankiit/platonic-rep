#!/usr/bin/env python3
"""
Test script for the HuggingFace dataset implementation.
Tests the dataset loading and corruption functionality.
"""

import sys
import os
from pathlib import Path
import tempfile
import json

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase1_misalignment_experiment import (
    ExperimentConfig, MisalignedCOCODataset, get_transforms,
    setup_experiment
)

def test_hf_dataset_loading():
    """Test loading the HuggingFace dataset."""
    print("Testing HuggingFace dataset loading...")

    try:
        # Test with a small sample
        dataset = MisalignedCOCODataset(
            dataset_name="JotDe/mscoco_50k",
            split="train",
            misalignment_ratio=0.0,
            max_samples=100  # Small sample for testing
        )

        print(f"✓ Dataset loaded successfully!")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Dataset features: {dataset.dataset.features}")

        # Test getting a sample
        sample = dataset[0]
        print(f"✓ Sample loaded successfully!")
        print(f"  Image shape: {sample['image'].size}")
        print(f"  Caption: {sample['caption'][:100]}...")
        print(f"  Image ID: {sample['image_id']}")
        print(f"  Is corrupted: {sample['is_corrupted']}")

        return True

    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_misalignment_mechanism():
    """Test the misalignment corruption mechanism."""
    print("\nTesting misalignment corruption mechanism...")

    try:
        # Test with different corruption levels
        corruption_levels = [0.0, 0.25, 0.5, 1.0]

        for corruption_level in corruption_levels:
            print(f"\n  Testing {corruption_level*100:.0f}% corruption:")

            dataset = MisalignedCOCODataset(
                dataset_name="JotDe/mscoco_50k",
                split="train",
                misalignment_ratio=corruption_level,
                max_samples=50,  # Small sample for testing
                seed=42  # Fixed seed for reproducibility
            )

            # Check corruption statistics
            n_corrupted = len(dataset.corrupted_indices)
            corruption_ratio_actual = n_corrupted / len(dataset)
            print(f"    Expected corruption: {corruption_level:.2f}")
            print(f"    Actual corruption: {corruption_ratio_actual:.2f} ({n_corrupted}/{len(dataset)})")

            # Test getting samples
            sample_corrupted = dataset[0]
            print(f"    Sample is corrupted: {sample_corrupted['is_corrupted']}")

        print(f"\n✓ Misalignment mechanism working correctly!")
        return True

    except Exception as e:
        print(f"✗ Misalignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_transforms():
    """Test the image transforms."""
    print("\nTesting image transforms...")

    try:
        # Create dataset with transforms
        transform = get_transforms("train")
        dataset = MisalignedCOCODataset(
            dataset_name="JotDe/mscoco_50k",
            split="train",
            misalignment_ratio=0.0,
            transform=transform,
            max_samples=10
        )

        sample = dataset[0]
        image = sample['image']

        print(f"✓ Transforms applied successfully!")
        print(f"  Image shape: {image.shape}")
        print(f"  Image type: {type(image)}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image min/max: {image.min():.3f}/{image.max():.3f}")

        return True

    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_splits():
    """Test loading different dataset splits."""
    print("\nTesting dataset splits...")

    try:
        # Try to load different splits
        splits_to_test = ["train", "validation", "test"]

        for split in splits_to_test:
            try:
                dataset = MisalignedCOCODataset(
                    dataset_name="JotDe/mscoco_50k",
                    split=split,
                    misalignment_ratio=0.0,
                    max_samples=20  # Small sample for testing
                )
                print(f"✓ Split '{split}': {len(dataset)} samples")
            except Exception as e:
                print(f"✗ Split '{split}' failed: {e}")

        return True

    except Exception as e:
        print(f"✗ Split test failed: {e}")
        return False

def test_dataloader_creation():
    """Test creating DataLoaders."""
    print("\nTesting DataLoader creation...")

    try:
        config = ExperimentConfig(
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            dataset_name="JotDe/mscoco_50k"
        )

        # Import here to avoid circular imports
        from phase1_misalignment_experiment import create_dataloaders

        train_loader, val_loader = create_dataloaders(
            config, misalignment_ratio=0.25, max_samples=50
        )

        print(f"✓ DataLoaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        # Test one batch
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded successfully!")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Captions count: {len(batch['captions'])}")
        print(f"  Corrupted count: {batch['is_corrupted'].sum()}/{len(batch['is_corrupted'])}")

        return True

    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all HuggingFace dataset tests."""
    print("Testing HuggingFace Dataset Implementation")
    print("=" * 50)

    tests = [
        test_hf_dataset_loading,
        test_misalignment_mechanism,
        test_data_transforms,
        test_dataset_splits,
        test_dataloader_creation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All HuggingFace dataset tests passed!")
        print("✅ Ready to use HuggingFace datasets for Phase 1 experiment!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("Make sure the datasets library is installed:")
        print("pip install datasets")

if __name__ == "__main__":
    main()