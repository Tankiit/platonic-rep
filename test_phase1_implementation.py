#!/usr/bin/env python3
"""
Quick test script for Phase 1 implementation.
Tests core functionality without requiring the full COCO dataset.
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import json
import sys
import os

# Add the current directory to path so we can import the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase1_misalignment_experiment import (
    ExperimentConfig, setup_experiment, CLIPModel, MetricComputer,
    VisionEncoder, TextEncoder, get_transforms
)

class DummyCOCODataset:
    """Dummy dataset for testing without actual COCO data."""

    def __init__(self, num_samples=100, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.rng = np.random.RandomState(42)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random RGB image
        image = torch.rand(3, 224, 224)
        if self.transform:
            image = self.transform(image)

        # Create dummy caption
        caption = f"This is image {idx}"

        return {
            'image': image,
            'caption': caption,
            'image_id': idx,
            'is_corrupted': False
        }

def test_basic_functionality():
    """Test basic model and metric functionality."""
    print("Testing basic functionality...")

    # Test device availability
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create small model for testing
    model = CLIPModel(
        vision_encoder="resnet18",
        text_encoder="distilbert",
        embedding_dim=128,
        temperature=0.07
    )
    model.to(device)
    model.eval()

    # Test forward pass
    with torch.no_grad():
        dummy_images = torch.rand(4, 3, 224, 224).to(device)
        dummy_texts = ["test caption 1", "test caption 2", "test caption 3", "test caption 4"]

        img_emb, txt_emb, logit_scale = model(dummy_images, dummy_texts, device)

        print(f"Image embeddings shape: {img_emb.shape}")
        print(f"Text embeddings shape: {txt_emb.shape}")
        print(f"Logit scale: {logit_scale.item()}")

        # Test loss computation
        loss = model.compute_loss(img_emb, txt_emb, logit_scale)
        print(f"CLIP loss: {loss.item():.4f}")

    print("✓ Basic functionality test passed")
    return True

def test_metrics():
    """Test metric computation."""
    print("\nTesting metric computation...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Create model and metric computer
    model = CLIPModel(embedding_dim=128)
    model.to(device)
    metric_computer = MetricComputer(model, device)

    # Create dummy embeddings
    batch_size = 20
    vision_emb = torch.randn(batch_size, 128).to(device)
    text_emb = torch.randn(batch_size, 128).to(device)

    # Normalize
    vision_emb = torch.nn.functional.normalize(vision_emb, p=2, dim=-1)
    text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)

    # Test CKA
    cka = metric_computer.compute_cka(vision_emb, text_emb)
    print(f"CKA similarity: {cka:.3f}")

    # Test mutual kNN
    mutual_knn = metric_computer.compute_mutual_knn(vision_emb, text_emb, k=5)
    print(f"Mutual kNN (k=5): {mutual_knn:.3f}")

    # Test AGOP
    dummy_grads_v = torch.randn(1000).to(device)
    dummy_grads_t = torch.randn(1000).to(device)

    metric_computer.accumulate_agop(dummy_grads_v, dummy_grads_t)
    agop_metrics = metric_computer.compute_agop_metrics()

    if agop_metrics:
        print(f"AGOP vision rank: {agop_metrics.get('vision_agop_rank', 'N/A')}")
        print(f"AGOP text rank: {agop_metrics.get('text_agop_rank', 'N/A')}")

    print("✓ Metric computation test passed")
    return True

def test_config_and_setup():
    """Test configuration and experiment setup."""
    print("\nTesting configuration and setup...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ExperimentConfig(
            coco_root="/fake/path",  # Won't be used in test
            output_dir=temp_dir,
            device="cpu",  # Use CPU for testing
            batch_size=8,
            num_epochs=2,
            misalignment_ratios=[0.0, 0.5]  # Test only two conditions
        )

        # Test setup
        setup_experiment(config)

        # Check if directories were created
        output_path = Path(temp_dir)
        assert output_path.exists(), "Output directory not created"
        assert (output_path / "checkpoints").exists(), "Checkpoints directory not created"
        assert (output_path / "metrics").exists(), "Metrics directory not created"
        assert (output_path / "figures").exists(), "Figures directory not created"
        assert (output_path / "experiment.log").exists(), "Log file not created"

        print("✓ Configuration and setup test passed")
        return True

def test_text_encoder():
    """Test text encoder separately."""
    print("\nTesting text encoder...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        encoder = TextEncoder(model_name="distilbert", embedding_dim=128)
        encoder.to(device)
        encoder.eval()

        with torch.no_grad():
            texts = ["A cat sitting on a mat", "A dog running in the park"]
            embeddings = encoder(texts, device)

            print(f"Text embeddings shape: {embeddings.shape}")
            print(f"Embeddings norm: {embeddings.norm(dim=-1).mean():.3f}")

        print("✓ Text encoder test passed")
        return True

    except Exception as e:
        print(f"✗ Text encoder test failed: {e}")
        print("Note: This requires transformers library to be installed")
        return False

def test_vision_encoder():
    """Test vision encoder separately."""
    print("\nTesting vision encoder...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    encoder = VisionEncoder(backbone="resnet18", embedding_dim=128)
    encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        dummy_images = torch.rand(4, 3, 224, 224).to(device)
        embeddings = encoder(dummy_images)

        print(f"Vision embeddings shape: {embeddings.shape}")
        print(f"Embeddings norm: {embeddings.norm(dim=-1).mean():.3f}")

    print("✓ Vision encoder test passed")
    return True

def main():
    """Run all tests."""
    print("Running Phase 1 Implementation Tests")
    print("=" * 50)

    tests = [
        test_config_and_setup,
        test_vision_encoder,
        test_text_encoder,
        test_basic_functionality,
        test_metrics,
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
        print("🎉 All tests passed! Implementation is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("Make sure all required dependencies are installed:")
        print("pip install -r requirements_phase1.txt")

if __name__ == "__main__":
    main()