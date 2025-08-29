#!/usr/bin/env python3
"""
Test script for multi-model analysis system
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

def test_feature_extraction():
    """Test feature extraction functionality"""
    print("Testing feature extraction...")
    
    # Create dummy features data
    num_samples = 100
    num_layers = 4
    feature_dim = 128
    
    features = torch.randn(num_samples, num_layers, feature_dim)
    layer_names = [f'layer_{i}' for i in range(num_layers)]
    targets = torch.randint(0, 10, (num_samples,))
    
    features_data = {
        'feats': features,
        'layer_names': layer_names,
        'targets': targets
    }
    
    print(f"✓ Created dummy features: {features.shape}")
    return features_data

def test_analysis_classes():
    """Test that analysis classes can be imported and used"""
    print("Testing analysis classes...")
    
    try:
        from macroscopic import MacroscopicAnalysis
        from mesoscopic import MesoscopicAnalysis
        print("✓ Successfully imported analysis classes")
        
        # Test macroscopic analysis
        macro_analyzer = MacroscopicAnalysis()
        print("✓ Created macroscopic analyzer")
        
        # Test mesoscopic analysis
        meso_analyzer = MesoscopicAnalysis()
        print("✓ Created mesoscopic analyzer")
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing analysis classes: {e}")
        return False

def test_multi_model_analyzer():
    """Test multi-model analyzer creation"""
    print("Testing multi-model analyzer...")
    
    try:
        from multi_model_analysis import MultiModelAnalyzer
        
        # Create analyzer with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = MultiModelAnalyzer(output_dir=temp_dir)
            print("✓ Created multi-model analyzer")
            
            # Test model availability
            available_models = analyzer.get_available_models()
            print(f"✓ Found {len(available_models)} available models")
            
            # Test dataset configs
            dataset_configs = analyzer.dataset_configs
            print(f"✓ Configured {len(dataset_configs)} datasets")
            
            return True
            
    except Exception as e:
        print(f"✗ Error creating multi-model analyzer: {e}")
        return False

def test_timm_integration():
    """Test timm library integration"""
    print("Testing timm integration...")
    
    try:
        import timm
        
        # Check if key models are available
        key_models = ['resnet18', 'vit_base_patch16_224', 'convnext_tiny']
        available_models = timm.list_models()
        
        found_models = []
        for model in key_models:
            if model in available_models:
                found_models.append(model)
        
        print(f"✓ Found {len(found_models)}/{len(key_models)} key models in timm")
        print(f"  Available: {found_models}")
        
        if len(found_models) > 0:
            # Test model creation
            model = timm.create_model(found_models[0], pretrained=False, num_classes=10)
            print(f"✓ Successfully created {found_models[0]} model")
            return True
        else:
            print("✗ No key models found in timm")
            return False
            
    except Exception as e:
        print(f"✗ Error with timm integration: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("Testing dataset loading...")
    
    try:
        from torchvision import datasets, transforms
        
        # Test CIFAR-10 loading
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Try to load from existing data directory
        dataset = datasets.CIFAR10('/Users/tanmoy/research/data', train=True, download=False, transform=transform)
        print(f"✓ Successfully loaded CIFAR-10 dataset with {len(dataset)} samples")
        
        # Test data loader creation
        from torch.utils.data import DataLoader
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Test one batch
        for batch_idx, (data, targets) in enumerate(data_loader):
            print(f"✓ Successfully loaded batch {batch_idx}: {data.shape}, {targets.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ Error with dataset loading: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Testing Multi-Model Analysis System ===\n")
    
    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Analysis Classes", test_analysis_classes),
        ("Multi-Model Analyzer", test_multi_model_analyzer),
        ("TIMM Integration", test_timm_integration),
        ("Dataset Loading", test_dataset_loading)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python example_multi_model_analysis.py")
        print("2. Or: python multi_model_analysis.py --models resnet18 --datasets cifar10")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
