#!/usr/bin/env python3
"""
Test the new feature extractor approach in isolation
"""
import torch
import timm
from torchvision import datasets, transforms
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    print(f"🔍 Memory: {mem.percent:.1f}% used ({mem.available / 1024**3:.1f}GB available)")

def test_feature_extractor():
    print("=== Testing Feature Extractor Approach ===")
    check_memory()
    
    device = 'mps'
    
    # Load model
    model = timm.create_model('resnet18', pretrained=False, num_classes=10)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Try feature extractor
    try:
        from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
        print("✓ Feature extraction imports available")
        
        # Get nodes
        _, eval_nodes = get_graph_node_names(model)
        print(f"✓ Found {len(eval_nodes)} nodes")
        print(f"First 10 nodes: {eval_nodes[:10]}")
        
        # Select minimal nodes
        return_nodes = {}
        key_patterns = ['layer1', 'layer2', 'layer3', 'layer4']
        
        for node in eval_nodes:
            for pattern in key_patterns:
                if pattern in node and len(return_nodes) < 4:
                    return_nodes[node] = f"{pattern}_{len(return_nodes)}"
                    break
        
        print(f"✓ Selected nodes: {return_nodes}")
        
        # Create extractor
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        print("✓ Feature extractor created")
        
        # Test with dummy data
        x = torch.randn(2, 3, 32, 32, device=device)
        print("✓ Created test input")
        
        with torch.no_grad():
            features = feature_extractor(x)
            print(f"✓ Extracted features: {list(features.keys())}")
            
            for name, feat in features.items():
                print(f"  {name}: {feat.shape}")
                
                # Move to CPU and flatten
                feat_cpu = feat.cpu()
                if len(feat_cpu.shape) > 2:
                    feat_flat = feat_cpu.view(feat_cpu.shape[0], -1)
                    print(f"    Flattened: {feat_flat.shape}")
        
        torch.mps.empty_cache()
        check_memory()
        print("✅ Feature extractor test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Feature extractor failed: {e}")
        return test_timm_fallback(model, device)

def test_timm_fallback(model, device):
    print("\n=== Testing Timm Fallback ===")
    
    try:
        x = torch.randn(2, 3, 32, 32, device=device)
        
        if hasattr(model, 'forward_features'):
            print("✓ Model has forward_features")
            with torch.no_grad():
                features = model.forward_features(x)
                print(f"✓ Got features: {features.shape}")
                
                feat_cpu = features.cpu()
                if len(feat_cpu.shape) > 2:
                    feat_flat = feat_cpu.view(feat_cpu.shape[0], -1)
                    print(f"✓ Flattened: {feat_flat.shape}")
        else:
            print("✓ Using regular forward")
            with torch.no_grad():
                output = model(x)
                print(f"✓ Got output: {output.shape}")
        
        torch.mps.empty_cache()
        check_memory()
        print("✅ Timm fallback test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Timm fallback failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing new feature extraction approaches")
    test_feature_extractor()