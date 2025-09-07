#!/usr/bin/env python3
"""
Minimal MPS debugging script to find the exact crash point
"""
import torch
import torch.nn as nn
import numpy as np
import timm
from torchvision import datasets, transforms
import psutil
import gc

def check_memory():
    """Check memory usage"""
    mem = psutil.virtual_memory()
    print(f"🔍 Memory: {mem.percent:.1f}% used ({mem.available / 1024**3:.1f}GB available)")

def test_basic_mps():
    """Test 1: Basic MPS operations"""
    print("=== TEST 1: Basic MPS Operations ===")
    check_memory()
    
    try:
        device = 'mps'
        x = torch.randn(2, 3, 32, 32, device=device)
        print(f"✓ Created tensor on MPS: {x.shape}")
        
        x_cpu = x.cpu()
        print(f"✓ Moved to CPU: {x_cpu.shape}")
        
        torch.mps.empty_cache()
        print("✓ Cleared MPS cache")
        
        del x, x_cpu
        gc.collect()
        check_memory()
        
    except Exception as e:
        print(f"❌ Basic MPS test failed: {e}")
        return False
    
    return True

def test_model_forward():
    """Test 2: Model forward pass"""
    print("\n=== TEST 2: Model Forward Pass ===")
    check_memory()
    
    try:
        device = 'mps'
        model = timm.create_model('resnet18', pretrained=False, num_classes=10)
        model = model.to(device)
        model.eval()
        print("✓ Model loaded on MPS")
        
        x = torch.randn(2, 3, 32, 32, device=device)
        print("✓ Input created")
        
        with torch.no_grad():
            output = model(x)
            print(f"✓ Forward pass: {output.shape}")
        
        torch.mps.empty_cache()
        print("✓ Cleared cache")
        
        del model, x, output
        gc.collect()
        check_memory()
        
    except Exception as e:
        print(f"❌ Model forward test failed: {e}")
        return False
    
    return True

def test_hook_registration():
    """Test 3: Hook registration and execution"""
    print("\n=== TEST 3: Hook Registration ===")
    check_memory()
    
    try:
        device = 'mps'
        model = timm.create_model('resnet18', pretrained=False, num_classes=10)
        model = model.to(device)
        model.eval()
        
        representations = {}
        hooks = []
        
        def get_hook(name):
            def hook(module, input, output):
                print(f"🔍 Hook triggered: {name}, shape: {output.shape}")
                # Immediate processing - no accumulation
                rep = output.detach()[:1]  # Only first sample
                if len(rep.shape) > 2:
                    rep = rep.view(rep.shape[0], -1)
                representations[name] = rep.cpu()
                print(f"🔍 Processed {name}: {rep.shape} -> CPU")
            return hook
        
        # Register only a few hooks
        hook_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and hook_count < 5:
                hooks.append(module.register_forward_hook(get_hook(name)))
                hook_count += 1
        
        print(f"✓ Registered {len(hooks)} hooks")
        
        x = torch.randn(2, 3, 32, 32, device=device)
        with torch.no_grad():
            _ = model(x)
        
        print(f"✓ Forward pass with hooks completed")
        print(f"✓ Captured {len(representations)} representations")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        torch.mps.empty_cache()
        del model, x, representations, hooks
        gc.collect()
        check_memory()
        
    except Exception as e:
        print(f"❌ Hook test failed: {e}")
        return False
    
    return True

def test_data_loader():
    """Test 4: Data loader iteration"""
    print("\n=== TEST 4: Data Loader ===")
    check_memory()
    
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.CIFAR10(
            root='/Users/tanmoy/research/data',
            train=False,
            download=False,
            transform=transform
        )
        
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=0
        )
        
        print("✓ Data loader created")
        
        for i, (x, y) in enumerate(data_loader):
            print(f"🔍 Batch {i}: {x.shape}")
            if i >= 2:  # Only process 3 batches
                break
            
            # Move to MPS
            x = x.to('mps')
            print(f"🔍 Moved to MPS: {x.shape}")
            
            # Move back to CPU
            x = x.cpu()
            print(f"🔍 Moved to CPU: {x.shape}")
            
            torch.mps.empty_cache()
            del x, y
        
        print("✓ Data loader test completed")
        check_memory()
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False
    
    return True

def test_numpy_operations():
    """Test 5: Numpy operations on representations"""
    print("\n=== TEST 5: Numpy Operations ===")
    check_memory()
    
    try:
        # Create small tensor
        x = torch.randn(10, 16)
        print(f"✓ Created tensor: {x.shape}")
        
        # Convert to numpy
        x_np = x.numpy()
        print(f"✓ Converted to numpy: {x_np.shape}")
        
        # Basic operations
        mean_val = np.mean(x_np)
        std_val = np.std(x_np)
        sparsity = np.mean(x_np == 0)
        
        print(f"✓ Computed stats: mean={mean_val:.3f}, std={std_val:.3f}, sparsity={sparsity:.3f}")
        
        del x, x_np
        gc.collect()
        check_memory()
        
    except Exception as e:
        print(f"❌ Numpy test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🔍 MPS Debug Test Suite")
    print(f"🔍 PyTorch version: {torch.__version__}")
    print(f"🔍 MPS available: {torch.backends.mps.is_available()}")
    
    check_memory()
    
    tests = [
        test_basic_mps,
        test_model_forward,
        test_hook_registration,
        test_data_loader,
        test_numpy_operations
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*50}")
        result = test()
        if not result:
            print(f"❌ Test {i} failed - stopping")
            break
        print(f"✅ Test {i} passed")
        
        # Force cleanup between tests
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    print(f"\n{'='*50}")
    print("🔍 Debug test suite completed")
    check_memory()

if __name__ == "__main__":
    main()