import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
from scipy.stats import entropy, wasserstein_distance
from typing import Dict, List, Tuple, Optional, Callable
import copy
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import warnings
import argparse
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Try to import timm, fall back gracefully if not available
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")

# Try to import thingsvision, fall back gracefully if not available
try:
    import thingsvision
    THINGSVISION_AVAILABLE = True
except ImportError:
    THINGSVISION_AVAILABLE = False
    print("Warning: thingsvision not available. Install with: pip install thingsvision")
    print("         thingsvision provides access to THINGS dataset and many pre-trained models.")
    print("         You can still use built-in architectures and other datasets.")

warnings.filterwarnings('ignore')

# Global tensorboard writer
tensorboard_writer = None

def check_device_compatibility(device):
    """Check if the requested device is available and compatible"""
    if device == 'cuda':
        if not torch.cuda.is_available():
            return False, "CUDA is not available on this system"
        return True, f"CUDA available: {torch.cuda.get_device_name()}"
    
    elif device == 'mps':
        if not hasattr(torch.backends, 'mps'):
            return False, "MPS backend not available (PyTorch version too old)"
        if not torch.backends.mps.is_available():
            return False, "MPS not available on this system"
        if not torch.backends.mps.is_built():
            return False, "MPS not built with this PyTorch installation"
        return True, "MPS available for Apple Silicon"
    
    elif device == 'cpu':
        return True, "CPU available"
    
    return False, f"Unknown device: {device}"

def get_device_info(device):
    """Get detailed information about the device"""
    if device == 'cuda':
        return {
            'name': torch.cuda.get_device_name(),
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
            'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,  # GB
            'memory_cached': torch.cuda.memory_reserved(0) / 1024**3,  # GB
        }
    elif device == 'mps':
        return {
            'name': 'Apple Silicon MPS',
            'memory_total': 'Unknown (MPS)',  # MPS doesn't expose memory info
            'memory_allocated': 'Unknown (MPS)',
            'memory_cached': 'Unknown (MPS)',
        }
    elif device == 'cpu':
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'name': 'CPU',
                'memory_total': memory.total / 1024**3,  # GB
                'memory_available': memory.available / 1024**3,  # GB
                'memory_percent': memory.percent,
            }
        except ImportError:
            return {
                'name': 'CPU',
                'memory_total': 'Unknown (psutil not available)',
                'memory_available': 'Unknown (psutil not available)',
                'memory_percent': 'Unknown (psutil not available)',
            }
    return {'name': 'Unknown', 'error': 'Device not recognized'}

def get_optimal_batch_size(device, base_batch_size=128):
    """Get optimal batch size for the device"""
    if device == 'mps':
        # MPS works well with moderate batch sizes
        return min(base_batch_size, 256)
    elif device == 'cuda':
        # CUDA can handle larger batches
        return min(base_batch_size, 512)
    else:
        # CPU works best with smaller batches
        return min(base_batch_size, 64)

def get_device_considerations(device):
    """Get device-specific considerations and tips"""
    considerations = []
    
    if device == 'mps':
        considerations.extend([
            "MPS provides good acceleration for Apple Silicon Macs",
            "Batch normalization may behave differently on MPS",
            "Some operations might fall back to CPU automatically",
            "Memory management is handled by the system",
            "Use float32 for best compatibility (avoid float16 on MPS)",
            "Some complex operations may be slower than CUDA"
        ])
    elif device == 'cuda':
        considerations.extend([
            "CUDA provides excellent acceleration for NVIDIA GPUs",
            "Monitor GPU memory usage to avoid OOM errors",
            "Use mixed precision training for better memory efficiency",
            "Consider gradient checkpointing for large models"
        ])
    elif device == 'cpu':
        considerations.extend([
            "CPU training is slower but more reliable",
            "Use smaller batch sizes for better memory efficiency",
            "Consider reducing model complexity for faster training",
            "Multi-threading can improve performance"
        ])
    
    return considerations

def get_training_considerations(epochs):
    """Get training-specific considerations based on number of epochs"""
    considerations = []
    
    if epochs >= 100:
        considerations.extend([
            f"Long training run ({epochs} epochs) - consider using early stopping",
            "Monitor for overfitting and save best models",
            "Use learning rate scheduling for better convergence",
            "Consider checkpointing models every 10-20 epochs"
        ])
    elif epochs >= 50:
        considerations.extend([
            f"Medium training run ({epochs} epochs) - monitor convergence",
            "Save intermediate checkpoints for analysis"
        ])
    else:
        considerations.extend([
            f"Quick training run ({epochs} epochs) - suitable for exploration"
        ])
    
    return considerations

def list_timm_models(pattern=None, limit=20):
    """List available timm models, optionally filtered by pattern"""
    if not TIMM_AVAILABLE:
        return []
    
    try:
        all_models = timm.list_models()
        if pattern:
            all_models = [m for m in all_models if pattern.lower() in m.lower()]
        
        # Group by model family
        model_families = {}
        for model in all_models[:limit]:
            family = model.split('_')[0] if '_' in model else 'other'
            if family not in model_families:
                model_families[family] = []
            model_families[family].append(model)
        
        return model_families
    except Exception as e:
        print(f"Error listing timm models: {e}")
        return []

def get_timm_model_info(model_name):
    """Get information about a specific timm model"""
    if not TIMM_AVAILABLE:
        return None
    
    try:
        # Try to get model info - timm API may vary by version
        model_info = {}
        
        # Try different ways to get model info
        try:
            # Newer timm versions
            if hasattr(timm.models, 'registry') and hasattr(timm.models.registry, 'model_entry'):
                info = timm.models.registry.model_entry(model_name)
                if info:
                    model_info.update(info)
        except:
            pass
        
        try:
            # Alternative approach - create model temporarily to get info
            temp_model = timm.create_model(model_name, pretrained=False, num_classes=1000)
            if hasattr(temp_model, 'num_features'):
                model_info['features'] = temp_model.num_features
            # Estimate parameters
            total_params = sum(p.numel() for p in temp_model.parameters())
            model_info['parameters'] = f"{total_params / 1e6:.1f}"
            del temp_model
        except:
            pass
        
        # Set defaults
        model_info.setdefault('name', model_name)
        model_info.setdefault('family', model_name.split('_')[0] if '_' in model_name else 'unknown')
        model_info.setdefault('parameters', 'unknown')
        
        return model_info
        
    except Exception as e:
        print(f"Error getting model info for {model_name}: {e}")
    
    return None

def get_popular_vit_models():
    """Get list of popular ViT models from timm"""
    if not TIMM_AVAILABLE:
        return []
    
    try:
        all_models = timm.list_models()
        vit_models = [m for m in all_models if 'vit' in m.lower() and 'patch' in m.lower()]
        
        # Sort by size (tiny -> small -> base -> large)
        size_order = {'tiny': 0, 'small': 1, 'base': 2, 'large': 3, 'huge': 4}
        
        def sort_key(model):
            for size, order in size_order.items():
                if size in model.lower():
                    return order
            return 5  # Default for unknown sizes
        
        vit_models.sort(key=sort_key)
        return vit_models[:20]  # Return top 20
    except Exception as e:
        print(f"Error getting ViT models: {e}")
        return []

def get_thingsvision_models():
    """Get list of available models from thingsvision"""
    if not THINGSVISION_AVAILABLE:
        return []
    
    try:
        from thingsvision import get_model
        from thingsvision.models import get_available_models
        
        # Get available models from thingsvision
        available_models = get_available_models()
        
        # Filter and categorize models
        model_categories = {
            'torchvision': [],
            'timm': [],
            'keras': [],
            'clip': [],
            'other': []
        }
        
        for model_name in available_models:
            if 'clip' in model_name.lower():
                model_categories['clip'].append(model_name)
            elif any(source in model_name.lower() for source in ['resnet', 'vgg', 'alexnet', 'inception']):
                model_categories['torchvision'].append(model_name)
            elif any(source in model_name.lower() for source in ['vit', 'efficientnet', 'densenet']):
                model_categories['timm'].append(model_name)
            elif any(source in model_name.lower() for source in ['vgg', 'resnet']):
                model_categories['keras'].append(model_name)
            else:
                model_categories['other'].append(model_name)
        
        return model_categories
        
    except Exception as e:
        print(f"Error getting thingsvision models: {e}")
        return {}

def safe_to_device(tensor, device):
    """Safely move tensor to device with MPS-specific handling"""
    try:
        if device == 'mps':
            # Ensure tensor is float32 for MPS compatibility
            if tensor.dtype == torch.float16:
                tensor = tensor.float()
            return tensor.to(device)
        else:
            return tensor.to(device)
    except Exception as e:
        print(f"Warning: Failed to move tensor to {device}, falling back to CPU: {e}")
        return tensor.cpu()

def setup_output_dirs(architecture, dataset_name, experiment_name, timestamp=None):
    """Setup organized output directories"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base output directory
    base_dir = "outputs"
    
    # Create structured directory hierarchy
    output_dir = os.path.join(base_dir, dataset_name, architecture, experiment_name, timestamp)
    plots_dir = os.path.join(output_dir, "plots")
    metrics_dir = os.path.join(output_dir, "metrics")
    models_dir = os.path.join(output_dir, "models")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    return output_dir, plots_dir, metrics_dir, models_dir

def setup_tensorboard(log_dir=None, architecture=None, dataset_name=None):
    """Setup tensorboard logging with organized structure"""
    global tensorboard_writer
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if architecture and dataset_name:
            log_dir = f"runs/{dataset_name}/{architecture}/geometric_alignment_{timestamp}"
        else:
            log_dir = f"runs/geometric_alignment_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir)
    print(f"Tensorboard logging to: {log_dir}")
    return tensorboard_writer

def log_metrics(metrics, step, prefix=""):
    """Log metrics to tensorboard"""
    global tensorboard_writer
    if tensorboard_writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensorboard_writer.add_scalar(f"{prefix}{key}", value, step)
            elif isinstance(value, dict):
                log_metrics(value, step, f"{prefix}{key}/")

def save_metrics(metrics, filepath, indent=2):
    """Save metrics to JSON file"""
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    try:
        # Convert numpy types
        metrics_serializable = convert_numpy(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=indent)
        print(f"Metrics saved to: {filepath}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

def save_experiment_summary(summary, filepath, indent=2):
    """Save experiment summary to JSON file"""
    import json
    
    try:
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=indent)
        print(f"Experiment summary saved to: {filepath}")
    except Exception as e:
        print(f"Error saving experiment summary: {e}")

def load_dataset(dataset_name, data_dir, batch_size, device):
    """Load dataset with appropriate transforms and settings"""
    from torchvision import datasets, transforms
    import os
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Dataset-specific configurations
    dataset_configs = {
        'cifar10': {
            'train_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'test_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'num_classes': 10,
            'in_channels': 3,
            'image_size': 32
        },
        'cifar100': {
            'train_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]),
            'test_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]),
            'num_classes': 100,
            'in_channels': 3,
            'image_size': 32
        },
        'mnist': {
            'train_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'test_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'num_classes': 10,
            'in_channels': 1,
            'image_size': 28
        },
        'fashion_mnist': {
            'train_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            'test_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            'num_classes': 10,
            'in_channels': 1,
            'image_size': 28
        },
        'stl10': {
            'train_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            'test_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            'num_classes': 10,
            'in_channels': 3,
            'image_size': 96
        },
        'svhn': {
            'train_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ]),
            'test_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ]),
            'num_classes': 10,
            'in_channels': 3,
            'image_size': 32
        },
        'things': {
            'train_transform': transforms.Compose([
                transforms.Resize(224),  # Standard size for many pre-trained models
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
            ]),
            'test_transform': transforms.Compose([
                transforms.Resize(224),  # Standard size for many pre-trained models
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
            ]),
            'num_classes': 1854,  # THINGS has 1854 unique concepts
            'in_channels': 3,
            'image_size': 224,
            'requires_thingsvision': True
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    config = dataset_configs[dataset_name]
    
    # Load dataset
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=config['train_transform'])
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=config['test_transform'])
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=config['train_transform'])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=config['test_transform'])
    elif dataset_name == 'mnist':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=config['train_transform'])
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=config['test_transform'])
    elif dataset_name == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=config['train_transform'])
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=config['test_transform'])
    elif dataset_name == 'stl10':
        train_dataset = datasets.STL10(data_dir, split='train', download=True, transform=config['train_transform'])
        test_dataset = datasets.STL10(data_dir, split='test', download=True, transform=config['train_transform'])
    elif dataset_name == 'svhn':
        train_dataset = datasets.SVHN(data_dir, split='train', download=True, transform=config['train_transform'])
        test_dataset = datasets.SVHN(data_dir, split='test', download=True, transform=config['test_transform'])
    elif dataset_name == 'things':
        if not THINGSVISION_AVAILABLE:
            raise ImportError("thingsvision not available. Install with: pip install thingsvision")
        
        try:
            # Load THINGS dataset using thingsvision
            from thingsvision import get_dataset
            train_dataset = get_dataset('THINGS', transform=config['train_transform'], split='train')
            test_dataset = get_dataset('THINGS', transform=config['test_transform'], split='test')
            
            # Add additional dataset info
            config['dataset_type'] = 'things'
            config['description'] = 'THINGS dataset with 1854 object concepts'
            
        except Exception as e:
            print(f"Error loading THINGS dataset: {e}")
            print("Falling back to CIFAR-10...")
            return load_dataset('cifar10', data_dir, batch_size, device)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Store dataset info globally for model creation
    global DATASET_INFO
    DATASET_INFO = {
        'name': dataset_name,
        'num_classes': config['num_classes'],
        'in_channels': config['in_channels'],
        'image_size': config['image_size'],
        'config': config
    }
    
    return train_dataset, test_dataset, train_loader, test_loader

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Geometric Alignment Analysis - Neural Collapse Research Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments with default settings (50 epochs)
  python geometric_alignment.py
  
  # Training duration presets
  python geometric_alignment.py --quick          # 10 epochs
  python geometric_alignment.py --standard       # 50 epochs (default)
  python geometric_alignment.py --thorough       # 100 epochs
  python geometric_alignment.py --extensive      # 200 epochs
  
  # Run only ETF analysis with custom epochs
  python geometric_alignment.py --experiment etf --epochs 100
  
  # Run AGOP intervention with custom parameters
  python geometric_alignment.py --experiment agop --lambda_rank 0.5 --target_rank 5
  
  # Run architecture universality with specific model
  python geometric_alignment.py --experiment universality --architecture resnet
  
  # Use timm models (if available)
  python geometric_alignment.py --architecture timm --timm_model resnet50
  python geometric_alignment.py --architecture timm --timm_model efficientnet_b0
  python geometric_alignment.py --architecture timm --timm_model vit_base_patch16_224
  
  # Use thingsvision models (if available)
  python geometric_alignment.py --architecture thingsvision --dataset things
  python geometric_alignment.py --architecture timm --timm_model resnet50 --dataset things
  
  # Use ViT architectures
  python geometric_alignment.py --architecture vit
  python geometric_alignment.py --architecture timm --timm_model vit_tiny_patch16_224
  python geometric_alignment.py --architecture timm --timm_model vit_small_patch16_224
  python geometric_alignment.py --architecture timm --timm_model vit_base_patch16_224
  python geometric_alignment.py --architecture timm --timm_model vit_large_patch16_224
  
  # Use specific device (MPS for Apple Silicon, CUDA for NVIDIA)
  python geometric_alignment.py --device mps
  python geometric_alignment.py --device cuda
  
  # Use different datasets
  python geometric_alignment.py --dataset mnist
  python geometric_alignment.py --dataset fashion_mnist
  python geometric_alignment.py --dataset cifar100
  python geometric_alignment.py --dataset stl10
  python geometric_alignment.py --dataset svhn
  python geometric_alignment.py --dataset things  # THINGS dataset (requires thingsvision)
  
  # Use custom data directory
  python geometric_alignment.py --data_dir /path/to/your/data
  
  # Disable tensorboard logging
  python geometric_alignment.py --no_tensorboard
  
  # Use specific log directory
  python geometric_alignment.py --log_dir ./my_logs
        """
    )
    
    # Experiment settings
    parser.add_argument('--experiment', type=str, default='all', 
                       choices=['all', 'etf', 'agop', 'universality'],
                       help='Which experiment to run (default: all)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size for training (default: 128)')
    
    # Training duration presets
    parser.add_argument('--quick', action='store_true',
                       help='Quick training: 10 epochs')
    parser.add_argument('--standard', action='store_true',
                       help='Standard training: 50 epochs (default)')
    parser.add_argument('--thorough', action='store_true',
                       help='Thorough training: 100 epochs')
    parser.add_argument('--extensive', action='store_true',
                       help='Extensive training: 200 epochs')
    
    # Model settings
    parser.add_argument('--architecture', type=str, default='convnet',
                       choices=['convnet', 'resnet', 'mlp', 'vit'] + (['timm'] if TIMM_AVAILABLE else []) + (['thingsvision'] if THINGSVISION_AVAILABLE else []),
                       help='Model architecture to use (default: convnet)')
    
    # Timm-specific settings
    if TIMM_AVAILABLE:
        parser.add_argument('--timm_model', type=str, default='resnet18',
                           help='Specific timm model to use (default: resnet18)')
        parser.add_argument('--timm_pretrained', action='store_true', default=True,
                           help='Use pretrained timm models (default: True)')
    
    # AGOP intervention settings
    parser.add_argument('--lambda_rank', type=float, default=0.1,
                       help='AGOP rank constraint strength (default: 0.1)')
    parser.add_argument('--target_rank', type=int, default=10,
                       help='Target rank for AGOP constraint (default: 10)')
    
    # Logging settings
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory for tensorboard logs (default: auto-generated)')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='Disable tensorboard logging')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Data settings
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'imagenet', 'stl10', 'svhn', 'things'],
                       help='Dataset to use (default: cifar10)')
    parser.add_argument('--data_dir', type=str, default='/Users/cril/tanmoy/research/data',
                       help='Directory for datasets (default: /Users/cril/tanmoy/research/data)')
    
    # Device settings
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (cuda/mps/cpu, default: auto-detect)')
    
    # Utility options
    parser.add_argument('--list_timm_models', action='store_true',
                       help='List available timm models and exit')
    parser.add_argument('--timm_pattern', type=str, default=None,
                       help='Pattern to filter timm models when listing')
    parser.add_argument('--list_vit_models', action='store_true',
                       help='List popular ViT models from timm and exit')
    parser.add_argument('--list_thingsvision_models', action='store_true',
                       help='List available models from thingsvision and exit')
    
    return parser.parse_args()

class GeometricAnalyzer:
    """Core toolkit for geometric analysis of neural representations"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics_history = {}
        
    def compute_etf_alignment(self, features: torch.Tensor, labels: torch.Tensor) -> Dict:
        """
        Compute Equiangular Tight Frame alignment metrics
        """
        features = features.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        # Center features
        features_centered = features - features.mean(axis=0)
        
        # Compute class means
        unique_labels = np.unique(labels)
        class_means = []
        for label in unique_labels:
            class_mask = labels == label
            class_means.append(features_centered[class_mask].mean(axis=0))
        
        class_means = np.array(class_means)
        
        # Normalize class means
        class_norms = np.linalg.norm(class_means, axis=1, keepdims=True)
        class_means_norm = class_means / (class_norms + 1e-8)
        
        # Compute Gram matrix
        gram_matrix = class_means_norm @ class_means_norm.T
        
        # ETF ideal: diagonal = 1, off-diagonal = -1/(C-1)
        C = len(unique_labels)
        ideal_off_diag = -1 / (C - 1) if C > 1 else 0
        
        # Extract off-diagonal elements
        off_diag_mask = ~np.eye(C, dtype=bool)
        actual_off_diag = gram_matrix[off_diag_mask]
        
        # Compute metrics
        mean_off_diag = np.mean(actual_off_diag)
        std_off_diag = np.std(actual_off_diag)
        etf_deviation = np.mean(np.abs(actual_off_diag - ideal_off_diag))
        
        # Angular uniformity
        angles = np.arccos(np.clip(actual_off_diag, -1, 1))
        angular_uniformity = np.std(angles)
        
        return {
            'etf_quality': 1 / (1 + etf_deviation),
            'angular_uniformity': 1 / (1 + angular_uniformity),
            'mean_cosine_sim': float(mean_off_diag),
            'cosine_std': float(std_off_diag),
            'ideal_deviation': float(etf_deviation)
        }
    
    def compute_gram_spectrum(self, features: torch.Tensor) -> Dict:
        """Analyze spectrum of Gram matrix"""
        features_np = features.detach().cpu().numpy()
        gram = features_np @ features_np.T
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Effective rank
        normalized_eig = eigenvalues / eigenvalues.sum()
        effective_rank = np.exp(entropy(normalized_eig))
        
        return {
            'effective_rank': effective_rank,
            'max_eigenvalue': eigenvalues[0],
            'min_eigenvalue': eigenvalues[-1],
            'eigenvalue_entropy': entropy(normalized_eig),
            'spectrum': eigenvalues
        }
    
    def compute_manifold_properties(self, features: torch.Tensor) -> Dict:
        """Compute manifold geometric properties"""
        features_np = features.detach().cpu().numpy()
        
        # Intrinsic dimension estimation
        pca = PCA()
        pca.fit(features_np)
        explained_variance = pca.explained_variance_ratio_
        
        # Find where cumulative variance reaches 95%
        cumulative_variance = np.cumsum(explained_variance)
        intrinsic_dim = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Local curvature estimation (simplified)
        distances = pdist(features_np[:1000])  # Subsample for efficiency
        curvature_estimate = np.std(distances) / np.mean(distances)
        
        return {
            'intrinsic_dimension': intrinsic_dim,
            'curvature_estimate': curvature_estimate,
            'pca_variance': explained_variance
        }
    
    def compute_within_class_variance(self, features: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Compute within-class and between-class variance"""
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        unique_labels = np.unique(labels_np)
        overall_mean = features_np.mean(axis=0)
        
        within_var = 0
        between_var = 0
        class_means = []
        
        for label in unique_labels:
            class_mask = labels_np == label
            class_features = features_np[class_mask]
            class_mean = class_features.mean(axis=0)
            class_means.append(class_mean)
            
            within_var += np.sum((class_features - class_mean) ** 2)
            between_var += np.sum((class_mean - overall_mean) ** 2) * class_mask.sum()
        
        within_var /= len(features_np)
        between_var /= len(features_np)
        
        return {
            'within_class_variance': within_var,
            'between_class_variance': between_var,
            'variance_ratio': between_var / (within_var + 1e-8),
            'class_collapse': 1 / (1 + between_var)  # Lower is better
        }
    
    def track_geometric_evolution(self, model: nn.Module, dataloader, 
                                 layers_to_track: List[str] = None) -> Dict:
        """
        Track geometric properties across layers and training time
        """
        model.eval()
        results = {}
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Analyzing geometric evolution", leave=False)):
                if batch_idx >= 10:  # Use first 10 batches for stability
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Register hooks for intermediate features
                activations = {}
                hooks = []
                
                def hook_fn(name):
                    def hook(module, input, output):
                        activations[name] = output.detach()
                    return hook
                
                # Attach hooks to specified layers
                if layers_to_track is None:
                    # Track all linear layers by default
                    layers_to_track = []
                    for name, module in model.named_modules():
                        if isinstance(module, (nn.Linear, nn.Conv2d)):
                            layers_to_track.append(name)
                            hooks.append(module.register_forward_hook(hook_fn(name)))
                else:
                    for name, module in model.named_modules():
                        if name in layers_to_track:
                            hooks.append(module.register_forward_hook(hook_fn(name)))
                
                # Forward pass
                _ = model(data)
                
                # Analyze each layer's activations
                for layer_name, activation in activations.items():
                    if activation.dim() > 2:
                        activation = activation.view(activation.size(0), -1)
                    
                    if layer_name not in results:
                        results[layer_name] = {
                            'etf_metrics': [], 'gram_metrics': [],
                            'manifold_metrics': [], 'variance_metrics': []
                        }
                    
                    # Compute all metrics
                    etf_metrics = self.compute_etf_alignment(activation, target)
                    gram_metrics = self.compute_gram_spectrum(activation)
                    manifold_metrics = self.compute_manifold_properties(activation)
                    variance_metrics = self.compute_within_class_variance(activation, target)
                    
                    results[layer_name]['etf_metrics'].append(etf_metrics)
                    results[layer_name]['gram_metrics'].append(gram_metrics)
                    results[layer_name]['manifold_metrics'].append(manifold_metrics)
                    results[layer_name]['variance_metrics'].append(variance_metrics)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
        
        # Average across batches
        for layer_name in results:
            for metric_type in results[layer_name]:
                if results[layer_name][metric_type]:
                    # Average dictionary values
                    avg_metrics = {}
                    for key in results[layer_name][metric_type][0].keys():
                        values = [m[key] for m in results[layer_name][metric_type] 
                                if isinstance(m[key], (int, float))]
                        if values:
                            avg_metrics[key] = np.mean(values)
                    results[layer_name][metric_type] = avg_metrics
        
        return results

class AGOPIntervention:
    """Interventions on AGOP structure"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_agop(self, model: nn.Module, data: torch.Tensor) -> Dict:
        """Compute AGOP for each layer"""
        model.eval()
        agop_results = {}
        
        # Register hooks for gradients
        gradients = {}
        hooks = []
        
        def forward_hook(name):
            def hook(module, input, output):
                gradients[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    gradients[f'{name}_grad'] = grad_output[0].detach()
            return hook
        
        # Attach hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(forward_hook(name)))
                hooks.append(module.register_backward_hook(backward_hook(name)))
        
        # Forward and backward pass
        output = model(data)
        loss = F.cross_entropy(output, torch.randint(0, output.size(1), (data.size(0),).to(self.device)))
        loss.backward()
        
        # Compute AGOP for each layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and f'{name}_grad' in gradients:
                activation = gradients[name]
                grad = gradients[f'{name}_grad']
                
                if activation.dim() > 2:
                    activation = activation.view(activation.size(0), -1)
                if grad.dim() > 2:
                    grad = grad.view(grad.size(0), -1)
                
                # AGOP = E[∇f(x) ∇f(x)^T]
                agop = torch.matmul(grad, grad.transpose(0, 1))
                agop_results[name] = {
                    'agop_matrix': agop,
                    'agop_rank': torch.matrix_rank(agop).item(),
                    'agop_norm': torch.norm(agop).item()
                }
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return agop_results
    
    def agop_regularization_loss(self, model: nn.Module, data: torch.Tensor, 
                                target_rank: int = None, lambda_rank: float = 0.1) -> torch.Tensor:
        """Add AGOP regularization to loss"""
        agop_results = self.compute_agop(model, data)
        reg_loss = 0
        
        for layer_name, agop_data in agop_results.items():
            agop = agop_data['agop_matrix']
            
            if target_rank is not None:
                # Encourage specific rank
                current_rank = agop_data['agop_rank']
                rank_deviation = abs(current_rank - target_rank)
                reg_loss += lambda_rank * rank_deviation
            else:
                # General regularization on AGOP norm
                reg_loss += lambda_rank * agop_data['agop_norm']
        
        return reg_loss
    
    def train_with_agop_constraints(self, model: nn.Module, train_loader, 
                                   epochs: int = 10, lambda_rank: float = 0.1,
                                   target_rank: Optional[int] = None):
        """Train model with AGOP constraints"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        geometric_analyzer = GeometricAnalyzer(self.device)
        
        history = {
            'train_loss': [], 'agop_metrics': [], 
            'geometric_metrics': [], 'test_accuracy': []
        }
        
        for epoch in tqdm(range(epochs), desc="AGOP Training", leave=False):
            model.train()
            epoch_loss = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                task_loss = F.cross_entropy(output, target)
                
                # AGOP regularization
                agop_reg = self.agop_regularization_loss(model, data, target_rank, lambda_rank)
                total_loss = task_loss + agop_reg
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
                # Track metrics every few batches
                if batch_idx % 100 == 0:
                    agop_metrics = self.compute_agop(model, data)
                    history['agop_metrics'].append(agop_metrics)
            
            # Track geometric evolution
            geometric_metrics = geometric_analyzer.track_geometric_evolution(model, train_loader)
            history['geometric_metrics'].append(geometric_metrics)
            history['train_loss'].append(epoch_loss / len(train_loader))
            
            print(f"Epoch {epoch+1}: Loss = {history['train_loss'][-1]:.4f}")
        
        return history

class ArchitectureUniversality:
    """Test geometric principles across architectures"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.geometric_analyzer = GeometricAnalyzer(device)
    
    def create_models(self, architecture_types: List[str]) -> Dict:
        """Create different architecture instances"""
        models = {}
        
        if 'resnet' in architecture_types:
            models['resnet'] = self._create_resnet()
        if 'vit' in architecture_types:
            models['vit'] = self._create_vit()
        if 'mlp' in architecture_types:
            models['mlp'] = self._create_mlp()
        if 'convnet' in architecture_types:
            models['convnet'] = self._create_convnet()
        if 'timm' in architecture_types and TIMM_AVAILABLE:
            models['timm'] = self._create_timm_model()
        if 'thingsvision' in architecture_types and THINGSVISION_AVAILABLE:
            models['thingsvision'] = self._create_thingsvision_model()
        
        return models
    
    def _create_timm_model(self, model_name='resnet18', pretrained=True):
        """Create a timm model"""
        if not TIMM_AVAILABLE:
            raise ImportError("timm is not available")
        
        try:
            # Get dataset info
            global DATASET_INFO
            if 'DATASET_INFO' not in globals():
                # Fallback to CIFAR-10 defaults
                num_classes = 10
                in_channels = 3
            else:
                num_classes = DATASET_INFO['num_classes']
                in_channels = DATASET_INFO['in_channels']
            
            # Create model with timm
            model = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=num_classes,
                in_chans=in_channels
            )
            
            # Move to device
            model = model.to(self.device)
            
            # Get model info
            info = get_timm_model_info(model_name)
            if info:
                print(f"Created timm model: {model_name}")
                print(f"  Dataset: {DATASET_INFO.get('name', 'unknown')}")
                print(f"  Classes: {num_classes}")
                print(f"  Input channels: {in_channels}")
                if 'parameters' in info and info['parameters'] != 'unknown':
                    print(f"  Parameters: {info['parameters']}M")
                if 'family' in info:
                    print(f"  Family: {info['family']}")
            else:
                print(f"Created timm model: {model_name}")
                print(f"  Dataset: {DATASET_INFO.get('name', 'unknown')}")
                print(f"  Classes: {num_classes}")
                print(f"  Input channels: {in_channels}")
            
            return model
            
        except Exception as e:
            print(f"Error creating timm model {model_name}: {e}")
            print("Falling back to default convnet")
            return self._create_convnet()
    
    def _create_thingsvision_model(self, model_name='resnet50', source='torchvision'):
        """Create a thingsvision model"""
        if not THINGSVISION_AVAILABLE:
            raise ImportError("thingsvision is not available")
        
        try:
            # Get dataset info
            global DATASET_INFO
            if 'DATASET_INFO' not in globals():
                # Fallback to CIFAR-10 defaults
                num_classes = 10
                in_channels = 3
            else:
                num_classes = DATASET_INFO['num_classes']
                in_channels = DATASET_INFO['in_channels']
            
            # Create model with thingsvision
            from thingsvision import get_model
            model = get_model(
                model_name, 
                source=source,
                device=self.device,
                pretrained=True
            )
            
            # Adapt the model for our dataset if needed
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
                if model.classifier.out_features != num_classes:
                    model.classifier.out_features = num_classes
            elif hasattr(model, 'head') and hasattr(model.head, 'out_features'):
                if model.head.out_features != num_classes:
                    model.head.out_features = num_classes
            elif hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
                if model.fc.out_features != num_classes:
                    model.fc.out_features = num_classes
            
            # Move to device
            model = model.to(self.device)
            
            print(f"Created thingsvision model: {model_name} (source: {source})")
            print(f"  Dataset: {DATASET_INFO.get('name', 'unknown')}")
            print(f"  Classes: {num_classes}")
            print(f"  Input channels: {in_channels}")
            
            return model
            
        except Exception as e:
            print(f"Error creating thingsvision model {model_name}: {e}")
            print("Falling back to default convnet")
            return self._create_convnet()
    
    def _create_resnet(self) -> nn.Module:
        """Simple ResNet variant"""
        class BasicBlock(nn.Module):
            def __init__(self, in_planes, planes, stride=1):
                super(BasicBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                    )
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out
        
        class ResNet(nn.Module):
            def __init__(self, num_blocks=[2, 2, 2], num_classes=10):
                super(ResNet, self).__init__()
                self.in_planes = 64
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
                self.linear = nn.Linear(256, num_classes)
            
            def _make_layer(self, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for stride in strides:
                    layers.append(BasicBlock(self.in_planes, planes, stride))
                    self.in_planes = planes
                return nn.Sequential(*layers)
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                out = self.linear(out)
                return out
        
        return ResNet().to(self.device)
    
    def _create_vit(self) -> nn.Module:
        """Simple Vision Transformer variant"""
        class PatchEmbedding(nn.Module):
            def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
                super().__init__()
                self.img_size = img_size
                self.patch_size = patch_size
                self.n_patches = (img_size // patch_size) ** 2
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
            def forward(self, x):
                x = self.proj(x)
                x = x.flatten(2).transpose(1, 2)
                return x
        
        class Attention(nn.Module):
            def __init__(self, dim, n_heads=4):
                super().__init__()
                self.n_heads = n_heads
                self.scale = (dim // n_heads) ** -0.5
                self.qkv = nn.Linear(dim, dim * 3)
                self.proj = nn.Linear(dim, dim)
            
            def forward(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                return x
        
        class Block(nn.Module):
            def __init__(self, dim, n_heads):
                super().__init__()
                self.attn = Attention(dim, n_heads)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)
            
            def forward(self, x):
                x = x + self.attn(self.norm1(x))
                x = x + self.mlp(self.norm2(x))
                return x
        
        class ViT(nn.Module):
            def __init__(self, img_size=32, patch_size=4, in_chans=3, n_classes=10, 
                         embed_dim=128, depth=4, n_heads=4):
                super().__init__()
                self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
                self.blocks = nn.ModuleList([Block(embed_dim, n_heads) for _ in range(depth)])
                self.norm = nn.LayerNorm(embed_dim)
                self.head = nn.Linear(embed_dim, n_classes)
            
            def forward(self, x):
                B = x.shape[0]
                x = self.patch_embed(x)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                x = x[:, 0]
                x = self.head(x)
                return x
        
        # Get dataset info
        global DATASET_INFO
        if 'DATASET_INFO' not in globals():
            img_size = 32
            in_chans = 3
            n_classes = 10
        else:
            img_size = DATASET_INFO['image_size']
            in_chans = DATASET_INFO['in_channels']
            n_classes = DATASET_INFO['num_classes']
        
        # Adjust ViT parameters based on dataset
        if img_size <= 32:
            # Small images: use smaller patch size and embedding
            patch_size = 4
            embed_dim = 128
            depth = 4
            n_heads = 4
        else:
            # Larger images: use larger patch size and embedding
            patch_size = 8
            embed_dim = 256
            depth = 6
            n_heads = 8
        
        model = ViT(img_size, patch_size, in_chans, n_classes, embed_dim, depth, n_heads)
        model = model.to(self.device)
        return model
    
    def _create_mlp(self) -> nn.Module:
        """Simple MLP for various datasets"""
        class MLP(nn.Module):
            def __init__(self, input_dim=3072, hidden_dims=[512, 256, 128], num_classes=10):
                super(MLP, self).__init__()
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.1))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, num_classes))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.network(x)
        
        # Get dataset info
        global DATASET_INFO
        if 'DATASET_INFO' not in globals():
            input_dim = 3072  # Default for 32x32x3
            num_classes = 10
        else:
            # Calculate input dimension based on dataset
            image_size = DATASET_INFO['image_size']
            in_channels = DATASET_INFO['in_channels']
            input_dim = image_size * image_size * in_channels
            num_classes = DATASET_INFO['num_classes']
        
        # Adjust hidden dimensions based on input size
        if input_dim <= 1024:
            hidden_dims = [256, 128, 64]
        elif input_dim <= 3072:
            hidden_dims = [512, 256, 128]
        else:
            hidden_dims = [1024, 512, 256]
        
        model = MLP(input_dim, hidden_dims, num_classes)
        model = model.to(self.device)
        return model
    
    def _create_convnet(self) -> nn.Module:
        """Simple ConvNet for various datasets"""
        class ConvNet(nn.Module):
            def __init__(self, num_classes=10, in_channels=3, image_size=32):
                super(ConvNet, self).__init__()
                
                # Adjust architecture based on image size
                if image_size <= 32:
                    # Small images (CIFAR, MNIST, SVHN)
                    self.features = nn.Sequential(
                        nn.Conv2d(in_channels, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                    # Calculate feature size: (image_size / 8) * (image_size / 8) * 128
                    # For 32x32: 4 * 4 * 128 = 2048
                    # For 28x28: 3 * 3 * 128 = 1152
                    feature_size = (image_size // 8) * (image_size // 8) * 128
                else:
                    # Larger images (STL-10)
                    self.features = nn.Sequential(
                        nn.Conv2d(in_channels, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                    # Calculate feature size for larger images
                    feature_size = (image_size // 16) * (image_size // 16) * 256
                
                self.classifier = nn.Sequential(
                    nn.Linear(feature_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        # Get dataset info
        global DATASET_INFO
        if 'DATASET_INFO' not in globals():
            num_classes = 10
            in_channels = 3
            image_size = 32
        else:
            num_classes = DATASET_INFO['num_classes']
            in_channels = DATASET_INFO['in_channels']
            image_size = DATASET_INFO['image_size']
        
        model = ConvNet(num_classes, in_channels, image_size)
        model = model.to(self.device)
        return model
    
    def compare_architectures(self, architectures: List[str], train_loader, test_loader, 
                             epochs: int = 20) -> Dict:
        """Compare geometric evolution across architectures"""
        models = self.create_models(architectures)
        results = {}
        
        for arch_name, model in models.items():
            print(f"Training {arch_name}...")
            
            # Train model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            geometric_history = []
            accuracy_history = []
            
            for epoch in tqdm(range(epochs), desc=f"Training {arch_name}", leave=False):
                model.train()
                epoch_loss = 0
                
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Evaluate geometric properties
                model.eval()
                geometric_metrics = self.geometric_analyzer.track_geometric_evolution(model, test_loader)
                geometric_history.append(geometric_metrics)
                
                # Test accuracy
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in tqdm(test_loader, desc="Evaluating", leave=False):
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                
                accuracy = 100. * correct / total
                accuracy_history.append(accuracy)
                
                # Log to tensorboard if available
                if 'tensorboard_writer' in globals() and tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(f"training/{arch_name}/loss", epoch_loss/len(train_loader), epoch)
                    tensorboard_writer.add_scalar(f"training/{arch_name}/accuracy", accuracy, epoch)
                
                print(f"{arch_name} Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}, Acc = {accuracy:.2f}%")
            
            results[arch_name] = {
                'geometric_history': geometric_history,
                'accuracy_history': accuracy_history,
                'final_model': model
            }
        
        return results
    
    def analyze_universality(self, results: Dict) -> Dict:
        """Analyze geometric universality across architectures"""
        universality_metrics = {}
        
        # Compare final geometric properties
        final_geometrics = {}
        for arch_name, arch_data in results.items():
            final_geometrics[arch_name] = arch_data['geometric_history'][-1]
        
        # Compute similarity between architectures
        arch_names = list(final_geometrics.keys())
        similarity_matrix = np.zeros((len(arch_names), len(arch_names)))
        
        for i, arch1 in enumerate(arch_names):
            for j, arch2 in enumerate(arch_names):
                if i != j:
                    # Compare geometric properties
                    similarity = self._compare_geometric_profiles(
                        final_geometrics[arch1], 
                        final_geometrics[arch2]
                    )
                    similarity_matrix[i, j] = similarity
        
        universality_metrics['similarity_matrix'] = similarity_matrix
        universality_metrics['arch_names'] = arch_names
        
        # Analyze convergence patterns
        convergence_analysis = self._analyze_convergence_patterns(results)
        universality_metrics.update(convergence_analysis)
        
        return universality_metrics
    
    def _compare_geometric_profiles(self, metrics1: Dict, metrics2: Dict) -> float:
        """Compare two geometric profiles"""
        similarity = 0
        count = 0
        
        # Compare common layers
        common_layers = set(metrics1.keys()) & set(metrics2.keys())
        
        for layer in common_layers:
            for metric_type in ['etf_metrics', 'gram_metrics', 'manifold_metrics', 'variance_metrics']:
                if metric_type in metrics1[layer] and metric_type in metrics2[layer]:
                    for key in metrics1[layer][metric_type]:
                        if key in metrics2[layer][metric_type]:
                            val1 = metrics1[layer][metric_type][key]
                            val2 = metrics2[layer][metric_type][key]
                            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                                similarity += 1 / (1 + abs(val1 - val2))
                                count += 1
        
        return similarity / count if count > 0 else 0
    
    def _analyze_convergence_patterns(self, results: Dict) -> Dict:
        """Analyze convergence patterns across architectures"""
        convergence_metrics = {}
        
        # Track when each architecture reaches geometric stability
        stability_epochs = {}
        for arch_name, arch_data in results.items():
            geometric_history = arch_data['geometric_history']
            
            # Find epoch where geometric changes become small
            changes = []
            for i in range(1, len(geometric_history)):
                change = self._compute_geometric_change(
                    geometric_history[i-1], geometric_history[i]
                )
                changes.append(change)
            
            # Find stabilization point (where changes drop below threshold)
            threshold = 0.1  # Adjust based on scale
            for epoch, change in enumerate(changes):
                if change < threshold:
                    stability_epochs[arch_name] = epoch + 1
                    break
            else:
                stability_epochs[arch_name] = len(changes)
        
        convergence_metrics['stability_epochs'] = stability_epochs
        
        return convergence_metrics
    
    def _compute_geometric_change(self, metrics1: Dict, metrics2: Dict) -> float:
        """Compute change between two geometric states"""
        total_change = 0
        count = 0
        
        common_layers = set(metrics1.keys()) & set(metrics2.keys())
        
        for layer in common_layers:
            for metric_type in ['etf_metrics', 'gram_metrics', 'manifold_metrics', 'variance_metrics']:
                if metric_type in metrics1[layer] and metric_type in metrics2[layer]:
                    for key in metrics1[layer][metric_type]:
                        if key in metrics2[layer][metric_type]:
                            val1 = metrics1[layer][metric_type][key]
                            val2 = metrics2[layer][metric_type][key]
                            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                                total_change += abs(val1 - val2)
                                count += 1
        
        return total_change / count if count > 0 else 0

# Example usage and visualization
def run_high_priority_experiments(args=None):
    """Run all high-priority experiments"""
    if args is None:
        args = parse_args()
    
    # Handle training duration presets
    if args.quick:
        args.epochs = 10
        training_mode = "Quick"
    elif args.thorough:
        args.epochs = 100
        training_mode = "Thorough"
    elif args.extensive:
        args.epochs = 200
        training_mode = "Extensive"
    else:
        training_mode = "Standard"
    
    print("=" * 60)
    print("Geometric Alignment Analysis - Neural Collapse Research")
    print("=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Training Mode: {training_mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Architecture: {args.architecture}")
    if args.experiment == 'agop':
        print(f"AGOP lambda_rank: {args.lambda_rank}")
        print(f"AGOP target_rank: {args.target_rank}")
    print("=" * 60)
    
    # Setup device
    if args.device is None:
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    # Validate device compatibility
    is_compatible, message = check_device_compatibility(device)
    if not is_compatible:
        print(f"Warning: {message}")
        if device != 'cpu':
            print("Falling back to CPU")
            device = 'cpu'
        else:
            print("No compatible device found, exiting")
            return
    
    print(f"Using device: {device}")
    
    # Get and display device information
    device_info = get_device_info(device)
    if device == 'mps':
        print("Note: Using Apple Silicon MPS acceleration")
        print(f"  - Device: {device_info['name']}")
        # MPS-specific optimizations
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have empty_cache, but we can set some optimizations
            print("  - MPS optimizations enabled")
    elif device == 'cuda':
        print(f"Note: CUDA device: {device_info['name']}")
        print(f"  - Total memory: {device_info['memory_total']:.1f} GB")
        print(f"  - Allocated memory: {device_info['memory_allocated']:.1f} GB")
        print(f"  - Cached memory: {device_info['memory_cached']:.1f} GB")
        # CUDA-specific optimizations
        torch.cuda.empty_cache()
        print("  - CUDA cache cleared")
    elif device == 'cpu':
        print("Note: Using CPU (slower training)")
        if 'memory_total' in device_info and isinstance(device_info['memory_total'], (int, float)):
            print(f"  - Total memory: {device_info['memory_total']:.1f} GB")
            print(f"  - Available memory: {device_info['memory_available']:.1f} GB")
            print(f"  - Memory usage: {device_info['memory_percent']:.1f}%")
    
    # Setup output directories
    dataset_name = args.dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup tensorboard logging
    if not args.no_tensorboard:
        setup_tensorboard(args.log_dir, args.architecture, dataset_name)
        print("Tensorboard logging enabled")
    else:
        print("Tensorboard logging disabled")
    
    # Get optimal batch size for the device
    optimal_batch_size = get_optimal_batch_size(device, args.batch_size)
    if optimal_batch_size != args.batch_size:
        print(f"Note: Adjusted batch size from {args.batch_size} to {optimal_batch_size} for optimal {device} performance")
        args.batch_size = optimal_batch_size
    
    # Display device considerations
    considerations = get_device_considerations(device)
    if considerations:
        print("Device considerations:")
        for consideration in considerations:
            print(f"  - {consideration}")
    
    # Display training considerations
    training_considerations = get_training_considerations(args.epochs)
    if considerations:
        print("Training considerations:")
        for consideration in training_considerations:
            print(f"  - {consideration}")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name, args.data_dir, args.batch_size, device
    )
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    # Experiment selection based on args
    if args.experiment in ['all', 'etf']:
        # Setup ETF experiment output directories
        etf_output_dir, etf_plots_dir, etf_metrics_dir, etf_models_dir = setup_output_dirs(
            args.architecture, dataset_name, "etf", timestamp
        )
        
        # Experiment 1: ETF Emergence Tracking
        print("=== Experiment 1: ETF Emergence Tracking ===")
        geometric_analyzer = GeometricAnalyzer(device)
        
        # Create a sample model
        model = ArchitectureUniversality(device)._create_convnet()
        
        # Track geometric evolution
        geometric_metrics = geometric_analyzer.track_geometric_evolution(model, test_loader)
        
        # Log ETF metrics to tensorboard
        if not args.no_tensorboard:
            for layer, metrics in geometric_metrics.items():
                log_metrics(metrics['etf_metrics'], 0, f"etf/{layer}/")
        
        # Save ETF metrics to JSON
        etf_metrics_file = os.path.join(etf_metrics_dir, "etf_metrics.json")
        save_metrics(geometric_metrics, etf_metrics_file)
        
        # Visualize ETF quality across layers
        layers = list(geometric_metrics.keys())
        etf_qualities = [geometric_metrics[layer]['etf_metrics']['etf_quality'] for layer in layers]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(layers)), etf_qualities)
        plt.xlabel('Layer')
        plt.ylabel('ETF Quality')
        plt.title('ETF Quality Across Layers')
        plt.xticks(range(len(layers)), layers, rotation=45)
        plt.tight_layout()
        
        # Save plot to organized directory
        etf_plot_file = os.path.join(etf_plots_dir, "etf_quality_across_layers.png")
        plt.savefig(etf_plot_file)
        plt.close()
        
        # Log final ETF quality to tensorboard
        if not args.no_tensorboard:
            for i, (layer, quality) in enumerate(zip(layers, etf_qualities)):
                tensorboard_writer.add_scalar(f"etf/final_quality/{layer}", quality, 0)
        
        # Save experiment summary
        etf_summary = {
            'experiment': 'etf',
            'architecture': args.architecture,
            'dataset': dataset_name,
            'timestamp': timestamp,
            'device': device,
            'layers_analyzed': layers,
            'etf_qualities': etf_qualities,
            'output_directory': etf_output_dir
        }
        etf_summary_file = os.path.join(etf_metrics_dir, "experiment_summary.json")
        save_experiment_summary(etf_summary, etf_summary_file)
    
    if args.experiment in ['all', 'agop']:
        # Setup AGOP experiment output directories
        agop_output_dir, agop_plots_dir, agop_metrics_dir, agop_models_dir = setup_output_dirs(
            args.architecture, dataset_name, "agop", timestamp
        )
        
        # Experiment 2: AGOP Intervention
        print("=== Experiment 2: AGOP Intervention ===")
        agop_intervention = AGOPIntervention(device)
        
        # Train with different AGOP constraints
        print("Training with low-rank AGOP constraint...")
        model_low_rank = ArchitectureUniversality(device)._create_convnet()
        history_low_rank = agop_intervention.train_with_agop_constraints(
            model_low_rank, train_loader, epochs=args.epochs, 
            lambda_rank=args.lambda_rank, target_rank=args.target_rank
        )
        
        print("Training with high-rank AGOP constraint...")
        model_high_rank = ArchitectureUniversality(device)._create_convnet()
        history_high_rank = agop_intervention.train_with_agop_constraints(
            model_high_rank, train_loader, epochs=args.epochs, 
            lambda_rank=-args.lambda_rank
        )
    
        # Save AGOP training history
        agop_history_file = os.path.join(agop_metrics_dir, "agop_training_history.json")
        agop_history = {
            'low_rank': history_low_rank,
            'high_rank': history_high_rank,
            'lambda_rank': args.lambda_rank,
            'target_rank': args.target_rank
        }
        save_metrics(agop_history, agop_history_file)
        
        # Compare results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_low_rank['train_loss'], label='Low Rank AGOP')
        plt.plot(history_high_rank['train_loss'], label='High Rank AGOP')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss')
        
        # Log training loss to tensorboard
        if not args.no_tensorboard:
            for epoch, (loss_low, loss_high) in enumerate(zip(history_low_rank['train_loss'], history_high_rank['train_loss'])):
                tensorboard_writer.add_scalar("agop/training_loss/low_rank", loss_low, epoch)
                tensorboard_writer.add_scalar("agop/training_loss/high_rank", loss_high, epoch)
        
        plt.subplot(1, 2, 2)
        # Compare final ETF quality
        final_etf_low = geometric_analyzer.track_geometric_evolution(model_low_rank, test_loader)
        final_etf_high = geometric_analyzer.track_geometric_evolution(model_high_rank, test_loader)
        
        # Save final ETF metrics
        final_etf_file = os.path.join(agop_metrics_dir, "final_etf_comparison.json")
        final_etf_data = {
            'low_rank': final_etf_low,
            'high_rank': final_etf_high
        }
        save_metrics(final_etf_data, final_etf_file)
        
        layers = list(final_etf_low.keys())
        etf_low = [final_etf_low[layer]['etf_metrics']['etf_quality'] for layer in layers]
        etf_high = [final_etf_high[layer]['etf_metrics']['etf_quality'] for layer in layers]
        
        # Log final ETF quality to tensorboard
        if not args.no_tensorboard:
            for layer, (quality_low, quality_high) in zip(layers, zip(etf_low, etf_high)):
                tensorboard_writer.add_scalar(f"agop/final_etf_quality/low_rank/{layer}", quality_low, 0)
                tensorboard_writer.add_scalar(f"agop/final_etf_quality/high_rank/{layer}", quality_high, 0)
        
        x = np.arange(len(layers))
        width = 0.35
        plt.bar(x - width/2, etf_low, width, label='Low Rank AGOP')
        plt.bar(x + width/2, etf_high, width, label='High Rank AGOP')
        plt.xlabel('Layer')
        plt.ylabel('ETF Quality')
        plt.legend()
        plt.title('Final ETF Quality')
        plt.xticks(x, layers, rotation=45)
        
        plt.tight_layout()
        
        # Save plot to organized directory
        agop_plot_file = os.path.join(agop_plots_dir, "agop_intervention_results.png")
        plt.savefig(agop_plot_file)
        plt.close()
    
    if args.experiment in ['all', 'universality']:
        # Setup universality experiment output directories
        univ_output_dir, univ_plots_dir, univ_metrics_dir, univ_models_dir = setup_output_dirs(
            args.architecture, dataset_name, "universality", timestamp
        )
        
        # Experiment 3: Architecture Universality
        print("=== Experiment 3: Architecture Universality ===")
        universality = ArchitectureUniversality(device)
        
        # Use specified architecture or default list
        if args.architecture != 'convnet':
            if args.architecture == 'timm' and TIMM_AVAILABLE:
                architectures = ['timm']
                print(f"Using timm model: {args.timm_model}")
            else:
                architectures = [args.architecture]
        else:
            architectures = ['convnet', 'resnet', 'mlp']  # Skip ViT for faster training
        
        # For timm models, we need to pass the specific model name
        if 'timm' in architectures and TIMM_AVAILABLE:
            # Override the _create_timm_model method temporarily to use the specified model
            original_method = universality._create_timm_model
            universality._create_timm_model = lambda: original_method(args.timm_model, args.timm_pretrained)
        
        results = universality.compare_architectures(architectures, train_loader, test_loader, epochs=args.epochs)
    
        # Analyze universality
        universality_metrics = universality.analyze_universality(results)
        
        # Save universality metrics
        univ_metrics_file = os.path.join(univ_metrics_dir, "universality_metrics.json")
        save_metrics(universality_metrics, univ_metrics_file)
        
        # Save training results
        univ_results_file = os.path.join(univ_metrics_dir, "training_results.json")
        # Convert results to serializable format
        serializable_results = {}
        for arch_name, arch_data in results.items():
            serializable_results[arch_name] = {
                'accuracy_history': arch_data['accuracy_history'],
                'geometric_history': arch_data['geometric_history']
            }
        save_metrics(serializable_results, univ_results_file)
        
        # Log universality metrics to tensorboard
        if not args.no_tensorboard:
            # Log similarity matrix
            similarity_matrix = universality_metrics['similarity_matrix']
            for i, arch1 in enumerate(universality_metrics['arch_names']):
                for j, arch2 in enumerate(universality_metrics['arch_names']):
                    tensorboard_writer.add_scalar(f"universality/similarity/{arch1}_{arch2}", 
                                                similarity_matrix[i, j], 0)
            
            # Log stability epochs
            for arch, epoch in universality_metrics['stability_epochs'].items():
                tensorboard_writer.add_scalar(f"universality/stability_epoch/{arch}", epoch, 0)
        
        # Plot similarity matrix
        plt.figure(figsize=(8, 6))
        similarity_matrix = universality_metrics['similarity_matrix']
        arch_names = universality_metrics['arch_names']
        
        sns.heatmap(similarity_matrix, annot=True, xticklabels=arch_names, 
                    yticklabels=arch_names, cmap='viridis')
        plt.title('Geometric Similarity Between Architectures')
        plt.tight_layout()
        
        # Save plot to organized directory
        univ_sim_plot_file = os.path.join(univ_plots_dir, "architecture_similarity.png")
        plt.savefig(univ_sim_plot_file)
        plt.close()
    
        # Plot convergence patterns
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        stability_epochs = universality_metrics['stability_epochs']
        plt.bar(range(len(stability_epochs)), list(stability_epochs.values()))
        plt.xlabel('Architecture')
        plt.ylabel('Stability Epoch')
        plt.title('Geometric Stabilization Epoch')
        plt.xticks(range(len(stability_epochs)), list(stability_epochs.keys()), rotation=45)
        
        plt.subplot(1, 2, 2)
        # Plot accuracy convergence
        for arch_name, arch_data in results.items():
            plt.plot(arch_data['accuracy_history'], label=arch_name)
            # Log accuracy convergence to tensorboard
            if not args.no_tensorboard:
                for epoch, acc in enumerate(arch_data['accuracy_history']):
                    tensorboard_writer.add_scalar(f"universality/accuracy/{arch_name}", acc, epoch)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Convergence')
        
        plt.tight_layout()
        
        # Save plot to organized directory
        univ_conv_plot_file = os.path.join(univ_plots_dir, "convergence_analysis.png")
        plt.savefig(univ_conv_plot_file)
        plt.close()
    
    # Close tensorboard writer
    if not args.no_tensorboard and tensorboard_writer is not None:
        tensorboard_writer.close()
        print("Tensorboard logging completed.")
    
    print("=" * 60)
    print("All experiments completed successfully!")
    print("\nResults saved to organized directories:")
    
    if args.experiment in ['all', 'etf']:
        print(f"  ETF Experiment: {etf_output_dir}")
        print(f"    - Plots: {etf_plots_dir}")
        print(f"    - Metrics: {etf_metrics_dir}")
        print(f"    - Models: {etf_models_dir}")
    
    if args.experiment in ['all', 'agop']:
        print(f"  AGOP Experiment: {agop_output_dir}")
        print(f"    - Plots: {agop_plots_dir}")
        print(f"    - Metrics: {agop_metrics_dir}")
        print(f"    - Models: {agop_models_dir}")
    
    if args.experiment in ['all', 'universality']:
        print(f"  Universality Experiment: {univ_output_dir}")
        print(f"    - Plots: {univ_plots_dir}")
        print(f"    - Metrics: {univ_metrics_dir}")
        print(f"    - Models: {univ_models_dir}")
    
    if not args.no_tensorboard:
        print(f"\nTensorboard logs available in: {tensorboard_writer.log_dir}")
        print("Run 'tensorboard --logdir <log_dir>' to view results")
    
    print("\nDirectory structure:")
    print("  outputs/")
    print(f"    {dataset_name}/")
    print(f"      {args.architecture}/")
    print("        [experiment_name]/")
    print("          [timestamp]/")
    print("            plots/     # PNG visualizations")
    print("            metrics/   # JSON data files")
    print("            models/    # Saved model files")
    print("=" * 60)

if __name__ == "__main__":
    args = parse_args()
    
    # Handle timm model listing
    if args.list_timm_models:
        if not TIMM_AVAILABLE:
            print("Error: timm is not available. Install with: pip install timm")
            sys.exit(1)
        
        print("=" * 60)
        print("Available Timm Models")
        print("=" * 60)
        
        model_families = list_timm_models(args.timm_pattern, limit=50)
        if model_families:
            for family, models in model_families.items():
                print(f"\n{family.upper()} Models:")
                for model in models:
                    info = get_timm_model_info(model)
                    if info and 'parameters' in info and info['parameters'] != 'unknown':
                        print(f"  {model} ({info['parameters']}M params)")
                    else:
                        print(f"  {model}")
        else:
            print("No models found.")
        
        print("\n" + "=" * 60)
        print("Usage examples:")
        print("  python geometric_alignment.py --architecture timm --timm_model resnet50")
        print("  python geometric_alignment.py --architecture timm --timm_model efficientnet_b0")
        print("  python geometric_alignment.py --architecture timm --timm_model vit_base_patch16_224")
        print("=" * 60)
        sys.exit(0)
    
    # Handle ViT model listing
    if args.list_vit_models:
        if not TIMM_AVAILABLE:
            print("Error: timm is not available. Install with: pip install timm")
            sys.exit(1)
        
        print("=" * 60)
        print("Popular ViT Models from Timm")
        print("=" * 60)
        
        vit_models = get_popular_vit_models()
        if vit_models:
            print("Available ViT models (sorted by size):")
            for model in vit_models:
                info = get_timm_model_info(model)
                if info and 'parameters' in info and info['parameters'] != 'unknown':
                    print(f"  {model} ({info['parameters']}M params)")
                else:
                    print(f"  {model}")
        else:
            print("No ViT models found.")
        
        print("\n" + "=" * 60)
        print("Usage examples:")
        print("  python geometric_alignment.py --architecture vit")
        print("  python geometric_alignment.py --architecture timm --timm_model vit_base_patch16_224")
        print("  python geometric_alignment.py --architecture timm --timm_model vit_small_patch16_224")
        print("=" * 60)
        sys.exit(0)
    
    # Handle thingsvision model listing
    if args.list_thingsvision_models:
        if not THINGSVISION_AVAILABLE:
            print("Error: thingsvision is not available. Install with: pip install thingsvision")
            sys.exit(1)
        
        print("=" * 60)
        print("Available Models from Thingsvision")
        print("=" * 60)
        
        model_categories = get_thingsvision_models()
        if model_categories:
            for category, models in model_categories.items():
                if models:
                    print(f"\n{category.upper()} Models:")
                    for model in models[:10]:  # Show first 10 per category
                        print(f"  {model}")
                    if len(models) > 10:
                        print(f"  ... and {len(models) - 10} more")
        else:
            print("No models found.")
        
        print("\n" + "=" * 60)
        print("Usage examples:")
        print("  python geometric_alignment.py --dataset things --architecture timm --timm_model resnet50")
        print("  python geometric_alignment.py --dataset things --architecture timm --timm_model vit_base_patch16_224")
        print("  python geometric_alignment.py --dataset things --architecture convnet")
        print("=" * 60)
        sys.exit(0)
    
    run_high_priority_experiments(args)