#!/usr/bin/env python3
"""
Configuration file for comprehensive multi-model analysis
Customize these parameters to control your analysis
"""

# Model configurations
MODELS = {
    'resnet': [
        'resnet18',
        'resnet34', 
        'resnet50'
    ],
    'vit': [
        'vit_base_patch16_224',
        'vit_small_patch16_224',
        'deit_base_patch16_224'
    ],
    'cnn': [
        'convnext_tiny',
        'convnext_small',
        'efficientnet_b0'
    ],
    'mlp': [
        'mlp_mixer_b16_224',
        'mlp_mixer_b32_224'
    ]
}

# Dataset configurations
DATASETS = {
    'cifar10': {
        'name': 'cifar10',
        'num_classes': 10,
        'description': 'CIFAR-10 (10-class image classification)'
    },
    'cifar100': {
        'name': 'cifar100', 
        'num_classes': 100,
        'description': 'CIFAR-100 (100-class image classification)'
    },
    'svhn': {
        'name': 'svhn',
        'num_classes': 10,
        'description': 'Street View House Numbers (10-class digit recognition)'
    }
}

# Analysis settings
ANALYSIS_CONFIG = {
    'pretrained': True,           # Use pretrained models
    'save_features': True,        # Save extracted features
    'save_intermediate': True,    # Save intermediate analysis results
    'generate_plots': True,       # Generate comparison plots
    'tensorboard_logging': True,  # Enable TensorBoard logging
    
    # Performance settings
    'max_batches': 8,             # Number of batches to process (8 * 64 = 512 samples)
    'batch_size': 64,             # Batch size for data loading
    'num_workers': 4,             # Number of data loading workers
    
    # Device settings
    'force_device': None,         # Force specific device (None = auto-detect)
    'prefer_mps': True,          # Prefer MPS on Apple Silicon
    
    # Analysis depth
    'run_macroscopic': True,      # Run macroscopic analysis
    'run_mesoscopic': True,       # Run mesoscopic analysis
    'run_comparisons': True,      # Run cross-model comparisons
}

# Output settings
OUTPUT_CONFIG = {
    'base_dir': './results/',     # Base output directory
    'timestamp_format': '%Y%m%d_%H%M%S',  # Timestamp format for folders
    'save_formats': ['json', 'pt', 'png'],  # File formats to save
    
    # Folder structure
    'create_model_folders': True,  # Create separate folders for each model
    'create_dataset_folders': True, # Create separate folders for each dataset
    'save_tensorboard_logs': True,  # Save TensorBoard logs
}

# Model selection (choose which to analyze)
SELECTED_MODELS = [
    'resnet18',           # ResNet architecture
    'vit_base_patch16_224',  # Vision Transformer  
    'convnext_tiny',      # ConvNeXt (CNN)
    'mixer_b16_224'   # MLP-Mixer
]

SELECTED_DATASETS = [
    'cifar10',
    'cifar100', 
    'svhn'
]

# Quick analysis mode (for testing)
QUICK_MODE = {
    'enabled': False,     # Set to True for faster testing
    'max_batches': 4,     # Reduce to 4 batches (256 samples)
    'batch_size': 32,     # Smaller batch size
    'models': ['resnet18'],  # Only test one model
    'datasets': ['cifar10']  # Only test one dataset
}

# Advanced settings
ADVANCED_CONFIG = {
    'memory_efficient': False,    # Use memory-efficient processing
    'parallel_processing': True,  # Enable parallel model analysis
    'checkpoint_saving': True,    # Save checkpoints during analysis
    'error_recovery': True,       # Continue analysis after errors
    'progress_tracking': True,    # Track progress and timing
}

def get_analysis_config():
    """Get the current analysis configuration"""
    if QUICK_MODE['enabled']:
        return {
            'models': QUICK_MODE['models'],
            'datasets': QUICK_MODE['datasets'],
            'max_batches': QUICK_MODE['max_batches'],
            'batch_size': QUICK_MODE['batch_size'],
            **ANALYSIS_CONFIG
        }
    else:
        return {
            'models': SELECTED_MODELS,
            'datasets': SELECTED_DATASETS,
            **ANALYSIS_CONFIG
        }

def print_config():
    """Print the current configuration"""
    config = get_analysis_config()
    
    print("=== Analysis Configuration ===")
    print(f"Models: {config['models']}")
    print(f"Datasets: {config['datasets']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Max batches: {config['max_batches']}")
    print(f"Total samples: {config['batch_size'] * config['max_batches']}")
    print(f"Device: {config['force_device'] or 'Auto-detect'}")
    print(f"Pretrained: {config['pretrained']}")
    print(f"Quick mode: {QUICK_MODE['enabled']}")
    
    if QUICK_MODE['enabled']:
        print("⚠️  QUICK MODE ENABLED - Limited analysis for testing")

if __name__ == "__main__":
    print_config()
