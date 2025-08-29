# Multi-Model Analysis: Macroscopic and Mesoscopic Analysis

This system provides comprehensive analysis of neural network representations across different architectures (MLP, CNN, ResNet, ViT) and datasets (CIFAR-10/100, SVHN) using the timm library.

## Overview

The analysis combines two complementary approaches:

1. **Macroscopic Analysis**: Information bottleneck theory, phase transitions, and critical layer identification
2. **Mesoscopic Analysis**: NTK spectrum analysis, feature evolution, and representational dynamics

## Supported Models

### Architecture Types
- **MLP**: `mlp_mixer_b16_224`, `mlp_mixer_b32_224`
- **CNN**: `convnext_tiny`, `convnext_small`, `efficientnet_b0`
- **ResNet**: `resnet18`, `resnet34`, `resnet50`
- **ViT**: `vit_base_patch16_224`, `vit_small_patch16_224`, `deit_base_patch16_224`

### Datasets
- **CIFAR-10**: 10-class image classification (32x32 RGB)
- **CIFAR-100**: 100-class image classification (32x32 RGB)
- **SVHN**: Street View House Numbers (32x32 RGB)

## Quick Start

### 1. Basic Usage

```python
from multi_model_analysis import MultiModelAnalyzer

# Initialize analyzer
analyzer = MultiModelAnalyzer(output_dir="./results/my_analysis/")

# Run analysis on specific models and datasets
results = analyzer.run_comprehensive_analysis(
    models=['resnet18', 'vit_base_patch16_224', 'convnext_tiny'],
    datasets=['cifar10', 'cifar100'],
    pretrained=True,
    device='cpu'  # or 'cuda' for GPU
)
```

### 2. Command Line Interface

```bash
# Analyze default models on all datasets
python multi_model_analysis.py

# Analyze specific models
python multi_model_analysis.py --models resnet18 vit_base_patch16_224 convnext_tiny

# Analyze specific datasets
python multi_model_analysis.py --datasets cifar10 svhn

# Use GPU
python multi_model_analysis.py --device cuda

# Custom output directory
python multi_model_analysis.py --output_dir ./my_results/
```

### 3. Example Script

```bash
python example_multi_model_analysis.py
```

## Analysis Components

### Macroscopic Analysis
- **Information Bottleneck Trajectory**: I(X;T) vs I(Y;T) across layers
- **Phase Transitions**: Detection of fitting → compression phases
- **Critical Layers**: Identification of key information processing layers
- **Information Dynamics**: Velocity, acceleration, and path analysis

### Mesoscopic Analysis
- **NTK Spectrum**: Empirical neural tangent kernel analysis
- **Feature Evolution**: Layer-to-layer similarity and convergence
- **Feature Dynamics**: Intrinsic dimension and complexity measures
- **Representational Change**: Cumulative drift and topology analysis

## Output Structure

```
results/
├── multi_model_analysis/
│   ├── {model}_{dataset}_features.pt      # Extracted features
│   ├── {model}_{dataset}_analysis.json    # Combined results
│   ├── {dataset}_model_comparison.png     # Model comparison plots
│   ├── {model}_dataset_comparison.png     # Dataset comparison plots
│   └── architecture_comparison.png        # Architecture comparison
```

### JSON Results Format

```json
{
  "model": "resnet18",
  "dataset": "cifar10",
  "pretrained": true,
  "macroscopic": {
    "information_flow": {...},
    "phase_analysis": {...},
    "critical_transitions": {...},
    "information_dynamics": {...}
  },
  "mesoscopic": {
    "ntk": {...},
    "evolution": {...},
    "dynamics": {...},
    "representational_change": {...}
  },
  "metadata": {
    "num_samples": 1600,
    "num_layers": 4,
    "feature_dim": 512,
    "layer_names": ["stage1", "stage2", "stage3", "stage4"]
  }
}
```

## Key Metrics

### Information Flow
- **Total Compression**: Reduction in input information
- **Task Info Gain**: Increase in task-relevant information
- **Peak Task Info**: Maximum task information achieved
- **Information Efficiency**: I(Y;T) / I(X;T) ratio

### NTK Properties
- **Effective Rank**: Participation ratio of eigenvalues
- **Spectral Decay**: Rate of eigenvalue falloff
- **Kernel Alignment**: Coherence of feature representations

### Feature Dynamics
- **Intrinsic Dimension**: Effective dimensionality of features
- **Feature Complexity**: Entropy of singular values
- **Manifold Capacity**: Estimated capacity for classification

## Customization

### Adding New Models

```python
# Add to supported_models in MultiModelAnalyzer
self.supported_models['new_arch'] = ['new_model_1', 'new_model_2']

# Customize feature extraction points
def get_feature_extractor(self, model, model_name):
    if 'new_arch' in model_name.lower():
        return_nodes = {'custom_layer': 'custom_name'}
        return create_feature_extractor(model, return_nodes)
```

### Adding New Datasets

```python
# Add to dataset_configs
self.dataset_configs['new_dataset'] = {
    'num_classes': 20,
    'input_size': 64,
    'channels': 3
}

# Customize data loading
def create_data_loader(self, dataset, batch_size, split):
    if dataset == 'new_dataset':
        # Custom dataset loading logic
        pass
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.1.2`
- `timm` (for model loading)
- `torchvision` (for datasets)
- `scikit-learn` (for metrics)
- `matplotlib` (for visualization)
- `tensorboard` (for experiment logging)

## Performance Tips

1. **Device Auto-Detection**: The system automatically detects the best available device (CUDA > MPS > CPU)
2. **GPU Usage**: Use `device='cuda'` for faster feature extraction on NVIDIA GPUs
3. **MPS Support**: Apple Silicon Macs will automatically use MPS for acceleration
4. **Batch Size**: Adjust batch size based on available memory
5. **Sample Limit**: Analysis uses ~1600 samples by default (adjustable)
6. **Parallel Processing**: Multiple workers for data loading

## Experiment Logging with TensorBoard

The system automatically logs all experiments to TensorBoard for easy monitoring and comparison:

### Logged Metrics

**Macroscopic Analysis:**
- Information bottleneck trajectories (I(X;T), I(Y;T))
- Phase transition detection
- Critical layer identification
- Information dynamics (velocity, acceleration)

**Mesoscopic Analysis:**
- NTK spectrum properties
- Feature evolution metrics
- Intrinsic dimensionality
- Representational change

**Metadata:**
- Model and dataset information
- Processing time and sample counts
- Device utilization

### Viewing Logs

```bash
# Launch TensorBoard
python launch_tensorboard.py

# Or manually
tensorboard --logdir=./results/multi_model_analysis/tensorboard_logs

# View in browser at: http://localhost:6006
```

### Log Structure

```
tensorboard_logs/
├── run_20241201_143022/
│   ├── resnet18_cifar10/
│   │   ├── macroscopic/
│   │   ├── mesoscopic/
│   │   ├── layers/
│   │   └── metadata/
│   ├── vit_base_patch16_224_cifar10/
│   └── ...
└── experiment_log_20241201_143022.json
```

## Troubleshooting

### Common Issues

1. **Model Not Found**: Check if model name exists in timm
   ```python
   import timm
   print(timm.list_models())
   ```

2. **Memory Issues**: Reduce batch size or use CPU
   ```python
   analyzer.run_analysis(model, dataset, device='cpu')
   ```

3. **Dataset Download**: Ensure internet connection for first run
   ```python
   # Datasets are loaded from /Users/tanmoy/research/data/
   ```

4. **Feature Extraction Errors**: Check model architecture compatibility
   ```python
   # Some models may need custom feature extraction points
   ```

## Research Applications

This analysis system is particularly useful for:

- **Architecture Comparison**: Understanding representation differences across model types
- **Dataset Analysis**: How different datasets affect representation learning
- **Training Dynamics**: Phase transition analysis during training
- **Model Interpretability**: Identifying critical layers and information flow
- **Transfer Learning**: Understanding representation transfer across tasks

## Citation

If you use this analysis system in your research, please cite:

```bibtex
@software{multi_model_analysis,
  title={Multi-Model Analysis: Macroscopic and Mesoscopic Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional model architectures
- New analysis metrics
- Enhanced visualizations
- Performance optimizations
- Additional datasets

## License

[Your License Here]
