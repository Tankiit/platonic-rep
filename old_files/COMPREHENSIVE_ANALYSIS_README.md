# Comprehensive Multi-Model Analysis System

This system provides **comprehensive analysis** of neural network representations across different architectures (MLP, CNN, ResNet, ViT) and datasets (CIFAR-10/100, SVHN) with **organized output folders** and **intermediate file saving**.

## 🚀 **Quick Start**

### **1. Run Quick Test (Recommended First)**
```bash
python run_analysis.py --quick
```
- **Duration**: ~10-15 minutes
- **Models**: 1 model (resnet18)
- **Datasets**: 1 dataset (cifar10)
- **Samples**: 256 samples

### **2. Run Full Analysis**
```bash
python run_analysis.py
```
- **Duration**: 1-2 hours
- **Models**: 4 models (resnet18, vit_base_patch16_224, convnext_tiny, mlp_mixer_b16_224)
- **Datasets**: 3 datasets (cifar10, cifar100, svhn)
- **Samples**: 512 samples per model-dataset combination

### **3. View Configuration**
```bash
python run_analysis.py --config
```

## 📁 **Output Structure**

The system creates a **timestamped output directory** with organized folders:

```
results/comprehensive_analysis_20241201_143022/
├── experiment_log.json                    # Complete experiment log
├── comprehensive_results.json             # All results combined
├── ANALYSIS_SUMMARY.md                   # Human-readable summary
├── tensorboard_logs/                     # TensorBoard experiment logs
│   └── run_20241201_143022/             # Timestamped run
└── [model]_[dataset]/                    # Individual model-dataset folders
    ├── analysis_results.json             # Complete analysis results
    ├── extracted_features.pt             # Raw extracted features
    ├── macroscopic_analysis.json         # Macroscopic analysis results
    └── mesoscopic_analysis.json          # Mesoscopic analysis results
```

### **Example Output Folders:**
- `resnet18_cifar10/` - ResNet18 on CIFAR-10
- `vit_base_patch16_224_cifar10/` - ViT on CIFAR-10
- `convnext_tiny_cifar100/` - ConvNeXt on CIFAR-100
- `mlp_mixer_b16_224_svhn/` - MLP-Mixer on SVHN

## 🔧 **Configuration**

### **Edit `analysis_config.py` to customize:**

```python
# Model selection
SELECTED_MODELS = [
    'resnet18',           # ResNet architecture
    'vit_base_patch16_224',  # Vision Transformer
    'convnext_tiny',      # ConvNeXt (CNN)
    'mlp_mixer_b16_224'   # MLP-Mixer
]

# Dataset selection
SELECTED_DATASETS = [
    'cifar10',
    'cifar100', 
    'svhn'
]

# Performance settings
ANALYSIS_CONFIG = {
    'max_batches': 8,     # Number of batches (8 * 64 = 512 samples)
    'batch_size': 64,     # Batch size for data loading
    'pretrained': True,   # Use pretrained models
}
```

### **Quick Mode Configuration:**
```python
QUICK_MODE = {
    'enabled': True,      # Enable for testing
    'max_batches': 4,     # 4 batches (256 samples)
    'batch_size': 32,     # Smaller batch size
    'models': ['resnet18'],  # Only one model
    'datasets': ['cifar10']  # Only one dataset
}
```

## 📊 **What Gets Analyzed**

### **Macroscopic Analysis:**
- **Information Bottleneck Trajectory**: I(X;T) vs I(Y;T) across layers
- **Phase Transitions**: Fitting → compression phase detection
- **Critical Layers**: Key information processing layers
- **Information Dynamics**: Velocity, acceleration, path analysis

### **Mesoscopic Analysis:**
- **NTK Spectrum**: Empirical neural tangent kernel analysis
- **Feature Evolution**: Layer-to-layer similarity and convergence
- **Feature Dynamics**: Intrinsic dimension and complexity
- **Representational Change**: Cumulative drift and topology

### **Cross-Model Comparisons:**
- **Architecture Comparison**: MLP vs CNN vs ResNet vs ViT
- **Dataset Comparison**: CIFAR-10 vs CIFAR-100 vs SVHN
- **Performance Metrics**: Compression, task information, efficiency

## 🎯 **Usage Examples**

### **Basic Usage:**
```bash
# Quick test
python run_analysis.py --quick

# Full analysis
python run_analysis.py

# Show configuration
python run_analysis.py --config
```

### **Custom Analysis:**
```python
from run_comprehensive_analysis import run_comprehensive_analysis

# Run with custom configuration
results, log = run_comprehensive_analysis()

# Access results
for dataset in results:
    for model in results[dataset]:
        print(f"{model} on {dataset}:")
        print(f"  Compression: {results[dataset][model]['macroscopic']['information_flow']['summary']['total_compression']}")
```

### **Individual Model Analysis:**
```python
from multi_model_analysis import MultiModelAnalyzer

analyzer = MultiModelAnalyzer(output_dir="./my_analysis/")

# Analyze single model-dataset
result = analyzer.run_analysis(
    model_name='resnet18',
    dataset='cifar10',
    pretrained=True,
    device=None  # Auto-detect
)
```

## 📈 **Monitoring Progress**

### **Real-time Progress:**
- **Console output** shows progress for each model-dataset
- **Timing information** for each analysis
- **Error handling** with detailed error messages

### **TensorBoard Logging:**
```bash
# Launch TensorBoard
tensorboard --logdir=./results/comprehensive_analysis_[timestamp]/tensorboard_logs

# Or use the launcher script
python launch_tensorboard.py --log_dir ./results/comprehensive_analysis_[timestamp]/tensorboard_logs
```

### **Log Files:**
- `experiment_log.json` - Complete experiment metadata
- `ANALYSIS_SUMMARY.md` - Human-readable summary report
- Individual model-dataset results in separate folders

## 🔍 **Analyzing Results**

### **1. View Summary Report:**
```bash
cat results/comprehensive_analysis_[timestamp]/ANALYSIS_SUMMARY.md
```

### **2. Examine Individual Results:**
```bash
# View macroscopic analysis
cat results/comprehensive_analysis_[timestamp]/resnet18_cifar10/macroscopic_analysis.json

# View mesoscopic analysis  
cat results/comprehensive_analysis_[timestamp]/resnet18_cifar10/mesoscopic_analysis.json
```

### **3. Load Features for Further Analysis:**
```python
import torch

# Load extracted features
features_data = torch.load('./results/comprehensive_analysis_[timestamp]/resnet18_cifar10/extracted_features.pt')

# Access features
features = features_data['features']  # [N, L, D]
layer_names = features_data['layer_names']
feature_dims = features_data['feature_dims']
```

### **4. Compare Across Models:**
```python
import json

# Load comprehensive results
with open('./results/comprehensive_analysis_[timestamp]/comprehensive_results.json', 'r') as f:
    all_results = json.load(f)

# Compare compression across models
for dataset in all_results:
    print(f"\n{dataset.upper()}:")
    for model in all_results[dataset]:
        compression = all_results[dataset][model]['macroscopic']['information_flow']['summary']['total_compression']
        print(f"  {model}: {compression:.3f}")
```

## ⚡ **Performance Optimization**

### **For Faster Testing:**
```python
# In analysis_config.py
QUICK_MODE = {
    'enabled': True,
    'max_batches': 2,      # 2 batches = 128 samples
    'batch_size': 32,      # Smaller batches
    'models': ['resnet18'], # Only one model
    'datasets': ['cifar10'] # Only one dataset
}
```

### **For Production Use:**
```python
ANALYSIS_CONFIG = {
    'max_batches': 16,     # 16 batches = 1024 samples
    'batch_size': 128,     # Larger batches
    'num_workers': 8,      # More parallel workers
}
```

### **Device Optimization:**
- **MPS** (Apple Silicon): ~2-3x faster than CPU
- **CUDA** (NVIDIA): ~5-10x faster than CPU
- **CPU**: Fallback option, slower but reliable

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Out of Memory:**
   ```python
   # Reduce batch size and sample count
   'max_batches': 4,      # Reduce from 8 to 4
   'batch_size': 32,      # Reduce from 64 to 32
   ```

2. **Model Not Found:**
   ```bash
   # Check available models
   python -c "import timm; print(timm.list_models()[:10])"
   ```

3. **Dataset Download Issues:**
   ```python
   # Ensure data directory exists
   data_dir = '/Users/tanmoy/research/data'
   ```

4. **Analysis Errors:**
   - Check individual model-dataset folders for error logs
   - Review `experiment_log.json` for detailed error information
   - Use quick mode to isolate issues

### **Debug Mode:**
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 **Advanced Usage**

### **Custom Model Analysis:**
```python
# Add custom models to analysis_config.py
CUSTOM_MODELS = {
    'custom': ['my_model_1', 'my_model_2']
}

# Update SELECTED_MODELS
SELECTED_MODELS.extend(CUSTOM_MODELS['custom'])
```

### **Custom Dataset Analysis:**
```python
# Add custom datasets
CUSTOM_DATASETS = {
    'my_dataset': {
        'name': 'my_dataset',
        'num_classes': 20,
        'description': 'Custom dataset'
    }
}
```

### **Parallel Processing:**
```python
# Enable parallel model analysis
ADVANCED_CONFIG = {
    'parallel_processing': True,
    'max_workers': 4
}
```

## 🎉 **Success Indicators**

### **Complete Analysis:**
- ✅ All model-dataset combinations completed
- ✅ Individual result folders created
- ✅ Comparison plots generated
- ✅ TensorBoard logs saved
- ✅ Summary report created

### **Expected Output:**
```
🎉 Analysis completed successfully!
Total duration: 3600.00 seconds (60.0 minutes)
Successful analyses: 12
Failed analyses: 0
Output directory: ./results/comprehensive_analysis_20241201_143022/
```

## 🔗 **Integration with Existing Code**

### **Use with Platonic Analysis:**
```python
from platonic import Alignment

# Load your extracted features
features_data = torch.load('./results/comprehensive_analysis_[timestamp]/resnet18_cifar10/extracted_features.pt')

# Use with platonic metric
platonic_metric = Alignment(dataset="minhuh/prh", subset="wit_1024")
score = platonic_metric.score(features_data['features'], metric="mutual_knn")
```

### **Use with Custom Analysis:**
```python
# Load results for further analysis
with open('./results/comprehensive_analysis_[timestamp]/comprehensive_results.json', 'r') as f:
    all_results = json.load(f)

# Your custom analysis here
for dataset in all_results:
    for model in all_results[dataset]:
        # Process results
        pass
```

## 📞 **Support**

### **Getting Help:**
1. Check the `ANALYSIS_SUMMARY.md` for detailed results
2. Review individual model-dataset folders for specific issues
3. Check `experiment_log.json` for error details
4. Use quick mode to isolate problems

### **Reporting Issues:**
- Include the complete error message
- Share the relevant section of `experiment_log.json`
- Specify your system configuration (device, memory, etc.)

---

**Happy Analyzing! 🚀**

This system will give you comprehensive insights into how different neural network architectures process information across various datasets, with all intermediate results saved for further analysis.
