# Feature Representations Directory

This directory contains extracted features from various MLX models organized by model name and dataset.

## Directory Structure

```
representations/
├── model_name/
│   ├── dataset_name_model_name_features.h5
│   └── ...
└── ...
```

## File Naming Convention

Features are saved as HDF5 files with the naming pattern:
`{dataset_name}_{model_name}_features.h5`

Where:
- `dataset_name`: Name of the dataset (e.g., "Flickr8k_Sample", "MS_COCO_Captions_1K")
- `model_name`: Name of the model with special characters replaced (e.g., "nanoLLaVA_1_5", "Phi_3_mini")

## File Contents

Each HDF5 file contains:

### Datasets:
- `vision_features`: Vision features (if available) - shape: (n_samples, feature_dim)
- `text_features`: Text features - shape: (n_samples, feature_dim)

### Attributes:
- `model_name`: Original model name
- `dataset_name`: Dataset name
- `extraction_time`: Time taken for extraction (seconds)
- `n_samples`: Number of samples
- `feature_dim`: Feature dimensionality

### Groups:
- `dataset_metadata`: Contains dataset-specific metadata

## Model Categories

### Small Models (8-16GB MacBooks)
- **VLMs**: nanoLLaVA-1.5, SmolVLM-Instruct, Qwen2-VL-2B-Instruct
- **Language**: Phi-3-mini-4k, Qwen2.5-1.5B-Instruct, Gemma-2-2B
- **Vision**: CLIP-ViT-Base-32, CLIP-ViT-Base-16

### Medium Models (16-32GB MacBooks)
- **VLMs**: LLaVA-v1.6-Mistral-7B, Llama-3.2-11B-Vision, Qwen2-VL-7B-Instruct
- **Language**: Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct, CodeLlama-7B-Instruct
- **Vision**: CLIP-ViT-Large-14, OpenCLIP-ViT-B-32

## Dataset Categories

### Tiny Datasets (<100MB, <1K samples)
- MNIST-Text: Handwritten digits with descriptions
- Flickr8k-Sample: 100 images with 5 captions each
- CIFAR-10-Captions: CIFAR images with generated captions

### Small Datasets (100MB-1GB, 1K-10K samples)
- MS-COCO-Captions-1K: High-quality images with multiple captions
- Conceptual-Captions-3K: Web images with alt-text captions
- Flickr30k-Entities-Sample: Images with detailed entity annotations
- WIT-Sample: Wikipedia images with contextual text

### Medium Datasets (1GB-5GB, 10K-50K samples)
- MS-COCO-2017-Val: COCO validation set with 5 captions per image
- Visual-Genome-QA: Complex scene graphs and QA pairs
- LAION-5K: Large-scale web crawl image-text pairs

## Usage Example

```python
import h5py
import numpy as np

# Load features
with h5py.File('representations/nanoLLaVA_1_5/Flickr8k_Sample_nanoLLaVA_1_5_features.h5', 'r') as f:
    vision_features = np.array(f['vision_features'])
    text_features = np.array(f['text_features'])

    print(f"Model: {f.attrs['model_name']}")
    print(f"Dataset: {f.attrs['dataset_name']}")
    print(f"Samples: {f.attrs['n_samples']}")
    print(f"Feature dim: {f.attrs['feature_dim']}")
```

## Memory Requirements

- **Tiny datasets**: ~10-100MB per model
- **Small datasets**: ~100MB-1GB per model
- **Medium datasets**: ~1-5GB per model

Total storage requirements:
- Small model category: ~5-10GB
- Medium model category: ~20-50GB
- Large model category: ~50-100GB