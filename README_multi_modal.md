# Multi-Modal Embedding Comparison System

A comprehensive system for comparing language and visual embeddings across multiple medium and small models, analyzing cross-modal alignment and representation similarities.

## Features

- **Multiple Vision Models**: Support for ResNet, EfficientNet, MobileNet, Vision Transformers, and CLIP models
- **Multiple Language Models**: Support for BERT family, GPT-2, T5, and Sentence Transformers
- **Cross-Modal Alignment Metrics**: CKA, Cosine Similarity, Procrustes Alignment, Mutual Nearest Neighbors
- **Comprehensive Visualizations**: Alignment matrices, embedding statistics, representation similarity matrices
- **Flexible Configurations**: Predefined configs from minimal to comprehensive comparisons
- **Memory Efficient**: Batch processing and automatic device detection (CUDA/MPS/CPU)

## Installation

```bash
# Install dependencies
pip install -r requirements_multi_modal.txt
```

## Quick Start

### 1. Test with Minimal Configuration (2x2 models)
```bash
python run_multi_modal_comparison.py --config minimal --num-samples 100
```

### 2. Small Models Only (3x3 models)
```bash
python run_multi_modal_comparison.py --config small --num-samples 500
```

### 3. Medium Models (4x4 models)
```bash
python run_multi_modal_comparison.py --config medium --num-samples 500
```

### 4. Mixed Small and Medium (5x5 models)
```bash
python run_multi_modal_comparison.py --config mixed --num-samples 500
```

### 5. CLIP-Focused Comparison
```bash
python run_multi_modal_comparison.py --config clip_focused --num-samples 500
```

### 6. Comprehensive Analysis (10x9 models)
```bash
python run_multi_modal_comparison.py --config comprehensive --num-samples 1000
```

## Custom Model Selection

You can specify your own model combinations:

```bash
# Custom vision and language models
python run_multi_modal_comparison.py \
    --vision-models resnet18 mobilenet_v2 vit_tiny_patch16_224 \
    --language-models distilbert-base gpt2 sentence-bert-base \
    --num-samples 500
```

## Available Models

### Vision Models

**Small Models:**
- `resnet18` - ResNet-18 (11M params)
- `mobilenet_v2` - MobileNetV2 (3.5M params)
- `efficientnet_b0` - EfficientNet-B0 (5.3M params)

**Medium Models:**
- `resnet34` - ResNet-34 (21M params)
- `resnet50` - ResNet-50 (25M params)
- `efficientnet_b1` - EfficientNet-B1 (7.8M params)
- `vit_tiny_patch16_224` - Vision Transformer Tiny (5.7M params)
- `vit_small_patch16_224` - Vision Transformer Small (22M params)

**CLIP Models:**
- `clip_rn50` - CLIP with ResNet-50 backbone
- `clip_vit_b32` - CLIP ViT-B/32
- `clip_vit_b16` - CLIP ViT-B/16

### Language Models

**Small Models:**
- `distilbert-base` - DistilBERT (66M params)
- `albert-base-v2` - ALBERT Base v2 (11M params)
- `gpt2` - GPT-2 Small (124M params)

**Medium Models:**
- `bert-base` - BERT Base (110M params)
- `roberta-base` - RoBERTa Base (125M params)
- `gpt2-medium` - GPT-2 Medium (355M params)
- `t5-small` - T5 Small (60M params)

**Sentence Transformers:**
- `sentence-bert-base` - all-MiniLM-L6-v2 (22M params)
- `sentence-bert-large` - all-mpnet-base-v2 (110M params)

## Predefined Configurations

- **minimal**: 2 vision × 2 language models (4 comparisons)
- **small**: 3 vision × 3 language models (9 comparisons)
- **medium**: 4 vision × 4 language models (16 comparisons)
- **mixed**: 5 vision × 5 language models (25 comparisons)
- **clip_focused**: 4 vision × 4 language models including CLIP
- **comprehensive**: 10 vision × 9 language models (90 comparisons)

## Outputs

The system generates the following outputs in the results directory:

1. **Alignment Matrices** (`alignment_matrix_*.png`): Heatmaps showing cross-modal alignment scores
2. **Embedding Statistics** (`*_embedding_stats.png`): Statistical analysis of embeddings
3. **RSM Plots** (`rsm_*.png`): Representation Similarity Matrices
4. **JSON Results** (`comparison_results.json`): Complete numerical results

## System Information

Check your system capabilities:
```bash
python run_multi_modal_comparison.py --check-system
```

List available configurations:
```bash
python run_multi_modal_comparison.py --list-configs
```

## Programmatic Usage

```python
from multi_modal_embedding_comparison import MultiModalComparison

# Create comparison instance
comparison = MultiModalComparison(output_dir='./results')

# Run comparison
results = comparison.run_comparison(
    vision_models=['resnet18', 'vit_tiny_patch16_224'],
    language_models=['bert-base', 'gpt2'],
    dataset_name='synthetic',
    num_samples=500
)

# Access results
alignment_scores = results['alignment_results']
vision_features = results['vision_features']
text_features = results['text_features']
```

## Memory Requirements

- **Minimal config**: ~2GB GPU memory
- **Small config**: ~4GB GPU memory
- **Medium config**: ~8GB GPU memory
- **Comprehensive config**: ~16GB GPU memory

The system automatically detects and uses the best available device (CUDA > MPS > CPU).

## Datasets

- **synthetic**: Fast synthetic data for testing (default)
- **mscoco**: MS COCO captions (requires internet)
- **conceptual_captions**: Conceptual Captions dataset

## Tips for Large-Scale Comparisons

1. Start with smaller configurations to test your setup
2. Use smaller `--num-samples` for initial experiments
3. The system processes models sequentially to manage memory
4. Results are saved incrementally

## Troubleshooting

If you encounter memory issues:
1. Reduce `--num-samples`
2. Use smaller model configurations
3. Process models in smaller batches
4. Ensure GPU memory is cleared between runs

## Citation

This system builds upon the Platonic Representation Hypothesis research. If you use this code, please cite the original work and this implementation.