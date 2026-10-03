# Phase 1 Experiment with HuggingFace Datasets

## Overview

The Phase 1 controlled misalignment experiment has been updated to use HuggingFace datasets, making it much easier to run without manually downloading and processing COCO data.

## Key Features

- ✅ **HuggingFace Dataset Integration**: Uses `JotDe/mscoco_50k` dataset by default
- ✅ **Automatic Validation Split**: Creates train/validation splits when needed
- ✅ **Controlled Misalignment**: Applies caption corruption at specified levels
- ✅ **MPS Support**: Full Apple Silicon compatibility
- ✅ **Flexible Dataset Limits**: Easy testing with smaller subsets

## Quick Start

### 1. Install Dependencies

```bash
# Make sure you're in the torch-multimodal conda environment
conda activate torch-multimodal

# Dependencies should already be installed, but if needed:
pip install datasets transformers torch torchvision timm
```

### 2. Run the Experiment

```bash
# Quick test with small dataset (1000 train, 200 val samples)
python run_phase1_example.py --mode small

# Full experiment with all available samples
python run_phase1_example.py --mode full

# Custom configuration
python run_phase1_example.py \
    --dataset "JotDe/mscoco_50k" \
    --max-train-samples 5000 \
    --max-val-samples 500 \
    --device mps \
    --batch-size 32
```

## Configuration Options

### Dataset Settings
- `--dataset`: HuggingFace dataset name (default: "JotDe/mscoco_50k")
- `--max-train-samples`: Limit training samples (None = all available)
- `--max-val-samples`: Limit validation samples (None = all available)

### Training Settings
- `--device`: "mps", "cuda", or "cpu"
- `--batch-size`: Override default batch size
- `--mode`: "small" (quick test) or "full" (complete experiment)

## What the Experiment Does

1. **Dataset Loading**: Loads image-caption pairs from HuggingFace
2. **Controlled Misalignment**: Corrupts a fraction of training pairs by reassigning captions
3. **Model Training**: Trains CLIP-style models with different corruption levels
4. **Metric Tracking**: Computes both representation and optimization metrics
5. **Analysis**: Compares which metrics better detect semantic misalignment

## Misalignment Levels

The experiment tests these corruption levels:
- **0%**: No corruption (baseline)
- **25%**: 1/4 of training pairs corrupted
- **50%**: 1/2 of training pairs corrupted
- **75%**: 3/4 of training pairs corrupted
- **100%**: All training pairs corrupted

## Metrics Tracked

### Representation Metrics
- **CKA**: Centered Kernel Alignment
- **Mutual kNN**: Mutual k-nearest neighbors overlap

### Optimization Metrics
- **NTK Similarity**: Neural Tangent Kernel similarity
- **AGOP**: Average Gradient Outer Product analysis

### Performance Metrics
- **Retrieval**: Image-text retrieval performance (R@1, R@5)

## Expected Results

The experiment tests whether:
- Representation metrics fail to detect semantic misalignment
- Optimization metrics correctly identify misalignment
- Which metric class better predicts downstream performance

## Output Structure

```
phase1_results/
├── checkpoints/          # Model checkpoints
├── metrics/             # Metric histories per condition
├── figures/             # Analysis plots
└── experiment.log       # Detailed log file
```

## Troubleshooting

### Dataset Issues
- Dataset downloads automatically on first run
- Ensure internet connection for initial download
- Dataset is cached locally after first download

### Memory Issues
- Reduce `--batch-size` for GPU memory constraints
- Use `--max-train-samples` to limit dataset size
- Use `--device cpu` if GPU issues occur

### MPS Issues
- MPS fallback is automatically enabled for eigenvalue operations
- Some operations will use CPU for compatibility

## Custom Datasets

To use a different dataset:

```bash
python run_phase1_example.py --dataset "your-dataset-name"
```

The dataset should have:
- Image data in 'image', 'jpg', or similar field
- Caption data in 'caption', 'text', or similar field

## Next Steps

1. Run the quick test to verify everything works
2. Check the generated analysis plots
3. Run the full experiment for complete results
4. Examine the correlation between metrics and performance