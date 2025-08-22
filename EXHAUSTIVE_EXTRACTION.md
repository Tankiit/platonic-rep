# Exhaustive Feature Extraction

This document describes the enhanced feature extraction capabilities that support comprehensive model sets.

## New Features

### Exhaustive Model Set
The `exhaustive` modelset includes comprehensive coverage of major model families:

**Language Models (70+ models):**
- BLOOM family (560M - 7B parameters)
- LLaMA family (7B - 70B parameters) 
- Gemma (2B - 7B parameters)
- Mistral/Mixtral (7B - 8x7B parameters)
- GPT-2/Neo/J (125M - 6B parameters)
- Pythia (70M - 12B parameters)
- CodeGen (350M - 16B parameters)
- Falcon (7B - 40B parameters)
- MPT (7B - 30B parameters)
- OPT (125M - 66B parameters)

**Vision Models (30+ models):**
- Vision Transformers (ViT) - various sizes and pretraining
- Data-efficient Image Transformers (DeiT)
- CLIP models with different backbones
- Masked Autoencoders (MAE)
- DINOv2 self-supervised models
- DINO models

### Enhanced Pooling Options
- `avg`: Average pooling over sequence/spatial dimensions
- `cls`: CLS token extraction (ViTs) or first token (LLMs)
- `last`: Last token extraction (LLMs only)
- `max`: Max pooling over dimensions

### Robust Error Handling
- Graceful handling of model loading failures
- Automatic fallback for unsupported architectures
- Detailed error reporting and logging
- Memory cleanup after each model

## Usage Examples

### Basic Exhaustive Extraction
```bash
# Extract all models (language + vision)
python extract_features.py --modelset exhaustive --modality all

# Extract only language models
python extract_features.py --modelset exhaustive --modality language --pool avg

# Extract only vision models  
python extract_features.py --modelset exhaustive --modality vision --pool cls
```

### Using the Convenience Script
```bash
# Run full exhaustive extraction with default settings
./run_exhaustive_extraction.sh

# Run with custom dataset and batch size
./run_exhaustive_extraction.sh prh wit_1024 8 2048
```

### Advanced Options
```bash
# Force re-extraction with QLoRA quantization
python extract_features.py --modelset exhaustive --force_remake --qlora

# Custom output directory and larger batches
python extract_features.py --modelset exhaustive --output_dir ./my_features --batch_size 16
```

## Model Selection Guidelines

- **val**: Balanced set for validation/development (12 LLMs + 17 vision models)
- **test**: Modern large models for final evaluation (8 LLMs)
- **exhaustive**: Comprehensive coverage for research (70+ LLMs + 30+ vision models)
- **custom**: Minimal set for testing/debugging (2 LLMs + 1 vision model)

## Hardware Requirements

### For Exhaustive Extraction:
- **GPU Memory**: 24GB+ recommended (some models require significant memory)
- **System RAM**: 32GB+ recommended
- **Storage**: 500GB+ for all features
- **Time**: Several days for complete exhaustive extraction

### Resource Optimization:
- Use `--qlora` for 4-bit quantization to reduce memory usage
- Adjust `--batch_size` based on available GPU memory
- Run modalities separately (`--modality language` or `--modality vision`)
- Use `--force_remake` selectively to avoid re-processing

## Output Structure

Features are saved as PyTorch tensors with metadata:
```
results/features/
├── dataset/
│   ├── subset/
│   │   ├── model_name_pool-strategy.pt
│   │   └── ...
```

Each feature file contains:
- `feats`: Extracted features tensor [samples, layers, hidden_dim]
- `num_params`: Model parameter count
- `loss`: Average loss (LLMs only)
- `bpb`: Bits per byte (LLMs only)
- `mask`: Attention mask (LLMs only)

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size or use QLoRA
2. **Model Not Found**: Check internet connection and model availability
3. **Slow Processing**: Use smaller model sets or increase batch size
4. **Disk Space**: Monitor storage usage, clean up temporary files

### Performance Tips:
- Process language and vision models separately
- Use background processing for long runs
- Monitor GPU memory usage with `nvidia-smi`
- Clean up intermediate files regularly
