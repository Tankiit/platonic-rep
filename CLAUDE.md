# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the implementation repository for "The Platonic Representation Hypothesis" research project. The codebase enables feature extraction from language and vision models, measurement of cross-modal alignment, and evaluation against the Platonic Representation Hypothesis.

## Common Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install as pip package for library usage
pip install -e .
```

### Feature Extraction

**Start with smallest models for MPS/limited GPU memory:**
```bash
# Direct model specification (NEW: no need to modify tasks.py!)
python extract_features.py --models bigscience/bloomz-560m --pool avg
python extract_features.py --models vit_tiny_patch16_224.augreg_in21k --pool cls

# Extract both smallest language and vision models together
python extract_features.py --models bigscience/bloomz-560m vit_tiny_patch16_224.augreg_in21k

# Multiple small language models
python extract_features.py --models bigscience/bloomz-560m bigscience/bloomz-1b1 --pool avg
```

**Legacy approach using modelsets:**
```bash
# Custom modelset (560M, 1.1B language + giant vision model)
python extract_features.py --modelset custom --modality language --pool avg

# Validation set (12 LLMs, 17 vision models)
python extract_features.py --modelset val --modality language --pool avg
python extract_features.py --modelset val --modality vision --pool cls

# Memory-efficient large-scale extraction
python extract_features.py --modelset exhaustive --modality language --batch_size 4 --qlora
./run_exhaustive_extraction.sh
```

### Alignment Measurement
```bash
# Measure vision-language alignment
python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset val \
    --modality_x language --pool_x avg --modality_y vision --pool_y cls

# Results stored in ./results/alignment/
```

### SLURM Cluster Usage

**Single GPU job for small models:**
```bash
#!/bin/bash
#SBATCH --job-name=platonic-extract
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load python/3.11 cuda/11.8
source venv/bin/activate

# Extract specific small models
python extract_features.py --models bigscience/bloomz-560m bigscience/bloomz-1b1 --pool avg --batch_size 8
python extract_features.py --models vit_tiny_patch16_224.augreg_in21k vit_small_patch16_224.augreg_in21k --pool cls --batch_size 16
```

**High-memory GPU job for large models:**
```bash
#!/bin/bash
#SBATCH --job-name=platonic-exhaustive
#SBATCH --output=logs/%j.out  
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-high-mem
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load python/3.11 cuda/11.8
source venv/bin/activate

# Exhaustive extraction with QLoRA for memory efficiency
python extract_features.py --modelset exhaustive --modality language --pool avg --batch_size 4 --qlora
python extract_features.py --modelset exhaustive --modality vision --pool cls --batch_size 8
```

**Array job for parallel model processing:**
```bash
#!/bin/bash
#SBATCH --job-name=platonic-array
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10

# Define model lists
LANG_MODELS=(
    "bigscience/bloomz-560m"
    "bigscience/bloomz-1b1"
    "bigscience/bloomz-3b"
    "openlm-research/open_llama_3b"
    "openlm-research/open_llama_7b"
)

VISION_MODELS=(
    "vit_tiny_patch16_224.augreg_in21k"
    "vit_small_patch16_224.augreg_in21k"
    "vit_base_patch16_224.augreg_in21k"
    "vit_small_patch14_dinov2.lvd142m"
    "vit_base_patch14_dinov2.lvd142m"
)

module load python/3.11 cuda/11.8
source venv/bin/activate

# Process different models based on array index
if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
    MODEL=${LANG_MODELS[$((SLURM_ARRAY_TASK_ID-1))]}
    python extract_features.py --models $MODEL --pool avg --batch_size 8
else
    MODEL=${VISION_MODELS[$((SLURM_ARRAY_TASK_ID-6))]}
    python extract_features.py --models $MODEL --pool cls --batch_size 16
fi
```

### Library Usage
```python
import platonic

# Setup metric for evaluation
platonic_metric = platonic.Alignment(
    dataset="minhuh/prh", 
    subset="wit_1024",
    models=["dinov2_g", "clip_h"]
)

# Score your model features
score = platonic_metric.score(features, metric="mutual_knn", topk=10, normalize=True)
```

## Architecture Overview

### Core Components

**Feature Extraction Pipeline**
- `extract_features.py`: Main extraction script supporting 100+ models
- `models.py`: Model loading utilities with QLoRA support and dtype management
- `tasks.py`: Model set definitions (val, test, exhaustive, custom) with 70+ LLMs and 30+ vision models

**Alignment Measurement**  
- `measure_alignment.py`: Cross-modal alignment computation
- `metrics.py`: Implementation of 8 alignment metrics (mutual_knn, cycle_knn, CKA, etc.)
- `platonic/alignment.py`: Library interface for real-time model evaluation

**Model Sets**
- `val`: Balanced development set (12 LLMs + 17 vision models)
- `test`: Modern large models (8 LLMs)  
- `exhaustive`: Comprehensive research set (70+ LLMs + 30+ vision models)
- `custom`: Minimal debug set (2 LLMs + 1 vision model)

### Data Flow
1. Features extracted per layer/block → `results/features/`
2. Cross-modal alignment computed → `results/alignment/`
3. Library provides real-time scoring during model training/evaluation

### Key Files
- `tasks.py:68-288`: Model definitions and size estimation
- `extract_features.py:48-150`: LLM feature extraction with pooling strategies
- `metrics.py:15-26`: Supported alignment metrics
- `platonic/__init__.py:4-111`: Precomputed feature download URLs

## Model Support

**Language Models**: BLOOM, LLaMA, Gemma, Mistral, GPT-2/Neo/J, Pythia, CodeGen, Falcon, MPT, OPT families

**Vision Models**: ViT variants, DeiT, CLIP, MAE, DINOv2, DINO models

**Pooling Strategies**: avg (average), cls (CLS token), last (last token), max (max pooling)

## Hardware Requirements

**For exhaustive extraction**:
- GPU Memory: 24GB+ recommended
- System RAM: 32GB+ 
- Storage: 500GB+ for all features
- Time: Several days for complete extraction

**Optimization options**: `--qlora` for 4-bit quantization, adjust `--batch_size`, process modalities separately

## Development Notes

- No test framework detected - validation through feature extraction and alignment scoring
- Memory management critical for large model processing
- Automatic model availability checking and graceful error handling
- Features saved as PyTorch tensors with metadata (num_params, loss, bpb, mask)
- Supports both research (exhaustive analysis) and production (real-time scoring) use cases