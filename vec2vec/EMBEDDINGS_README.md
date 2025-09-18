# Multi-Model Embedding Extraction System

This system provides comprehensive tools for extracting and analyzing embeddings from multiple models across diverse datasets.

## Features

### 1. **extract_embeddings.py** - Basic embedding extraction
- Extract from sentence-transformers models
- Batch processing support
- Multiple output formats (npz, pkl, pt, h5)

### 2. **extract_hf_embeddings.py** - Advanced HuggingFace extraction
- Extract from any HuggingFace model
- Layer-wise extraction
- Multiple pooling strategies (mean, cls, max)
- Organized model families for systematic testing

### 3. **run_extraction_experiments.py** - Analysis pipeline
- Similarity analysis across models
- Cross-dataset consistency testing
- Visualization generation

## Installation

```bash
# Install required packages
pip install transformers sentence-transformers datasets torch numpy pandas matplotlib seaborn scikit-learn
```

## Usage Examples

### Quick Test
```bash
# Run a quick test with minimal data
python run_extraction_experiments.py --mode quick_test
```

### Extract Embeddings from Specific Models
```bash
# Using sentence-transformers models
python extract_embeddings.py --input texts.txt --models sbert gte e5 --compute-stats

# Using HuggingFace models with specific configuration
python extract_hf_embeddings.py \
    --models bert-base-uncased gpt2 \
    --datasets general questions \
    --layers -1 -2 \
    --pooling mean cls \
    --output-dir ./embeddings
```

### Extract from Model Families
```bash
# Extract from all BERT variants
python extract_hf_embeddings.py --model-family bert_variants --datasets all

# Extract from all model families
python extract_hf_embeddings.py --model-family all --datasets all --max-samples 100
```

### Custom Text Input
```bash
# From file
python extract_embeddings.py --input my_texts.txt --models all

# Direct text
python extract_embeddings.py --texts "text1" "text2" "text3" --models sbert gte

# Custom texts with HF models
python extract_hf_embeddings.py --custom-texts my_data.txt --models bert-base-uncased gpt2
```

### Full Pipeline
```bash
# Run extraction, analysis, and visualization
python run_extraction_experiments.py --mode full --embeddings-dir ./my_embeddings
```

### Analysis Only
```bash
# Analyze existing embeddings
python run_extraction_experiments.py --mode analyze --embeddings-dir ./embeddings

# Generate visualizations
python run_extraction_experiments.py --mode visualize --embeddings-dir ./embeddings
```

## Output Structure

```
embeddings/
├── metadata.json                     # Extraction configuration
├── extraction_results.json           # Detailed extraction log
├── similarity_analysis.json          # Analysis results
├── visualizations/
│   ├── similarity_heatmap.png       # Model similarity matrix
│   └── pca_visualization.png        # PCA projection of embedding spaces
└── [dataset]_[model]_[layer]_[pooling].npz  # Individual embedding files
```

## Available Models

### Sentence-Transformers (extract_embeddings.py)
- `sbert`, `gte`, `gtr`, `e5`, `stella`, `jina`, `clip`, etc.
- Run `python extract_embeddings.py --list-models` for full list

### HuggingFace Models (extract_hf_embeddings.py)

**BERT Family:**
- bert-base-uncased, bert-large-uncased
- roberta-base, roberta-large
- distilbert-base-uncased
- microsoft/deberta-v3-base

**GPT Family:**
- gpt2, gpt2-medium, gpt2-large
- EleutherAI/gpt-neo-1.3B

**T5 Family:**
- google/t5-v1_1-small/base/large
- google/flan-t5-base

**Specialized:**
- allenai/scibert_scivocab_uncased (Scientific)
- microsoft/codebert-base (Code)
- emilyalsentzer/Bio_ClinicalBERT (Biomedical)

## Available Datasets

- **general**: Wikipedia text
- **questions**: Natural Questions
- **scientific**: PubMed abstracts
- **code**: GitHub code snippets
- **conversation**: Daily dialog

## Key Parameters

### Layers
- `-1`: Last layer (default)
- `-2, -3, -4`: Earlier layers
- Layer choice affects abstraction level

### Pooling Strategies
- `mean`: Average of token embeddings (default)
- `cls`: [CLS] token embedding
- `max`: Max pooling across tokens

### Batch Size
- Adjust based on GPU memory
- Default: 32

## Analysis Metrics

The system computes:
1. **Within-family similarity**: How similar are models from the same family?
2. **Cross-family similarity**: Do different architectures learn similar representations?
3. **Dataset consistency**: Are representations consistent across different text types?
4. **Layer-wise analysis**: How does similarity change across layers?

## Tips for Large-Scale Extraction

1. **Start small**: Test with `--max-samples 100` first
2. **Use GPU**: Add `CUDA_VISIBLE_DEVICES=0` before command
3. **Monitor memory**: Some large models require significant RAM/VRAM
4. **Incremental extraction**: Extract one family at a time
5. **Save frequently**: Results are saved after each model

## Example Workflow

```bash
# 1. Quick test to ensure everything works
python run_extraction_experiments.py --mode quick_test

# 2. Extract embeddings from specific models
python extract_hf_embeddings.py \
    --model-family bert_variants \
    --datasets general questions \
    --max-samples 500 \
    --output-dir ./bert_embeddings

# 3. Analyze the results
python run_extraction_experiments.py --mode analyze --embeddings-dir ./bert_embeddings

# 4. Create visualizations
python run_extraction_experiments.py --mode visualize --embeddings-dir ./bert_embeddings

# 5. Scale up to more models and data
python extract_hf_embeddings.py \
    --model-family all \
    --datasets all \
    --layers -1 -2 -3 \
    --pooling mean cls \
    --max-samples 1000 \
    --output-dir ./full_embeddings
```

## Loading Saved Embeddings

```python
import numpy as np

# Load individual embedding file
data = np.load('embeddings/general_bert_base_uncased_layer-1_mean.npz')
embeddings = data['embeddings']
model_name = str(data['model_name'])
dataset_name = str(data['dataset_name'])

# Load multiple embeddings
from pathlib import Path
import json

embeddings_dir = Path('./embeddings')
with open(embeddings_dir / 'extraction_results.json', 'r') as f:
    results = json.load(f)

for result in results:
    if 'filepath' in result:
        data = np.load(result['filepath'])
        embeddings = data['embeddings']
        print(f"{result['model']} on {result['dataset']}: {embeddings.shape}")
```

## Troubleshooting

**Out of Memory:**
- Reduce batch_size
- Use smaller models
- Reduce max_samples
- Use CPU instead of GPU for small models

**Slow Extraction:**
- Use GPU: ensure CUDA is available
- Increase batch_size if memory allows
- Use smaller models or fewer layers

**Dataset Loading Issues:**
- Some datasets require authentication
- Internet connection needed for first download
- Datasets are cached after first use

**Model Loading Issues:**
- First download can be slow
- Models are cached in ./hf_cache
- Some models require `trust_remote_code=True`