# Geometric Vec2Vec Alignment

A non-adversarial approach to embedding space alignment using geometric transformations based on platonic representation insights.

## Overview

Instead of using adversarial training (GANs) to align embedding spaces, this approach directly computes geometric transformations that preserve universal structure. The key insight is that we don't need to fight against a discriminator - we can directly optimize for geometric alignment using mathematical principles.

## Features

- **Multiple alignment methods**: Procrustes, CCA, Low-rank alignment
- **Adaptive method selection**: Automatically chooses the best method based on data characteristics
- **Platonic insights integration**: Can incorporate spectral predictions for method selection
- **Visualization support**: Built-in t-SNE visualization of alignments
- **Comprehensive evaluation**: Cosine similarity and MSE metrics

## Installation

The script uses the existing codebase infrastructure. Make sure you have the required dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn torch sentence-transformers
```

## Usage

### Basic Usage

```bash
python geometric_alignment.py --model_a stella --model_b gte --n_samples 2000 --visualize
```

### Command Line Options

- `--model_a`: First model name (e.g., 'stella', 'gte', 'gist')
- `--model_b`: Second model name
- `--dataset`: Dataset to use ('nq', 'mimic', etc.)
- `--n_samples`: Number of samples to use for alignment
- `--method`: Alignment method ('auto', 'procrustes', 'cca', 'lowrank')
- `--device`: Device to use ('cpu', 'cuda')
- `--visualize`: Create visualization
- `--output_dir`: Directory to save results

### Python API

```python
from geometric_alignment import GeometricVec2Vec, extract_embeddings, load_sample_texts

# Load data
texts = load_sample_texts('nq', 1000)

# Extract embeddings
embeddings_A = extract_embeddings('stella', texts)
embeddings_B = extract_embeddings('gte', texts)

# Fit alignment
aligner = GeometricVec2Vec(alignment_method='auto')
aligner.fit(embeddings_A, embeddings_B)

# Transform embeddings
A_transformed = aligner.transform(embeddings_A, direction='A_to_B')

# Evaluate alignment
metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
print(f"Cosine similarity: {metrics['cosine_similarity']:.4f}")
```

## Alignment Methods

### 1. Procrustes Alignment
- Best for: Models in the same "phase" (similar geometric structure)
- Finds optimal orthogonal transformation
- Preserves angles and relative distances
- Computationally efficient

### 2. Canonical Correlation Analysis (CCA)
- Best for: Different dimensional spaces with shared semantic structure
- Finds maximally correlated linear combinations
- Handles dimension mismatches well
- Good for finding shared semantic axes

### 3. Low-rank Alignment
- Best for: Data with universal low-rank structure
- Projects to low-rank subspace before alignment
- More robust to noise
- Leverages platonic representation insights

### 4. Adaptive Selection
- Automatically chooses method based on:
  - Dimension ratios
  - Effective rank of data
  - Data characteristics

## Testing

Run the test suite to verify functionality:

```bash
python test_geometric_alignment.py
```

This will test:
- Synthetic data with known transformations
- Different dimensional spaces
- Low-rank data structures

## Output

The script saves:
- `alignment_<model_a>_<model_b>.json`: Alignment results and metrics
- `alignment_viz_<model_a>_<model_b>.png`: Visualization (if --visualize is used)

## Integration with Existing Codebase

The script is designed to work with the existing vec2vec infrastructure:
- Uses `utils.model_utils` for loading encoders
- Uses `utils.streaming_utils` for loading datasets  
- Compatible with existing model flags and configurations
- Can be extended to work with trained translators

## Platonic Insights Integration

The alignment can incorporate spectral predictions:

```python
spectral_predictions = {
    'phase_transition': True,  # Use low-rank alignment
    'high_correlation': False,
    'shared_structure': 0.85
}

aligner = GeometricVec2Vec(
    alignment_method='auto',
    use_platonic_insights=True
)
aligner.fit(embeddings_A, embeddings_B, spectral_predictions)
```

## Advantages over Adversarial Training

1. **No adversarial dynamics**: Direct optimization without discriminator
2. **Faster convergence**: Closed-form or simple iterative solutions
3. **More stable**: No mode collapse or training instabilities
4. **Interpretable**: Clear geometric meaning of transformations
5. **Universal structure preserving**: Leverages mathematical insights

## Future Extensions

- Integration with spectral analysis for automatic method selection
- Support for multiple embedding spaces simultaneously  
- Hierarchical alignment for different granularities
- Online/incremental alignment updates
- GPU acceleration for large-scale alignments