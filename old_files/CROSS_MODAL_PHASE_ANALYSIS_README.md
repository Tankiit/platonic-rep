# Cross-Modal Phase Diagram Analysis

This module extends the NeuREPs phase diagram analysis to cross-modal settings, allowing you to analyze the phase transitions and alignment properties between vision and text models.

## Overview

The cross-modal phase analysis examines three key metrics:

1. **Cross-Modal NTK Stability**: Measures how well the cross-modal neural tangent kernel preserves structural information
2. **Cross-Modal AGOP Magnitude**: Approximates the alignment gradient outer product magnitude across modalities
3. **Cross-Modal Alignment**: Computes representation alignment using platonic metrics or cosine similarity

## Phase Regions

Based on NTK stability thresholds, models are classified into three phases:

- **Lazy Phase** (NTK ≥ 0.9): Models maintain stable representations with minimal adaptation
- **Optimal Phase** (0.7 ≤ NTK < 0.9): Models show balanced adaptation and stability
- **Chaotic Phase** (NTK < 0.7): Models exhibit high adaptation but potential instability

## Usage

### 1. Integrated with Multi-Model Analysis

```python
from multi_model_analysis import MultiModelAnalyzer

# Initialize analyzer
analyzer = MultiModelAnalyzer(output_dir="./results/")

# Run cross-modal phase analysis
results = analyzer.run_cross_modal_phase_analysis(
    vision_models=['resnet18', 'vit_base_patch16_224'],
    text_models=['bert_base', 'roberta_base'],
    dataset='cifar10',
    pretrained=True
)
```

### 2. Standalone Cross-Modal Analysis

```python
from cross_modal_phase_analysis import CrossModalPhaseAnalyzer

# Initialize analyzer
analyzer = CrossModalPhaseAnalyzer(output_dir="./results/cross_modal/")

# Prepare features (in practice, extract from actual models)
vision_features = {
    'resnet18': torch.randn(100, 512),
    'vit_base': torch.randn(100, 768)
}

text_features = {
    'bert_base': torch.randn(100, 768),
    'roberta_base': torch.randn(100, 768)
}

# Run analysis
results = analyzer.analyze_cross_modal_representations(
    vision_features, text_features
)
```

### 3. Command Line Usage

```bash
# Run with cross-modal analysis enabled
python multi_model_analysis.py --cross_modal \
    --vision_models resnet18 vit_base_patch16_224 \
    --text_models bert_base roberta_base \
    --datasets cifar10

# Run standalone cross-modal analysis
python cross_modal_phase_analysis.py
```

### 4. Manual Metric Computation

```python
from cross_modal_phase_analysis import CrossModalPhaseAnalyzer

analyzer = CrossModalPhaseAnalyzer()

# Compute individual metrics
ntk_stability = analyzer.compute_cross_modal_ntk_stability(vision_features, text_features)
agop_magnitude = analyzer.compute_cross_modal_agop_proxy(vision_features, text_features)
alignment = analyzer.compute_cross_modal_alignment(vision_features, text_features)

print(f"NTK Stability: {ntk_stability:.4f}")
print(f"AGOP Magnitude: {agop_magnitude:.4f}")
print(f"Alignment Score: {alignment:.4f}")
```

## Output

The analysis generates:

1. **Phase Diagram Visualization**: 6-panel comprehensive visualization including:
   - Cross-modal alignment heatmap
   - Model pairs in phase space
   - Alignment vs NTK stability relationship
   - Phase region distribution
   - AGOP vs NTK relationship
   - Summary statistics

2. **JSON Results**: Detailed analysis results including:
   - Individual model pair metrics
   - Phase region classifications
   - Summary statistics
   - Metadata

3. **TensorBoard Logging**: Real-time metrics logging for monitoring

## Example Output

```
=== Cross-Modal Phase Analysis ===
Vision models: ['resnet18', 'vit_base_patch16_224']
Text models: ['bert_base', 'roberta_base']
Dataset: cifar10

Computing cross-modal phase diagram for 4 model pairs...
Analyzing model pairs: 100%|██████████| 4/4 [00:02<00:00]

=== Analysis Complete ===
Average metrics across all pairs:
  NTK Stability: 0.8234
  AGOP Magnitude: 0.1567
  Alignment Score: 0.7345

Phase distribution:
  Optimal: 3 pairs
  Lazy: 1 pairs
```

## Key Features

- **Automatic Device Detection**: Supports CUDA, MPS (Apple Silicon), and CPU
- **Flexible Model Support**: Works with any vision and text model features
- **Comprehensive Visualization**: 6-panel phase diagram with detailed analysis
- **TensorBoard Integration**: Real-time logging and monitoring
- **Fallback Mechanisms**: Graceful handling of missing dependencies
- **Extensible Design**: Easy to add new metrics and visualizations

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- Seaborn
- SciPy
- tqdm
- TensorBoard (optional)

## File Structure

```
cross_modal_phase_analysis.py    # Main analysis module
example_cross_modal_analysis.py  # Usage examples
multi_model_analysis.py         # Integrated analysis (updated)
```

## Integration with Existing Framework

The cross-modal phase analysis is fully integrated with the existing multi-model analysis framework:

- Extends `MultiModelAnalyzer` with cross-modal capabilities
- Reuses existing feature extraction infrastructure
- Maintains consistent output formats
- Integrates with TensorBoard logging
- Follows the same experiment tracking patterns

## Advanced Usage

### Custom Model Pairs

```python
# Analyze specific model pairs
model_pairs = [
    ('resnet18', 'bert_base'),
    ('vit_base_patch16_224', 'roberta_base'),
    ('convnext_tiny', 'gpt2_medium')
]

results = analyzer.analyze_cross_modal_representations(
    vision_features, text_features, model_pairs=model_pairs
)
```

### Custom Alignment Metrics

```python
# Use different alignment metrics
alignment = analyzer.compute_cross_modal_alignment(
    vision_features, text_features,
    metric="cycle_knn",  # Options: mutual_knn, cycle_knn, cka, etc.
    topk=15
)
```

### Custom Phase Thresholds

```python
# Modify phase thresholds
analyzer.ntk_thresholds = {
    'lazy': 0.95,
    'optimal': 0.75,
    'chaotic': 0.55
}
```

## Troubleshooting

### Missing Alignment Dependencies

If the platonic alignment package is not available, the system automatically falls back to cosine similarity computation.

### Memory Issues

For large models, consider:
- Reducing batch sizes
- Using fewer samples
- Processing model pairs sequentially

### Feature Dimension Mismatches

The system automatically handles different feature dimensions by:
- Padding smaller features with zeros
- Truncating larger features to match
- Computing metrics on aligned representations

## Citation

If you use this cross-modal phase analysis in your research, please cite:

```bibtex
@article{neurreps2024,
  title={NeuREPs: Neural Representation Phase Transitions},
  author={...},
  journal={...},
  year={2024}
}
```
