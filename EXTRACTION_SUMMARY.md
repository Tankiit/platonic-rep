# Multi-Layer Feature Extraction Results

## Overview
Successfully extracted embeddings from multiple layers of vision and language models as requested. All features are saved as numpy arrays in the `complete_features/` directory.

## Vision Models Processed

### 1. ResNet-18 (`resnet18/`)
- **conv1**: (30, 64) - Initial convolutional layer
- **layer1**: (30, 64) - First residual block
- **layer2**: (30, 128) - Second residual block
- **layer3**: (30, 256) - Third residual block
- **layer4**: (30, 512) - Fourth residual block
- **avgpool**: (30, 512) - Global average pooled features

### 2. MobileNet-V2 (`mobilenet_v2/`)
- **stem**: (30, 32) - Initial convolution
- **block_3**: (30, 24) - Early inverted residual block
- **block_6**: (30, 32) - Mid inverted residual block
- **block_13**: (30, 96) - Late inverted residual block
- **block_17**: (30, 320) - Final inverted residual block
- **final**: (30, 1280) - Final feature map

### 3. SqueezeNet (`squeezenet/`)
- **conv1**: (30, 64) - Initial convolution
- **fire2**: (30, 128) - Fire module 2
- **fire4**: (30, 128) - Fire module 4
- **fire7**: (30, 256) - Fire module 7
- **fire9**: (30, 384) - Fire module 9
- **conv10**: (30, 512) - Final convolution

### 4. EfficientNet-B0 (`efficientnet_b0/`)
- **stem**: (30, 32) - Stem convolution
- **block_1**: (30, 16) - MBConv block 1
- **block_2**: (30, 24) - MBConv block 2
- **block_4**: (30, 80) - MBConv block 4
- **block_6**: (30, 192) - MBConv block 6
- **avgpool**: (30, 1280) - Global average pooled features

## Language Models Processed

### 1. DistilBERT (`distilbert/`)
- **embeddings**: (30, 768) - Input embeddings
- **layer_0**: (30, 768) - Transformer layer 0
- **layer_1**: (30, 768) - Transformer layer 1
- **layer_2**: (30, 768) - Transformer layer 2
- **layer_3**: (30, 768) - Transformer layer 3
- **layer_4**: (30, 768) - Transformer layer 4
- **layer_5**: (30, 768) - Transformer layer 5
- **final**: (30, 768) - Final hidden states

### 2. ALBERT-Base-V2 (`albert/`)
- **embeddings**: (30, 768) - Input embeddings
- **layer_0**: (30, 768) - ALBERT layer 0
- **layer_1**: (30, 768) - ALBERT layer 1
- **layer_2**: (30, 768) - ALBERT layer 2
- **layer_3**: (30, 768) - ALBERT layer 3
- **layer_4**: (30, 768) - ALBERT layer 4
- **layer_5**: (30, 768) - ALBERT layer 5
- **final**: (30, 768) - Final hidden states

## Model Configuration Details

### Vision Models (from torchvision)
```python
vision_models = {
    'mobilenet_v2': torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1'),
    'squeezenet': torchvision.models.squeezenet1_1(weights='IMAGENET1K_V1'),
    'efficientnet_b0': torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1'),
    'resnet18': torchvision.models.resnet18(weights='IMAGENET1K_V1')
}
```

### Language Models (from transformers)
```python
language_models = {
    'distilbert': AutoModel.from_pretrained('distilbert-base-uncased'),
    'albert': AutoModel.from_pretrained('albert-base-v2'),
}
```

## Test Data
- **30 synthetic image-text pairs** with varied visual patterns
- **Images**: 224x224 RGB with different patterns (solid colors, stripes, gradients, noise, etc.)
- **Texts**: Descriptive sentences matching the visual patterns

## Feature Processing
- **Vision**: Global average pooling applied to spatial feature maps
- **Language**: Mean pooling across sequence length dimension
- **Format**: All features saved as `.npy` files with shape (samples, features)

## Statistics
- ✅ **6 models processed** (4 vision + 2 language)
- ✅ **40 total layers** extracted
- ✅ **561,120 total feature values**
- ✅ All features successfully saved to disk

## Files Generated
- `feature_extraction_config.json` - Configuration for all models and datasets
- `complete_feature_extractor.py` - Working extraction script
- `complete_features/` - Directory containing all extracted features
- Model subdirectories with layer-wise `.npy` files

## Next Steps
The extracted multi-layer embeddings can now be used for:
1. Cross-modal alignment analysis
2. Layer-wise representation similarity studies
3. Platonic representation hypothesis testing
4. Multi-scale geometric analysis

All requested models have been successfully processed with their embeddings extracted across multiple layers for both vision (images) and language (text) modalities.