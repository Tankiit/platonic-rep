#!/usr/bin/env python3
"""
Complete multi-layer feature extraction for all specified models
"""

import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteFeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def create_test_data(self, n_samples: int = 50):
        """Create diverse test data"""
        logger.info(f"Creating {n_samples} test samples")

        images = []
        texts = []

        for i in range(n_samples):
            # Create varied images
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)

            pattern_type = i % 8
            if pattern_type == 0:  # Red
                img_array[:, :, 0] = 255
                text = "A bright red colored image"
            elif pattern_type == 1:  # Green
                img_array[:, :, 1] = 255
                text = "A vibrant green colored image"
            elif pattern_type == 2:  # Blue
                img_array[:, :, 2] = 255
                text = "A deep blue colored image"
            elif pattern_type == 3:  # Stripes
                img_array[::10, :] = [255, 255, 0]  # Yellow stripes
                text = "An image with horizontal yellow stripes"
            elif pattern_type == 4:  # Checkerboard
                img_array[::20, ::20] = [255, 0, 255]  # Purple squares
                text = "A checkerboard pattern with purple squares"
            elif pattern_type == 5:  # Gradient
                for j in range(224):
                    img_array[j, :] = [j, 100, 224-j]
                text = "A smooth color gradient from red to blue"
            elif pattern_type == 6:  # Noise
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                text = "A random colorful noise pattern with many colors"
            else:  # Mixed
                img_array[:112, :] = [255, 0, 0]  # Top red
                img_array[112:, :] = [0, 0, 255]  # Bottom blue
                text = "An image split between red and blue sections"

            images.append(Image.fromarray(img_array))
            texts.append(text)

        return images, texts

    def extract_resnet18_features(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """Extract ResNet18 features from multiple layers"""
        logger.info("Extracting ResNet18 features...")

        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model = model.to(self.device)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        features = {
            'conv1': [], 'layer1': [], 'layer2': [], 'layer3': [], 'layer4': [], 'avgpool': []
        }

        batch_size = 8
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack([preprocess(img) for img in batch])
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                x = batch_tensor
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                features['conv1'].append(x.mean(dim=(2,3)).cpu().numpy())
                x = model.maxpool(x)

                x = model.layer1(x)
                features['layer1'].append(x.mean(dim=(2,3)).cpu().numpy())
                x = model.layer2(x)
                features['layer2'].append(x.mean(dim=(2,3)).cpu().numpy())
                x = model.layer3(x)
                features['layer3'].append(x.mean(dim=(2,3)).cpu().numpy())
                x = model.layer4(x)
                features['layer4'].append(x.mean(dim=(2,3)).cpu().numpy())
                x = model.avgpool(x)
                features['avgpool'].append(x.flatten(1).cpu().numpy())

        return {k: np.vstack(v) for k, v in features.items()}

    def extract_mobilenet_v2_features(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """Extract MobileNetV2 features"""
        logger.info("Extracting MobileNetV2 features...")

        model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        model = model.to(self.device)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        features = {'stem': [], 'block_3': [], 'block_6': [], 'block_13': [], 'block_17': [], 'final': []}

        batch_size = 8
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack([preprocess(img) for img in batch])
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                x = batch_tensor
                for idx, layer in enumerate(model.features):
                    x = layer(x)
                    if idx == 0: features['stem'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 3: features['block_3'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 6: features['block_6'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 13: features['block_13'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 17: features['block_17'].append(x.mean(dim=(2,3)).cpu().numpy())
                features['final'].append(x.mean(dim=(2,3)).cpu().numpy())

        return {k: np.vstack(v) for k, v in features.items()}

    def extract_squeezenet_features(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """Extract SqueezeNet features"""
        logger.info("Extracting SqueezeNet features...")

        model = torchvision.models.squeezenet1_1(weights='IMAGENET1K_V1')
        model = model.to(self.device)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        features = {'conv1': [], 'fire2': [], 'fire4': [], 'fire7': [], 'fire9': [], 'conv10': []}

        batch_size = 8
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack([preprocess(img) for img in batch])
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                x = batch_tensor
                for idx, layer in enumerate(model.features):
                    x = layer(x)
                    if idx == 0: features['conv1'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 3: features['fire2'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 5: features['fire4'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 8: features['fire7'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 10: features['fire9'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 12: features['conv10'].append(x.mean(dim=(2,3)).cpu().numpy())

        return {k: np.vstack(v) for k, v in features.items()}

    def extract_efficientnet_b0_features(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """Extract EfficientNet-B0 features"""
        logger.info("Extracting EfficientNet-B0 features...")

        model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        model = model.to(self.device)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        features = {'stem': [], 'block_1': [], 'block_2': [], 'block_4': [], 'block_6': [], 'avgpool': []}

        batch_size = 8
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack([preprocess(img) for img in batch])
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                x = batch_tensor
                for idx, layer in enumerate(model.features):
                    x = layer(x)
                    if idx == 0: features['stem'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 1: features['block_1'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 2: features['block_2'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 4: features['block_4'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 6: features['block_6'].append(x.mean(dim=(2,3)).cpu().numpy())

                x = model.avgpool(x)
                features['avgpool'].append(x.flatten(1).cpu().numpy())

        return {k: np.vstack(v) for k, v in features.items()}

    def extract_distilbert_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract DistilBERT features (fixed version)"""
        logger.info("Extracting DistilBERT features...")

        try:
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            model = AutoModel.from_pretrained('distilbert-base-uncased')
            model = model.to(self.device)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load DistilBERT: {e}")
            return {}

        features = {
            'embeddings': [], 'layer_0': [], 'layer_1': [], 'layer_2': [],
            'layer_3': [], 'layer_4': [], 'layer_5': [], 'final': []
        }

        batch_size = 4
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]

            encoded = tokenizer(batch, padding=True, truncation=True,
                              max_length=128, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                # Get full model output with hidden states
                outputs = model(**encoded, output_hidden_states=True)

                # Hidden states include: embeddings + all layer outputs
                hidden_states = outputs.hidden_states

                # Extract features from each layer
                for layer_idx, layer_output in enumerate(hidden_states):
                    # Mean pool over sequence dimension
                    pooled = layer_output.mean(dim=1).cpu().numpy()

                    if layer_idx == 0:
                        features['embeddings'].append(pooled)
                    elif layer_idx <= 6:
                        features[f'layer_{layer_idx-1}'].append(pooled)

                # Final layer is the last hidden state
                features['final'].append(hidden_states[-1].mean(dim=1).cpu().numpy())

        return {k: np.vstack(v) for k, v in features.items() if v}

    def extract_albert_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract ALBERT features"""
        logger.info("Extracting ALBERT features...")

        try:
            tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
            model = AutoModel.from_pretrained('albert-base-v2')
            model = model.to(self.device)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load ALBERT: {e}")
            return {}

        features = {
            'embeddings': [], 'layer_0': [], 'layer_1': [], 'layer_2': [],
            'layer_3': [], 'layer_4': [], 'layer_5': [], 'final': []
        }

        batch_size = 4
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]

            encoded = tokenizer(batch, padding=True, truncation=True,
                              max_length=128, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                for layer_idx, layer_output in enumerate(hidden_states):
                    pooled = layer_output.mean(dim=1).cpu().numpy()

                    if layer_idx == 0:
                        features['embeddings'].append(pooled)
                    elif layer_idx <= 6:
                        features[f'layer_{layer_idx-1}'].append(pooled)

                features['final'].append(hidden_states[-1].mean(dim=1).cpu().numpy())

        return {k: np.vstack(v) for k, v in features.items() if v}

    def run_complete_extraction(self, n_samples: int = 40):
        """Run extraction for all models"""
        logger.info(f"Starting complete feature extraction with {n_samples} samples")

        images, texts = self.create_test_data(n_samples)
        results = {}

        # Vision models
        vision_models = {
            'resnet18': self.extract_resnet18_features,
            'mobilenet_v2': self.extract_mobilenet_v2_features,
            'squeezenet': self.extract_squeezenet_features,
            'efficientnet_b0': self.extract_efficientnet_b0_features,
        }

        for model_name, extract_func in vision_models.items():
            try:
                results[model_name] = extract_func(images)
                logger.info(f"✓ {model_name} extraction successful")
            except Exception as e:
                logger.error(f"✗ {model_name} failed: {e}")

        # Language models
        language_models = {
            'distilbert': self.extract_distilbert_features,
            'albert': self.extract_albert_features,
        }

        for model_name, extract_func in language_models.items():
            try:
                results[model_name] = extract_func(texts)
                if results[model_name]:  # Only log if we got features
                    logger.info(f"✓ {model_name} extraction successful")
                else:
                    logger.warning(f"⚠ {model_name} returned empty features")
            except Exception as e:
                logger.error(f"✗ {model_name} failed: {e}")

        return results

def main():
    extractor = CompleteFeatureExtractor()
    results = extractor.run_complete_extraction(n_samples=30)

    print("\n" + "="*80)
    print("COMPLETE FEATURE EXTRACTION RESULTS")
    print("="*80)

    vision_models = ['resnet18', 'mobilenet_v2', 'squeezenet', 'efficientnet_b0']
    language_models = ['distilbert', 'albert']

    print("\nVISION MODELS:")
    for model_name in vision_models:
        if model_name in results:
            print(f"\n{model_name.upper()}:")
            for layer_name, features in results[model_name].items():
                print(f"  {layer_name}: {features.shape}")

    print("\nLANGUAGE MODELS:")
    for model_name in language_models:
        if model_name in results:
            print(f"\n{model_name.upper()}:")
            for layer_name, features in results[model_name].items():
                print(f"  {layer_name}: {features.shape}")

    # Save all features
    output_dir = Path("complete_features")
    output_dir.mkdir(exist_ok=True)

    for model_name, features in results.items():
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)

        for layer_name, layer_features in features.items():
            np.save(model_dir / f"{layer_name}.npy", layer_features)

    print(f"\nAll features saved to {output_dir}/")

    # Summary statistics
    total_models = len(results)
    total_layers = sum(len(features) for features in results.values())
    total_features = sum(sum(f.size for f in features.values()) for features in results.values())

    print(f"\nSUMMARY:")
    print(f"✓ Models processed: {total_models}")
    print(f"✓ Total layers: {total_layers}")
    print(f"✓ Total feature values: {total_features:,}")

if __name__ == "__main__":
    main()