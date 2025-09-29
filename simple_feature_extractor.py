#!/usr/bin/env python3
"""
Simple multi-layer feature extraction that actually works
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

class SimpleFeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def create_test_data(self, n_samples: int = 50):
        """Create simple test data"""
        logger.info(f"Creating {n_samples} test samples")

        images = []
        texts = []

        for i in range(n_samples):
            # Create simple colored images
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)

            color_type = i % 4
            if color_type == 0:  # Red
                img_array[:, :, 0] = 255
                text = "A red image"
            elif color_type == 1:  # Green
                img_array[:, :, 1] = 255
                text = "A green image"
            elif color_type == 2:  # Blue
                img_array[:, :, 2] = 255
                text = "A blue image"
            else:  # Random
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                text = "A colorful random image"

            images.append(Image.fromarray(img_array))
            texts.append(text)

        return images, texts

    def extract_resnet_features(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """Extract ResNet18 features from multiple layers"""
        logger.info("Extracting ResNet18 features...")

        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model = model.to(self.device)
        model.eval()

        # Preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        features = {
            'conv1': [],
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
            'avgpool': []
        }

        batch_size = 8
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack([preprocess(img) for img in batch])
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                x = batch_tensor

                # Conv1
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                features['conv1'].append(x.mean(dim=(2,3)).cpu().numpy())  # Global avg pool
                x = model.maxpool(x)

                # Layer blocks
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

        # Concatenate all batches
        return {k: np.vstack(v) for k, v in features.items()}

    def extract_mobilenet_features(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
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

        features = {
            'stem': [],
            'early_layers': [],
            'mid_layers': [],
            'late_layers': [],
            'final': []
        }

        batch_size = 8
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack([preprocess(img) for img in batch])
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                x = batch_tensor

                # Process through feature layers
                for idx, layer in enumerate(model.features):
                    x = layer(x)

                    if idx == 0:  # Stem
                        features['stem'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 3:  # Early
                        features['early_layers'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 10:  # Mid
                        features['mid_layers'].append(x.mean(dim=(2,3)).cpu().numpy())
                    elif idx == 17:  # Late
                        features['late_layers'].append(x.mean(dim=(2,3)).cpu().numpy())

                # Final features
                features['final'].append(x.mean(dim=(2,3)).cpu().numpy())

        return {k: np.vstack(v) for k, v in features.items()}

    def extract_distilbert_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract DistilBERT features"""
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
            'embeddings': [],
            'layer_0': [],
            'layer_1': [],
            'layer_2': [],
            'layer_3': [],
            'layer_4': [],
            'layer_5': [],
            'final': []
        }

        batch_size = 4
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]

            encoded = tokenizer(batch, padding=True, truncation=True,
                              max_length=128, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                # Get embeddings
                embeddings = model.embeddings(encoded['input_ids'])
                features['embeddings'].append(embeddings.mean(dim=1).cpu().numpy())

                # Process through transformer layers
                hidden_states = embeddings
                for layer_idx, layer in enumerate(model.transformer.layer):
                    layer_output = layer(hidden_states, attention_mask=encoded['attention_mask'])
                    if isinstance(layer_output, tuple):
                        hidden_states = layer_output[0]
                    else:
                        hidden_states = layer_output

                    if layer_idx < 6:  # Store first 6 layers
                        features[f'layer_{layer_idx}'].append(hidden_states.mean(dim=1).cpu().numpy())

                features['final'].append(hidden_states.mean(dim=1).cpu().numpy())

        return {k: np.vstack(v) for k, v in features.items() if v}

    def run_extraction(self, n_samples: int = 30):
        """Run the complete extraction pipeline"""
        logger.info(f"Starting feature extraction with {n_samples} samples")

        # Create test data
        images, texts = self.create_test_data(n_samples)

        results = {}

        # Extract vision features
        try:
            results['resnet18'] = self.extract_resnet_features(images)
            logger.info("✓ ResNet18 extraction successful")
        except Exception as e:
            logger.error(f"✗ ResNet18 failed: {e}")

        try:
            results['mobilenet_v2'] = self.extract_mobilenet_features(images)
            logger.info("✓ MobileNetV2 extraction successful")
        except Exception as e:
            logger.error(f"✗ MobileNetV2 failed: {e}")

        # Extract language features
        try:
            results['distilbert'] = self.extract_distilbert_features(texts)
            logger.info("✓ DistilBERT extraction successful")
        except Exception as e:
            logger.error(f"✗ DistilBERT failed: {e}")

        return results

def main():
    extractor = SimpleFeatureExtractor()
    results = extractor.run_extraction(n_samples=20)

    print("\n" + "="*60)
    print("FEATURE EXTRACTION RESULTS")
    print("="*60)

    for model_name, features in results.items():
        print(f"\n{model_name.upper()}:")
        for layer_name, layer_features in features.items():
            print(f"  {layer_name}: {layer_features.shape}")

    # Save results
    output_dir = Path("simple_features")
    output_dir.mkdir(exist_ok=True)

    for model_name, features in results.items():
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)

        for layer_name, layer_features in features.items():
            np.save(model_dir / f"{layer_name}.npy", layer_features)

    print(f"\nFeatures saved to {output_dir}/")

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    total_layers = sum(len(features) for features in results.values())
    total_features = sum(sum(f.size for f in features.values()) for features in results.values())
    print(f"Total models: {len(results)}")
    print(f"Total layers: {total_layers}")
    print(f"Total feature values: {total_features:,}")

if __name__ == "__main__":
    main()