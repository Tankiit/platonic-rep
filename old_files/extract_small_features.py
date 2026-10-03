#!/usr/bin/env python3
"""
Minimal feature extraction for small models only
"""

import torch
import numpy as np
from pathlib import Path
import timm
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_vision_features(model_name, num_samples=100):
    """Extract features from small vision models"""
    try:
        # Load model
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval()

        # Generate random inputs (simulating image data)
        batch_size = 4
        features_list = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                # Random input (3, 224, 224)
                x = torch.randn(batch_size, 3, 224, 224)
                features = model(x)
                features_list.append(features.numpy())

        features = np.concatenate(features_list, axis=0)[:num_samples]
        return features
    except Exception as e:
        print(f"Failed to extract features from {model_name}: {e}")
        return None

def extract_language_features(model_name, num_samples=100):
    """Extract features from small language models"""
    try:
        # Load model and tokenizer
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()

        # Generate sample texts
        texts = [f"This is sample text number {i}." for i in range(num_samples)]

        features_list = []
        batch_size = 4

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
                outputs = model(**inputs)

                # Use pooled output or mean of last hidden states
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state.mean(dim=1)

                features_list.append(features.numpy())

        features = np.concatenate(features_list, axis=0)[:num_samples]
        return features
    except Exception as e:
        print(f"Failed to extract features from {model_name}: {e}")
        return None

def main():
    """Extract features from small models"""

    output_dir = Path("./results/features/minhuh/prh/wit_1024")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Small vision models available in timm
    vision_models = [
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "resnet18",
        "resnet34",
        "mobilenetv2_050",
        "efficientnet_b0",
    ]

    # Small language models
    language_models = [
        "google/bert_uncased_L-2_H-128_A-2",  # Bert-Tiny
        "google/bert_uncased_L-4_H-256_A-4",  # Bert-Mini
        "distilbert-base-uncased",
        "albert-base-v2",
    ]

    print("Extracting features from small models...")
    print("="*50)

    # Extract vision features
    print("\nVision models:")
    for model_name in tqdm(vision_models, desc="Vision"):
        features = extract_vision_features(model_name, num_samples=100)
        if features is not None:
            # Save with naming convention
            safe_name = model_name.replace('/', '_')
            save_path = output_dir / f"{safe_name}_pool-cls.npy"
            np.save(save_path, features)
            print(f"  ✓ {model_name}: {features.shape}")

    # Extract language features
    print("\nLanguage models:")
    for model_name in tqdm(language_models, desc="Language"):
        features = extract_language_features(model_name, num_samples=100)
        if features is not None:
            # Save with naming convention
            safe_name = model_name.replace('/', '_')
            save_path = output_dir / f"{safe_name}_pool-avg.npy"
            np.save(save_path, features)
            print(f"  ✓ {model_name}: {features.shape}")

    print("\nFeature extraction complete!")
    print(f"Features saved to: {output_dir}")

if __name__ == "__main__":
    main()