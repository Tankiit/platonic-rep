#!/usr/bin/env python3
"""
Super simple MLX feature extraction
"""

import os
import mlx.core as mx
from mlx_lm import load as load_lm
import numpy as np
from datasets import load_dataset
from pathlib import Path

def main():
    print("Loading model...")
    model, tokenizer = load_lm("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

    print("Loading dataset...")
    dataset = load_dataset("imdb", split='train')
    texts = [x['text'] for x in dataset.select(range(100))]  # Just 100 samples

    print("Extracting features...")
    features = []

    for i, text in enumerate(texts[:10]):  # Just 10 for testing
        print(f"Processing {i+1}/10")

        # Tokenize
        tokens = tokenizer.encode(text)
        input_ids = mx.array([tokens])

        # Forward pass to get hidden states
        output = model(input_ids, output_hidden_states=True)

        # Get last hidden state mean
        if hasattr(output, 'hidden_states'):
            last_hidden = output.hidden_states[-1]  # Last layer
            pooled = mx.mean(last_hidden, axis=1)    # Mean over sequence
            features.append(np.array(pooled))

        if i % 5 == 0:
            mx.clear_cache()

    # Save results
    if features:
        final_features = np.vstack(features)
        output_dir = Path("representations/simple_features")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / "imdb_qwen_features.npz"
        np.savez(save_path, features=final_features)

        print(f"✅ Saved features: {final_features.shape}")
        print(f"📁 Location: {save_path}")
    else:
        print("❌ No features extracted")

if __name__ == "__main__":
    main()