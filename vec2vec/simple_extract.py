#!/usr/bin/env python3
"""
Simplified extraction script using local text data.
"""

import os
import numpy as np
from pathlib import Path
import json
from extract_hf_embeddings import HuggingFaceEmbeddingExtractor, save_embeddings

def create_sample_datasets():
    """Create simple sample datasets for testing."""
    datasets = {
        'general': [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming technology.",
            "Climate change affects global weather patterns.",
            "The economy shows signs of recovery this quarter.",
            "Scientists discover new exoplanet in distant galaxy.",
            "Machine learning algorithms improve daily.",
            "The stock market reached new highs today.",
            "Renewable energy sources become more efficient.",
            "Medical researchers develop innovative treatments.",
            "Technology companies invest in quantum computing."
        ] * 10,  # Repeat to get 100 samples

        'questions': [
            "What is the meaning of life?",
            "How does machine learning work?",
            "Why is the sky blue?",
            "When was the internet invented?",
            "Where do black holes lead?",
            "Who invented the telephone?",
            "What causes earthquakes?",
            "How do vaccines work?",
            "Why do we dream?",
            "What is quantum entanglement?"
        ] * 10,

        'scientific': [
            "The hypothesis suggests a correlation between variables.",
            "Experimental results confirm theoretical predictions.",
            "DNA sequencing reveals genetic variations.",
            "Protein folding determines biological function.",
            "Neural networks mimic brain processes.",
            "Quantum mechanics describes subatomic behavior.",
            "Chemical reactions follow thermodynamic principles.",
            "Cell division involves complex molecular machinery.",
            "Evolution shapes species through natural selection.",
            "Photosynthesis converts light into chemical energy."
        ] * 10
    }

    return datasets

def main():
    # Setup
    output_dir = Path("./embeddings_bert")
    output_dir.mkdir(exist_ok=True)

    # Initialize extractor
    extractor = HuggingFaceEmbeddingExtractor()

    # Get BERT variants
    bert_models = [
        'bert-base-uncased',
        'roberta-base',
        'distilbert-base-uncased'
    ]

    # Create simple datasets
    print("Creating sample datasets...")
    datasets = create_sample_datasets()

    # Track results
    results = []

    print(f"\nExtracting embeddings from {len(bert_models)} BERT variants")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Output directory: {output_dir}\n")

    # Extract embeddings
    for model_name in bert_models:
        for dataset_name, texts in datasets.items():
            try:
                print(f"\n{'='*60}")
                print(f"Extracting: {model_name} on {dataset_name}")
                print(f"{'='*60}")

                # Extract embeddings
                embeddings = extractor.extract_embeddings(
                    model_name=model_name,
                    texts=texts[:50],  # Use only 50 samples for speed
                    layer=-1,
                    pooling='mean',
                    batch_size=16
                )

                # Save embeddings
                filepath = save_embeddings(
                    embeddings=embeddings,
                    output_dir=output_dir,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    layer=-1,
                    pooling='mean'
                )

                print(f"✓ Shape: {embeddings.shape}")
                print(f"✓ Saved: {filepath}")

                results.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'shape': embeddings.shape,
                    'filepath': str(filepath)
                })

            except Exception as e:
                print(f"✗ Failed: {e}")
                results.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'error': str(e)
                })

    # Save results
    with open(output_dir / 'extraction_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(results)} extractions")
    print(f"Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"Failed: {sum(1 for r in results if 'error' in r)}")
    print(f"\nResults saved to: {output_dir}")

    # Quick similarity analysis
    print(f"\n{'='*60}")
    print("QUICK SIMILARITY ANALYSIS")
    print(f"{'='*60}")

    # Load embeddings and compute similarities
    for dataset_name in datasets.keys():
        print(f"\nDataset: {dataset_name}")

        dataset_embeddings = {}
        for model_name in bert_models:
            filename = f"{dataset_name}_{model_name.replace('/', '_').replace('-', '_')}_layer-1_mean.npz"
            filepath = output_dir / filename
            if filepath.exists():
                data = np.load(filepath)
                dataset_embeddings[model_name] = data['embeddings']

        # Compute pairwise similarities
        if len(dataset_embeddings) >= 2:
            models = list(dataset_embeddings.keys())
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    emb1 = dataset_embeddings[models[i]]
                    emb2 = dataset_embeddings[models[j]]

                    # Simple cosine similarity
                    min_samples = min(emb1.shape[0], emb2.shape[0])
                    min_dim = min(emb1.shape[1], emb2.shape[1])

                    # Normalize
                    emb1_norm = emb1[:min_samples, :min_dim] / np.linalg.norm(emb1[:min_samples, :min_dim], axis=1, keepdims=True)
                    emb2_norm = emb2[:min_samples, :min_dim] / np.linalg.norm(emb2[:min_samples, :min_dim], axis=1, keepdims=True)

                    # Average cosine similarity
                    similarities = np.sum(emb1_norm * emb2_norm, axis=1)
                    avg_sim = np.mean(similarities)

                    model1_short = models[i].split('/')[-1]
                    model2_short = models[j].split('/')[-1]
                    print(f"  {model1_short} vs {model2_short}: {avg_sim:.4f}")

if __name__ == '__main__':
    main()