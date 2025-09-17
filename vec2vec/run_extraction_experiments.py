#!/usr/bin/env python3
"""
Run systematic embedding extraction experiments and analyze results.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import argparse


def analyze_embeddings_similarity(embeddings_dir: str = "./embeddings"):
    """
    Analyze similarity patterns across extracted embeddings.
    """
    embeddings_path = Path(embeddings_dir)

    # Load extraction results
    with open(embeddings_path / 'extraction_results.json', 'r') as f:
        results = json.load(f)

    # Group results by dataset and model family
    analysis_results = {
        'within_family': {},
        'cross_family': {},
        'cross_dataset': {},
        'layer_analysis': {}
    }

    # Load all embeddings
    embeddings_cache = {}
    for result in results:
        if 'filepath' in result:
            filepath = result['filepath']
            data = np.load(filepath)
            embeddings_cache[filepath] = data['embeddings']

    print(f"Loaded {len(embeddings_cache)} embedding files")

    # 1. Within-family similarity analysis
    print("\n" + "="*60)
    print("Within-Family Similarity Analysis")
    print("="*60)

    family_groups = {
        'bert': ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'],
        'gpt': ['gpt2', 'gpt2-medium'],
        't5': ['google/t5-v1_1-small', 'google/t5-v1_1-base']
    }

    for family_name, models in family_groups.items():
        family_similarities = []

        for dataset_name in ['general', 'questions', 'scientific']:
            # Find embeddings for this family and dataset
            family_embeddings = []
            model_names = []

            for result in results:
                if ('error' not in result and
                    result['dataset'] == dataset_name and
                    any(model in result['model'] for model in models)):

                    filepath = result['filepath']
                    if filepath in embeddings_cache:
                        family_embeddings.append(embeddings_cache[filepath])
                        model_names.append(result['model'])

            if len(family_embeddings) >= 2:
                # Compute pairwise similarities
                for i in range(len(family_embeddings)):
                    for j in range(i+1, len(family_embeddings)):
                        # Align dimensions if needed
                        min_dim = min(family_embeddings[i].shape[1],
                                    family_embeddings[j].shape[1])
                        emb1 = family_embeddings[i][:, :min_dim]
                        emb2 = family_embeddings[j][:, :min_dim]

                        # Compute average cosine similarity
                        sim = np.mean([cosine_similarity([e1], [e2])[0, 0]
                                     for e1, e2 in zip(emb1[:100], emb2[:100])])

                        family_similarities.append({
                            'family': family_name,
                            'dataset': dataset_name,
                            'model1': model_names[i],
                            'model2': model_names[j],
                            'similarity': sim
                        })

        if family_similarities:
            avg_sim = np.mean([s['similarity'] for s in family_similarities])
            std_sim = np.std([s['similarity'] for s in family_similarities])
            print(f"\n{family_name.upper()} Family:")
            print(f"  Average similarity: {avg_sim:.4f} ± {std_sim:.4f}")

            analysis_results['within_family'][family_name] = {
                'mean': avg_sim,
                'std': std_sim,
                'details': family_similarities
            }

    # 2. Cross-family similarity analysis
    print("\n" + "="*60)
    print("Cross-Family Similarity Analysis")
    print("="*60)

    cross_family_pairs = [
        ('bert-base-uncased', 'gpt2'),
        ('bert-base-uncased', 'google/t5-v1_1-base'),
        ('gpt2', 'google/t5-v1_1-base')
    ]

    for model1, model2 in cross_family_pairs:
        pair_similarities = []

        for dataset_name in ['general', 'questions', 'scientific']:
            # Find embeddings
            emb1_data = None
            emb2_data = None

            for result in results:
                if ('error' not in result and
                    result['dataset'] == dataset_name):

                    if result['model'] == model1:
                        filepath = result['filepath']
                        if filepath in embeddings_cache:
                            emb1_data = embeddings_cache[filepath]
                    elif result['model'] == model2:
                        filepath = result['filepath']
                        if filepath in embeddings_cache:
                            emb2_data = embeddings_cache[filepath]

            if emb1_data is not None and emb2_data is not None:
                # Align dimensions
                min_dim = min(emb1_data.shape[1], emb2_data.shape[1])
                emb1 = emb1_data[:, :min_dim]
                emb2 = emb2_data[:, :min_dim]

                # Compute similarity
                sim = np.mean([cosine_similarity([e1], [e2])[0, 0]
                             for e1, e2 in zip(emb1[:100], emb2[:100])])

                pair_similarities.append({
                    'model1': model1,
                    'model2': model2,
                    'dataset': dataset_name,
                    'similarity': sim
                })

        if pair_similarities:
            avg_sim = np.mean([s['similarity'] for s in pair_similarities])
            print(f"\n{model1} vs {model2}:")
            print(f"  Average similarity: {avg_sim:.4f}")

            for ps in pair_similarities:
                print(f"    {ps['dataset']}: {ps['similarity']:.4f}")

    # 3. Dataset consistency analysis
    print("\n" + "="*60)
    print("Dataset Consistency Analysis")
    print("="*60)

    model_dataset_consistency = {}

    for model_name in ['bert-base-uncased', 'gpt2']:
        dataset_embeddings = {}

        for result in results:
            if ('error' not in result and
                result['model'] == model_name):

                dataset = result['dataset']
                filepath = result['filepath']
                if filepath in embeddings_cache:
                    dataset_embeddings[dataset] = embeddings_cache[filepath]

        if len(dataset_embeddings) >= 2:
            # Compute cross-dataset similarities
            datasets = list(dataset_embeddings.keys())
            cross_dataset_sims = []

            for i in range(len(datasets)):
                for j in range(i+1, len(datasets)):
                    emb1 = dataset_embeddings[datasets[i]]
                    emb2 = dataset_embeddings[datasets[j]]

                    # Use PCA to project to common space
                    min_samples = min(emb1.shape[0], emb2.shape[0], 100)
                    min_dim = min(emb1.shape[1], emb2.shape[1], 50)

                    pca = PCA(n_components=min_dim)
                    emb1_pca = pca.fit_transform(emb1[:min_samples])
                    emb2_pca = pca.transform(emb2[:min_samples])

                    # Compute similarity in PCA space
                    sim = np.mean([cosine_similarity([e1], [e2])[0, 0]
                                 for e1, e2 in zip(emb1_pca, emb2_pca)])

                    cross_dataset_sims.append({
                        'dataset1': datasets[i],
                        'dataset2': datasets[j],
                        'similarity': sim
                    })

            if cross_dataset_sims:
                avg_sim = np.mean([s['similarity'] for s in cross_dataset_sims])
                std_sim = np.std([s['similarity'] for s in cross_dataset_sims])

                print(f"\n{model_name}:")
                print(f"  Cross-dataset similarity: {avg_sim:.4f} ± {std_sim:.4f}")

                model_dataset_consistency[model_name] = {
                    'mean': avg_sim,
                    'std': std_sim,
                    'details': cross_dataset_sims
                }

    # Save analysis results
    with open(embeddings_path / 'similarity_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print("\n" + "="*60)
    print("Analysis complete! Results saved to similarity_analysis.json")

    return analysis_results


def visualize_embedding_landscape(embeddings_dir: str = "./embeddings"):
    """
    Create visualizations of the embedding landscape.
    """
    embeddings_path = Path(embeddings_dir)
    output_dir = embeddings_path / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    # Load results
    with open(embeddings_path / 'extraction_results.json', 'r') as f:
        results = json.load(f)

    # 1. Create similarity heatmap
    print("Creating similarity heatmap...")

    models = list(set(r['model'] for r in results if 'error' not in r))[:5]  # Limit to 5 models
    datasets = list(set(r['dataset'] for r in results if 'error' not in r))[:3]  # Limit to 3 datasets

    similarity_matrix = np.zeros((len(models), len(models)))

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i <= j:
                similarities = []

                for dataset in datasets:
                    # Find embeddings
                    emb1 = None
                    emb2 = None

                    for result in results:
                        if ('error' not in result and
                            result['dataset'] == dataset):

                            if result['model'] == model1:
                                filepath = result['filepath']
                                data = np.load(filepath)
                                emb1 = data['embeddings']
                            elif result['model'] == model2:
                                filepath = result['filepath']
                                data = np.load(filepath)
                                emb2 = data['embeddings']

                    if emb1 is not None and emb2 is not None:
                        # Compute similarity
                        min_dim = min(emb1.shape[1], emb2.shape[1])
                        min_samples = min(emb1.shape[0], emb2.shape[0], 50)

                        sim = np.mean([cosine_similarity([e1[:min_dim]], [e2[:min_dim]])[0, 0]
                                     for e1, e2 in zip(emb1[:min_samples], emb2[:min_samples])])
                        similarities.append(sim)

                if similarities:
                    avg_sim = np.mean(similarities)
                    similarity_matrix[i, j] = avg_sim
                    similarity_matrix[j, i] = avg_sim

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix,
                xticklabels=[m.split('/')[-1] for m in models],
                yticklabels=[m.split('/')[-1] for m in models],
                annot=True, fmt='.3f', cmap='coolwarm',
                vmin=0, vmax=1)
    plt.title('Model Embedding Similarity Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_heatmap.png', dpi=150)
    plt.close()

    print(f"Visualizations saved to {output_dir}")

    # 2. Create PCA visualization of embedding spaces
    print("Creating PCA visualization...")

    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
    if len(datasets) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        for model_idx, model in enumerate(models[:3]):  # Limit to 3 models for clarity
            # Find embeddings
            for result in results:
                if ('error' not in result and
                    result['dataset'] == dataset and
                    result['model'] == model):

                    filepath = result['filepath']
                    data = np.load(filepath)
                    embeddings = data['embeddings']

                    # PCA projection
                    pca = PCA(n_components=2)
                    embeddings_2d = pca.fit_transform(embeddings[:100])

                    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=[colors[model_idx]], alpha=0.6,
                             label=model.split('/')[-1], s=20)
                    break

        ax.set_title(f'Dataset: {dataset}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(fontsize=8)

    plt.suptitle('Embedding Space Visualization (PCA)')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_visualization.png', dpi=150)
    plt.close()

    print("Visualizations complete!")


def run_quick_test():
    """
    Run a quick test with minimal models and data.
    """
    from extract_hf_embeddings import extract_all_embeddings

    print("Running quick test extraction...")

    # Use small models and limited data for testing
    test_models = [
        'distilbert-base-uncased',
        'gpt2',
        'google/t5-v1_1-small'
    ]

    test_datasets = {
        'test_general': ["This is a test sentence."] * 10,
        'test_questions': ["What is the meaning of life?"] * 10,
        'test_code': ["def hello(): print('hello')"] * 10
    }

    results = extract_all_embeddings(
        models=test_models,
        datasets=test_datasets,
        output_dir="./test_embeddings",
        layers=[-1],
        pooling_strategies=['mean'],
        max_samples=10,
        batch_size=5
    )

    print("\nTest extraction complete!")
    print(f"Successfully extracted {len([r for r in results if 'error' not in r])} embeddings")

    # Run analysis
    print("\nRunning similarity analysis...")
    analysis = analyze_embeddings_similarity("./test_embeddings")

    print("\nCreating visualizations...")
    visualize_embedding_landscape("./test_embeddings")

    print("\nQuick test complete! Check ./test_embeddings for results.")


def main():
    parser = argparse.ArgumentParser(description="Run embedding extraction experiments")

    parser.add_argument('--mode', choices=['extract', 'analyze', 'visualize', 'quick_test', 'full'],
                       default='quick_test',
                       help='Operation mode')
    parser.add_argument('--embeddings-dir', type=str, default='./embeddings',
                       help='Directory containing embeddings')

    args = parser.parse_args()

    if args.mode == 'quick_test':
        run_quick_test()
    elif args.mode == 'analyze':
        analyze_embeddings_similarity(args.embeddings_dir)
    elif args.mode == 'visualize':
        visualize_embedding_landscape(args.embeddings_dir)
    elif args.mode == 'full':
        # Full pipeline
        print("Running full extraction pipeline...")
        from extract_hf_embeddings import main as extract_main
        extract_main()

        print("\nRunning analysis...")
        analyze_embeddings_similarity(args.embeddings_dir)

        print("\nCreating visualizations...")
        visualize_embedding_landscape(args.embeddings_dir)
    else:
        print(f"Mode {args.mode} not yet implemented")


if __name__ == '__main__':
    main()