#!/usr/bin/env python3
"""
Visualization tools for embedding alignment analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from pathlib import Path
import json
from typing import Dict, List, Optional
import argparse

from hyperprocrustes import HyperProcrustes, HyperProcrustesTrainer


def visualize_embedding_alignment(
    embeddings_A: np.ndarray,
    embeddings_B: np.ndarray,
    transformed_A: Optional[np.ndarray] = None,
    model_A: str = "Model A",
    model_B: str = "Model B",
    method: str = 'pca',
    save_path: Optional[str] = None
):
    """
    Visualize embedding alignment using dimensionality reduction.
    """
    # Prepare data
    n_samples = min(embeddings_A.shape[0], embeddings_B.shape[0], 500)
    embeddings_A = embeddings_A[:n_samples]
    embeddings_B = embeddings_B[:n_samples]

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fit on combined data for consistent projection
    if transformed_A is not None:
        transformed_A = transformed_A[:n_samples]
        combined = np.vstack([embeddings_A, embeddings_B, transformed_A])
    else:
        combined = np.vstack([embeddings_A, embeddings_B])

    print(f"Reducing dimensions using {method.upper()}...")
    reduced = reducer.fit_transform(combined)

    # Split back
    reduced_A = reduced[:n_samples]
    reduced_B = reduced[n_samples:2*n_samples]
    reduced_transformed = reduced[2*n_samples:] if transformed_A is not None else None

    # Create visualization
    fig, axes = plt.subplots(1, 3 if transformed_A is not None else 2, figsize=(15, 5))

    # Plot original embeddings
    ax = axes[0]
    ax.scatter(reduced_A[:, 0], reduced_A[:, 1], alpha=0.6, c='blue', label=model_A, s=20)
    ax.scatter(reduced_B[:, 0], reduced_B[:, 1], alpha=0.6, c='red', label=model_B, s=20)
    ax.set_title('Original Embeddings')
    ax.legend()
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')

    # Plot overlay
    ax = axes[1]
    ax.scatter(reduced_A[:, 0], reduced_A[:, 1], alpha=0.3, c='blue', s=20)
    ax.scatter(reduced_B[:, 0], reduced_B[:, 1], alpha=0.3, c='red', s=20)

    # Draw connections for nearest neighbors
    for i in range(min(50, n_samples)):
        ax.plot([reduced_A[i, 0], reduced_B[i, 0]],
               [reduced_A[i, 1], reduced_B[i, 1]],
               'gray', alpha=0.1, linewidth=0.5)

    ax.set_title('Correspondence (First 50 points)')
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')

    # Plot transformed if available
    if transformed_A is not None:
        ax = axes[2]
        ax.scatter(reduced_transformed[:, 0], reduced_transformed[:, 1],
                  alpha=0.6, c='green', label=f'{model_A} → {model_B}', s=20)
        ax.scatter(reduced_B[:, 0], reduced_B[:, 1],
                  alpha=0.6, c='red', label=model_B, s=20)
        ax.set_title('After Alignment')
        ax.legend()
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')

    plt.suptitle(f'Embedding Alignment Visualization ({method.upper()})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    return fig


def plot_similarity_matrix(embeddings_dir: str, save_path: Optional[str] = None):
    """
    Create a heatmap of pairwise similarities between all models.
    """
    embeddings_path = Path(embeddings_dir)

    # Load all embeddings
    embeddings_dict = {}
    for npz_file in embeddings_path.glob('*.npz'):
        data = np.load(npz_file)
        model_name = str(data.get('model_name', npz_file.stem))
        embeddings_dict[model_name] = data['embeddings']

    if not embeddings_dict:
        print("No embeddings found!")
        return

    # Compute pairwise similarities
    models = list(embeddings_dict.keys())
    n_models = len(models)
    similarity_matrix = np.zeros((n_models, n_models))

    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            emb_i = embeddings_dict[model_i]
            emb_j = embeddings_dict[model_j]

            # Use subset for efficiency
            n_samples = min(emb_i.shape[0], emb_j.shape[0], 100)
            emb_i = emb_i[:n_samples]
            emb_j = emb_j[:n_samples]

            # Normalize
            emb_i_norm = emb_i / (np.linalg.norm(emb_i, axis=1, keepdims=True) + 1e-8)
            emb_j_norm = emb_j / (np.linalg.norm(emb_j, axis=1, keepdims=True) + 1e-8)

            # Compute average cosine similarity
            if emb_i.shape[1] == emb_j.shape[1]:
                similarity = np.mean(np.sum(emb_i_norm * emb_j_norm, axis=1))
            else:
                # Different dimensions - use correlation of norms
                norms_i = np.linalg.norm(emb_i, axis=1)
                norms_j = np.linalg.norm(emb_j, axis=1)
                similarity = np.corrcoef(norms_i, norms_j)[0, 1]

            similarity_matrix[i, j] = similarity

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix,
                xticklabels=[m.split('/')[-1] for m in models],
                yticklabels=[m.split('/')[-1] for m in models],
                annot=True, fmt='.3f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0)
    plt.title('Model Embedding Similarity Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity matrix to {save_path}")
    else:
        plt.show()

    return similarity_matrix, models


def visualize_transformation_quality(
    model: HyperProcrustes,
    embeddings_dir: str,
    device: str = 'cuda',
    save_dir: Optional[str] = None
):
    """
    Visualize how well the model aligns different embedding pairs.
    """
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
    else:
        save_path = None

    embeddings_path = Path(embeddings_dir)

    # Test on specific pairs
    test_pairs = [
        ('bert_base_uncased', 'distilbert_base_uncased'),  # Should align well
        ('bert_base_uncased', 'roberta_base'),  # Same architecture, different training
        ('bert_base_uncased', 'gpt2'),  # Different architectures
    ]

    for model_A_name, model_B_name in test_pairs:
        print(f"\nProcessing {model_A_name} -> {model_B_name}")

        # Find files
        files_A = list(embeddings_path.glob(f"*{model_A_name}*"))
        files_B = list(embeddings_path.glob(f"*{model_B_name}*"))

        if not files_A or not files_B:
            print(f"  Skipping - embeddings not found")
            continue

        # Load embeddings
        data_A = np.load(files_A[0])
        data_B = np.load(files_B[0])

        embeddings_A = torch.FloatTensor(data_A['embeddings'][:100])
        embeddings_B = torch.FloatTensor(data_B['embeddings'][:100])

        # Get transformation
        model.eval()
        with torch.no_grad():
            embeddings_A_batch = embeddings_A.unsqueeze(0).to(device)
            embeddings_B_batch = embeddings_B.unsqueeze(0).to(device)

            params = model(embeddings_A_batch, embeddings_B_batch)
            transformed_A = model.transform(embeddings_A_batch, params, direction='A_to_B')
            transformed_A = transformed_A.squeeze(0).cpu().numpy()

        embeddings_A_np = embeddings_A.numpy()
        embeddings_B_np = embeddings_B.numpy()

        # Visualize
        for method in ['pca', 'tsne']:
            output_path = None
            if save_path:
                output_path = save_path / f"{model_A_name}_to_{model_B_name}_{method}.png"

            visualize_embedding_alignment(
                embeddings_A_np,
                embeddings_B_np,
                transformed_A,
                model_A_name,
                model_B_name,
                method=method,
                save_path=output_path
            )

        # Compute and print metrics
        cos_sim = np.mean([
            np.dot(transformed_A[i], embeddings_B_np[i]) /
            (np.linalg.norm(transformed_A[i]) * np.linalg.norm(embeddings_B_np[i]) + 1e-8)
            for i in range(len(transformed_A))
        ])

        print(f"  Alignment quality: {params['predicted_quality'].mean().item():.4f}")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(f"  Scale factor: {params['scale'].mean().item():.4f}")


def plot_training_curves(log_dir: str):
    """
    Plot training curves from TensorBoard logs.
    """
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_path = Path(log_dir)

    # Find the latest run
    runs = sorted([d for d in log_path.iterdir() if d.is_dir()])
    if not runs:
        print("No training runs found!")
        return

    latest_run = runs[-1]
    print(f"Loading logs from {latest_run}")

    # Load events
    ea = EventAccumulator(str(latest_run))
    ea.Reload()

    # Extract loss curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss components
    loss_types = [
        'Loss/distribution_matching',
        'Loss/cycle_consistency',
        'Loss/orthogonality',
        'Loss/total'
    ]

    for idx, loss_type in enumerate(loss_types):
        ax = axes[idx // 2, idx % 2]

        if loss_type in ea.Tags()['scalars']:
            events = ea.Scalars(loss_type)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            ax.plot(steps, values, label=loss_type.split('/')[-1])
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(loss_type.split('/')[-1].replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.suptitle('Training Curves', fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding alignments")

    parser.add_argument('--embeddings-dir', type=str, default='./embeddings_bert',
                       help='Directory containing embeddings')
    parser.add_argument('--checkpoint', type=str,
                       help='Model checkpoint path')
    parser.add_argument('--log-dir', type=str, default='./runs/hyperprocrustes',
                       help='TensorBoard log directory')
    parser.add_argument('--save-dir', type=str,
                       help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    # Visualization options
    parser.add_argument('--plot-similarity', action='store_true',
                       help='Plot similarity matrix')
    parser.add_argument('--plot-training', action='store_true',
                       help='Plot training curves')
    parser.add_argument('--plot-alignment', action='store_true',
                       help='Visualize alignment transformations')

    args = parser.parse_args()

    if args.plot_similarity:
        print("Creating similarity matrix...")
        save_path = None
        if args.save_dir:
            save_path = Path(args.save_dir) / 'similarity_matrix.png'
        plot_similarity_matrix(args.embeddings_dir, save_path)

    if args.plot_training:
        print("Plotting training curves...")
        plot_training_curves(args.log_dir)

    if args.plot_alignment:
        if not args.checkpoint:
            print("Warning: No checkpoint provided, using random initialization")

        # Load model
        model = HyperProcrustes()
        if args.checkpoint:
            trainer = HyperProcrustesTrainer(model=model, device=args.device)
            trainer.load_checkpoint(args.checkpoint)

        print("Visualizing alignments...")
        visualize_transformation_quality(
            model,
            args.embeddings_dir,
            args.device,
            args.save_dir
        )


if __name__ == '__main__':
    main()