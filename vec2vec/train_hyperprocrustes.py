#!/usr/bin/env python3
"""
Training script for HyperProcrustes using extracted embeddings.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
import random

from hyperprocrustes import HyperProcrustes, HyperProcrustesTrainer


class EmbeddingPairDataset(Dataset):
    """
    Dataset for loading pairs of embeddings from different models.
    """

    def __init__(self, embeddings_dir: str, num_samples: int = 50, mode: str = 'train'):
        self.embeddings_dir = Path(embeddings_dir)
        self.num_samples = num_samples
        self.mode = mode

        # Load metadata
        results_file = self.embeddings_dir / 'extraction_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)
        else:
            # Scan directory for npz files
            self.results = []
            for npz_file in self.embeddings_dir.glob('*.npz'):
                self.results.append({'filepath': str(npz_file)})

        # Filter successful extractions
        self.valid_results = [r for r in self.results if 'error' not in r and 'filepath' in r]

        # Create pairs of different models on the same dataset
        self.pairs = self._create_pairs()

        # Split for train/val
        random.shuffle(self.pairs)
        split_idx = int(0.8 * len(self.pairs))
        if mode == 'train':
            self.pairs = self.pairs[:split_idx]
        else:
            self.pairs = self.pairs[split_idx:]

        print(f"Created {len(self.pairs)} {mode} pairs from {len(self.valid_results)} embeddings")

    def _create_pairs(self) -> List[Tuple[Dict, Dict]]:
        """Create pairs of embeddings from different models on same datasets."""
        pairs = []

        # Group by dataset
        dataset_groups = {}
        for result in self.valid_results:
            if 'dataset' in result:
                dataset = result['dataset']
            else:
                # Parse from filename
                filename = Path(result['filepath']).stem
                dataset = filename.split('_')[0]

            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(result)

        # Create pairs within each dataset
        for dataset, results in dataset_groups.items():
            if len(results) >= 2:
                # Create all possible pairs
                for i in range(len(results)):
                    for j in range(i + 1, len(results)):
                        pairs.append((results[i], results[j]))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        result_A, result_B = self.pairs[idx]

        # Load embeddings
        data_A = np.load(result_A['filepath'])
        data_B = np.load(result_B['filepath'])

        embeddings_A = data_A['embeddings'][:self.num_samples]
        embeddings_B = data_B['embeddings'][:self.num_samples]

        # Convert to tensors
        embeddings_A = torch.FloatTensor(embeddings_A)
        embeddings_B = torch.FloatTensor(embeddings_B)

        # Get metadata
        model_A = str(data_A.get('model_name', 'unknown'))
        model_B = str(data_B.get('model_name', 'unknown'))
        dataset_name = str(data_A.get('dataset_name', 'unknown'))

        return {
            'embeddings_A': embeddings_A,
            'embeddings_B': embeddings_B,
            'model_A': model_A,
            'model_B': model_B,
            'dataset': dataset_name
        }


def evaluate_alignment(model: HyperProcrustes,
                       embeddings_A: torch.Tensor,
                       embeddings_B: torch.Tensor,
                       device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate alignment quality using multiple metrics.
    """
    model.eval()
    with torch.no_grad():
        embeddings_A = embeddings_A.to(device)
        embeddings_B = embeddings_B.to(device)

        # Get transformation parameters
        params = model(embeddings_A.unsqueeze(0), embeddings_B.unsqueeze(0))

        # Transform A to B
        A_transformed = model.transform(embeddings_A.unsqueeze(0), params, direction='A_to_B')
        A_transformed = A_transformed.squeeze(0)

        # Compute metrics
        # 1. Mean Squared Error
        mse = torch.mean((A_transformed - embeddings_B) ** 2).item()

        # 2. Cosine similarity
        cos_sim_list = []
        for i in range(min(A_transformed.shape[0], embeddings_B.shape[0])):
            cos_sim = torch.nn.functional.cosine_similarity(
                A_transformed[i].unsqueeze(0),
                embeddings_B[i].unsqueeze(0)
            )
            cos_sim_list.append(cos_sim.item())
        avg_cos_sim = np.mean(cos_sim_list)

        # 3. Procrustes distance (canonical metric)
        # Center both sets
        A_t_centered = A_transformed - A_transformed.mean(dim=0)
        B_centered = embeddings_B - embeddings_B.mean(dim=0)

        # Frobenius norm of difference
        procrustes_dist = torch.norm(A_t_centered - B_centered, 'fro').item()

        # 4. Correlation of pairwise distances (geometry preservation)
        dist_A = torch.cdist(A_transformed[:50], A_transformed[:50])
        dist_B = torch.cdist(embeddings_B[:50], embeddings_B[:50])

        # Flatten and compute correlation
        dist_A_flat = dist_A.flatten()
        dist_B_flat = dist_B.flatten()

        # Pearson correlation
        mean_A = dist_A_flat.mean()
        mean_B = dist_B_flat.mean()
        cov = ((dist_A_flat - mean_A) * (dist_B_flat - mean_B)).mean()
        std_A = dist_A_flat.std()
        std_B = dist_B_flat.std()
        distance_correlation = (cov / (std_A * std_B + 1e-8)).item()

    return {
        'mse': mse,
        'cosine_similarity': avg_cos_sim,
        'procrustes_distance': procrustes_dist,
        'distance_correlation': distance_correlation,
        'predicted_quality': params['predicted_quality'].mean().item()
    }


def analyze_model_families(model: HyperProcrustes,
                          embeddings_dir: str,
                          device: str = 'cuda'):
    """
    Analyze alignment quality across different model families.
    """
    print("\n" + "="*60)
    print("ANALYZING MODEL FAMILY ALIGNMENTS")
    print("="*60)

    embeddings_path = Path(embeddings_dir)

    # Define model families
    families = {
        'BERT': ['bert_base_uncased', 'roberta_base', 'distilbert_base_uncased'],
        'GPT': ['gpt2', 'gpt2_medium'],
        'T5': ['t5_v1_1_small', 't5_v1_1_base']
    }

    results = {}

    # Within-family alignment
    print("\nWithin-Family Alignments:")
    for family_name, models in families.items():
        family_results = []

        for i, model_A in enumerate(models):
            for j, model_B in enumerate(models):
                if i < j:
                    # Try to find embeddings
                    pattern_A = f"*{model_A}*"
                    pattern_B = f"*{model_B}*"

                    files_A = list(embeddings_path.glob(pattern_A))
                    files_B = list(embeddings_path.glob(pattern_B))

                    if files_A and files_B:
                        # Load first match
                        data_A = np.load(files_A[0])
                        data_B = np.load(files_B[0])

                        embeddings_A = torch.FloatTensor(data_A['embeddings'][:50])
                        embeddings_B = torch.FloatTensor(data_B['embeddings'][:50])

                        metrics = evaluate_alignment(model, embeddings_A, embeddings_B, device)

                        print(f"  {model_A} -> {model_B}:")
                        print(f"    Cosine Sim: {metrics['cosine_similarity']:.4f}")
                        print(f"    Distance Corr: {metrics['distance_correlation']:.4f}")
                        print(f"    Predicted Quality: {metrics['predicted_quality']:.4f}")

                        family_results.append(metrics)

        if family_results:
            avg_metrics = {
                key: np.mean([r[key] for r in family_results])
                for key in family_results[0].keys()
            }
            results[family_name] = avg_metrics

    # Cross-family alignment
    print("\nCross-Family Alignments:")
    cross_pairs = [
        ('bert_base_uncased', 'gpt2'),
        ('bert_base_uncased', 't5_v1_1_small'),
        ('gpt2', 't5_v1_1_small')
    ]

    cross_results = []
    for model_A, model_B in cross_pairs:
        pattern_A = f"*{model_A}*"
        pattern_B = f"*{model_B}*"

        files_A = list(embeddings_path.glob(pattern_A))
        files_B = list(embeddings_path.glob(pattern_B))

        if files_A and files_B:
            data_A = np.load(files_A[0])
            data_B = np.load(files_B[0])

            embeddings_A = torch.FloatTensor(data_A['embeddings'][:50])
            embeddings_B = torch.FloatTensor(data_B['embeddings'][:50])

            metrics = evaluate_alignment(model, embeddings_A, embeddings_B, device)

            print(f"  {model_A} -> {model_B}:")
            print(f"    Cosine Sim: {metrics['cosine_similarity']:.4f}")
            print(f"    Distance Corr: {metrics['distance_correlation']:.4f}")
            print(f"    Predicted Quality: {metrics['predicted_quality']:.4f}")

            cross_results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train HyperProcrustes on extracted embeddings")

    # Data arguments
    parser.add_argument('--embeddings-dir', type=str, default='./embeddings_bert',
                       help='Directory containing extracted embeddings')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of embedding samples per model')

    # Model arguments
    parser.add_argument('--max-dim', type=int, default=1024,
                       help='Maximum embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension of hypernetwork')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of hypernetwork layers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--log-dir', type=str, default='./runs/hyperprocrustes',
                       help='TensorBoard log directory')

    # Mode
    parser.add_argument('--mode', choices=['train', 'evaluate', 'analyze'],
                       default='train',
                       help='Operation mode')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint for evaluation')

    args = parser.parse_args()

    # Initialize model
    model = HyperProcrustes(
        max_embedding_dim=args.max_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    if args.mode == 'train':
        # Create datasets
        train_dataset = EmbeddingPairDataset(
            args.embeddings_dir,
            num_samples=args.num_samples,
            mode='train'
        )
        val_dataset = EmbeddingPairDataset(
            args.embeddings_dir,
            num_samples=args.num_samples,
            mode='val'
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )

        # Initialize trainer
        trainer = HyperProcrustesTrainer(
            model=model,
            device=args.device,
            log_dir=args.log_dir
        )

        # Train
        print(f"\nStarting training on {len(train_dataset)} train pairs, {len(val_dataset)} val pairs")
        print(f"Logs will be saved to {trainer.log_dir}")
        print("\nTo view training progress:")
        print(f"  tensorboard --logdir {args.log_dir}\n")

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_interval=10
        )

    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for evaluation")

        # Load checkpoint
        trainer = HyperProcrustesTrainer(model=model, device=args.device)
        trainer.load_checkpoint(args.checkpoint)

        # Create validation dataset
        val_dataset = EmbeddingPairDataset(
            args.embeddings_dir,
            num_samples=args.num_samples,
            mode='val'
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False
        )

        # Evaluate
        print("\nEvaluating model...")
        all_metrics = []

        for batch in tqdm(val_loader, desc="Evaluating"):
            metrics = evaluate_alignment(
                model,
                batch['embeddings_A'][0],
                batch['embeddings_B'][0],
                args.device
            )
            all_metrics.append(metrics)

            # Print sample results
            if len(all_metrics) <= 5:
                print(f"\n{batch['model_A'][0]} -> {batch['model_B'][0]} ({batch['dataset'][0]}):")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

        # Aggregate metrics
        print("\n" + "="*60)
        print("AGGREGATE METRICS")
        print("="*60)
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key:20s}: {mean_val:.4f} ± {std_val:.4f}")

    elif args.mode == 'analyze':
        if args.checkpoint:
            trainer = HyperProcrustesTrainer(model=model, device=args.device)
            trainer.load_checkpoint(args.checkpoint)

        # Analyze model families
        analyze_model_families(model, args.embeddings_dir, args.device)


if __name__ == '__main__':
    main()