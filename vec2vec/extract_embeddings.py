#!/usr/bin/env python3
"""
Extract embeddings from multiple models for given texts.
Supports batch processing and multiple output formats.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from utils.model_utils import load_encoder, HF_FLAGS


def extract_embeddings(
    texts: List[str],
    model_names: List[str],
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    mixed_precision: Optional[str] = 'fp16',
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from multiple models for given texts.

    Args:
        texts: List of input texts
        model_names: List of model names/identifiers
        batch_size: Batch size for encoding
        device: Device to use ('cuda' or 'cpu')
        mixed_precision: Mixed precision setting ('fp16', 'bf16', or None)
        show_progress: Whether to show progress bars

    Returns:
        Dictionary mapping model names to embedding arrays
    """
    embeddings_dict = {}

    for model_name in tqdm(model_names, desc="Processing models", disable=not show_progress):
        print(f"\nExtracting embeddings from: {model_name}")

        try:
            # Load model
            encoder = load_encoder(model_name, device=device, mixed_precision=mixed_precision)

            # Extract embeddings in batches
            all_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size),
                         desc=f"Encoding batches",
                         disable=not show_progress):
                batch_texts = texts[i:i + batch_size]

                with torch.no_grad():
                    batch_embeddings = encoder.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                all_embeddings.append(batch_embeddings)

            # Combine all batches
            embeddings = np.vstack(all_embeddings)
            embeddings_dict[model_name] = embeddings

            print(f"✓ Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

            # Clear GPU memory if using CUDA
            if device == 'cuda':
                del encoder
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ Failed to extract embeddings from {model_name}: {e}")
            continue

    return embeddings_dict


def save_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    output_path: str,
    format: str = 'npz',
    metadata: Optional[Dict] = None
):
    """
    Save embeddings to file in specified format.

    Args:
        embeddings_dict: Dictionary of model names to embeddings
        output_path: Output file path
        format: Output format ('npz', 'pkl', 'pt', 'h5')
        metadata: Optional metadata to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'npz':
        # Save as numpy archive
        save_dict = embeddings_dict.copy()
        if metadata:
            save_dict['metadata'] = metadata
        np.savez_compressed(output_path, **save_dict)

    elif format == 'pkl':
        # Save as pickle
        save_data = {'embeddings': embeddings_dict}
        if metadata:
            save_data['metadata'] = metadata
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)

    elif format == 'pt':
        # Save as PyTorch tensor
        tensor_dict = {k: torch.tensor(v) for k, v in embeddings_dict.items()}
        save_data = {'embeddings': tensor_dict}
        if metadata:
            save_data['metadata'] = metadata
        torch.save(save_data, output_path)

    elif format == 'h5':
        # Save as HDF5
        try:
            import h5py
            with h5py.File(output_path, 'w') as f:
                for model_name, embeddings in embeddings_dict.items():
                    f.create_dataset(model_name, data=embeddings, compression='gzip')
                if metadata:
                    f.attrs.update(metadata)
        except ImportError:
            raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")

    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved embeddings to {output_path}")


def load_texts(input_path: str, text_column: Optional[str] = None) -> List[str]:
    """
    Load texts from various file formats.

    Args:
        input_path: Path to input file
        text_column: Column name for CSV/JSON files

    Returns:
        List of texts
    """
    input_path = Path(input_path)

    if input_path.suffix == '.txt':
        with open(input_path, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]

    elif input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            if text_column and isinstance(data[0], dict):
                texts = [item[text_column] for item in data]
            else:
                texts = data
        else:
            raise ValueError("JSON file must contain a list")

    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
        if text_column:
            texts = df[text_column].tolist()
        else:
            # Use first column if not specified
            texts = df.iloc[:, 0].tolist()

    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    return texts


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from multiple models")

    # Input arguments
    parser.add_argument('--input', type=str,
                       help='Input file path (txt, json, or csv)')
    parser.add_argument('--text-column', type=str,
                       help='Column name for CSV/JSON files')
    parser.add_argument('--texts', nargs='+',
                       help='Direct text inputs (alternative to --input)')

    # Model arguments
    parser.add_argument('--models', nargs='+',
                       help='List of model names or "all" for all available models')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models')

    # Output arguments
    parser.add_argument('--output', type=str, default='embeddings.npz',
                       help='Output file path')
    parser.add_argument('--format', choices=['npz', 'pkl', 'pt', 'h5'],
                       default='npz',
                       help='Output format (default: npz)')

    # Processing arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--mixed-precision', choices=['fp16', 'bf16', 'no'],
                       default='fp16' if torch.cuda.is_available() else 'no',
                       help='Mixed precision setting')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')

    # Analysis arguments
    parser.add_argument('--compute-stats', action='store_true',
                       help='Compute and save embedding statistics')

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("Available models:")
        for i, (key, value) in enumerate(HF_FLAGS.items(), 1):
            print(f"  {i:2d}. {key:20s} -> {value}")
        return

    # Load texts
    if args.texts:
        texts = args.texts
    elif args.input:
        texts = load_texts(args.input, args.text_column)
    else:
        if not args.list_models:
            raise ValueError("Either --input or --texts must be provided")

    if not args.list_models:
        print(f"Loaded {len(texts)} texts")

        # Determine models to use
        if 'all' in args.models:
            model_names = list(HF_FLAGS.keys())
        else:
            model_names = args.models

        print(f"Will extract embeddings from {len(model_names)} models")

        # Extract embeddings
        embeddings_dict = extract_embeddings(
            texts=texts,
            model_names=model_names,
            batch_size=args.batch_size,
            device=args.device,
            mixed_precision=args.mixed_precision,
            show_progress=not args.no_progress
        )

        # Prepare metadata
        metadata = {
            'num_texts': len(texts),
            'models': list(embeddings_dict.keys()),
            'device': args.device,
            'mixed_precision': args.mixed_precision
        }

        # Compute statistics if requested
        if args.compute_stats:
            stats = {}
            for model_name, embeddings in embeddings_dict.items():
                stats[model_name] = {
                    'mean': float(np.mean(embeddings)),
                    'std': float(np.std(embeddings)),
                    'min': float(np.min(embeddings)),
                    'max': float(np.max(embeddings)),
                    'shape': embeddings.shape
                }
            metadata['stats'] = stats

            print("\nEmbedding Statistics:")
            for model_name, model_stats in stats.items():
                print(f"\n{model_name}:")
                print(f"  Shape: {model_stats['shape']}")
                print(f"  Mean:  {model_stats['mean']:.4f}")
                print(f"  Std:   {model_stats['std']:.4f}")
                print(f"  Range: [{model_stats['min']:.4f}, {model_stats['max']:.4f}]")

        # Save embeddings
        save_embeddings(
            embeddings_dict=embeddings_dict,
            output_path=args.output,
            format=args.format,
            metadata=metadata
        )

        print(f"\n✅ Successfully extracted embeddings from {len(embeddings_dict)} models")


if __name__ == '__main__':
    main()