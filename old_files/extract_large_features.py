#!/usr/bin/env python3
"""
Large-scale feature extraction with memory optimization and checkpointing.
Designed for extracting features from large models (>1B parameters).
"""

import gc
import os
import argparse
import warnings
import json
from tqdm import trange
from pathlib import Path
import psutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import utils

# Large model configurations
LARGE_MODELS = {
    'vision': {
        # Giant models (>1B params)
        'eva_giant_patch14_560': {'batch_size': 1, 'gradient_checkpointing': True},
        'vit_giant_patch14_224': {'batch_size': 1, 'gradient_checkpointing': True},
        'vit_gigantic_patch14_224': {'batch_size': 1, 'gradient_checkpointing': True},

        # Large models (300M-1B params)
        'vit_large_patch14_clip_336': {'batch_size': 2, 'gradient_checkpointing': True},
        'vit_huge_patch14_clip_336': {'batch_size': 1, 'gradient_checkpointing': True},
        'convnext_xxlarge': {'batch_size': 2, 'gradient_checkpointing': True},
        'swin_large_patch4_window12_384': {'batch_size': 2, 'gradient_checkpointing': False},
        'beit_large_patch16_384': {'batch_size': 2, 'gradient_checkpointing': False},
        'dinov2_vitg14': {'batch_size': 1, 'gradient_checkpointing': True},
    },
    'language': {
        # Ultra-large models (>65B params)
        'meta-llama/Llama-2-70b-hf': {'batch_size': 1, 'gradient_checkpointing': True, 'load_in_8bit': True},
        'meta-llama/Meta-Llama-3-70B': {'batch_size': 1, 'gradient_checkpointing': True, 'load_in_8bit': True},
        'tiiuae/falcon-180B': {'batch_size': 1, 'gradient_checkpointing': True, 'load_in_4bit': True},

        # Large models (7B-65B params)
        'meta-llama/Llama-2-7b-hf': {'batch_size': 4, 'gradient_checkpointing': True},
        'meta-llama/Llama-2-13b-hf': {'batch_size': 2, 'gradient_checkpointing': True},
        'meta-llama/Llama-2-30b-hf': {'batch_size': 1, 'gradient_checkpointing': True},
        'mistralai/Mistral-7B-v0.1': {'batch_size': 4, 'gradient_checkpointing': True},
        'mistralai/Mixtral-8x7B-v0.1': {'batch_size': 1, 'gradient_checkpointing': True, 'load_in_8bit': True},
        'tiiuae/falcon-40b': {'batch_size': 1, 'gradient_checkpointing': True},
        'EleutherAI/gpt-neox-20b': {'batch_size': 2, 'gradient_checkpointing': True},
        'bigscience/bloom': {'batch_size': 1, 'gradient_checkpointing': True, 'load_in_8bit': True},
    }
}


class MemoryMonitor:
    """Monitor and manage GPU memory during extraction."""

    def __init__(self, threshold_gb=0.5):
        self.threshold_gb = threshold_gb
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None

    def get_memory_stats(self):
        """Get current memory statistics."""
        if self.device is None:
            return {'ram_gb': psutil.virtual_memory().available / 1e9}

        return {
            'gpu_allocated_gb': torch.cuda.memory_allocated(self.device) / 1e9,
            'gpu_reserved_gb': torch.cuda.memory_reserved(self.device) / 1e9,
            'gpu_free_gb': (torch.cuda.get_device_properties(self.device).total_memory -
                           torch.cuda.memory_allocated(self.device)) / 1e9,
            'ram_gb': psutil.virtual_memory().available / 1e9
        }

    def should_clear_cache(self):
        """Check if we should clear GPU cache."""
        if self.device is None:
            return False

        stats = self.get_memory_stats()
        return stats['gpu_free_gb'] < self.threshold_gb

    def clear_memory(self):
        """Clear GPU and CPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class CheckpointManager:
    """Manage checkpointing for interrupted extraction."""

    def __init__(self, checkpoint_dir, model_name):
        self.checkpoint_dir = Path(checkpoint_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name.replace('/', '_')
        self.checkpoint_file = self.checkpoint_dir / f"{self.model_name}_checkpoint.json"

    def save_checkpoint(self, batch_idx, features_path, metadata):
        """Save extraction checkpoint."""
        checkpoint = {
            'batch_idx': batch_idx,
            'features_path': str(features_path),
            'metadata': metadata,
            'timestamp': time.time()
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

    def load_checkpoint(self):
        """Load existing checkpoint if available."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None

    def remove_checkpoint(self):
        """Remove checkpoint after successful completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


def load_large_language_model(model_name, config):
    """Load large language model with optimizations."""
    print(f"Loading {model_name} with optimizations...")

    model_config = AutoConfig.from_pretrained(model_name)

    # Quantization settings
    load_in_8bit = config.get('load_in_8bit', False)
    load_in_4bit = config.get('load_in_4bit', False)

    # Device map for model parallelism
    device_map = 'auto' if torch.cuda.device_count() > 1 else None

    # Load model with appropriate settings
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    # Enable gradient checkpointing if specified
    if config.get('gradient_checkpointing', False) and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def extract_large_llm_features(model_name, dataset, args):
    """Extract features from large language models with memory optimization."""

    # Get model configuration
    config = LARGE_MODELS['language'].get(model_name, {'batch_size': 1, 'gradient_checkpointing': True})
    batch_size = min(args.batch_size, config['batch_size'])

    # Initialize managers
    memory_monitor = MemoryMonitor()
    checkpoint_manager = CheckpointManager(args.output_dir, model_name)

    # Check for existing checkpoint
    checkpoint = checkpoint_manager.load_checkpoint()
    start_idx = 0
    if checkpoint and not args.force_remake:
        print(f"Resuming from checkpoint at batch {checkpoint['batch_idx']}")
        start_idx = checkpoint['batch_idx']

    # Setup save path
    save_path = utils.to_feature_filename(
        args.output_dir, args.dataset, args.subset, model_name,
        pool=args.pool, prompt=args.prompt, caption_idx=args.caption_idx,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path) and not args.force_remake:
        print(f"Features already exist at {save_path}. Skipping...")
        return

    print(f"Extracting features for {model_name}")
    print(f"Memory stats: {memory_monitor.get_memory_stats()}")

    try:
        # Load model
        model, tokenizer = load_large_language_model(model_name, config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {param_count/1e9:.2f}B parameters")

        # Prepare texts
        texts = [str(x['text'][args.caption_idx]) for x in dataset][:args.num_samples]

        # Tokenize in batches to save memory
        all_features = []
        all_losses = []

        # Process in smaller chunks
        chunk_size = min(100, args.num_samples)

        for chunk_start in trange(start_idx, len(texts), chunk_size, desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]

            # Tokenize chunk
            tokens = tokenizer(
                chunk_texts,
                padding="longest",
                truncation=True,
                max_length=512,  # Limit sequence length for memory
                return_tensors="pt"
            )

            chunk_features = []
            chunk_losses = []

            # Process chunk in batches
            for batch_start in range(0, len(chunk_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(chunk_texts))

                # Move batch to device
                device = next(model.parameters()).device
                batch_tokens = {k: v[batch_start:batch_end].to(device) for k, v in tokens.items()}

                with torch.no_grad():
                    outputs = model(
                        input_ids=batch_tokens["input_ids"],
                        attention_mask=batch_tokens["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True
                    )

                    # Extract features based on pooling strategy
                    if args.pool == 'avg':
                        hidden_states = torch.stack(outputs.hidden_states)
                        # Average over sequence length
                        mask = batch_tokens["attention_mask"].unsqueeze(0).unsqueeze(-1)
                        features = (hidden_states * mask).sum(2) / mask.sum(2)
                        features = features.permute(1, 0, 2)  # (batch, layers, dim)
                    elif args.pool == 'last':
                        # Get last token features
                        hidden_states = torch.stack([h[:, -1, :] for h in outputs.hidden_states])
                        features = hidden_states.permute(1, 0, 2)
                    else:
                        raise ValueError(f"Unknown pooling: {args.pool}")

                    chunk_features.append(features.cpu())

                    # Calculate loss if needed
                    if hasattr(outputs, 'loss'):
                        chunk_losses.append(outputs.loss.cpu())

                # Clear memory after each batch
                del batch_tokens, outputs
                if memory_monitor.should_clear_cache():
                    memory_monitor.clear_memory()

            # Concatenate chunk results
            all_features.extend(torch.cat(chunk_features, dim=0))
            if chunk_losses:
                all_losses.extend(chunk_losses)

            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                chunk_end,
                save_path,
                {'param_count': param_count, 'processed_samples': chunk_end}
            )

            # Clear memory after chunk
            memory_monitor.clear_memory()

        # Save final features
        save_dict = {
            "feats": torch.stack(all_features) if len(all_features) > 1 else all_features[0],
            "num_params": param_count,
            "model_config": config,
        }

        if all_losses:
            save_dict["loss"] = torch.stack(all_losses).mean()

        torch.save(save_dict, save_path)
        print(f"Features saved to {save_path}")

        # Remove checkpoint on success
        checkpoint_manager.remove_checkpoint()

    except Exception as e:
        print(f"Error extracting features for {model_name}: {e}")
        raise

    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        memory_monitor.clear_memory()


def extract_large_vision_features(model_name, dataset, args):
    """Extract features from large vision models with memory optimization."""

    # Get model configuration
    config = LARGE_MODELS['vision'].get(model_name, {'batch_size': 2, 'gradient_checkpointing': False})
    batch_size = min(args.batch_size, config['batch_size'])

    # Initialize managers
    memory_monitor = MemoryMonitor()
    checkpoint_manager = CheckpointManager(args.output_dir, model_name)

    # Setup save path
    save_path = utils.to_feature_filename(
        args.output_dir, args.dataset, args.subset, model_name,
        pool=args.pool, prompt=None, caption_idx=None,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path) and not args.force_remake:
        print(f"Features already exist at {save_path}. Skipping...")
        return

    print(f"Extracting features for {model_name}")
    print(f"Memory stats: {memory_monitor.get_memory_stats()}")

    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = timm.create_model(model_name, pretrained=True)

        # Enable gradient checkpointing if specified
        if config.get('gradient_checkpointing', False) and hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)

        model = model.to(device).eval()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {param_count/1e6:.2f}M parameters")

        # Setup transform
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        # Setup feature extraction
        if hasattr(model, 'blocks'):
            return_nodes = [f"blocks.{i}" for i in range(len(model.blocks))]
            model = create_feature_extractor(model, return_nodes=return_nodes)

        all_features = []

        # Process in batches
        num_samples = min(args.num_samples, len(dataset))

        for batch_start in trange(0, num_samples, batch_size, desc="Processing batches"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = list(range(batch_start, batch_end))

            # Load and transform batch
            if hasattr(dataset[0], '__getitem__'):
                # Torchvision dataset
                images = torch.stack([transform(dataset[i][0]) for i in batch_indices])
            else:
                # HuggingFace dataset
                images = torch.stack([transform(dataset[i]['image']) for i in batch_indices])

            images = images.to(device)

            with torch.no_grad():
                if hasattr(model, 'extract_features'):
                    outputs = model.extract_features(images)
                else:
                    outputs = model(images)

                # Handle different output formats
                if isinstance(outputs, dict):
                    # Feature extractor output
                    if args.pool == "cls":
                        features = [v[:, 0, :] for v in outputs.values() if v.dim() >= 3]
                    elif args.pool == "avg":
                        features = [v.mean(dim=1) if v.dim() >= 3 else v for v in outputs.values()]
                    else:
                        features = list(outputs.values())

                    if features:
                        features = torch.stack(features).permute(1, 0, 2)
                    else:
                        features = list(outputs.values())[0]
                else:
                    features = outputs

                all_features.append(features.cpu())

            # Clear memory after batch
            del images, outputs
            if memory_monitor.should_clear_cache():
                memory_monitor.clear_memory()

        # Save features
        save_dict = {
            "feats": torch.cat(all_features, dim=0),
            "num_params": param_count,
            "model_config": config,
        }

        torch.save(save_dict, save_path)
        print(f"Features saved to {save_path}")

    except Exception as e:
        print(f"Error extracting features for {model_name}: {e}")
        raise

    finally:
        # Cleanup
        if 'model' in locals():
            del model
        memory_monitor.clear_memory()


def main():
    parser = argparse.ArgumentParser(description="Extract features from large models")

    parser.add_argument("--models", type=str, nargs="+", required=True,
                       help="Models to extract features from")
    parser.add_argument("--modality", type=str, required=True,
                       choices=["vision", "language"],
                       help="Model modality")
    parser.add_argument("--dataset", type=str, default="minhuh/prh",
                       help="Dataset to use")
    parser.add_argument("--subset", type=str, default="wit_1024",
                       help="Dataset subset")
    parser.add_argument("--num_samples", type=int, default=1024,
                       help="Number of samples to process")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Maximum batch size")
    parser.add_argument("--pool", type=str, default='avg',
                       choices=['avg', 'cls', 'last'],
                       help="Pooling strategy")
    parser.add_argument("--output_dir", type=str, default="./results/features",
                       help="Output directory")
    parser.add_argument("--force_remake", action="store_true",
                       help="Force remake existing files")
    parser.add_argument("--prompt", action="store_true",
                       help="Use prompting for language models")
    parser.add_argument("--caption_idx", type=int, default=0,
                       help="Caption index for multi-caption datasets")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset {args.dataset}/{args.subset}...")
    dataset = load_dataset(args.dataset, revision=args.subset, split='train')

    # Process each model
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")

        try:
            if args.modality == "language":
                extract_large_llm_features(model_name, dataset, args)
            else:
                extract_large_vision_features(model_name, dataset, args)
        except Exception as e:
            print(f"Failed to process {model_name}: {e}")
            continue


if __name__ == "__main__":
    main()