#!/usr/bin/env python3
"""
Exhaustive Multi-Dataset Feature Extractor
Extends the original extract_features.py to handle multiple datasets systematically
"""

import gc
import os
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import yaml
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from PIL import Image
import numpy as np

# Import existing utilities
from old_files.tasks import get_models, sort_models_by_size
from old_files.models import load_llm, load_tokenizer
from multi_dataset_config import MultiDatasetConfig, DatasetConfig, DatasetType
import old_files.utils as utils


class ExhaustiveMultiDatasetExtractor:
    """
    Exhaustive feature extractor that handles multiple datasets systematically
    Supports both vision and language datasets with comprehensive error handling
    """

    def __init__(self, output_dir: str = "./results/features", config_dir: str = "./config"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize multi-dataset configuration
        self.config = MultiDatasetConfig(config_dir)

        # Processing statistics
        self.stats = {
            "total_datasets": 0,
            "total_models": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "skipped_extractions": 0,
            "start_time": None,
            "end_time": None,
            "errors": []
        }

    def create_model_dataset_output_dir(self, model_name: str, dataset_name: str, subset: str = "") -> Path:
        """Create output directory structure: representations/{model_name}/{dataset_name}/"""
        # Clean model name for filesystem
        clean_model_name = model_name.replace("/", "_").replace("-", "_")

        if subset:
            model_dataset_path = self.output_dir / clean_model_name / f"{dataset_name}_{subset}"
        else:
            model_dataset_path = self.output_dir / clean_model_name / dataset_name

        model_dataset_path.mkdir(parents=True, exist_ok=True)
        return model_dataset_path

    def load_dataset_safely(self, dataset_config: DatasetConfig, split: str = "train") -> Optional[Dataset]:
        """Safely load a dataset with error handling"""
        try:
            print(f"Loading dataset: {dataset_config.name}")

            # Handle different dataset loading patterns
            if dataset_config.subset:
                if ":" in dataset_config.subset:
                    # Handle revision-based loading (e.g., "wit_1024" as revision)
                    dataset = load_dataset(
                        dataset_config.name,
                        revision=dataset_config.subset,
                        split=split
                    )
                else:
                    # Handle subset-based loading
                    dataset = load_dataset(
                        dataset_config.name,
                        dataset_config.subset,
                        split=split
                    )
            else:
                dataset = load_dataset(dataset_config.name, split=split)

            # Limit samples if specified
            if dataset_config.num_samples and len(dataset) > dataset_config.num_samples:
                dataset = dataset.select(range(dataset_config.num_samples))

            print(f"  Loaded {len(dataset)} samples")
            return dataset

        except Exception as e:
            error_msg = f"Failed to load dataset {dataset_config.name}: {e}"
            print(f"  ❌ {error_msg}")
            self.stats["errors"].append(error_msg)
            return None

    def extract_vision_features_multi_dataset(self, vision_models: List[str],
                                            dataset_configs: List[DatasetConfig],
                                            args: argparse.Namespace):
        """Extract vision features across multiple datasets"""

        print(f"\n{'='*80}")
        print(f"EXTRACTING VISION FEATURES ACROSS {len(dataset_configs)} DATASETS")
        print(f"{'='*80}")

        for dataset_config in dataset_configs:
            if dataset_config.dataset_type not in [DatasetType.VISION, DatasetType.MULTIMODAL]:
                print(f"Skipping {dataset_config.name} - not a vision dataset")
                continue

            print(f"\n{'='*50}")
            print(f"DATASET: {dataset_config.name} ({dataset_config.dataset_type.value})")
            print(f"{'='*50}")

            # Load dataset
            dataset = self.load_dataset_safely(dataset_config)
            if dataset is None:
                continue

            # Extract features for each vision model with model-specific directories
            self._extract_vision_features_single_dataset(vision_models, dataset, dataset_config, args)

    def _extract_vision_features_single_dataset(self, vision_models: List[str],
                                              dataset: Dataset,
                                              dataset_config: DatasetConfig,
                                              args: argparse.Namespace):
        """Extract vision features for a single dataset"""

        # Sort models by size (smallest first for memory efficiency)
        sorted_models = sort_models_by_size(vision_models, model_type="lvm")

        for model_name in sorted_models:
            print(f"\nProcessing vision model: {model_name}")

            # Create model-dataset specific output directory
            model_output_dir = self.create_model_dataset_output_dir(
                model_name, dataset_config.name, dataset_config.subset
            )

            # Generate save path
            save_path = model_output_dir / f"features_{args.pool}.pt"

            if save_path.exists() and not args.force_remake:
                print(f"  ✓ Features already exist, skipping")
                self.stats["skipped_extractions"] += 1
                continue

            try:
                success = self._extract_single_vision_model_features(
                    model_name, dataset, dataset_config, save_path, args
                )

                if success:
                    self.stats["successful_extractions"] += 1
                    print(f"  ✓ Successfully extracted features to {save_path}")
                else:
                    self.stats["failed_extractions"] += 1

            except Exception as e:
                error_msg = f"Failed to extract features for {model_name} on {dataset_config.name}: {e}"
                print(f"  ❌ {error_msg}")
                self.stats["errors"].append(error_msg)
                self.stats["failed_extractions"] += 1
                continue

    def _extract_single_vision_model_features(self, model_name: str,
                                            dataset: Dataset,
                                            dataset_config: DatasetConfig,
                                            save_path: Path,
                                            args: argparse.Namespace) -> bool:
        """Extract features for a single vision model"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Using device: {device}")

            # Load vision model
            vision_model = timm.create_model(model_name, pretrained=True).to(device).eval()
            model_param_count = sum(p.numel() for p in vision_model.parameters())

            # Create transform
            transform = create_transform(
                **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
            )

            # Setup feature extraction
            if "vit" in model_name.lower() or "deit" in model_name.lower():
                # ViT-based models
                if hasattr(vision_model, 'blocks'):
                    return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
                else:
                    return_nodes = None
            else:
                return_nodes = None

            if return_nodes:
                vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

            # Extract features
            all_features = []
            batch_size = args.batch_size

            for i in trange(0, len(dataset), batch_size, desc=f"  Extracting {model_name}"):
                batch_end = min(i + batch_size, len(dataset))
                batch_indices = list(range(i, batch_end))

                try:
                    # Prepare batch of images
                    images = []
                    for idx in batch_indices:
                        img = dataset[idx]['image'] if 'image' in dataset[idx] else dataset[idx]['img']
                        if not isinstance(img, Image.Image):
                            img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                        images.append(transform(img))

                    batch_tensor = torch.stack(images).to(device)

                    with torch.no_grad():
                        if return_nodes:
                            output = vision_model(batch_tensor)

                            # Apply pooling strategy
                            if args.pool == "cls":
                                # Use CLS token (first token)
                                feats = [v[:, 0, :] for v in output.values()]
                            elif args.pool == "avg":
                                # Average pooling
                                feats = [v.mean(dim=1) for v in output.values()]
                            elif args.pool == "max":
                                # Max pooling
                                feats = [v.max(dim=1)[0] for v in output.values()]
                            else:
                                raise ValueError(f"Unknown pooling strategy: {args.pool}")

                            feats = torch.stack(feats).permute(1, 0, 2)
                        else:
                            # Fallback for non-ViT models
                            output = vision_model(batch_tensor)
                            if isinstance(output, torch.Tensor):
                                feats = output.unsqueeze(1)  # Add layer dimension
                            else:
                                feats = output[0].unsqueeze(1) if isinstance(output, (list, tuple)) else output

                        all_features.append(feats.cpu())

                except Exception as e:
                    print(f"    Warning: Batch {i}-{batch_end} failed: {e}")
                    continue

                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if all_features:
                # Save features
                final_features = torch.cat(all_features, dim=0)
                save_dict = {
                    "feats": final_features,
                    "num_params": model_param_count,
                    "dataset_name": dataset_config.name,
                    "dataset_subset": dataset_config.subset,
                    "num_samples": len(dataset),
                    "pooling_strategy": args.pool,
                    "model_name": model_name
                }

                torch.save(save_dict, save_path)
                return True
            else:
                print(f"    No features extracted for {model_name}")
                return False

        finally:
            # Cleanup
            if 'vision_model' in locals():
                del vision_model
            if 'transform' in locals():
                del transform
            if 'all_features' in locals():
                del all_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

    def extract_language_features_multi_dataset(self, language_models: List[str],
                                               dataset_configs: List[DatasetConfig],
                                               args: argparse.Namespace):
        """Extract language features across multiple datasets"""

        print(f"\n{'='*80}")
        print(f"EXTRACTING LANGUAGE FEATURES ACROSS {len(dataset_configs)} DATASETS")
        print(f"{'='*80}")

        for dataset_config in dataset_configs:
            if dataset_config.dataset_type not in [DatasetType.LANGUAGE, DatasetType.MULTIMODAL]:
                print(f"Skipping {dataset_config.name} - not a language dataset")
                continue

            print(f"\n{'='*50}")
            print(f"DATASET: {dataset_config.name} ({dataset_config.dataset_type.value})")
            print(f"{'='*50}")

            # Load dataset
            dataset = self.load_dataset_safely(dataset_config)
            if dataset is None:
                continue

            # Extract features for each language model with model-specific directories
            self._extract_language_features_single_dataset(language_models, dataset, dataset_config, args)

    def _extract_language_features_single_dataset(self, language_models: List[str],
                                                dataset: Dataset,
                                                dataset_config: DatasetConfig,
                                                args: argparse.Namespace):
        """Extract language features for a single dataset"""

        # Sort models by size (smallest first for memory efficiency)
        sorted_models = sort_models_by_size(language_models, model_type="llm")

        for model_name in sorted_models:
            print(f"\nProcessing language model: {model_name}")

            # Create model-dataset specific output directory
            model_output_dir = self.create_model_dataset_output_dir(
                model_name, dataset_config.name, dataset_config.subset
            )

            # Generate save path
            save_path = model_output_dir / f"features_{args.pool}.pt"

            if save_path.exists() and not args.force_remake:
                print(f"  ✓ Features already exist, skipping")
                self.stats["skipped_extractions"] += 1
                continue

            try:
                success = self._extract_single_language_model_features(
                    model_name, dataset, dataset_config, save_path, args
                )

                if success:
                    self.stats["successful_extractions"] += 1
                    print(f"  ✓ Successfully extracted features to {save_path}")
                else:
                    self.stats["failed_extractions"] += 1

            except Exception as e:
                error_msg = f"Failed to extract features for {model_name} on {dataset_config.name}: {e}"
                print(f"  ❌ {error_msg}")
                self.stats["errors"].append(error_msg)
                self.stats["failed_extractions"] += 1
                continue

    def _extract_single_language_model_features(self, model_name: str,
                                              dataset: Dataset,
                                              dataset_config: DatasetConfig,
                                              save_path: Path,
                                              args: argparse.Namespace) -> bool:
        """Extract features for a single language model"""
        try:
            # Load model and tokenizer
            language_model = load_llm(model_name, qlora=args.qlora, force_download=args.force_download)
            model_param_count = sum(p.numel() for p in language_model.parameters())
            tokenizer = load_tokenizer(model_name)

            device = next(language_model.parameters()).device
            print(f"  Using device: {device}")

            # Prepare texts from dataset
            texts = self._extract_texts_from_dataset(dataset, dataset_config, args)
            if not texts:
                print("  No texts found in dataset")
                return False

            # Tokenize all texts
            tokens = tokenizer(texts, padding="longest", return_tensors="pt", truncation=True, max_length=512)

            all_features = []
            losses = []
            bpb_losses = []
            batch_size = args.batch_size

            for i in trange(0, len(texts), batch_size, desc=f"  Extracting {model_name}"):
                batch_end = min(i + batch_size, len(texts))

                # Get batch tokens
                batch_tokens = {k: v[i:batch_end].to(device).long() for k, v in tokens.items()}

                try:
                    with torch.no_grad():
                        # Get model output with hidden states
                        output = language_model(
                            input_ids=batch_tokens["input_ids"],
                            attention_mask=batch_tokens["attention_mask"],
                            output_hidden_states=True,
                        )

                        # Calculate loss
                        loss, avg_loss = utils.cross_entropy_loss(batch_tokens, output)
                        losses.extend(avg_loss.cpu())

                        # Calculate bits per byte
                        bpb = utils.cross_entropy_to_bits_per_unit(
                            loss.cpu(), texts[i:batch_end], unit="byte"
                        )
                        bpb_losses.extend(bpb)

                        # Apply pooling strategy
                        if args.pool == 'avg':
                            # Average pooling with attention mask
                            feats = torch.stack(output["hidden_states"]).permute(1, 0, 2, 3)
                            mask = batch_tokens["attention_mask"].unsqueeze(-1).unsqueeze(1)
                            feats = (feats * mask).sum(2) / mask.sum(2)
                        elif args.pool == 'last':
                            # Last token pooling
                            feats = [v[:, -1, :] for v in output["hidden_states"]]
                            feats = torch.stack(feats).permute(1, 0, 2)
                        else:
                            raise NotImplementedError(f"Unknown pooling strategy: {args.pool}")

                        all_features.append(feats.cpu())

                except Exception as e:
                    print(f"    Warning: Batch {i}-{batch_end} failed: {e}")
                    continue

                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if all_features:
                # Save features
                final_features = torch.cat(all_features, dim=0)
                save_dict = {
                    "feats": final_features,
                    "num_params": model_param_count,
                    "mask": tokens["attention_mask"].cpu(),
                    "loss": torch.stack(losses).mean() if losses else torch.tensor(0.0),
                    "bpb": torch.stack(bpb_losses).mean() if bpb_losses else torch.tensor(0.0),
                    "dataset_name": dataset_config.name,
                    "dataset_subset": dataset_config.subset,
                    "num_samples": len(texts),
                    "pooling_strategy": args.pool,
                    "model_name": model_name
                }

                torch.save(save_dict, save_path)
                print(f"  Average loss: {save_dict['loss']:.4f}")
                return True
            else:
                print(f"  No features extracted for {model_name}")
                return False

        finally:
            # Cleanup
            if 'language_model' in locals():
                del language_model
            if 'tokenizer' in locals():
                del tokenizer
            if 'all_features' in locals():
                del all_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

    def _extract_texts_from_dataset(self, dataset: Dataset,
                                  dataset_config: DatasetConfig,
                                  args: argparse.Namespace) -> List[str]:
        """Extract texts from dataset based on its structure"""
        texts = []

        # Common text field names to check
        text_fields = ['text', 'sentence', 'content', 'caption', 'review', 'comment']

        for item in dataset:
            text = None

            # Try different text field extraction strategies
            for field in text_fields:
                if field in item:
                    candidate = item[field]

                    # Handle different text formats
                    if isinstance(candidate, str):
                        text = candidate
                        break
                    elif isinstance(candidate, list) and len(candidate) > 0:
                        # Use specified caption index or first item
                        idx = min(args.caption_idx, len(candidate) - 1)
                        text = str(candidate[idx])
                        break
                    elif hasattr(candidate, 'decode'):  # bytes
                        text = candidate.decode('utf-8')
                        break

            # Fallback: convert entire item to string
            if text is None:
                text = str(item)

            texts.append(text)

        return texts

    def run_exhaustive_extraction(self, experiment_plan: Dict[str, Any], args: argparse.Namespace):
        """Run exhaustive feature extraction based on experiment plan"""

        self.stats["start_time"] = datetime.now()
        print(f"\n{'='*100}")
        print(f"STARTING EXHAUSTIVE MULTI-DATASET FEATURE EXTRACTION")
        print(f"Start time: {self.stats['start_time']}")
        print(f"{'='*100}")

        # Print experiment overview
        metadata = experiment_plan["experiment_metadata"]
        print(f"Total experiments: {len(experiment_plan['experiments'])}")
        print(f"Total datasets: {metadata['total_datasets']}")
        print(f"Total combinations: {metadata['total_model_dataset_combinations']}")
        print(f"Estimated memory: {metadata['estimated_total_memory_gb']:.1f} GB")
        print(f"Estimated compute: {metadata['estimated_total_compute_hours']:.1f} hours")

        # Process each experiment
        for i, experiment in enumerate(experiment_plan["experiments"]):
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i+1}/{len(experiment_plan['experiments'])}: {experiment['dataset_name']}")
            print(f"Priority: {experiment['experiment_metadata']['priority']:.2f}")
            print(f"{'='*80}")

            # Reconstruct dataset config
            dataset_config_dict = experiment["dataset_config"]
            dataset_config_dict["dataset_type"] = DatasetType(dataset_config_dict["dataset_type"])
            dataset_config = DatasetConfig(**dataset_config_dict)

            # Extract model names
            model_names = [model["model_name"] for model in experiment["models"]]

            # Separate by modality
            llm_models = [m for m in model_names if any(model["model_type"] == "llm"
                         for model in experiment["models"] if model["model_name"] == m)]
            lvm_models = [m for m in model_names if any(model["model_type"] == "lvm"
                         for model in experiment["models"] if model["model_name"] == m)]

            # Extract features based on dataset type and selected modality
            if args.modality in ["all", "language"] and llm_models:
                if dataset_config.dataset_type in [DatasetType.LANGUAGE, DatasetType.MULTIMODAL]:
                    self.extract_language_features_multi_dataset(llm_models, [dataset_config], args)

            if args.modality in ["all", "vision"] and lvm_models:
                if dataset_config.dataset_type in [DatasetType.VISION, DatasetType.MULTIMODAL]:
                    self.extract_vision_features_multi_dataset(lvm_models, [dataset_config], args)

        self.stats["end_time"] = datetime.now()
        self._print_final_statistics()

    def _print_final_statistics(self):
        """Print final extraction statistics"""
        self.stats["total_time"] = self.stats["end_time"] - self.stats["start_time"]

        print(f"\n{'='*100}")
        print(f"EXHAUSTIVE EXTRACTION COMPLETED")
        print(f"{'='*100}")
        print(f"Start time: {self.stats['start_time']}")
        print(f"End time: {self.stats['end_time']}")
        print(f"Total time: {self.stats['total_time']}")
        print(f"\nStatistics:")
        print(f"  Successful extractions: {self.stats['successful_extractions']}")
        print(f"  Failed extractions: {self.stats['failed_extractions']}")
        print(f"  Skipped extractions: {self.stats['skipped_extractions']}")
        print(f"  Total errors: {len(self.stats['errors'])}")

        if self.stats["errors"]:
            print(f"\nFirst 10 errors:")
            for error in self.stats["errors"][:10]:
                print(f"  • {error}")

        # Save statistics
        stats_path = self.output_dir / "extraction_statistics.json"
        with open(stats_path, 'w') as f:
            # Convert datetime objects for JSON serialization
            stats_copy = self.stats.copy()
            stats_copy["start_time"] = str(stats_copy["start_time"])
            stats_copy["end_time"] = str(stats_copy["end_time"])
            stats_copy["total_time"] = str(stats_copy["total_time"])
            json.dump(stats_copy, f, indent=2)

        print(f"\nStatistics saved to: {stats_path}")


def main():
    """Main function for exhaustive multi-dataset feature extraction"""

    parser = argparse.ArgumentParser(
        description="Exhaustive Multi-Dataset Feature Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run exhaustive extraction on all vision datasets with small models
  python exhaustive_multi_dataset_extractor.py --max_memory_gb 16 --modality vision

  # Run on language datasets only with medium-sized models
  python exhaustive_multi_dataset_extractor.py --max_memory_gb 32 --modality language --max_models_per_dataset 5

  # Run comprehensive extraction with large models (requires significant GPU memory)
  python exhaustive_multi_dataset_extractor.py --max_memory_gb 64 --modality all --max_models_per_dataset 10

  # Test run with minimal resources
  python exhaustive_multi_dataset_extractor.py --max_memory_gb 8 --max_models_per_dataset 2 --batch_size 2
        """)

    parser.add_argument("--force_download", action="store_true", help="Force download of models")
    parser.add_argument("--force_remake", action="store_true", help="Force remake of existing feature files")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--pool", type=str, default='avg', choices=['avg', 'cls', 'last', 'max'],
                       help="Pooling strategy")
    parser.add_argument("--caption_idx", type=int, default=0, help="Caption index for multi-caption datasets")
    parser.add_argument("--modality", type=str, default="all", choices=["vision", "language", "all"],
                       help="Modality to extract features for")
    parser.add_argument("--output_dir", type=str, default="./results/features_exhaustive", help="Output directory")
    parser.add_argument("--config_dir", type=str, default="./config", help="Configuration directory")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA quantization for language models")

    # Multi-dataset specific arguments
    parser.add_argument("--max_memory_gb", type=int, default=32,
                       help="Maximum memory per model in GB")
    parser.add_argument("--max_models_per_dataset", type=int, default=8,
                       help="Maximum models to test per dataset")
    parser.add_argument("--dataset_types", nargs="+", default=["vision", "language"],
                       choices=["vision", "language", "multimodal"],
                       help="Dataset types to include")
    parser.add_argument("--priority_threshold", type=float, default=0.0,
                       help="Minimum priority threshold for experiments")

    args = parser.parse_args()

    # Initialize extractor
    extractor = ExhaustiveMultiDatasetExtractor(args.output_dir, args.config_dir)

    # Generate experiment plan
    dataset_types = [DatasetType(dt) for dt in args.dataset_types]
    experiment_plan = extractor.config.generate_exhaustive_experiment_plan(
        dataset_types=dataset_types,
        max_memory_gb=args.max_memory_gb,
        max_models_per_dataset=args.max_models_per_dataset,
        include_multimodal="multimodal" in args.dataset_types
    )

    # Filter by priority threshold
    if args.priority_threshold > 0:
        original_count = len(experiment_plan["experiments"])
        experiment_plan["experiments"] = [
            exp for exp in experiment_plan["experiments"]
            if exp["experiment_metadata"]["priority"] >= args.priority_threshold
        ]
        print(f"Filtered experiments: {original_count} -> {len(experiment_plan['experiments'])} "
              f"(priority >= {args.priority_threshold})")

    # Save experiment plan
    plan_path = Path(args.output_dir) / "experiment_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, 'w') as f:
        json.dump(experiment_plan, f, indent=2, default=str)
    print(f"Experiment plan saved to: {plan_path}")

    # Run exhaustive extraction
    extractor.run_exhaustive_extraction(experiment_plan, args)


if __name__ == "__main__":
    main()