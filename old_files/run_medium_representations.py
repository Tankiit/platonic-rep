#!/usr/bin/env python3
"""
Run Medium Models Across Vision, Language, and Multimodal Datasets
Saves features in organized folder structure: representations/{model_name}/{dataset_name}/
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from multi_dataset_config import MultiDatasetConfig, DatasetType
from exhaustive_multi_dataset_extractor import ExhaustiveMultiDatasetExtractor


def get_medium_models():
    """Get medium-sized models for comprehensive analysis"""

    # Medium vision models (5M - 100M parameters)
    medium_vision_models = [
        # ViT medium variants
        "vit_small_patch16_224.augreg_in21k",
        "vit_base_patch16_224.augreg_in21k",
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",

        # CLIP medium variants
        "vit_base_patch16_clip_224.laion2b",
        "vit_base_patch16_clip_224.openai",

        # DeiT variants
        "deit_small_patch16_224.fb_in1k",
        "deit_base_patch16_224.fb_in1k",

        # MAE variants
        "vit_base_patch16_224.mae",
    ]

    # Medium language models (500M - 8B parameters)
    medium_language_models = [
        # BLOOM medium variants
        "bigscience/bloomz-560m",
        "bigscience/bloomz-1b1",
        "bigscience/bloomz-3b",

        # LLaMA/OpenLLaMA medium variants
        "openlm-research/open_llama_3b",
        "openlm-research/open_llama_7b",
        "huggyllama/llama-7b",

        # Gemma medium variants
        "google/gemma-2b",
        "google/gemma-7b",

        # Mistral variants
        "mistralai/Mistral-7B-v0.1",

        # GPT variants
        "gpt2",
        "gpt2-medium",
        "gpt2-large",

        # Pythia medium variants
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
    ]

    return medium_vision_models, medium_language_models


class MediumRepresentationExtractor:
    """Extract representations from medium models across diverse datasets"""

    def __init__(self, output_dir="./representations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize configuration
        self.config = MultiDatasetConfig()

        # Get medium models
        self.medium_vision_models, self.medium_language_models = get_medium_models()

        # Statistics
        self.stats = {
            "start_time": datetime.now(),
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "datasets_processed": set(),
            "models_processed": set()
        }

    def create_model_dataset_output_dir(self, model_name, dataset_name):
        """Create output directory for specific model-dataset combination"""
        # Clean model name for filesystem
        clean_model_name = model_name.replace("/", "_").replace("-", "_")
        model_dataset_dir = self.output_dir / clean_model_name / dataset_name
        model_dataset_dir.mkdir(parents=True, exist_ok=True)
        return model_dataset_dir

    def run_vision_representations(self, max_models=None):
        """Extract representations from vision models across vision datasets"""
        print(f"\n{'='*80}")
        print("EXTRACTING VISION REPRESENTATIONS")
        print(f"{'='*80}")

        # Get vision datasets
        vision_datasets = list(self.config.get_datasets_by_type(DatasetType.VISION).values())

        # Add multimodal datasets that have images
        multimodal_datasets = list(self.config.get_datasets_by_type(DatasetType.MULTIMODAL).values())
        all_vision_datasets = vision_datasets + multimodal_datasets

        # Limit models if specified
        models_to_process = self.medium_vision_models[:max_models] if max_models else self.medium_vision_models

        print(f"Processing {len(models_to_process)} vision models across {len(all_vision_datasets)} datasets")

        for model_name in models_to_process:
            print(f"\n--- Processing Vision Model: {model_name} ---")

            for dataset_config in all_vision_datasets:
                print(f"  Dataset: {dataset_config.name}")

                # Create output directory
                output_dir = self.create_model_dataset_output_dir(model_name, dataset_config.name)

                # Check if already exists
                expected_file = output_dir / f"features_cls.pt"
                if expected_file.exists():
                    print(f"    ✓ Features already exist, skipping")
                    continue

                try:
                    # Extract features using the enhanced extractor
                    self._extract_single_model_dataset(
                        model_name, dataset_config, output_dir, "lvm", "cls"
                    )

                    self.stats["successful_extractions"] += 1
                    self.stats["models_processed"].add(model_name)
                    self.stats["datasets_processed"].add(dataset_config.name)
                    print(f"    ✓ Success")

                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    self.stats["failed_extractions"] += 1

                self.stats["total_extractions"] += 1

    def run_language_representations(self, max_models=None):
        """Extract representations from language models across language datasets"""
        print(f"\n{'='*80}")
        print("EXTRACTING LANGUAGE REPRESENTATIONS")
        print(f"{'='*80}")

        # Get language datasets
        language_datasets = list(self.config.get_datasets_by_type(DatasetType.LANGUAGE).values())

        # Add multimodal datasets that have text
        multimodal_datasets = list(self.config.get_datasets_by_type(DatasetType.MULTIMODAL).values())
        all_language_datasets = language_datasets + multimodal_datasets

        # Limit models if specified
        models_to_process = self.medium_language_models[:max_models] if max_models else self.medium_language_models

        print(f"Processing {len(models_to_process)} language models across {len(all_language_datasets)} datasets")

        for model_name in models_to_process:
            print(f"\n--- Processing Language Model: {model_name} ---")

            for dataset_config in all_language_datasets:
                print(f"  Dataset: {dataset_config.name}")

                # Create output directory
                output_dir = self.create_model_dataset_output_dir(model_name, dataset_config.name)

                # Check if already exists
                expected_file = output_dir / f"features_avg.pt"
                if expected_file.exists():
                    print(f"    ✓ Features already exist, skipping")
                    continue

                try:
                    # Extract features using the enhanced extractor
                    self._extract_single_model_dataset(
                        model_name, dataset_config, output_dir, "llm", "avg"
                    )

                    self.stats["successful_extractions"] += 1
                    self.stats["models_processed"].add(model_name)
                    self.stats["datasets_processed"].add(dataset_config.name)
                    print(f"    ✓ Success")

                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    self.stats["failed_extractions"] += 1

                self.stats["total_extractions"] += 1

    def run_multimodal_representations(self, max_models=None):
        """Extract cross-modal representations"""
        print(f"\n{'='*80}")
        print("EXTRACTING MULTIMODAL REPRESENTATIONS")
        print(f"{'='*80}")

        # Get multimodal datasets
        multimodal_datasets = list(self.config.get_datasets_by_type(DatasetType.MULTIMODAL).values())

        # Limit models
        vision_models = self.medium_vision_models[:max_models] if max_models else self.medium_vision_models[:5]
        language_models = self.medium_language_models[:max_models] if max_models else self.medium_language_models[:5]

        print(f"Processing {len(vision_models)} vision + {len(language_models)} language models")
        print(f"Across {len(multimodal_datasets)} multimodal datasets")

        # Extract vision representations on multimodal data
        for model_name in vision_models:
            print(f"\n--- Vision Model on Multimodal Data: {model_name} ---")

            for dataset_config in multimodal_datasets:
                print(f"  Dataset: {dataset_config.name}")

                output_dir = self.create_model_dataset_output_dir(f"vision_{model_name}", dataset_config.name)
                expected_file = output_dir / f"features_cls.pt"

                if expected_file.exists():
                    print(f"    ✓ Features already exist, skipping")
                    continue

                try:
                    self._extract_single_model_dataset(
                        model_name, dataset_config, output_dir, "lvm", "cls"
                    )
                    print(f"    ✓ Success")
                    self.stats["successful_extractions"] += 1
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    self.stats["failed_extractions"] += 1

                self.stats["total_extractions"] += 1

        # Extract language representations on multimodal data
        for model_name in language_models:
            print(f"\n--- Language Model on Multimodal Data: {model_name} ---")

            for dataset_config in multimodal_datasets:
                print(f"  Dataset: {dataset_config.name}")

                output_dir = self.create_model_dataset_output_dir(f"language_{model_name}", dataset_config.name)
                expected_file = output_dir / f"features_avg.pt"

                if expected_file.exists():
                    print(f"    ✓ Features already exist, skipping")
                    continue

                try:
                    self._extract_single_model_dataset(
                        model_name, dataset_config, output_dir, "llm", "avg"
                    )
                    print(f"    ✓ Success")
                    self.stats["successful_extractions"] += 1
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    self.stats["failed_extractions"] += 1

                self.stats["total_extractions"] += 1

    def _extract_single_model_dataset(self, model_name, dataset_config, output_dir, model_type, pooling_strategy):
        """Extract features for a single model-dataset combination"""

        # Create a temporary extractor for this specific extraction
        temp_extractor = ExhaustiveMultiDatasetExtractor(output_dir=str(output_dir.parent))

        # Load the dataset
        dataset = temp_extractor.load_dataset_safely(dataset_config)
        if dataset is None:
            raise ValueError(f"Could not load dataset {dataset_config.name}")

        # Mock args object with required parameters
        class MockArgs:
            def __init__(self):
                self.batch_size = 4
                self.pool = pooling_strategy
                self.caption_idx = 0
                self.force_remake = False
                self.qlora = False
                self.force_download = False

        args = MockArgs()

        # Extract features based on model type
        if model_type == "lvm":
            success = temp_extractor._extract_single_vision_model_features(
                model_name, dataset, dataset_config, output_dir / f"features_{pooling_strategy}.pt", args
            )
        else:  # llm
            success = temp_extractor._extract_single_language_model_features(
                model_name, dataset, dataset_config, output_dir / f"features_{pooling_strategy}.pt", args
            )

        if not success:
            raise ValueError("Feature extraction failed")

    def save_extraction_summary(self):
        """Save summary of extraction process"""
        self.stats["end_time"] = datetime.now()
        self.stats["total_time"] = str(self.stats["end_time"] - self.stats["start_time"])

        # Convert sets to lists for JSON serialization
        summary = {
            **self.stats,
            "datasets_processed": list(self.stats["datasets_processed"]),
            "models_processed": list(self.stats["models_processed"]),
            "start_time": str(self.stats["start_time"]),
            "end_time": str(self.stats["end_time"]),
            "success_rate": self.stats["successful_extractions"] / max(1, self.stats["total_extractions"])
        }

        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nExtraction summary saved to: {summary_path}")
        return summary

    def print_final_statistics(self):
        """Print final statistics"""
        print(f"\n{'='*80}")
        print("REPRESENTATION EXTRACTION COMPLETED")
        print(f"{'='*80}")
        print(f"Total extractions attempted: {self.stats['total_extractions']}")
        print(f"Successful extractions: {self.stats['successful_extractions']}")
        print(f"Failed extractions: {self.stats['failed_extractions']}")
        print(f"Success rate: {100 * self.stats['successful_extractions'] / max(1, self.stats['total_extractions']):.1f}%")
        print(f"Unique models processed: {len(self.stats['models_processed'])}")
        print(f"Unique datasets processed: {len(self.stats['datasets_processed'])}")
        print(f"Output directory: {self.output_dir}")

        print(f"\nDirectory structure:")
        print(f"representations/")
        for model in sorted(self.stats['models_processed'])[:5]:  # Show first 5
            clean_model = model.replace("/", "_").replace("-", "_")
            print(f"  {clean_model}/")
            for dataset in sorted(self.stats['datasets_processed'])[:3]:  # Show first 3 per model
                print(f"    {dataset}/")
                print(f"      features_*.pt")
        if len(self.stats['models_processed']) > 5:
            print(f"  ... and {len(self.stats['models_processed']) - 5} more models")


def main():
    """Main function for medium representation extraction"""
    parser = argparse.ArgumentParser(
        description="Extract Medium Model Representations Across Vision, Language, and Multimodal Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all medium representations
  python run_medium_representations.py

  # Limit number of models for testing
  python run_medium_representations.py --max_models 3

  # Only vision representations
  python run_medium_representations.py --modality vision

  # Only language representations
  python run_medium_representations.py --modality language

  # Custom output directory
  python run_medium_representations.py --output_dir ./my_representations
        """)

    parser.add_argument("--output_dir", type=str, default="./representations",
                       help="Output directory for representations")
    parser.add_argument("--modality", choices=["vision", "language", "multimodal", "all"],
                       default="all", help="Which modality to extract")
    parser.add_argument("--max_models", type=int, default=None,
                       help="Maximum number of models per modality (for testing)")

    args = parser.parse_args()

    # Initialize extractor
    extractor = MediumRepresentationExtractor(args.output_dir)

    print(f"{'='*80}")
    print("MEDIUM MODEL REPRESENTATION EXTRACTION")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"Modality: {args.modality}")
    if args.max_models:
        print(f"Max models per modality: {args.max_models}")

    # Run extractions based on modality
    if args.modality in ["vision", "all"]:
        extractor.run_vision_representations(args.max_models)

    if args.modality in ["language", "all"]:
        extractor.run_language_representations(args.max_models)

    if args.modality in ["multimodal", "all"]:
        extractor.run_multimodal_representations(args.max_models)

    # Save summary and print statistics
    extractor.save_extraction_summary()
    extractor.print_final_statistics()


if __name__ == "__main__":
    main()