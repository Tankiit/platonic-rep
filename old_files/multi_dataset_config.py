"""
Multi-Dataset Configuration System
Extends the existing analysis to handle exhaustive features across multiple datasets for vision and language models.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class DatasetType(Enum):
    VISION = "vision"
    LANGUAGE = "language"
    MULTIMODAL = "multimodal"


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str
    subset: str
    dataset_type: DatasetType
    num_samples: int = 1024
    caption_idx: int = 0
    description: str = ""
    supported_tasks: List[str] = None
    preprocessing_args: Dict[str, Any] = None

    def __post_init__(self):
        if self.supported_tasks is None:
            self.supported_tasks = []
        if self.preprocessing_args is None:
            self.preprocessing_args = {}


@dataclass
class ModelConfig:
    """Configuration for model sets"""
    model_name: str
    model_type: str  # "llm" or "lvm"
    estimated_size_mb: int
    supported_tasks: List[str]
    optimal_batch_size: int = 4
    memory_requirements_gb: int = 8
    requires_special_handling: bool = False
    notes: str = ""


class MultiDatasetConfig:
    """
    Manages configurations for multiple datasets and models
    Supports exhaustive analysis across diverse datasets
    """

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize predefined datasets
        self.vision_datasets = self._initialize_vision_datasets()
        self.language_datasets = self._initialize_language_datasets()
        self.multimodal_datasets = self._initialize_multimodal_datasets()

        # Model configurations
        self.model_configs = self._initialize_model_configs()

    def _initialize_vision_datasets(self) -> Dict[str, DatasetConfig]:
        """Initialize comprehensive vision datasets"""
        return {
            "cifar10": DatasetConfig(
                name="cifar10",
                subset="",
                dataset_type=DatasetType.VISION,
                num_samples=10000,
                description="CIFAR-10: 10-class natural image classification dataset",
                supported_tasks=["classification", "representation_learning", "transfer_learning"]
            ),

            "cifar100": DatasetConfig(
                name="cifar100",
                subset="",
                dataset_type=DatasetType.VISION,
                num_samples=10000,
                description="CIFAR-100: 100-class fine-grained image classification",
                supported_tasks=["fine_grained_classification", "representation_learning"]
            ),

            "imagenet1k": DatasetConfig(
                name="imagenet-1k",
                subset="validation",
                dataset_type=DatasetType.VISION,
                num_samples=50000,
                description="ImageNet-1K validation: 1000-class large-scale image classification",
                supported_tasks=["large_scale_classification", "transfer_learning", "representation_learning"]
            ),

            "food101": DatasetConfig(
                name="food101",
                subset="",
                dataset_type=DatasetType.VISION,
                num_samples=25250,
                description="Food-101: Food image classification with 101 categories",
                supported_tasks=["domain_specific_classification", "transfer_learning"]
            ),

            "oxford_flowers102": DatasetConfig(
                name="nelorth/oxford-flowers",
                subset="",
                dataset_type=DatasetType.VISION,
                num_samples=8189,
                description="Oxford Flowers-102: Fine-grained flower classification",
                supported_tasks=["fine_grained_classification", "few_shot_learning"]
            ),

            "oxford_pets": DatasetConfig(
                name="oxford-iiit-pet",
                subset="test",
                dataset_type=DatasetType.VISION,
                num_samples=3669,
                description="Oxford-IIIT Pet Dataset: 37 pet breed classification",
                supported_tasks=["fine_grained_classification", "few_shot_learning"]
            ),

            "stl10": DatasetConfig(
                name="stl10",
                subset="test",
                dataset_type=DatasetType.VISION,
                num_samples=8000,
                description="STL-10: 10-class image classification with unlabeled data",
                supported_tasks=["semi_supervised_learning", "representation_learning"]
            ),

            "caltech101": DatasetConfig(
                name="caltech101",
                subset="",
                dataset_type=DatasetType.VISION,
                num_samples=3060,
                description="Caltech-101: 101 object categories for object recognition",
                supported_tasks=["object_recognition", "few_shot_learning"]
            ),

            "svhn": DatasetConfig(
                name="svhn",
                subset="test",
                dataset_type=DatasetType.VISION,
                num_samples=26032,
                description="SVHN: Street View House Numbers digit recognition",
                supported_tasks=["digit_recognition", "domain_adaptation"]
            ),

            "eurosat": DatasetConfig(
                name="timm/eurosat-rgb",
                subset="",
                dataset_type=DatasetType.VISION,
                num_samples=5400,
                description="EuroSAT: Satellite image classification of land use/land cover",
                supported_tasks=["satellite_imagery", "domain_specific_classification"]
            )
        }

    def _initialize_language_datasets(self) -> Dict[str, DatasetConfig]:
        """Initialize comprehensive language datasets"""
        return {
            "openwebtext": DatasetConfig(
                name="openwebtext",
                subset="",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=8013769,
                description="OpenWebText: Large-scale web text corpus for language modeling",
                supported_tasks=["language_modeling", "representation_learning", "text_generation"]
            ),

            "wikitext103": DatasetConfig(
                name="wikitext",
                subset="wikitext-103-v1",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=1801350,
                description="WikiText-103: Long-range dependency modeling from Wikipedia articles",
                supported_tasks=["language_modeling", "long_range_dependencies"]
            ),

            "bookcorpus": DatasetConfig(
                name="bookcorpus",
                subset="",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=74004228,
                description="BookCorpus: Collection of over 11,000 books for language modeling",
                supported_tasks=["language_modeling", "long_form_text", "narrative_understanding"]
            ),

            "c4": DatasetConfig(
                name="c4",
                subset="en",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=364868892,
                description="C4: Cleaned Common Crawl corpus for T5 pretraining",
                supported_tasks=["language_modeling", "large_scale_pretraining"]
            ),

            "pile": DatasetConfig(
                name="EleutherAI/pile",
                subset="",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=210607728,
                description="The Pile: Diverse text dataset for language modeling",
                supported_tasks=["language_modeling", "diverse_domains", "knowledge_representation"]
            ),

            "imdb": DatasetConfig(
                name="imdb",
                subset="",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=50000,
                description="IMDB Movie Reviews: Binary sentiment classification",
                supported_tasks=["sentiment_analysis", "text_classification"]
            ),

            "glue_sst2": DatasetConfig(
                name="glue",
                subset="sst2",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=67349,
                description="Stanford Sentiment Treebank: Fine-grained sentiment analysis",
                supported_tasks=["sentiment_analysis", "fine_grained_classification"]
            ),

            "amazon_reviews": DatasetConfig(
                name="amazon_reviews_multi",
                subset="en",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=200000,
                description="Amazon Reviews: Multilingual product review sentiment",
                supported_tasks=["sentiment_analysis", "multilingual_processing"]
            ),

            "ag_news": DatasetConfig(
                name="ag_news",
                subset="",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=120000,
                description="AG News: News article topic classification",
                supported_tasks=["text_classification", "news_categorization"]
            ),

            "squad": DatasetConfig(
                name="squad",
                subset="",
                dataset_type=DatasetType.LANGUAGE,
                num_samples=87599,
                description="SQuAD: Stanford Question Answering Dataset",
                supported_tasks=["question_answering", "reading_comprehension"]
            )
        }

    def _initialize_multimodal_datasets(self) -> Dict[str, DatasetConfig]:
        """Initialize multimodal datasets"""
        return {
            "prh_wit": DatasetConfig(
                name="minhuh/prh",
                subset="wit_1024",
                dataset_type=DatasetType.MULTIMODAL,
                num_samples=1024,
                description="PRH WIT-1024: Platonic Representation Hypothesis evaluation dataset",
                supported_tasks=["cross_modal_alignment", "representation_analysis"]
            ),

            "conceptual_captions": DatasetConfig(
                name="conceptual_captions",
                subset="",
                dataset_type=DatasetType.MULTIMODAL,
                num_samples=3318333,
                description="Conceptual Captions: Image-caption pairs for vision-language learning",
                supported_tasks=["image_captioning", "cross_modal_learning"]
            ),

            "coco_captions": DatasetConfig(
                name="coco",
                subset="2017",
                dataset_type=DatasetType.MULTIMODAL,
                num_samples=118287,
                description="COCO Captions: Object detection dataset with rich captions",
                supported_tasks=["image_captioning", "object_detection", "cross_modal_alignment"]
            ),

            "flickr30k": DatasetConfig(
                name="flickr30k",
                subset="",
                dataset_type=DatasetType.MULTIMODAL,
                num_samples=31783,
                description="Flickr30K: Image-sentence retrieval benchmark",
                supported_tasks=["image_text_retrieval", "cross_modal_alignment"]
            ),

            "vqa_v2": DatasetConfig(
                name="HuggingFaceM4/VQAv2",
                subset="",
                dataset_type=DatasetType.MULTIMODAL,
                num_samples=214354,
                description="VQA v2: Visual Question Answering dataset",
                supported_tasks=["visual_question_answering", "multimodal_reasoning"]
            ),

            "winoground": DatasetConfig(
                name="facebook/winoground",
                subset="",
                dataset_type=DatasetType.MULTIMODAL,
                num_samples=400,
                description="Winoground: Compositional reasoning benchmark",
                supported_tasks=["compositional_reasoning", "fine_grained_understanding"]
            )
        }

    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations with memory and performance estimates"""
        configs = {}

        # Language models from tasks.py
        llm_models = [
            # Small models (< 1B parameters)
            ("bigscience/bloomz-560m", 560, 4, 4),
            ("EleutherAI/pythia-70m", 70, 2, 2),
            ("EleutherAI/pythia-160m", 160, 2, 2),
            ("EleutherAI/pythia-410m", 410, 4, 3),

            # Medium models (1B-7B parameters)
            ("bigscience/bloomz-1b1", 1100, 8, 6),
            ("bigscience/bloomz-3b", 3000, 16, 12),
            ("openlm-research/open_llama_3b", 3000, 16, 12),
            ("EleutherAI/pythia-1.4b", 1400, 8, 6),
            ("google/gemma-2b", 2000, 12, 8),

            # Large models (7B+ parameters)
            ("bigscience/bloomz-7b1", 7100, 32, 24),
            ("openlm-research/open_llama_7b", 7000, 32, 24),
            ("mistralai/Mistral-7B-v0.1", 7000, 32, 24),
            ("NousResearch/Meta-Llama-3-8B", 8000, 40, 32),

            # Very large models (13B+ parameters)
            ("huggyllama/llama-13b", 13000, 64, 48),
            ("openlm-research/open_llama_13b", 13000, 64, 48),
            ("huggyllama/llama-30b", 30000, 128, 96),
            ("NousResearch/Meta-Llama-3-70B", 70000, 256, 192),
        ]

        for model_name, size_mb, memory_gb, batch_size in llm_models:
            configs[model_name] = ModelConfig(
                model_name=model_name,
                model_type="llm",
                estimated_size_mb=size_mb,
                memory_requirements_gb=memory_gb,
                optimal_batch_size=batch_size,
                supported_tasks=["language_modeling", "text_generation", "representation_learning"]
            )

        # Vision models from tasks.py
        lvm_models = [
            # Tiny models
            ("vit_tiny_patch16_224.augreg_in21k", 5, 4, 32),
            ("deit_tiny_patch16_224.fb_in1k", 5, 4, 32),

            # Small models
            ("vit_small_patch16_224.augreg_in21k", 22, 6, 16),
            ("vit_small_patch14_dinov2.lvd142m", 22, 6, 16),
            ("deit_small_patch16_224.fb_in1k", 22, 6, 16),

            # Base models
            ("vit_base_patch16_224.augreg_in21k", 86, 8, 8),
            ("vit_base_patch14_dinov2.lvd142m", 86, 8, 8),
            ("vit_base_patch16_224.mae", 86, 8, 8),
            ("vit_base_patch16_clip_224.laion2b", 86, 8, 8),

            # Large models
            ("vit_large_patch16_224.augreg_in21k", 307, 16, 4),
            ("vit_large_patch14_dinov2.lvd142m", 307, 16, 4),
            ("vit_large_patch16_224.mae", 307, 16, 4),
            ("vit_large_patch14_clip_224.laion2b", 307, 16, 4),

            # Huge/Giant models
            ("vit_huge_patch14_224.mae", 632, 32, 2),
            ("vit_giant_patch14_dinov2.lvd142m", 1137, 48, 1),
            ("vit_huge_patch14_clip_224.laion2b", 632, 32, 2),
        ]

        for model_name, size_mb, memory_gb, batch_size in lvm_models:
            configs[model_name] = ModelConfig(
                model_name=model_name,
                model_type="lvm",
                estimated_size_mb=size_mb,
                memory_requirements_gb=memory_gb,
                optimal_batch_size=batch_size,
                supported_tasks=["image_classification", "representation_learning", "transfer_learning"]
            )

        return configs

    def get_datasets_by_type(self, dataset_type: DatasetType) -> Dict[str, DatasetConfig]:
        """Get all datasets of a specific type"""
        if dataset_type == DatasetType.VISION:
            return self.vision_datasets
        elif dataset_type == DatasetType.LANGUAGE:
            return self.language_datasets
        elif dataset_type == DatasetType.MULTIMODAL:
            return self.multimodal_datasets
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def get_all_datasets(self) -> Dict[str, DatasetConfig]:
        """Get all datasets across all types"""
        all_datasets = {}
        all_datasets.update(self.vision_datasets)
        all_datasets.update(self.language_datasets)
        all_datasets.update(self.multimodal_datasets)
        return all_datasets

    def get_compatible_models(self, dataset_config: DatasetConfig,
                            max_memory_gb: Optional[int] = None,
                            preferred_batch_size: Optional[int] = None) -> List[ModelConfig]:
        """Get models compatible with a dataset, optionally filtered by memory/batch size"""
        compatible = []

        # Determine compatible model types
        if dataset_config.dataset_type == DatasetType.VISION:
            target_model_type = "lvm"
        elif dataset_config.dataset_type == DatasetType.LANGUAGE:
            target_model_type = "llm"
        else:  # MULTIMODAL
            # Return both types for multimodal
            target_model_type = None

        for model_name, model_config in self.model_configs.items():
            # Type compatibility
            if target_model_type is not None and model_config.model_type != target_model_type:
                continue

            # Memory filtering
            if max_memory_gb is not None and model_config.memory_requirements_gb > max_memory_gb:
                continue

            # Batch size filtering
            if preferred_batch_size is not None and model_config.optimal_batch_size > preferred_batch_size:
                continue

            compatible.append(model_config)

        # Sort by memory requirements (smallest first)
        compatible.sort(key=lambda x: x.memory_requirements_gb)
        return compatible

    def generate_exhaustive_experiment_plan(self,
                                          dataset_types: List[DatasetType] = None,
                                          max_memory_gb: int = 32,
                                          max_models_per_dataset: int = 10,
                                          include_multimodal: bool = True) -> Dict[str, Any]:
        """Generate a comprehensive experiment plan for exhaustive analysis"""

        if dataset_types is None:
            dataset_types = [DatasetType.VISION, DatasetType.LANGUAGE]
            if include_multimodal:
                dataset_types.append(DatasetType.MULTIMODAL)

        experiment_plan = {
            "experiment_metadata": {
                "total_datasets": 0,
                "total_model_dataset_combinations": 0,
                "estimated_total_memory_gb": 0,
                "estimated_total_compute_hours": 0
            },
            "experiments": []
        }

        total_combinations = 0
        total_memory = 0

        for dataset_type in dataset_types:
            datasets = self.get_datasets_by_type(dataset_type)
            experiment_plan["experiment_metadata"]["total_datasets"] += len(datasets)

            for dataset_name, dataset_config in datasets.items():
                compatible_models = self.get_compatible_models(
                    dataset_config,
                    max_memory_gb=max_memory_gb
                )[:max_models_per_dataset]

                if not compatible_models:
                    continue

                experiment = {
                    "dataset_name": dataset_name,
                    "dataset_config": asdict(dataset_config),
                    "models": [asdict(model) for model in compatible_models],
                    "experiment_metadata": {
                        "num_models": len(compatible_models),
                        "estimated_memory_gb": max(m.memory_requirements_gb for m in compatible_models),
                        "estimated_runtime_hours": len(compatible_models) * 0.5,  # Rough estimate
                        "priority": self._calculate_experiment_priority(dataset_config, compatible_models)
                    }
                }

                experiment_plan["experiments"].append(experiment)
                total_combinations += len(compatible_models)
                total_memory += experiment["experiment_metadata"]["estimated_memory_gb"]

        experiment_plan["experiment_metadata"]["total_model_dataset_combinations"] = total_combinations
        experiment_plan["experiment_metadata"]["estimated_total_memory_gb"] = total_memory
        experiment_plan["experiment_metadata"]["estimated_total_compute_hours"] = sum(
            exp["experiment_metadata"]["estimated_runtime_hours"] for exp in experiment_plan["experiments"]
        )

        # Sort experiments by priority
        experiment_plan["experiments"].sort(key=lambda x: x["experiment_metadata"]["priority"], reverse=True)

        return experiment_plan

    def _calculate_experiment_priority(self, dataset_config: DatasetConfig, models: List[ModelConfig]) -> float:
        """Calculate priority score for an experiment (higher = more important)"""
        priority = 1.0

        # Dataset size factor (larger datasets get higher priority)
        if dataset_config.num_samples > 100000:
            priority += 2.0
        elif dataset_config.num_samples > 10000:
            priority += 1.0
        elif dataset_config.num_samples > 1000:
            priority += 0.5

        # Model diversity factor
        unique_sizes = len(set(m.estimated_size_mb for m in models))
        priority += unique_sizes * 0.2

        # Dataset type factor
        if dataset_config.dataset_type == DatasetType.MULTIMODAL:
            priority += 1.5  # Multimodal is more interesting
        elif dataset_config.dataset_type == DatasetType.VISION:
            priority += 1.0
        else:
            priority += 0.8

        # Well-known benchmark datasets get higher priority
        benchmark_keywords = ["cifar", "imagenet", "squad", "imdb", "coco"]
        if any(keyword in dataset_config.name.lower() for keyword in benchmark_keywords):
            priority += 1.0

        return priority

    def save_config(self, filename: str = "multi_dataset_config.json"):
        """Save configuration to file"""
        config_path = self.config_dir / filename

        config_data = {
            "vision_datasets": {k: asdict(v) for k, v in self.vision_datasets.items()},
            "language_datasets": {k: asdict(v) for k, v in self.language_datasets.items()},
            "multimodal_datasets": {k: asdict(v) for k, v in self.multimodal_datasets.items()},
            "model_configs": {k: asdict(v) for k, v in self.model_configs.items()}
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

        print(f"Configuration saved to: {config_path}")
        return config_path

    def load_config(self, filename: str = "multi_dataset_config.json"):
        """Load configuration from file"""
        config_path = self.config_dir / filename

        if not config_path.exists():
            print(f"Config file {config_path} not found. Using defaults.")
            return

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Reconstruct dataclasses from dict
        self.vision_datasets = {
            k: DatasetConfig(**{**v, 'dataset_type': DatasetType(v['dataset_type'])})
            for k, v in config_data.get('vision_datasets', {}).items()
        }

        self.language_datasets = {
            k: DatasetConfig(**{**v, 'dataset_type': DatasetType(v['dataset_type'])})
            for k, v in config_data.get('language_datasets', {}).items()
        }

        self.multimodal_datasets = {
            k: DatasetConfig(**{**v, 'dataset_type': DatasetType(v['dataset_type'])})
            for k, v in config_data.get('multimodal_datasets', {}).items()
        }

        self.model_configs = {
            k: ModelConfig(**v) for k, v in config_data.get('model_configs', {}).items()
        }

        print(f"Configuration loaded from: {config_path}")


def main():
    """Demonstration of multi-dataset configuration system"""
    print("=== Multi-Dataset Configuration System ===")

    # Initialize configuration
    config = MultiDatasetConfig()

    # Show available datasets
    print(f"\nAvailable datasets:")
    print(f"  Vision: {len(config.vision_datasets)} datasets")
    print(f"  Language: {len(config.language_datasets)} datasets")
    print(f"  Multimodal: {len(config.multimodal_datasets)} datasets")
    print(f"  Total models: {len(config.model_configs)} models")

    # Generate exhaustive experiment plan
    print(f"\n=== Generating Exhaustive Experiment Plan ===")
    experiment_plan = config.generate_exhaustive_experiment_plan(
        max_memory_gb=32,
        max_models_per_dataset=5
    )

    metadata = experiment_plan["experiment_metadata"]
    print(f"Total datasets: {metadata['total_datasets']}")
    print(f"Total combinations: {metadata['total_model_dataset_combinations']}")
    print(f"Estimated memory: {metadata['estimated_total_memory_gb']:.1f} GB")
    print(f"Estimated compute: {metadata['estimated_total_compute_hours']:.1f} hours")

    # Show top priority experiments
    print(f"\n=== Top Priority Experiments ===")
    for i, exp in enumerate(experiment_plan["experiments"][:5]):
        print(f"{i+1}. {exp['dataset_name']} ({exp['dataset_config']['dataset_type']}) "
              f"- {exp['experiment_metadata']['num_models']} models "
              f"(Priority: {exp['experiment_metadata']['priority']:.1f})")

    # Save configuration
    config_path = config.save_config()

    return config, experiment_plan


if __name__ == "__main__":
    config, experiment_plan = main()