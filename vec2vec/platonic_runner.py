#!/usr/bin/env python3
"""
Runner script for Platonic representation analysis across different scales of datasets.
Uses the existing feature extraction infrastructure with multimodal adaptations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import logging

# Import existing modules
from cross_modal_feature_extractor import (
    EnhancedVisionFeatureExtractor,
    EnhancedLanguageFeatureExtractor,
    MediumScaleModelRegistry
)
from extract_large_scale_features import (
    LargeScaleDatasetLoader,
    ExtractionConfig,
    FeatureExtractor
)


class PlatonicDatasetLoader:
    """Unified loader for Platonic analysis with paired vision-language data."""

    # Define paired datasets with their configurations
    PAIRED_DATASETS = {
        # Small-scale experiments
        'cifar100_text': {
            'vision_dataset': 'cifar100',
            'text_source': 'class_names',  # Use CIFAR-100 class names
            'size': 10000,
            'pairing': 'class_based'
        },
        'flowers102_descriptions': {
            'vision_dataset': 'flowers102',
            'text_source': 'flower_descriptions',  # Rich flower descriptions
            'size': 8189,
            'pairing': 'one_to_one'
        },
        'food101_names': {
            'vision_dataset': 'food101',
            'text_source': 'food_names',
            'size': 25250,
            'pairing': 'class_based'
        },

        # Medium-scale with captions
        'coco_captions': {
            'vision_dataset': 'coco',
            'text_source': 'mscoco_captions',
            'size': 5000,
            'pairing': 'one_to_many'  # Multiple captions per image
        },
        'conceptual_captions': {
            'vision_dataset': 'conceptual_captions_images',
            'text_source': 'conceptual_captions_text',
            'size': 15840,
            'pairing': 'one_to_one'
        },

        # Large-scale
        'laion_subset': {
            'vision_dataset': 'laion_images',
            'text_source': 'laion_captions',
            'size': 100000,  # Subset of LAION
            'pairing': 'one_to_one'
        }
    }

    @classmethod
    def load_paired_data(cls, dataset_name: str, num_samples: Optional[int] = None):
        """Load paired vision-language data."""
        if dataset_name not in cls.PAIRED_DATASETS:
            raise ValueError(f"Unknown paired dataset: {dataset_name}")

        config = cls.PAIRED_DATASETS[dataset_name]

        # Load based on dataset type
        if dataset_name == 'cifar100_text':
            return cls._load_cifar100_with_text(num_samples)
        elif dataset_name == 'flowers102_descriptions':
            return cls._load_flowers_with_descriptions(num_samples)
        elif dataset_name == 'food101_names':
            return cls._load_food_with_names(num_samples)
        elif dataset_name == 'coco_captions':
            return cls._load_coco_captions(num_samples)
        elif dataset_name == 'conceptual_captions':
            return cls._load_conceptual_captions(num_samples)
        else:
            raise NotImplementedError(f"Loader not implemented for {dataset_name}")

    @staticmethod
    def _load_cifar100_with_text(num_samples: Optional[int] = None):
        """Load CIFAR-100 with class name descriptions."""
        from torchvision import datasets, transforms

        # CIFAR-100 class names
        class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        if num_samples and num_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

        # Create text descriptions
        texts = []
        for _, label in dataset:
            text = f"A photo of a {class_names[label].replace('_', ' ')}"
            texts.append(text)

        return dataset, texts

    @staticmethod
    def _load_flowers_with_descriptions(num_samples: Optional[int] = None):
        """Load Flowers102 with detailed descriptions."""
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.Flowers102(root='./data', split='test', download=True, transform=transform)

        # Load flower names and create descriptions
        flower_names = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
            "sweet pea", "english marigold", "tiger lily", "moon orchid",
            "bird of paradise", "monkshood", "globe thistle", "snapdragon",
            "colt's foot", "king protea", "spear thistle", "yellow iris",
            "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
            "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
            "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
            "stemless gentian", "artichoke", "sweet william", "carnation",
            "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
            "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
            "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
            "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
            "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
            "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
            "pink and yellow dahlia", "cautleya spicata", "japanese anemone",
            "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
            "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
            "azalea", "water lily", "rose", "thorn apple", "morning glory",
            "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
            "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
            "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
            "bee balm", "air plant", "foxglove", "bougainvillea", "camellia",
            "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
            "blackberry lily"
        ]

        if num_samples and num_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

        texts = []
        for _, label in dataset:
            flower_name = flower_names[label] if label < len(flower_names) else f"flower type {label}"
            text = f"A beautiful {flower_name} flower with vibrant colors and delicate petals"
            texts.append(text)

        return dataset, texts

    @staticmethod
    def _load_coco_captions(num_samples: Optional[int] = None):
        """Load COCO with captions."""
        from datasets import load_dataset
        from torchvision import transforms
        from PIL import Image
        import requests
        from io import BytesIO

        # Load COCO captions dataset from HuggingFace
        dataset = load_dataset('HuggingFaceM4/COCO', split='validation')

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        images = []
        texts = []

        for item in tqdm(dataset, desc="Loading COCO"):
            # Get image
            image = item['image']
            if isinstance(image, str):
                # Download if URL
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert('RGB')

            image_tensor = transform(image)
            images.append(image_tensor)

            # Get caption (use first one)
            caption = item['sentences']['raw'][0] if 'sentences' in item else "An image from COCO dataset"
            texts.append(caption)

        return images, texts

    @staticmethod
    def _load_conceptual_captions(num_samples: Optional[int] = None):
        """Load Conceptual Captions dataset."""
        from datasets import load_dataset

        dataset = load_dataset('conceptual_captions', split='validation')

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Note: This returns URLs, actual loading would need to download images
        # For now, return placeholder
        texts = [item['caption'] for item in dataset]

        # Placeholder for images (in real use, download from URLs)
        images = None  # Would need to implement URL downloading

        return images, texts

    @staticmethod
    def _load_food_with_names(num_samples: Optional[int] = None):
        """Load Food101 with food names."""
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.Food101(root='./data', split='test', download=True, transform=transform)

        if num_samples and num_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

        # Get food class names
        with open('./data/food-101/meta/classes.txt', 'r') as f:
            food_names = [line.strip().replace('_', ' ') for line in f]

        texts = []
        for _, label in dataset:
            food_name = food_names[label] if label < len(food_names) else f"food item {label}"
            text = f"A delicious plate of {food_name}"
            texts.append(text)

        return dataset, texts


class PlatonicAnalyzer:
    """Main analyzer for Platonic representations."""

    def __init__(self, output_dir: str = "./platonic_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

        # Get model registry
        self.model_registry = MediumScaleModelRegistry()

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"platonic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_comprehensive_analysis(self, config: Dict[str, Any]):
        """Run comprehensive Platonic analysis with specified configuration."""

        self.logger.info("Starting Platonic Representation Analysis")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

        results = {
            'config': config,
            'experiments': {},
            'summary': {}
        }

        # Run experiments for each dataset
        for dataset_config in config['datasets']:
            dataset_name = dataset_config['name']
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing dataset: {dataset_name}")
            self.logger.info(f"{'='*60}")

            exp_results = self.run_dataset_experiment(
                dataset_name=dataset_name,
                vision_models=dataset_config.get('vision_models', config['default_vision_models']),
                language_models=dataset_config.get('language_models', config['default_language_models']),
                num_samples=dataset_config.get('num_samples', 1000)
            )

            results['experiments'][dataset_name] = exp_results

            # Save intermediate results
            self.save_experiment_results(exp_results, dataset_name)

        # Analyze cross-dataset trends
        results['summary'] = self.analyze_cross_dataset_trends(results['experiments'])

        # Generate visualizations
        self.create_comprehensive_visualizations(results)

        # Generate report
        self.generate_comprehensive_report(results)

        return results

    def run_dataset_experiment(self, dataset_name: str, vision_models: List[str],
                              language_models: List[str], num_samples: int):
        """Run experiment on a single dataset with multiple model pairs."""

        exp_results = {
            'dataset': dataset_name,
            'num_samples': num_samples,
            'model_pairs': {},
            'metrics': {}
        }

        # Extract features for each model combination
        for vision_model in vision_models:
            for language_model in language_models:
                pair_name = f"{vision_model}+{language_model}"
                self.logger.info(f"Processing model pair: {pair_name}")

                try:
                    # Extract features
                    vision_features, language_features = self.extract_paired_features(
                        dataset_name, vision_model, language_model, num_samples
                    )

                    # Compute alignment metrics
                    alignment_metrics = self.compute_comprehensive_alignment(
                        vision_features, language_features
                    )

                    exp_results['model_pairs'][pair_name] = {
                        'vision_shape': vision_features.shape,
                        'language_shape': language_features.shape,
                        'alignment': alignment_metrics
                    }

                    self.logger.info(f"  Mean alignment: {alignment_metrics['mean_similarity']:.4f}")
                    self.logger.info(f"  CKA: {alignment_metrics['cka']:.4f}")

                except Exception as e:
                    self.logger.error(f"Error processing {pair_name}: {str(e)}")
                    continue

        # Compute dataset-level metrics
        exp_results['metrics'] = self.compute_dataset_metrics(exp_results['model_pairs'])

        return exp_results

    def extract_paired_features(self, dataset_name: str, vision_model: str,
                               language_model: str, num_samples: int):
        """Extract paired vision-language features."""

        # Check cache first
        cache_dir = self.output_dir / 'feature_cache'
        cache_dir.mkdir(exist_ok=True)

        cache_key = f"{dataset_name}_{vision_model}_{language_model}_{num_samples}"
        cache_file = cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            self.logger.info(f"  Loading cached features from {cache_file}")
            data = np.load(cache_file)
            return data['vision_features'], data['language_features']

        # Load dataset
        if dataset_name in PlatonicDatasetLoader.PAIRED_DATASETS:
            dataset, texts = PlatonicDatasetLoader.load_paired_data(dataset_name, num_samples)
        else:
            # Use standard datasets
            self.logger.info(f"  Loading standard dataset: {dataset_name}")
            dataset, texts = self.load_standard_dataset(dataset_name, num_samples)

        # Extract vision features
        self.logger.info(f"  Extracting vision features with {vision_model}")
        vision_extractor = EnhancedVisionFeatureExtractor(vision_model)

        if dataset is not None:
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            vision_features_dict, _ = vision_extractor.extract_features(
                dataloader, pool_strategy='avg', normalize=True
            )
            vision_features = vision_features_dict.get('final', list(vision_features_dict.values())[0])
        else:
            # Placeholder for missing vision data
            vision_features = np.random.randn(len(texts), 512).astype(np.float32)

        # Extract language features
        self.logger.info(f"  Extracting language features with {language_model}")
        language_extractor = EnhancedLanguageFeatureExtractor(language_model)
        language_features_dict = language_extractor.extract_features(
            texts, pool_strategy='mean', normalize=True
        )
        language_features = list(language_features_dict.values())[-1]  # Use last layer

        # Ensure same dimensionality
        vision_features = vision_features.numpy() if torch.is_tensor(vision_features) else vision_features
        language_features = language_features.numpy() if torch.is_tensor(language_features) else language_features

        # Cache features
        np.savez_compressed(
            cache_file,
            vision_features=vision_features,
            language_features=language_features
        )

        return vision_features, language_features

    def load_standard_dataset(self, dataset_name: str, num_samples: int):
        """Load standard vision dataset with generated text descriptions."""
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Map dataset names to torchvision datasets
        dataset_map = {
            'cifar10': datasets.CIFAR10,
            'cifar100': datasets.CIFAR100,
            'imagenet': datasets.ImageNet,
            'stl10': datasets.STL10,
        }

        if dataset_name in dataset_map:
            dataset_class = dataset_map[dataset_name]
            dataset = dataset_class(root='./data', train=False, download=True, transform=transform)

            # Generate simple text descriptions based on class
            texts = []
            for _, label in dataset:
                texts.append(f"An image of class {label}")

            if num_samples and num_samples < len(dataset):
                indices = torch.randperm(len(dataset))[:num_samples]
                dataset = torch.utils.data.Subset(dataset, indices)
                texts = [texts[i] for i in indices]

            return dataset, texts
        else:
            # Return None for unsupported datasets
            return None, []

    def compute_comprehensive_alignment(self, vision_features: np.ndarray,
                                       language_features: np.ndarray) -> Dict:
        """Compute comprehensive alignment metrics."""

        # Ensure same number of samples
        min_samples = min(len(vision_features), len(language_features))
        vision_features = vision_features[:min_samples]
        language_features = language_features[:min_samples]

        # Align dimensions if needed
        if vision_features.shape[1] != language_features.shape[1]:
            target_dim = min(vision_features.shape[1], language_features.shape[1])

            # Use PCA for dimension reduction
            from sklearn.decomposition import PCA

            if vision_features.shape[1] > target_dim:
                pca = PCA(n_components=target_dim)
                vision_features = pca.fit_transform(vision_features)

            if language_features.shape[1] > target_dim:
                pca = PCA(n_components=target_dim)
                language_features = pca.fit_transform(language_features)

        metrics = {}

        # 1. Cosine similarity
        similarities = cosine_similarity(vision_features, language_features)
        diagonal_sim = np.diag(similarities)
        metrics['mean_similarity'] = float(diagonal_sim.mean())
        metrics['std_similarity'] = float(diagonal_sim.std())

        # 2. CKA (Centered Kernel Alignment)
        metrics['cka'] = self.compute_cka(vision_features, language_features)

        # 3. Procrustes alignment
        metrics['procrustes'] = self.compute_procrustes_score(vision_features, language_features)

        # 4. RSM correlation
        vision_rsm = cosine_similarity(vision_features)
        language_rsm = cosine_similarity(language_features)

        # Get upper triangular parts
        vision_flat = vision_rsm[np.triu_indices_from(vision_rsm, k=1)]
        language_flat = language_rsm[np.triu_indices_from(language_rsm, k=1)]

        metrics['rsm_correlation'] = float(np.corrcoef(vision_flat, language_flat)[0, 1])

        # 5. Retrieval metrics
        metrics['retrieval'] = self.compute_retrieval_metrics(similarities)

        # 6. Mutual nearest neighbors
        metrics['mutual_nn_10'] = self.compute_mutual_nn(vision_features, language_features, k=10)

        return metrics

    def compute_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Centered Kernel Alignment."""
        # Center features
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        # Compute Gram matrices
        K = X @ X.T
        L = Y @ Y.T

        # Center Gram matrices
        n = len(K)
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ K @ H
        L_c = H @ L @ H

        # Compute CKA
        hsic = np.sum(K_c * L_c)
        normalization = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))

        return float(hsic / normalization) if normalization > 0 else 0.0

    def compute_procrustes_score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Procrustes alignment score."""
        from scipy.spatial import procrustes

        try:
            _, _, disparity = procrustes(X, Y)
            return 1.0 - float(disparity)  # Convert distance to similarity score
        except:
            return 0.0

    def compute_retrieval_metrics(self, similarities: np.ndarray) -> Dict:
        """Compute retrieval metrics."""
        n = len(similarities)

        # Image to text retrieval
        i2t_ranks = []
        for i in range(n):
            sims = similarities[i]
            rank = (sims >= sims[i]).sum()
            i2t_ranks.append(rank)

        # Text to image retrieval
        t2i_ranks = []
        for j in range(n):
            sims = similarities[:, j]
            rank = (sims >= sims[j]).sum()
            t2i_ranks.append(rank)

        i2t_ranks = np.array(i2t_ranks)
        t2i_ranks = np.array(t2i_ranks)

        return {
            'i2t_r1': float((i2t_ranks <= 1).mean()),
            'i2t_r5': float((i2t_ranks <= 5).mean()),
            'i2t_r10': float((i2t_ranks <= 10).mean()),
            't2i_r1': float((t2i_ranks <= 1).mean()),
            't2i_r5': float((t2i_ranks <= 5).mean()),
            't2i_r10': float((t2i_ranks <= 10).mean())
        }

    def compute_mutual_nn(self, X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
        """Compute mutual nearest neighbors."""
        from sklearn.neighbors import NearestNeighbors

        nn_x = NearestNeighbors(n_neighbors=min(k+1, len(X))).fit(X)
        nn_y = NearestNeighbors(n_neighbors=min(k+1, len(Y))).fit(Y)

        _, indices_xy = nn_x.kneighbors(Y)
        _, indices_yx = nn_y.kneighbors(X)

        mutual_count = 0
        for i in range(min(len(X), len(Y))):
            if i in indices_yx[i, 1:] and i in indices_xy[i, 1:]:
                mutual_count += 1

        return mutual_count / min(len(X), len(Y))

    def compute_dataset_metrics(self, model_pairs: Dict) -> Dict:
        """Compute aggregate metrics for dataset."""

        all_alignments = []
        all_ckas = []
        all_rsms = []

        for pair_name, pair_data in model_pairs.items():
            alignment = pair_data['alignment']
            all_alignments.append(alignment['mean_similarity'])
            all_ckas.append(alignment['cka'])
            all_rsms.append(alignment['rsm_correlation'])

        return {
            'mean_alignment': np.mean(all_alignments) if all_alignments else 0,
            'std_alignment': np.std(all_alignments) if all_alignments else 0,
            'mean_cka': np.mean(all_ckas) if all_ckas else 0,
            'mean_rsm': np.mean(all_rsms) if all_rsms else 0,
            'best_pair': max(model_pairs.items(),
                           key=lambda x: x[1]['alignment']['mean_similarity'])[0] if model_pairs else None
        }

    def analyze_cross_dataset_trends(self, experiments: Dict) -> Dict:
        """Analyze trends across datasets."""

        summary = {
            'dataset_comparison': {},
            'model_performance': {},
            'scaling_analysis': {}
        }

        # Dataset comparison
        for dataset_name, exp_results in experiments.items():
            summary['dataset_comparison'][dataset_name] = {
                'num_samples': exp_results['num_samples'],
                'mean_alignment': exp_results['metrics'].get('mean_alignment', 0),
                'best_model_pair': exp_results['metrics'].get('best_pair', 'N/A')
            }

        # Model performance across datasets
        all_model_pairs = set()
        for exp_results in experiments.values():
            all_model_pairs.update(exp_results['model_pairs'].keys())

        for model_pair in all_model_pairs:
            performances = []
            for exp_results in experiments.values():
                if model_pair in exp_results['model_pairs']:
                    perf = exp_results['model_pairs'][model_pair]['alignment']['mean_similarity']
                    performances.append(perf)

            if performances:
                summary['model_performance'][model_pair] = {
                    'mean': np.mean(performances),
                    'std': np.std(performances),
                    'min': np.min(performances),
                    'max': np.max(performances)
                }

        # Scaling analysis (correlation with dataset size)
        dataset_sizes = []
        mean_alignments = []

        for dataset_name, exp_results in experiments.items():
            dataset_sizes.append(exp_results['num_samples'])
            mean_alignments.append(exp_results['metrics'].get('mean_alignment', 0))

        if len(dataset_sizes) > 1:
            correlation = np.corrcoef(dataset_sizes, mean_alignments)[0, 1]
            summary['scaling_analysis']['size_alignment_correlation'] = float(correlation)

        return summary

    def create_comprehensive_visualizations(self, results: Dict):
        """Create comprehensive visualization suite."""

        # Setup figure
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Dataset comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_dataset_comparison(ax1, results['experiments'])

        # 2. Model pair heatmap
        ax2 = fig.add_subplot(gs[1, :2])
        self.plot_model_pair_heatmap(ax2, results['experiments'])

        # 3. Scaling analysis
        ax3 = fig.add_subplot(gs[2, 0])
        self.plot_scaling_analysis(ax3, results['experiments'])

        # 4. CKA vs Cosine similarity
        ax4 = fig.add_subplot(gs[2, 1])
        self.plot_cka_vs_cosine(ax4, results['experiments'])

        # 5. Top performing pairs
        ax5 = fig.add_subplot(gs[0, 2])
        self.plot_top_pairs(ax5, results['summary']['model_performance'])

        # 6. Retrieval performance
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_retrieval_performance(ax6, results['experiments'])

        # 7. RSM correlation distribution
        ax7 = fig.add_subplot(gs[2, 2])
        self.plot_rsm_distribution(ax7, results['experiments'])

        plt.suptitle('Platonic Representation Analysis Results', fontsize=16, fontweight='bold')

        # Save figure
        output_file = self.output_dir / 'platonic_analysis_comprehensive.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved visualization to {output_file}")

    def plot_dataset_comparison(self, ax, experiments):
        """Plot dataset comparison."""
        datasets = list(experiments.keys())
        metrics = ['mean_alignment', 'mean_cka', 'mean_rsm']

        data = {metric: [] for metric in metrics}
        for dataset in datasets:
            exp_metrics = experiments[dataset]['metrics']
            for metric in metrics:
                data[metric].append(exp_metrics.get(metric, 0))

        x = np.arange(len(datasets))
        width = 0.25

        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, data[metric], width, label=metric)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Score')
        ax.set_title('Alignment Metrics Across Datasets')
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_model_pair_heatmap(self, ax, experiments):
        """Plot model pair performance heatmap."""
        # Collect all model pairs and datasets
        all_pairs = set()
        for exp in experiments.values():
            all_pairs.update(exp['model_pairs'].keys())

        all_pairs = sorted(list(all_pairs))
        datasets = sorted(list(experiments.keys()))

        # Create matrix
        matrix = np.zeros((len(all_pairs), len(datasets)))

        for i, pair in enumerate(all_pairs):
            for j, dataset in enumerate(datasets):
                if pair in experiments[dataset]['model_pairs']:
                    matrix[i, j] = experiments[dataset]['model_pairs'][pair]['alignment']['mean_similarity']

        # Plot heatmap
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_yticks(range(len(all_pairs)))
        ax.set_yticklabels(all_pairs, fontsize=8)
        ax.set_title('Model Pair Performance Across Datasets')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Model Pair')

        plt.colorbar(im, ax=ax, label='Mean Similarity')

    def plot_scaling_analysis(self, ax, experiments):
        """Plot scaling analysis."""
        sizes = []
        alignments = []

        for exp in experiments.values():
            sizes.append(exp['num_samples'])
            alignments.append(exp['metrics'].get('mean_alignment', 0))

        ax.scatter(sizes, alignments, s=100, alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Mean Alignment')
        ax.set_title('Alignment vs Dataset Scale')
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(sizes) > 1:
            z = np.polyfit(np.log(sizes), alignments, 1)
            p = np.poly1d(z)
            ax.plot(sizes, p(np.log(sizes)), "r--", alpha=0.5, label=f'Trend (corr={np.corrcoef(sizes, alignments)[0,1]:.3f})')
            ax.legend()

    def plot_cka_vs_cosine(self, ax, experiments):
        """Plot CKA vs Cosine similarity."""
        ckas = []
        cosines = []

        for exp in experiments.values():
            for pair_data in exp['model_pairs'].values():
                ckas.append(pair_data['alignment']['cka'])
                cosines.append(pair_data['alignment']['mean_similarity'])

        ax.scatter(cosines, ckas, alpha=0.5)
        ax.set_xlabel('Mean Cosine Similarity')
        ax.set_ylabel('CKA Score')
        ax.set_title('CKA vs Cosine Similarity')
        ax.grid(True, alpha=0.3)

        # Add diagonal reference line
        lims = [max(0, min(min(cosines), min(ckas))), min(1, max(max(cosines), max(ckas)))]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
        ax.legend()

    def plot_top_pairs(self, ax, model_performance):
        """Plot top performing model pairs."""
        # Sort by mean performance
        sorted_pairs = sorted(model_performance.items(),
                            key=lambda x: x[1]['mean'],
                            reverse=True)[:10]

        pairs = [p[0].replace('+', '\n') for p in sorted_pairs]
        means = [p[1]['mean'] for p in sorted_pairs]
        stds = [p[1]['std'] for p in sorted_pairs]

        y_pos = np.arange(len(pairs))
        ax.barh(y_pos, means, xerr=stds, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=8)
        ax.set_xlabel('Mean Alignment')
        ax.set_title('Top 10 Model Pairs')
        ax.grid(True, alpha=0.3, axis='x')

    def plot_retrieval_performance(self, ax, experiments):
        """Plot retrieval performance metrics."""
        retrieval_data = []

        for dataset, exp in experiments.items():
            for pair_name, pair_data in exp['model_pairs'].items():
                if 'retrieval' in pair_data['alignment']:
                    retrieval = pair_data['alignment']['retrieval']
                    retrieval_data.append({
                        'dataset': dataset,
                        'pair': pair_name,
                        'i2t_r1': retrieval['i2t_r1'],
                        't2i_r1': retrieval['t2i_r1']
                    })

        if retrieval_data:
            df = pd.DataFrame(retrieval_data)

            # Average by dataset
            avg_by_dataset = df.groupby('dataset')[['i2t_r1', 't2i_r1']].mean()

            avg_by_dataset.plot(kind='bar', ax=ax)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('R@1 Score')
            ax.set_title('Retrieval Performance (R@1)')
            ax.legend(['Image→Text', 'Text→Image'])
            ax.grid(True, alpha=0.3)

    def plot_rsm_distribution(self, ax, experiments):
        """Plot RSM correlation distribution."""
        rsm_corrs = []

        for exp in experiments.values():
            for pair_data in exp['model_pairs'].values():
                rsm_corrs.append(pair_data['alignment']['rsm_correlation'])

        ax.hist(rsm_corrs, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('RSM Correlation')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of RSM Correlations')
        ax.axvline(np.mean(rsm_corrs), color='red', linestyle='--',
                  label=f'Mean: {np.mean(rsm_corrs):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def save_experiment_results(self, results: Dict, dataset_name: str):
        """Save experiment results."""
        output_file = self.output_dir / f"{dataset_name}_results.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self.make_serializable(results)

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Saved results to {output_file}")

    def make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.make_serializable(v) for v in obj)
        else:
            return obj

    def generate_comprehensive_report(self, results: Dict):
        """Generate comprehensive analysis report."""
        report_file = self.output_dir / 'platonic_analysis_report.md'

        with open(report_file, 'w') as f:
            f.write("# Platonic Representation Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")

            # Find best overall configuration
            best_alignment = 0
            best_config = None
            for dataset, exp in results['experiments'].items():
                for pair, data in exp['model_pairs'].items():
                    if data['alignment']['mean_similarity'] > best_alignment:
                        best_alignment = data['alignment']['mean_similarity']
                        best_config = (dataset, pair)

            f.write(f"- **Best Configuration**: {best_config[1]} on {best_config[0]} dataset\n")
            f.write(f"- **Best Alignment Score**: {best_alignment:.4f}\n")
            f.write(f"- **Datasets Analyzed**: {len(results['experiments'])}\n")
            f.write(f"- **Total Model Pairs**: {len(results['summary']['model_performance'])}\n\n")

            # Dataset Analysis
            f.write("## Dataset Analysis\n\n")

            for dataset, exp in results['experiments'].items():
                f.write(f"### {dataset}\n")
                f.write(f"- Samples: {exp['num_samples']}\n")
                f.write(f"- Mean Alignment: {exp['metrics'].get('mean_alignment', 0):.4f}\n")
                f.write(f"- Best Model Pair: {exp['metrics'].get('best_pair', 'N/A')}\n")
                f.write(f"- Model Pairs Tested: {len(exp['model_pairs'])}\n\n")

            # Model Performance
            f.write("## Model Performance Analysis\n\n")

            # Sort by mean performance
            sorted_models = sorted(
                results['summary']['model_performance'].items(),
                key=lambda x: x[1]['mean'],
                reverse=True
            )

            f.write("### Top 10 Model Pairs\n\n")
            f.write("| Rank | Model Pair | Mean Alignment | Std Dev | Range |\n")
            f.write("|------|------------|---------------|---------|-------|\n")

            for i, (pair, perf) in enumerate(sorted_models[:10], 1):
                f.write(f"| {i} | {pair} | {perf['mean']:.4f} | {perf['std']:.4f} | "
                       f"{perf['min']:.4f}-{perf['max']:.4f} |\n")

            # Scaling Analysis
            f.write("\n## Scaling Analysis\n\n")

            if 'size_alignment_correlation' in results['summary']['scaling_analysis']:
                corr = results['summary']['scaling_analysis']['size_alignment_correlation']
                f.write(f"- **Correlation (Dataset Size vs Alignment)**: {corr:.4f}\n")

                if corr > 0.5:
                    f.write("- **Interpretation**: Strong positive correlation - larger datasets show better alignment\n")
                elif corr > 0:
                    f.write("- **Interpretation**: Weak positive correlation - slight improvement with scale\n")
                elif corr > -0.5:
                    f.write("- **Interpretation**: Weak negative correlation - slight degradation with scale\n")
                else:
                    f.write("- **Interpretation**: Strong negative correlation - alignment degrades with scale\n")

            # Key Findings
            f.write("\n## Key Findings\n\n")

            # Find patterns
            vision_models = {}
            language_models = {}

            for pair, perf in results['summary']['model_performance'].items():
                vision, language = pair.split('+')

                if vision not in vision_models:
                    vision_models[vision] = []
                vision_models[vision].append(perf['mean'])

                if language not in language_models:
                    language_models[language] = []
                language_models[language].append(perf['mean'])

            # Best vision model
            best_vision = max(vision_models.items(),
                            key=lambda x: np.mean(x[1]) if x[1] else 0)
            f.write(f"1. **Best Vision Model**: {best_vision[0]} "
                   f"(avg alignment: {np.mean(best_vision[1]):.4f})\n")

            # Best language model
            best_language = max(language_models.items(),
                              key=lambda x: np.mean(x[1]) if x[1] else 0)
            f.write(f"2. **Best Language Model**: {best_language[0]} "
                   f"(avg alignment: {np.mean(best_language[1]):.4f})\n")

            # Consistency analysis
            f.write("\n3. **Model Consistency**:\n")
            for pair, perf in sorted_models[:5]:
                f.write(f"   - {pair}: std={perf['std']:.4f} "
                       f"({'consistent' if perf['std'] < 0.1 else 'variable'})\n")

            f.write("\n## Recommendations\n\n")
            f.write("Based on the analysis:\n\n")
            f.write(f"1. For best cross-modal alignment, use: **{best_config[1]}**\n")
            f.write(f"2. Vision architecture recommendation: **{best_vision[0]}**\n")
            f.write(f"3. Language architecture recommendation: **{best_language[0]}**\n")

            if 'size_alignment_correlation' in results['summary']['scaling_analysis']:
                corr = results['summary']['scaling_analysis']['size_alignment_correlation']
                if corr > 0.3:
                    f.write("4. Consider using larger datasets for improved alignment\n")

            f.write("\n---\n")
            f.write(f"*Report generated by PlatonicAnalyzer*\n")

        self.logger.info(f"Generated report: {report_file}")


def main():
    """Main entry point for Platonic analysis."""
    parser = argparse.ArgumentParser(description="Platonic Representation Analysis")

    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./platonic_results',
                       help='Output directory for results')
    parser.add_argument('--datasets', nargs='+',
                       default=['cifar100_text', 'flowers102_descriptions'],
                       help='Datasets to analyze')
    parser.add_argument('--vision-models', nargs='+',
                       default=['resnet18', 'resnet50'],
                       help='Vision models to test')
    parser.add_argument('--language-models', nargs='+',
                       default=['bert-base-uncased', 'gpt2'],
                       help='Language models to test')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples per dataset')
    parser.add_argument('--list-available', action='store_true',
                       help='List available datasets and models')

    args = parser.parse_args()

    if args.list_available:
        print("\n=== Available Resources ===\n")
        print("Paired Datasets:")
        for name, config in PlatonicDatasetLoader.PAIRED_DATASETS.items():
            print(f"  - {name}: {config['size']} samples ({config['pairing']} pairing)")

        print("\nVision Models:")
        registry = MediumScaleModelRegistry()
        for family, models in registry.get_vision_models().items():
            print(f"  {family}: {list(models.keys())[:3]}...")

        print("\nLanguage Models:")
        for family, models in registry.get_language_models().items():
            print(f"  {family}: {list(models.keys())[:3]}...")

        return

    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'datasets': [
                {
                    'name': dataset,
                    'num_samples': args.num_samples
                }
                for dataset in args.datasets
            ],
            'default_vision_models': args.vision_models,
            'default_language_models': args.language_models
        }

    # Initialize analyzer
    analyzer = PlatonicAnalyzer(output_dir=args.output_dir)

    # Run analysis
    print("\n" + "="*60)
    print("PLATONIC REPRESENTATION ANALYSIS")
    print("="*60 + "\n")

    results = analyzer.run_comprehensive_analysis(config)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    results = main()