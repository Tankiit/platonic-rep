#!/usr/bin/env python3
"""
Dataset loaders for cross-modal alignment testing on MacBook
Organized by size and memory requirements
"""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import requests
import json
from pathlib import Path
import numpy as np
from PIL import Image
import io
from tqdm import tqdm

# For dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Install datasets with: pip install datasets")
    DATASETS_AVAILABLE = False

@dataclass
class DatasetConfig:
    name: str
    dataset_id: str  # HuggingFace dataset ID
    subset: Optional[str]
    split: str
    size_category: str  # 'tiny', 'small', 'medium'
    approx_size_mb: int
    n_samples: int
    description: str

# Recommended datasets for MacBook testing
CROSSMODAL_DATASETS = {
    'tiny': [  # < 100MB, < 1K samples - great for initial testing
        DatasetConfig(
            name="MNIST-Text",
            dataset_id="brendanartley/mnist-text",
            subset=None,
            split="train[:100]",
            size_category="tiny",
            approx_size_mb=10,
            n_samples=100,
            description="Handwritten digits with text descriptions"
        ),
        DatasetConfig(
            name="Flickr8k-Sample",
            dataset_id="nlphuji/flickr8k",
            subset=None,
            split="train[:100]",
            size_category="tiny",
            approx_size_mb=50,
            n_samples=100,
            description="100 images with 5 captions each"
        ),
        DatasetConfig(
            name="CIFAR-10-Captions",
            dataset_id="keremberke/cifar10-classification",
            subset=None,
            split="train[:500]",
            size_category="tiny",
            approx_size_mb=30,
            n_samples=500,
            description="CIFAR images with generated captions"
        ),
    ],

    'small': [  # 100MB-1GB, 1K-10K samples
        DatasetConfig(
            name="MS-COCO-Captions-1K",
            dataset_id="HuggingFaceM4/COCO",
            subset="2017_captions",
            split="train[:1000]",
            size_category="small",
            approx_size_mb=200,
            n_samples=1000,
            description="High-quality images with multiple captions"
        ),
        DatasetConfig(
            name="Conceptual-Captions-3K",
            dataset_id="conceptual_captions",
            subset="unlabeled",
            split="train[:3000]",
            size_category="small",
            approx_size_mb=500,
            n_samples=3000,
            description="Web images with alt-text captions"
        ),
        DatasetConfig(
            name="Flickr30k-Entities-Sample",
            dataset_id="nlphuji/flickr30k",
            subset=None,
            split="train[:2000]",
            size_category="small",
            approx_size_mb=400,
            n_samples=2000,
            description="Images with detailed entity annotations"
        ),
        DatasetConfig(
            name="WIT-Sample",
            dataset_id="minhuh/prh",  # Your PRH dataset!
            subset="wit_1024",
            split="train[:1000]",
            size_category="small",
            approx_size_mb=300,
            n_samples=1000,
            description="Wikipedia images with contextual text"
        ),
    ],

    'medium': [  # 1GB-5GB, 10K-50K samples
        DatasetConfig(
            name="MS-COCO-2017-Val",
            dataset_id="HuggingFaceM4/COCO",
            subset="2017_captions",
            split="validation",
            size_category="medium",
            approx_size_mb=800,
            n_samples=5000,
            description="COCO validation set with 5 captions per image"
        ),
        DatasetConfig(
            name="Visual-Genome-QA",
            dataset_id="visual_genome",
            subset="question_answers_v1.2",
            split="train[:10000]",
            size_category="medium",
            approx_size_mb=2000,
            n_samples=10000,
            description="Complex scene graphs and QA pairs"
        ),
        DatasetConfig(
            name="LAION-5K",
            dataset_id="laion/laion400m",
            subset=None,
            split="train[:5000]",
            size_category="medium",
            approx_size_mb=1500,
            n_samples=5000,
            description="Large-scale web crawl image-text pairs"
        ),
    ],

    # Specialized datasets for alignment testing
    'specialized': [
        DatasetConfig(
            name="ImageNet-Captions",
            dataset_id="mrm8488/ImageNet1K-captions",
            subset=None,
            split="train[:1000]",
            size_category="small",
            approx_size_mb=400,
            n_samples=1000,
            description="ImageNet with generated captions - tests visual taxonomy alignment"
        ),
        DatasetConfig(
            name="TextCaps",
            dataset_id="HuggingFaceM4/TextCaps",
            subset=None,
            split="train[:1000]",
            size_category="small",
            approx_size_mb=300,
            n_samples=1000,
            description="Images with text/OCR - tests text understanding"
        ),
        DatasetConfig(
            name="VQAv2-Sample",
            dataset_id="HuggingFaceM4/VQAv2",
            subset=None,
            split="train[:2000]",
            size_category="small",
            approx_size_mb=400,
            n_samples=2000,
            description="Visual Question Answering - tests reasoning alignment"
        ),
    ]
}

class CrossModalDatasetLoader:
    """
    Efficient dataset loader for MacBook with memory management
    """

    def __init__(self, cache_dir: str = "./dataset_cache", max_image_size: int = 224):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_image_size = max_image_size

    def load_dataset_config(self, config: DatasetConfig,
                          limit_samples: Optional[int] = None) -> Dict:
        """Load a dataset with the given configuration"""

        print(f"\nLoading {config.name} ({config.approx_size_mb}MB, {config.n_samples} samples)")

        if not DATASETS_AVAILABLE:
            print("Using synthetic data as datasets library not available")
            return self._generate_synthetic_data(config, limit_samples)

        try:
            # Load from HuggingFace
            if config.subset:
                dataset = load_dataset(
                    config.dataset_id,
                    config.subset,
                    split=config.split,
                    cache_dir=self.cache_dir
                )
            else:
                dataset = load_dataset(
                    config.dataset_id,
                    split=config.split,
                    cache_dir=self.cache_dir
                )

            # Limit samples if requested
            if limit_samples:
                dataset = dataset.select(range(min(limit_samples, len(dataset))))

            return self._process_dataset(dataset, config)

        except Exception as e:
            print(f"Failed to load {config.name}: {e}")
            print("Falling back to synthetic data")
            return self._generate_synthetic_data(config, limit_samples)

    def _process_dataset(self, dataset, config: DatasetConfig) -> Dict:
        """Process different dataset formats into unified structure"""

        images = []
        texts = []

        # Handle different dataset formats
        if "coco" in config.dataset_id.lower():
            for item in tqdm(dataset, desc="Processing COCO"):
                if 'image' in item:
                    images.append(self._process_image(item['image']))
                    # COCO has multiple captions
                    if 'captions' in item:
                        texts.append(item['captions'][0])  # Take first caption
                    elif 'caption' in item:
                        texts.append(item['caption'])

        elif "flickr" in config.dataset_id.lower():
            for item in tqdm(dataset, desc="Processing Flickr"):
                if 'image' in item:
                    images.append(self._process_image(item['image']))
                    if 'caption' in item:
                        texts.append(item['caption'][0] if isinstance(item['caption'], list) else item['caption'])
                    elif 'text' in item:
                        texts.append(item['text'])

        elif "conceptual_captions" in config.dataset_id:
            for item in tqdm(dataset, desc="Processing CC"):
                if 'image_url' in item:
                    # Download image
                    img = self._download_image(item['image_url'])
                    if img:
                        images.append(img)
                        texts.append(item.get('caption', ''))

        elif config.dataset_id == "minhuh/prh":  # Your dataset
            for item in tqdm(dataset, desc="Processing PRH"):
                # Handle your specific format
                if 'image' in item:
                    images.append(self._process_image(item['image']))
                    texts.append(item.get('text', item.get('caption', '')))

        else:
            # Generic handler
            image_keys = ['image', 'img', 'pixel_values']
            text_keys = ['text', 'caption', 'sentence', 'question', 'answer']

            for item in tqdm(dataset, desc=f"Processing {config.name}"):
                # Find image
                for key in image_keys:
                    if key in item:
                        images.append(self._process_image(item[key]))
                        break

                # Find text
                for key in text_keys:
                    if key in item:
                        text = item[key]
                        if isinstance(text, list):
                            text = text[0]
                        texts.append(text)
                        break

        return {
            'name': config.name,
            'images': images,
            'texts': texts,
            'metadata': {
                'dataset_id': config.dataset_id,
                'n_samples': len(images),
                'size_category': config.size_category
            }
        }

    def _process_image(self, image) -> np.ndarray:
        """Process image to standard format"""
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, str):
            # Path or URL
            if image.startswith('http'):
                img = self._download_image(image)
            else:
                img = Image.open(image)
        else:
            img = image

        # Resize and convert
        if img:
            img = img.convert('RGB')
            img.thumbnail((self.max_image_size, self.max_image_size))
            return np.array(img)

        # Return placeholder if failed
        return np.zeros((self.max_image_size, self.max_image_size, 3), dtype=np.uint8)

    def _download_image(self, url: str, timeout: int = 5) -> Optional[Image.Image]:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=timeout)
            return Image.open(io.BytesIO(response.content))
        except:
            return None

    def _generate_synthetic_data(self, config: DatasetConfig,
                               limit_samples: Optional[int] = None) -> Dict:
        """Generate synthetic data for testing"""

        n_samples = limit_samples or min(config.n_samples, 100)

        # Generate synthetic images
        images = []
        texts = []

        for i in range(n_samples):
            # Create simple patterned images
            img = np.zeros((self.max_image_size, self.max_image_size, 3), dtype=np.uint8)

            # Add some patterns based on index
            pattern_type = i % 5
            if pattern_type == 0:  # Horizontal stripes
                img[::10, :] = [255, 0, 0]
                texts.append(f"An image with horizontal red stripes")
            elif pattern_type == 1:  # Vertical stripes
                img[:, ::10] = [0, 255, 0]
                texts.append(f"An image with vertical green stripes")
            elif pattern_type == 2:  # Checkerboard
                img[::20, ::20] = [0, 0, 255]
                texts.append(f"A blue checkerboard pattern")
            elif pattern_type == 3:  # Gradient
                for j in range(self.max_image_size):
                    img[j, :] = [j, j, j]
                texts.append(f"A grayscale gradient from top to bottom")
            else:  # Random noise
                img = np.random.randint(0, 255, img.shape, dtype=np.uint8)
                texts.append(f"Random colorful noise pattern")

            images.append(img)

        return {
            'name': f"{config.name}_synthetic",
            'images': images,
            'texts': texts,
            'metadata': {
                'dataset_id': 'synthetic',
                'n_samples': n_samples,
                'size_category': config.size_category
            }
        }

    def prepare_for_mlx(self, dataset: Dict) -> Tuple[List, List]:
        """Prepare dataset for MLX models"""

        # Convert images to PIL format if needed
        pil_images = []
        for img in dataset['images']:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)

        return pil_images, dataset['texts']

class DatasetBenchmark:
    """Benchmark different datasets for alignment testing"""

    def __init__(self):
        self.loader = CrossModalDatasetLoader()
        self.results = {}

    def benchmark_datasets(self, size_category: str = 'tiny'):
        """Benchmark all datasets in a size category"""

        datasets = CROSSMODAL_DATASETS.get(size_category, [])

        for config in datasets:
            print(f"\n{'='*60}")
            print(f"Benchmarking {config.name}")
            print(f"{'='*60}")

            # Load dataset
            start_time = time.time()
            dataset = self.loader.load_dataset_config(config, limit_samples=100)
            load_time = time.time() - start_time

            # Analyze dataset properties
            images, texts = self.loader.prepare_for_mlx(dataset)

            # Compute statistics
            text_lengths = [len(text.split()) for text in texts]

            stats = {
                'load_time': load_time,
                'n_samples': len(images),
                'avg_text_length': np.mean(text_lengths),
                'min_text_length': np.min(text_lengths),
                'max_text_length': np.max(text_lengths),
                'unique_texts': len(set(texts)),
                'text_diversity': len(set(texts)) / len(texts),
            }

            # Image statistics
            img_arrays = [np.array(img) for img in images[:10]]  # Sample
            stats['avg_brightness'] = np.mean([img.mean() for img in img_arrays])
            stats['avg_contrast'] = np.mean([img.std() for img in img_arrays])

            self.results[config.name] = stats

            # Print summary
            print(f"Loaded in {load_time:.1f}s")
            print(f"Samples: {stats['n_samples']}")
            print(f"Avg text length: {stats['avg_text_length']:.1f} words")
            print(f"Text diversity: {stats['text_diversity']:.2%}")

# Example usage functions
def quick_dataset_test():
    """Quick test of tiny datasets"""

    loader = CrossModalDatasetLoader()

    # Load a tiny dataset
    config = CROSSMODAL_DATASETS['tiny'][0]  # MNIST-Text
    dataset = loader.load_dataset_config(config)

    images, texts = loader.prepare_for_mlx(dataset)

    print(f"\nLoaded {len(images)} image-text pairs")
    print(f"Sample text: {texts[0]}")
    print(f"Image shape: {np.array(images[0]).shape}")

    return images, texts

def get_dataset_for_memory(available_gb: float) -> DatasetConfig:
    """Recommend dataset based on available memory"""

    if available_gb < 2:
        return CROSSMODAL_DATASETS['tiny'][0]
    elif available_gb < 4:
        return CROSSMODAL_DATASETS['tiny'][1]
    elif available_gb < 8:
        return CROSSMODAL_DATASETS['small'][0]
    elif available_gb < 16:
        return CROSSMODAL_DATASETS['small'][2]
    else:
        return CROSSMODAL_DATASETS['medium'][0]

# Integration with MLX verifier
def load_dataset_for_mlx_test(dataset_name: str = "Flickr8k-Sample",
                             n_samples: int = 100) -> Tuple[List, List]:
    """Load a specific dataset for MLX testing"""

    loader = CrossModalDatasetLoader()

    # Find dataset config
    config = None
    for category in CROSSMODAL_DATASETS.values():
        for ds in category:
            if ds.name == dataset_name:
                config = ds
                break

    if not config:
        print(f"Dataset {dataset_name} not found. Using default.")
        config = CROSSMODAL_DATASETS['tiny'][1]

    # Load and prepare
    dataset = loader.load_dataset_config(config, limit_samples=n_samples)
    return loader.prepare_for_mlx(dataset)

if __name__ == "__main__":
    import time
    import psutil

    # Check available memory
    available_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available memory: {available_gb:.1f}GB")

    # Get recommended dataset
    recommended = get_dataset_for_memory(available_gb)
    print(f"Recommended dataset: {recommended.name}")

    # Run quick test
    print("\nRunning quick dataset test...")
    images, texts = quick_dataset_test()

    # Benchmark tiny datasets
    print("\nBenchmarking tiny datasets...")
    benchmark = DatasetBenchmark()
    benchmark.benchmark_datasets('tiny')

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    for name, stats in benchmark.results.items():
        print(f"\n{name}:")
        print(f"  Load time: {stats['load_time']:.1f}s")
        print(f"  Text diversity: {stats['text_diversity']:.2%}")