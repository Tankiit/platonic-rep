"""
Comprehensive Multi-Modal Embedding Comparison System
Supports multiple medium and small models across vision and language modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Vision model imports
import timm
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b1, mobilenet_v2, vgg16

# Language model imports
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    BertModel, BertTokenizer,
    DistilBertModel, DistilBertTokenizer,
    RobertaModel, RobertaTokenizer,
    AlbertModel, AlbertTokenizer,
    GPT2Model, GPT2Tokenizer,
    T5Model, T5Tokenizer,
    BitsAndBytesConfig
)

# Multi-modal imports
import open_clip
from sentence_transformers import SentenceTransformer

# Dataset imports
from datasets import load_dataset
from PIL import Image


class MultiModalEmbeddingExtractor:
    """Extract embeddings from various vision and language models"""

    def __init__(self, device='auto', cache_dir='./model_cache'):
        if device == 'auto':
            self.device = self._auto_detect_device()
        else:
            self.device = torch.device(device)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device}")

        # Model configurations
        self.vision_models_config = {
            # Small vision models
            'resnet18': {'type': 'torchvision', 'output_dim': 512, 'input_size': 224},
            'mobilenet_v2': {'type': 'torchvision', 'output_dim': 1280, 'input_size': 224},
            'efficientnet_b0': {'type': 'timm', 'output_dim': 1280, 'input_size': 224},

            # Medium vision models
            'resnet34': {'type': 'torchvision', 'output_dim': 512, 'input_size': 224},
            'resnet50': {'type': 'torchvision', 'output_dim': 2048, 'input_size': 224},
            'efficientnet_b1': {'type': 'timm', 'output_dim': 1280, 'input_size': 240},
            'vit_tiny_patch16_224': {'type': 'timm', 'output_dim': 192, 'input_size': 224},
            'vit_small_patch16_224': {'type': 'timm', 'output_dim': 384, 'input_size': 224},

            # CLIP models
            'clip_rn50': {'type': 'openclip', 'model_name': 'RN50', 'pretrained': 'openai', 'output_dim': 1024},
            'clip_vit_b32': {'type': 'openclip', 'model_name': 'ViT-B-32', 'pretrained': 'openai', 'output_dim': 512},
            'clip_vit_b16': {'type': 'openclip', 'model_name': 'ViT-B-16', 'pretrained': 'openai', 'output_dim': 512},
        }

        self.language_models_config = {
            # Small language models
            'distilbert-base': {'type': 'huggingface', 'model_name': 'distilbert-base-uncased', 'output_dim': 768},
            'albert-base-v2': {'type': 'huggingface', 'model_name': 'albert-base-v2', 'output_dim': 768},
            'gpt2': {'type': 'huggingface', 'model_name': 'gpt2', 'output_dim': 768},

            # Medium language models
            'bert-base': {'type': 'huggingface', 'model_name': 'bert-base-uncased', 'output_dim': 768},
            'roberta-base': {'type': 'huggingface', 'model_name': 'roberta-base', 'output_dim': 768},
            'gpt2-medium': {'type': 'huggingface', 'model_name': 'gpt2-medium', 'output_dim': 1024},
            't5-small': {'type': 'huggingface', 'model_name': 't5-small', 'output_dim': 512},

            # Sentence transformers
            'sentence-bert-base': {'type': 'sentence-transformers', 'model_name': 'all-MiniLM-L6-v2', 'output_dim': 384},
            'sentence-bert-large': {'type': 'sentence-transformers', 'model_name': 'all-mpnet-base-v2', 'output_dim': 768},
        }

        self.loaded_models = {}

    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("MPS (Apple Silicon) available")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device

    def load_vision_model(self, model_key: str):
        """Load a vision model"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        config = self.vision_models_config.get(model_key)
        if not config:
            raise ValueError(f"Unknown vision model: {model_key}")

        print(f"Loading vision model: {model_key}")

        if config['type'] == 'torchvision':
            if model_key == 'resnet18':
                model = resnet18(pretrained=True)
            elif model_key == 'resnet34':
                model = resnet34(pretrained=True)
            elif model_key == 'resnet50':
                model = resnet50(pretrained=True)
            elif model_key == 'mobilenet_v2':
                model = mobilenet_v2(pretrained=True)
            elif model_key == 'efficientnet_b0':
                model = efficientnet_b0(pretrained=True)

            # Remove classification head
            if hasattr(model, 'fc'):
                model.fc = nn.Identity()
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    model.classifier[-1] = nn.Identity()
                else:
                    model.classifier = nn.Identity()

        elif config['type'] == 'timm':
            model = timm.create_model(model_key, pretrained=True, num_classes=0)

        elif config['type'] == 'openclip':
            model, _, preprocess = open_clip.create_model_and_transforms(
                config['model_name'],
                pretrained=config['pretrained'],
                device=self.device
            )
            self.loaded_models[f"{model_key}_preprocess"] = preprocess

        model = model.to(self.device)
        model.eval()
        self.loaded_models[model_key] = model

        return model

    def load_language_model(self, model_key: str):
        """Load a language model"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        config = self.language_models_config.get(model_key)
        if not config:
            raise ValueError(f"Unknown language model: {model_key}")

        print(f"Loading language model: {model_key}")

        if config['type'] == 'huggingface':
            model_name = config['model_name']

            if 'distilbert' in model_name:
                model = DistilBertModel.from_pretrained(model_name)
                tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            elif 'albert' in model_name:
                model = AlbertModel.from_pretrained(model_name)
                tokenizer = AlbertTokenizer.from_pretrained(model_name)
            elif 'roberta' in model_name:
                model = RobertaModel.from_pretrained(model_name)
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
            elif 'bert' in model_name:
                model = BertModel.from_pretrained(model_name)
                tokenizer = BertTokenizer.from_pretrained(model_name)
            elif 'gpt2' in model_name:
                model = GPT2Model.from_pretrained(model_name)
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
            elif 't5' in model_name:
                model = T5Model.from_pretrained(model_name)
                tokenizer = T5Tokenizer.from_pretrained(model_name)

            self.loaded_models[f"{model_key}_tokenizer"] = tokenizer

        elif config['type'] == 'sentence-transformers':
            model = SentenceTransformer(config['model_name'])
            tokenizer = None

        model = model.to(self.device)
        model.eval()
        self.loaded_models[model_key] = model

        return model

    def extract_vision_features(self, model_key: str, images: torch.Tensor) -> torch.Tensor:
        """Extract features from vision model"""
        model = self.load_vision_model(model_key)
        config = self.vision_models_config[model_key]

        with torch.no_grad():
            if config['type'] == 'openclip':
                # For CLIP models
                if model_key + '_preprocess' in self.loaded_models:
                    # Images should be preprocessed already
                    features = model.encode_image(images)
                else:
                    features = model.encode_image(images)
            else:
                # For standard vision models
                features = model(images)

        return features

    def extract_text_features(self, model_key: str, texts: List[str],
                            max_length: int = 512) -> torch.Tensor:
        """Extract features from language model"""
        model = self.load_language_model(model_key)
        config = self.language_models_config[model_key]

        with torch.no_grad():
            if config['type'] == 'sentence-transformers':
                features = model.encode(texts, convert_to_tensor=True)

            elif config['type'] == 'huggingface':
                tokenizer = self.loaded_models[f"{model_key}_tokenizer"]

                # Tokenize
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)

                # Extract features
                outputs = model(**encoded)

                # Pool features
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    # Mean pooling
                    hidden_states = outputs.last_hidden_state
                    attention_mask = encoded['attention_mask'].unsqueeze(-1)
                    features = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)

        return features


class CrossModalAlignmentAnalyzer:
    """Analyze cross-modal alignment between vision and language embeddings"""

    def __init__(self):
        self.metrics = {}

    def compute_cosine_similarity(self, features1: torch.Tensor,
                                 features2: torch.Tensor) -> float:
        """Compute mean cosine similarity between paired features"""
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        similarities = (features1 * features2).sum(dim=1)
        return float(similarities.mean())

    def compute_cka(self, features1: torch.Tensor,
                   features2: torch.Tensor) -> float:
        """Compute Centered Kernel Alignment (CKA)"""
        features1 = features1 - features1.mean(dim=0, keepdim=True)
        features2 = features2 - features2.mean(dim=0, keepdim=True)

        K1 = features1 @ features1.T
        K2 = features2 @ features2.T

        hsic_12 = torch.trace(K1 @ K2)
        hsic_11 = torch.trace(K1 @ K1)
        hsic_22 = torch.trace(K2 @ K2)

        cka = hsic_12 / torch.sqrt(hsic_11 * hsic_22)
        return float(cka)

    def compute_procrustes_similarity(self, features1: torch.Tensor,
                                     features2: torch.Tensor) -> float:
        """Compute Procrustes similarity after optimal alignment"""
        # Center features
        features1 = features1 - features1.mean(dim=0, keepdim=True)
        features2 = features2 - features2.mean(dim=0, keepdim=True)

        # Normalize
        features1 = features1 / (features1.norm(dim=1, keepdim=True) + 1e-8)
        features2 = features2 / (features2.norm(dim=1, keepdim=True) + 1e-8)

        # Compute optimal rotation
        H = features1.T @ features2
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T

        # Apply rotation and compute similarity
        features1_aligned = features1 @ R.T
        similarity = (features1_aligned * features2).sum() / len(features1)

        return float(similarity)

    def compute_mutual_nn_accuracy(self, features1: torch.Tensor,
                                  features2: torch.Tensor, k: int = 5) -> float:
        """Compute mutual nearest neighbor accuracy"""
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = features1 @ features2.T

        # Find nearest neighbors
        _, nn1_indices = sim_matrix.topk(k, dim=1)
        _, nn2_indices = sim_matrix.T.topk(k, dim=1)

        # Check mutual nearest neighbors
        correct = 0
        for i in range(len(features1)):
            if i in nn2_indices[nn1_indices[i, 0]]:
                correct += 1

        accuracy = correct / len(features1)
        return float(accuracy)

    def compute_representation_similarity_matrix(self, features_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """Compute RSM across all feature sets"""
        model_names = list(features_dict.keys())
        n_models = len(model_names)
        rsm = np.zeros((n_models, n_models))

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i <= j:
                    sim = self.compute_cka(features_dict[model1], features_dict[model2])
                    rsm[i, j] = sim
                    rsm[j, i] = sim

        return rsm, model_names

    def analyze_alignment(self, vision_features: Dict[str, torch.Tensor],
                         text_features: Dict[str, torch.Tensor]) -> Dict:
        """Comprehensive alignment analysis"""
        results = {}

        # Ensure matching keys
        common_keys = set(vision_features.keys()) & set(text_features.keys())

        for key in common_keys:
            v_feat = vision_features[key]
            t_feat = text_features[key]

            # Ensure same number of samples
            min_samples = min(len(v_feat), len(t_feat))
            v_feat = v_feat[:min_samples]
            t_feat = t_feat[:min_samples]

            results[key] = {
                'cosine_similarity': self.compute_cosine_similarity(v_feat, t_feat),
                'cka': self.compute_cka(v_feat, t_feat),
                'procrustes': self.compute_procrustes_similarity(v_feat, t_feat),
                'mutual_nn_acc': self.compute_mutual_nn_accuracy(v_feat, t_feat)
            }

        return results


class MultiModalDatasetLoader:
    """Load and prepare multi-modal datasets"""

    def __init__(self, dataset_name: str = 'mscoco', batch_size: int = 32):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def load_paired_data(self, num_samples: int = 1000) -> Tuple[List, List]:
        """Load paired vision-language data"""

        if self.dataset_name == 'mscoco':
            # Load MS COCO captions
            dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
            dataset = dataset.select(range(min(num_samples, len(dataset))))

            images = []
            texts = []

            for item in dataset:
                images.append(item['image'])
                texts.append(item['text'])

        elif self.dataset_name == 'conceptual_captions':
            # Load Conceptual Captions
            dataset = load_dataset("conceptual_captions", split="train")
            dataset = dataset.select(range(min(num_samples, len(dataset))))

            images = []
            texts = []

            for item in dataset:
                if item['image_url'] and item['caption']:
                    # Note: Would need to download images from URLs
                    texts.append(item['caption'])

        else:
            # Use a simple synthetic dataset for testing
            print(f"Using synthetic data for testing")
            images = [Image.new('RGB', (224, 224), color=(i % 255, (i*2) % 255, (i*3) % 255))
                     for i in range(num_samples)]
            texts = [f"This is image number {i} in the dataset" for i in range(num_samples)]

        return images, texts

    def prepare_vision_batch(self, images: List, transform=None) -> torch.Tensor:
        """Prepare batch of images"""
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

        processed = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            processed.append(transform(img))

        return torch.stack(processed)


class MultiModalVisualization:
    """Visualization tools for multi-modal embeddings"""

    def __init__(self, output_dir: str = './results/multi_modal_comparison'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_alignment_matrix(self, alignment_results: Dict,
                             vision_models: List[str],
                             language_models: List[str],
                             metric: str = 'cka'):
        """Plot alignment matrix between vision and language models"""

        # Create matrix
        matrix = np.zeros((len(vision_models), len(language_models)))

        for i, v_model in enumerate(vision_models):
            for j, l_model in enumerate(language_models):
                key = f"{v_model}_{l_model}"
                if key in alignment_results and metric in alignment_results[key]:
                    matrix[i, j] = alignment_results[key][metric]

        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix,
                   xticklabels=language_models,
                   yticklabels=vision_models,
                   annot=True,
                   fmt='.3f',
                   cmap='coolwarm',
                   center=0.5,
                   vmin=0, vmax=1)

        plt.title(f'Cross-Modal Alignment ({metric.upper()})')
        plt.xlabel('Language Models')
        plt.ylabel('Vision Models')
        plt.tight_layout()

        output_file = self.output_dir / f'alignment_matrix_{metric}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Saved alignment matrix to {output_file}")

    def plot_embedding_comparison(self, features_dict: Dict[str, torch.Tensor],
                                 modality: str = 'vision'):
        """Plot embedding statistics comparison"""

        models = list(features_dict.keys())

        # Compute statistics
        stats = {
            'mean_norm': [],
            'std_norm': [],
            'sparsity': [],
            'dimensionality': []
        }

        for model in models:
            features = features_dict[model]
            norms = features.norm(dim=1)

            stats['mean_norm'].append(float(norms.mean()))
            stats['std_norm'].append(float(norms.std()))
            stats['sparsity'].append(float((features == 0).float().mean()))
            stats['dimensionality'].append(features.shape[1])

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Mean norm
        axes[0, 0].bar(range(len(models)), stats['mean_norm'])
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].set_title('Mean Feature Norm')
        axes[0, 0].set_ylabel('Norm')

        # Std norm
        axes[0, 1].bar(range(len(models)), stats['std_norm'])
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].set_title('Std Feature Norm')
        axes[0, 1].set_ylabel('Std')

        # Sparsity
        axes[1, 0].bar(range(len(models)), stats['sparsity'])
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].set_title('Feature Sparsity')
        axes[1, 0].set_ylabel('Proportion of Zeros')

        # Dimensionality
        axes[1, 1].bar(range(len(models)), stats['dimensionality'])
        axes[1, 1].set_xticks(range(len(models)))
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].set_title('Embedding Dimensionality')
        axes[1, 1].set_ylabel('Dimensions')

        plt.suptitle(f'{modality.capitalize()} Model Embedding Statistics')
        plt.tight_layout()

        output_file = self.output_dir / f'{modality}_embedding_stats.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Saved embedding statistics to {output_file}")

    def plot_rsm(self, rsm: np.ndarray, model_names: List[str], modality: str = 'combined'):
        """Plot Representation Similarity Matrix"""

        plt.figure(figsize=(12, 10))
        sns.heatmap(rsm,
                   xticklabels=model_names,
                   yticklabels=model_names,
                   annot=True,
                   fmt='.2f',
                   cmap='viridis',
                   vmin=0, vmax=1,
                   square=True)

        plt.title(f'Representation Similarity Matrix ({modality})')
        plt.tight_layout()

        output_file = self.output_dir / f'rsm_{modality}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Saved RSM to {output_file}")


class MultiModalComparison:
    """Main class for multi-modal embedding comparison"""

    def __init__(self, output_dir: str = './results/multi_modal_comparison'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = MultiModalEmbeddingExtractor()
        self.analyzer = CrossModalAlignmentAnalyzer()
        self.visualizer = MultiModalVisualization(output_dir)

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'vision_models': [],
            'language_models': [],
            'alignment_results': {},
            'vision_features': {},
            'text_features': {}
        }

    def run_comparison(self,
                       vision_models: List[str] = None,
                       language_models: List[str] = None,
                       dataset_name: str = 'synthetic',
                       num_samples: int = 500):
        """Run comprehensive multi-modal comparison"""

        # Default model selection
        if vision_models is None:
            vision_models = ['resnet18', 'mobilenet_v2', 'efficientnet_b0',
                           'resnet34', 'vit_tiny_patch16_224']

        if language_models is None:
            language_models = ['distilbert-base', 'albert-base-v2',
                             'bert-base', 'sentence-bert-base']

        print(f"\n{'='*60}")
        print("Multi-Modal Embedding Comparison")
        print(f"{'='*60}")
        print(f"Vision Models: {vision_models}")
        print(f"Language Models: {language_models}")
        print(f"Dataset: {dataset_name}")
        print(f"Num Samples: {num_samples}")

        # Load data
        print("\nLoading paired data...")
        loader = MultiModalDatasetLoader(dataset_name)
        images, texts = loader.load_paired_data(num_samples)

        # Extract vision features
        print("\nExtracting vision features...")
        vision_features_all = {}

        for v_model in tqdm(vision_models, desc="Vision models"):
            try:
                # Prepare images
                if 'clip' in v_model:
                    # Use CLIP preprocessing
                    self.extractor.load_vision_model(v_model)
                    preprocess = self.extractor.loaded_models.get(f"{v_model}_preprocess")
                    if preprocess:
                        processed_images = torch.stack([preprocess(img) for img in images])
                    else:
                        processed_images = loader.prepare_vision_batch(images)
                else:
                    processed_images = loader.prepare_vision_batch(images)

                # Extract features in batches
                features = []
                batch_size = 32

                for i in range(0, len(processed_images), batch_size):
                    batch = processed_images[i:i+batch_size].to(self.extractor.device)
                    batch_features = self.extractor.extract_vision_features(v_model, batch)
                    features.append(batch_features.cpu())

                vision_features_all[v_model] = torch.cat(features, dim=0)
                print(f"  {v_model}: {vision_features_all[v_model].shape}")

            except Exception as e:
                print(f"  Error with {v_model}: {e}")

        # Extract text features
        print("\nExtracting text features...")
        text_features_all = {}

        for l_model in tqdm(language_models, desc="Language models"):
            try:
                # Extract features in batches
                features = []
                batch_size = 32

                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_features = self.extractor.extract_text_features(l_model, batch_texts)
                    features.append(batch_features.cpu())

                text_features_all[l_model] = torch.cat(features, dim=0)
                print(f"  {l_model}: {text_features_all[l_model].shape}")

            except Exception as e:
                print(f"  Error with {l_model}: {e}")

        # Compute cross-modal alignment
        print("\nComputing cross-modal alignment...")
        alignment_results = {}

        for v_model in vision_features_all:
            for l_model in text_features_all:
                key = f"{v_model}_{l_model}"

                # Get features
                v_feat = vision_features_all[v_model]
                t_feat = text_features_all[l_model]

                # Ensure same number of samples
                min_samples = min(len(v_feat), len(t_feat))
                v_feat = v_feat[:min_samples]
                t_feat = t_feat[:min_samples]

                # Project to same dimension if needed
                if v_feat.shape[1] != t_feat.shape[1]:
                    common_dim = min(v_feat.shape[1], t_feat.shape[1])

                    # Use PCA-like projection
                    if v_feat.shape[1] > common_dim:
                        v_proj = nn.Linear(v_feat.shape[1], common_dim, bias=False)
                        v_feat = v_proj(v_feat)
                    else:
                        t_proj = nn.Linear(t_feat.shape[1], common_dim, bias=False)
                        t_feat = t_proj(t_feat)

                # Compute metrics
                alignment_results[key] = {
                    'cosine_similarity': self.analyzer.compute_cosine_similarity(v_feat, t_feat),
                    'cka': self.analyzer.compute_cka(v_feat, t_feat),
                    'procrustes': self.analyzer.compute_procrustes_similarity(v_feat, t_feat),
                    'mutual_nn_acc': self.analyzer.compute_mutual_nn_accuracy(v_feat, t_feat)
                }

        # Store results
        self.results['vision_models'] = list(vision_features_all.keys())
        self.results['language_models'] = list(text_features_all.keys())
        self.results['alignment_results'] = alignment_results
        self.results['vision_features'] = vision_features_all
        self.results['text_features'] = text_features_all

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Alignment matrices
        for metric in ['cka', 'cosine_similarity', 'procrustes', 'mutual_nn_acc']:
            self.visualizer.plot_alignment_matrix(
                alignment_results,
                self.results['vision_models'],
                self.results['language_models'],
                metric=metric
            )

        # Embedding statistics
        self.visualizer.plot_embedding_comparison(vision_features_all, 'vision')
        self.visualizer.plot_embedding_comparison(text_features_all, 'language')

        # RSM for vision models
        vision_rsm, v_names = self.analyzer.compute_representation_similarity_matrix(vision_features_all)
        self.visualizer.plot_rsm(vision_rsm, v_names, 'vision')

        # RSM for language models
        text_rsm, t_names = self.analyzer.compute_representation_similarity_matrix(text_features_all)
        self.visualizer.plot_rsm(text_rsm, t_names, 'language')

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

        return self.results

    def save_results(self):
        """Save results to JSON"""
        output_file = self.output_dir / 'comparison_results.json'

        # Convert tensors to lists for JSON serialization
        json_results = {
            'timestamp': self.results['timestamp'],
            'vision_models': self.results['vision_models'],
            'language_models': self.results['language_models'],
            'alignment_results': self.results['alignment_results']
        }

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to {output_file}")

    def print_summary(self):
        """Print summary of results"""
        print(f"\n{'='*60}")
        print("Summary of Results")
        print(f"{'='*60}")

        # Find best alignments
        best_alignments = {}
        for metric in ['cka', 'cosine_similarity', 'procrustes', 'mutual_nn_acc']:
            best_score = -1
            best_pair = None

            for key, values in self.results['alignment_results'].items():
                if values[metric] > best_score:
                    best_score = values[metric]
                    best_pair = key

            best_alignments[metric] = (best_pair, best_score)

        print("\nBest Cross-Modal Alignments:")
        for metric, (pair, score) in best_alignments.items():
            v_model, l_model = pair.rsplit('_', 1)
            print(f"  {metric.upper():20s}: {v_model:25s} <-> {l_model:20s} = {score:.4f}")

        # Average alignment by model
        print("\nAverage Alignment Scores by Vision Model (CKA):")
        for v_model in self.results['vision_models']:
            scores = []
            for key, values in self.results['alignment_results'].items():
                if key.startswith(v_model + '_'):
                    scores.append(values['cka'])
            if scores:
                print(f"  {v_model:30s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        print("\nAverage Alignment Scores by Language Model (CKA):")
        for l_model in self.results['language_models']:
            scores = []
            for key, values in self.results['alignment_results'].items():
                if key.endswith('_' + l_model):
                    scores.append(values['cka'])
            if scores:
                print(f"  {l_model:30s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


def main():
    """Main function to run multi-modal comparison"""

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Modal Embedding Comparison')
    parser.add_argument('--vision-models', nargs='+', default=None,
                       help='Vision models to compare')
    parser.add_argument('--language-models', nargs='+', default=None,
                       help='Language models to compare')
    parser.add_argument('--dataset', default='synthetic',
                       help='Dataset to use (synthetic, mscoco, conceptual_captions)')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of samples to use')
    parser.add_argument('--output-dir', default='./results/multi_modal_comparison',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create comparison instance
    comparison = MultiModalComparison(output_dir=args.output_dir)

    # Run comparison
    results = comparison.run_comparison(
        vision_models=args.vision_models,
        language_models=args.language_models,
        dataset_name=args.dataset,
        num_samples=args.num_samples
    )

    print(f"\n{'='*60}")
    print("Multi-Modal Comparison Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()