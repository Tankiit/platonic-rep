"""
Phase 1 Skeleton: PRH's Blind Spot - Controlled Misalignment Experiment
========================================================================

This skeleton provides the complete structure for testing whether representation
metrics (CKA, mutual kNN) fail to detect semantic misalignment while optimization
metrics (NTK similarity, AGOP) correctly identify it.

The experiment trains CLIP-style models on MS-COCO with controlled levels of
caption misalignment (0%, 25%, 50%, 75%, 100%) and tracks both metric classes
throughout training.

Structure:
    1. Configuration and Setup
    2. Data Pipeline with Controlled Corruption
    3. Model Architecture (CLIP-style)
    4. Metric Computation Infrastructure
    5. Training Loop with Instrumentation
    6. Evaluation and Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
from PIL import Image
import logging

# For metric computation
try:
    from torch.func import jacrev, vmap  # For NTK computation
    HAS_TORCH_FUNC = True
except ImportError:
    HAS_TORCH_FUNC = False
    logging.warning("torch.func not available. NTK computation will be disabled.")


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Central configuration for the entire Phase 1 experiment.
    All hyperparameters and settings in one place for reproducibility.
    """
    # Dataset settings
    dataset_name: str = "JotDe/mscoco_50k"  # HuggingFace dataset name
    coco_root: str = "/path/to/coco"  # Kept for backward compatibility
    train_annotation_file: str = "annotations/captions_train2017.json"  # Kept for compatibility
    val_annotation_file: str = "annotations/captions_val2017.json"  # Kept for compatibility

    # Dataset limits (useful for quick testing)
    max_train_samples: Optional[int] = None  # None for all samples
    max_val_samples: Optional[int] = None    # None for all samples

    # Misalignment conditions to run
    # Each value represents the fraction of training pairs that are corrupted
    misalignment_ratios: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # Model architecture
    vision_encoder: str = "resnet18"  # Options: resnet18, resnet34, vit_tiny
    text_encoder: str = "distilbert"  # Options: distilbert, bert_tiny
    embedding_dim: int = 256  # Shared embedding dimension

    # Training settings
    batch_size: int = 256
    num_epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07  # InfoNCE temperature

    # Metric computation frequency
    repr_metric_every_n_steps: int = 500   # CKA, mutual kNN
    optim_metric_every_n_epochs: int = 1    # NTK, AGOP
    checkpoint_epochs: List[int] = field(default_factory=lambda: [4, 8, 12, 15])

    # Computational settings
    device: str = "mps"  # Options: cuda, mps, cpu
    num_workers: int = 0  # Set to 0 to avoid multiprocessing issues
    seed: int = 42

    # Output paths
    output_dir: str = "./phase1_results"


def setup_experiment(config: ExperimentConfig) -> None:
    """Initialize experiment: set seeds, create directories, setup logging."""
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directories
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "metrics").mkdir(exist_ok=True)
    (output_path / "figures").mkdir(exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_path / "experiment.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Experiment initialized with config: {config}")


# =============================================================================
# SECTION 2: DATA PIPELINE WITH CONTROLLED CORRUPTION
# =============================================================================

class MisalignedCOCODataset(Dataset):
    """
    MS-COCO Captions dataset with controlled semantic misalignment using HuggingFace datasets.

    The key innovation here is the corruption mechanism: we select a fraction
    of images and reassign their captions to random other images. This breaks
    semantic correspondence while maintaining the same data distribution.

    Args:
        dataset_name: HuggingFace dataset name (default: "JotDe/mscoco_50k")
        split: Dataset split ('train' or 'validation')
        misalignment_ratio: Fraction of pairs to corrupt (0.0 to 1.0)
        transform: Image transforms to apply
        seed: Random seed for reproducible corruption
        max_samples: Maximum number of samples to use (None for all)
    """

    def __init__(
        self,
        dataset_name: str = "JotDe/mscoco_50k",
        split: str = "train",
        misalignment_ratio: float = 0.0,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
        max_samples: Optional[int] = None,
        create_val_split: bool = False,
        val_split_ratio: float = 0.1,
        train_indices: Optional[List[int]] = None
    ):
        from datasets import load_dataset

        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.misalignment_ratio = misalignment_ratio
        self.seed = seed
        self.create_val_split = create_val_split
        self.val_split_ratio = val_split_ratio
        self.train_indices = train_indices

        logging.info(f"Loading dataset {dataset_name} split: {split}")

        # Load dataset from HuggingFace
        try:
            # Load the full dataset
            full_dataset = load_dataset(dataset_name)

            # Get the appropriate split
            if split in full_dataset:
                self.dataset = full_dataset[split]
            else:
                # Try common split names
                split_mapping = {
                    'train': ['train', 'training'],
                    'validation': ['validation', 'val', 'test'],
                }
                found = False
                for alt_name in split_mapping.get(split, [split]):
                    if alt_name in full_dataset:
                        self.dataset = full_dataset[alt_name]
                        found = True
                        break

                # If validation split not found and we're creating one, use train_indices or create it
                if not found and split in ['validation', 'val', 'test']:
                    if 'train' in full_dataset:
                        if self.train_indices is not None:
                            # Use provided train indices for validation
                            self.dataset = full_dataset['train'].select(self.train_indices)
                            logging.info(f"Created validation set with {len(self.dataset)} samples using provided indices")
                            found = True
                        elif self.create_val_split:
                            logging.warning(f"Validation split not found in {dataset_name}. "
                                          f"Creating validation split from train data.")
                            # Use 10% of train data for validation
                            train_data = full_dataset['train']
                            total_size = len(train_data)
                            val_size = min(1000, max(100, total_size // 10))  # Use 10% but at least 100 samples

                            # Split the data - ensure we get consistent indices
                            import random
                            random.seed(seed + 1000)  # Different seed for validation
                            indices = random.sample(range(total_size), val_size)
                            self.dataset = train_data.select(indices)
                            logging.info(f"Created validation set with {len(self.dataset)} samples from train data")
                            found = True
                        else:
                            available = list(full_dataset.keys())
                            raise ValueError(f"Split '{split}' not found. Available: {available}")
                    else:
                        available = list(full_dataset.keys())
                        raise ValueError(f"Split '{split}' not found and no train data available. Available: {available}")

                if not found:
                    available = list(full_dataset.keys())
                    raise ValueError(f"Split '{split}' not found. Available: {available}")

        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

        # Limit samples if specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            logging.info(f"Limited dataset to {len(self.dataset)} samples")

        # Extract image IDs and captions
        self._extract_image_caption_pairs()

        # Apply corruption: reassign captions for selected images
        self._apply_corruption(seed)

    def _extract_image_caption_pairs(self) -> None:
        """
        Extract image IDs and captions from the HuggingFace dataset.

        This method handles different dataset formats and creates a consistent
        internal representation for the corruption mechanism.
        """
        # Initialize storage
        self.image_data = []
        self.captions = []
        self.image_ids = []

        # Process dataset
        for idx, sample in enumerate(self.dataset):
            # Extract image data
            if 'image' in sample:
                image_data = sample['image']
            elif 'jpg' in sample:
                image_data = sample['jpg']
            elif 'png' in sample:
                image_data = sample['png']
            else:
                # Try to find any image field
                image_fields = [k for k in sample.keys() if 'image' in k.lower() or k.lower() in ['jpg', 'png', 'jpeg']]
                if image_fields:
                    image_data = sample[image_fields[0]]
                else:
                    raise ValueError(f"No image field found in sample {idx}. Keys: {list(sample.keys())}")

            # Extract caption(s)
            caption_fields = ['caption', 'text', 'captions']
            caption_data = None

            for field in caption_fields:
                if field in sample:
                    caption_data = sample[field]
                    break

            if caption_data is None:
                # Try to find any text field
                text_fields = [k for k in sample.keys() if 'text' in k.lower() or 'caption' in k.lower()]
                if text_fields:
                    caption_data = sample[text_fields[0]]
                else:
                    raise ValueError(f"No caption field found in sample {idx}. Keys: {list(sample.keys())}")

            # Handle different caption formats
            if isinstance(caption_data, str):
                captions_list = [caption_data]
            elif isinstance(caption_data, list):
                # Flatten nested lists if needed
                if len(caption_data) > 0 and isinstance(caption_data[0], dict):
                    # Handle list of dicts like [{'caption': 'text1'}, {'caption': 'text2'}]
                    captions_list = [item.get('caption', item.get('text', str(item))) for item in caption_data]
                else:
                    captions_list = caption_data
            elif isinstance(caption_data, dict):
                captions_list = [caption_data.get('caption', caption_data.get('text', str(caption_data)))]
            else:
                captions_list = [str(caption_data)]

            # Store data
            self.image_data.append(image_data)
            self.captions.append(captions_list)
            self.image_ids.append(idx)  # Use dataset index as image ID

        logging.info(f"Extracted {len(self.image_data)} image-caption pairs")

    def _apply_corruption(self, seed: int) -> None:
        """
        Apply controlled misalignment to the dataset.

        For each image selected for corruption, we reassign its captions
        to captions from a randomly selected different image. This maintains
        the property that each image has captions (just not semantically
        related to that image).
        """
        if self.misalignment_ratio == 0.0:
            # No corruption needed
            self.corrupted_captions = self.captions.copy()
            self.corrupted_indices = set()
            return

        rng = np.random.RandomState(seed)
        n_images = len(self.image_ids)
        n_corrupt = int(n_images * self.misalignment_ratio)

        # Select which images to corrupt
        corrupt_indices = set(rng.choice(n_images, size=n_corrupt, replace=False))
        self.corrupted_indices = corrupt_indices

        # Create corrupted caption mapping
        self.corrupted_captions = self.captions.copy()

        # For corrupted images, assign captions from a random different image
        all_indices = list(range(n_images))
        for idx in corrupt_indices:
            # Select a random different image to steal captions from
            donor_idx = rng.randint(0, n_images)
            while donor_idx == idx:
                donor_idx = rng.randint(0, n_images)

            # Reassign captions (copy to avoid aliasing)
            self.corrupted_captions[idx] = self.captions[donor_idx].copy()

        logging.info(f"Applied {self.misalignment_ratio*100:.0f}% misalignment: "
                    f"{n_corrupt}/{n_images} images corrupted")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return an image-caption pair.

        For training, we randomly sample one of the available captions for each image.
        The caption may be corrupted (from a different image) depending on
        the misalignment ratio.
        """
        # Get image data
        image_data = self.image_data[idx]

        # Handle different image formats
        if isinstance(image_data, Image.Image):
            image = image_data.convert('RGB')
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data).convert('RGB')
        elif isinstance(image_data, str):
            # If it's a path, load the image
            image = Image.open(image_data).convert('RGB')
        else:
            # Try to convert to PIL Image
            try:
                image = Image.fromarray(image_data).convert('RGB')
            except Exception as e:
                logging.warning(f"Could not convert image data to PIL: {e}")
                # Create a dummy image if all else fails
                image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # Sample one caption (possibly corrupted)
        captions = self.corrupted_captions[idx]
        caption = np.random.choice(captions)

        # Track whether this pair is corrupted (for analysis)
        is_corrupted = idx in self.corrupted_indices

        return {
            'image': image,
            'caption': caption,
            'image_id': self.image_ids[idx],
            'is_corrupted': is_corrupted
        }


def get_transforms(split: str = "train") -> transforms.Compose:
    """Standard CLIP-style image transforms."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def collate_fn(batch):
    """Collate function for handling mixed image-caption batches."""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    is_corrupted = torch.tensor([item['is_corrupted'] for item in batch])
    return {
        'images': images,
        'captions': captions,
        'image_ids': image_ids,
        'is_corrupted': is_corrupted
    }


def create_dataloaders(
    config: ExperimentConfig,
    misalignment_ratio: float,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for a given misalignment level."""

    dataset_name = getattr(config, 'dataset_name', "JotDe/mscoco_50k")

    # First, load the full dataset to check available splits
    from datasets import load_dataset
    full_dataset = load_dataset(dataset_name)

    if 'validation' in full_dataset or 'val' in full_dataset or 'test' in full_dataset:
        # Dataset has proper train/validation splits
        train_dataset = MisalignedCOCODataset(
            dataset_name=dataset_name,
            split="train",
            misalignment_ratio=misalignment_ratio,  # Corruption applied here
            transform=get_transforms("train"),
            seed=config.seed,
            max_samples=max_samples
        )

        val_dataset = MisalignedCOCODataset(
            dataset_name=dataset_name,
            split="validation",
            misalignment_ratio=0.0,  # Validation is NEVER corrupted
            transform=get_transforms("val"),
            seed=config.seed,
            max_samples=max_samples
        )
    else:
        # Dataset only has train split, create validation split
        logging.warning(f"Dataset {dataset_name} only has train split. Creating train/validation split.")

        # Get the full train data
        train_data = full_dataset['train']
        total_size = len(train_data)

        # Limit samples if specified
        if max_samples is not None:
            total_size = min(total_size, max_samples)

        # Create train/validation split (90% train, 10% val)
        import random
        random.seed(config.seed)
        indices = list(range(total_size))
        random.shuffle(indices)

        # Split indices
        val_size = max(100, total_size // 10)  # At least 100 samples for validation
        val_size = min(val_size, 1000)  # At most 1000 samples for validation
        train_size = total_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]

        # Create datasets with appropriate indices
        train_dataset = MisalignedCOCODataset(
            dataset_name=dataset_name,
            split="train",
            misalignment_ratio=misalignment_ratio,  # Corruption applied here
            transform=get_transforms("train"),
            seed=config.seed,
            max_samples=None,  # We'll handle this with indices
            train_indices=train_indices
        )

        val_dataset = MisalignedCOCODataset(
            dataset_name=dataset_name,
            split="validation",
            misalignment_ratio=0.0,  # Validation is NEVER corrupted
            transform=get_transforms("val"),
            seed=config.seed,
            max_samples=None,  # We'll handle this with indices
            train_indices=val_indices
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


# =============================================================================
# SECTION 3: MODEL ARCHITECTURE
# =============================================================================

class VisionEncoder(nn.Module):
    """
    Vision encoder using a pretrained backbone with a projection head.

    We use pretrained weights as initialization but train end-to-end.
    The projection head maps backbone features to the shared embedding space.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 256,
        pretrained: bool = True
    ):
        super().__init__()

        if backbone == "resnet18":
            # Load pretrained ResNet-18
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification head

        elif backbone == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = resnet34(weights=weights)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == "vit_tiny":
            # Use timm for ViT variants
            import timm
            self.backbone = timm.create_model(
                'vit_tiny_patch16_224',
                pretrained=pretrained,
                num_classes=0  # Remove classification head
            )
            backbone_dim = self.backbone.embed_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Projection head: maps backbone features to shared embedding space
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning normalized embeddings.

        Args:
            x: Images of shape (batch_size, 3, 224, 224)

        Returns:
            Normalized embeddings of shape (batch_size, embedding_dim)
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        # L2 normalize for contrastive learning
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone features before projection (for metric analysis)."""
        return self.backbone(x)


class TextEncoder(nn.Module):
    """
    Text encoder using a pretrained transformer with a projection head.

    We use the [CLS] token representation from the transformer and project
    it to the shared embedding space.
    """

    def __init__(
        self,
        model_name: str = "distilbert",
        embedding_dim: int = 256,
        pretrained: bool = True
    ):
        super().__init__()

        from transformers import AutoModel, AutoTokenizer

        if model_name == "distilbert":
            model_path = "distilbert-base-uncased"
        elif model_name == "bert_tiny":
            model_path = "prajjwal1/bert-tiny"
        else:
            model_path = model_name

        self.transformer = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        hidden_dim = self.transformer.config.hidden_size

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(
        self,
        texts: List[str],
        device: torch.device
    ) -> torch.Tensor:
        """
        Forward pass returning normalized embeddings.

        Args:
            texts: List of caption strings
            device: Device to put tensors on

        Returns:
            Normalized embeddings of shape (batch_size, embedding_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP uses 77 tokens
            return_tensors="pt"
        ).to(device)

        # Get transformer output
        outputs = self.transformer(**encoded)

        # Use [CLS] token representation
        cls_features = outputs.last_hidden_state[:, 0, :]

        # Project and normalize
        embeddings = self.projection(cls_features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def get_features(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Get transformer features before projection (for metric analysis)."""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(device)
        outputs = self.transformer(**encoded)
        return outputs.last_hidden_state[:, 0, :]


class CLIPModel(nn.Module):
    """
    Complete CLIP-style model combining vision and text encoders.

    This model learns to align vision and text representations in a shared
    embedding space using contrastive learning.
    """

    def __init__(
        self,
        vision_encoder: str = "resnet18",
        text_encoder: str = "distilbert",
        embedding_dim: int = 256,
        temperature: float = 0.07
    ):
        super().__init__()

        self.vision_encoder = VisionEncoder(vision_encoder, embedding_dim)
        self.text_encoder = TextEncoder(text_encoder, embedding_dim)

        # Learnable temperature (initialized to given value)
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / temperature))
        )

    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing embeddings and similarity logits.

        Returns:
            image_embeddings: (batch_size, embedding_dim)
            text_embeddings: (batch_size, embedding_dim)
            logit_scale: Scalar temperature for logits
        """
        image_embeddings = self.vision_encoder(images)
        text_embeddings = self.text_encoder(texts, device)
        logit_scale = self.logit_scale.exp()

        return image_embeddings, text_embeddings, logit_scale

    def compute_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE contrastive loss.

        This is the standard CLIP loss: for each image, the correct text
        should have highest similarity, and vice versa.
        """
        # Compute similarity matrix
        logits = logit_scale * image_embeddings @ text_embeddings.t()

        # Symmetric loss: image-to-text and text-to-image
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return (loss_i2t + loss_t2i) / 2


# =============================================================================
# SECTION 4: METRIC COMPUTATION INFRASTRUCTURE
# =============================================================================

class MetricComputer:
    """
    Computes both representation and optimization metrics.

    Representation metrics (CKA, mutual kNN): measure similarity of final
    embeddings between vision and text modalities.

    Optimization metrics (NTK similarity, AGOP): measure compatibility of
    learning dynamics and gradient structure.
    """

    def __init__(self, model: CLIPModel, device: torch.device):
        self.model = model
        self.device = device

        # Storage for AGOP accumulation
        self.vision_agop_accumulator = None
        self.text_agop_accumulator = None
        self.agop_count = 0

    # =========================================================================
    # Representation Metrics
    # =========================================================================

    def compute_cka(
        self,
        vision_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> float:
        """
        Compute Centered Kernel Alignment between vision and text embeddings.

        CKA measures whether the geometry of relationships between points
        is similar across the two representation spaces.

        Args:
            vision_embeddings: (N, d_v) vision embeddings
            text_embeddings: (N, d_t) text embeddings (same N samples)

        Returns:
            CKA similarity score in [0, 1]
        """
        # Compute Gram matrices (kernel matrices)
        K_v = vision_embeddings @ vision_embeddings.t()
        K_t = text_embeddings @ text_embeddings.t()

        # Center the Gram matrices
        K_v = self._center_gram_matrix(K_v)
        K_t = self._center_gram_matrix(K_t)

        # Compute HSIC (Hilbert-Schmidt Independence Criterion)
        hsic = torch.sum(K_v * K_t)

        # Normalize
        norm_v = torch.sqrt(torch.sum(K_v * K_v))
        norm_t = torch.sqrt(torch.sum(K_t * K_t))

        cka = hsic / (norm_v * norm_t + 1e-8)
        return cka.item()

    def _center_gram_matrix(self, K: torch.Tensor) -> torch.Tensor:
        """Center a Gram matrix by removing row and column means."""
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def compute_mutual_knn(
        self,
        vision_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        k: int = 10
    ) -> float:
        """
        Compute mutual k-nearest neighbors between modalities.

        For each sample, find its k nearest neighbors in vision space and
        in text space. Count how many samples are mutual kNN in both spaces.

        Args:
            vision_embeddings: (N, d_v) vision embeddings
            text_embeddings: (N, d_t) text embeddings
            k: Number of nearest neighbors

        Returns:
            Fraction of samples that are mutual kNN in both spaces
        """
        n_samples = vision_embeddings.shape[0]

        # Compute pairwise distances
        dist_v = torch.cdist(vision_embeddings, vision_embeddings)
        dist_t = torch.cdist(text_embeddings, text_embeddings)

        # Set diagonal to infinity to exclude self
        dist_v.fill_diagonal_(float('inf'))
        dist_t.fill_diagonal_(float('inf'))

        # Find k nearest neighbors for each sample
        _, knn_v = torch.topk(dist_v, k, largest=False, dim=1)
        _, knn_t = torch.topk(dist_t, k, largest=False, dim=1)

        # For each sample, check if any of its k neighbors are mutual
        mutual_count = 0
        for i in range(n_samples):
            neighbors_v = set(knn_v[i].tolist())
            neighbors_t = set(knn_t[i].tolist())
            # Count mutual neighbors
            mutual = neighbors_v & neighbors_t
            mutual_count += len(mutual)

        # Normalize by maximum possible mutual neighbors
        max_mutual = n_samples * k
        return mutual_count / max_mutual

    # =========================================================================
    # Optimization Metrics
    # =========================================================================

    def compute_ntk_similarity(
        self,
        dataloader: DataLoader,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compute Neural Tangent Kernel similarity between modalities.

        The NTK captures how model outputs change with respect to parameter
        changes. Similar NTKs indicate compatible optimization dynamics.

        This is computationally expensive, so we subsample.

        Args:
            dataloader: Data loader to sample from
            n_samples: Number of samples to use

        Returns:
            Dictionary with NTK similarity scores
        """
        if not HAS_TORCH_FUNC:
            logging.warning("torch.func not available, skipping NTK computation")
            return {'ntk_similarity': 0.0}

        self.model.eval()

        # Collect samples
        images_list, texts_list = [], []
        for batch in dataloader:
            images_list.append(batch['images'])
            texts_list.extend(batch['captions'])
            if len(texts_list) >= n_samples:
                break

        images = torch.cat(images_list)[:n_samples].to(self.device)
        texts = texts_list[:n_samples]

        # Compute NTKs using the simplified method
        ntk_vision = self._compute_ntk_for_encoder(
            self.model.vision_encoder, images
        )
        ntk_text = self._compute_ntk_for_encoder_text(
            self.model.text_encoder, texts
        )

        # Compare NTKs using CKA or other similarity
        ntk_similarity = self._compare_ntk_matrices(ntk_vision, ntk_text)

        return {
            'ntk_similarity': ntk_similarity,
            'ntk_vision_rank': self._effective_rank(ntk_vision),
            'ntk_text_rank': self._effective_rank(ntk_text)
        }

    def _compute_ntk_for_encoder(
        self,
        encoder: nn.Module,
        inputs: torch.Tensor,
        n_subsample: int = 20
    ) -> torch.Tensor:
        """
        Compute a simplified approximation of the empirical NTK for an encoder.

        For computational efficiency, we use a finite difference approximation
        of the Jacobian and only compute it for a subset of parameters.
        """
        n_samples = min(inputs.shape[0], n_subsample)
        inputs_sub = inputs[:n_samples]

        # Get a subset of parameters (first few layers for efficiency)
        params = []
        param_indices = []
        for i, (name, p) in enumerate(encoder.named_parameters()):
            if len(params) >= 10:  # Limit number of parameter tensors
                break
            if p.requires_grad and len(p.shape) <= 2:  # Focus on dense layers
                params.append(p)
                param_indices.append(i)

        if not params:
            return torch.eye(n_samples, device=self.device)

        # Compute outputs
        encoder.eval()
        outputs = encoder(inputs_sub)

        # Use finite differences to approximate Jacobian
        eps = 1e-4
        ntk = torch.zeros(n_samples, n_samples, device=self.device)

        for param in params:
            # Store original value
            original = param.data.clone()

            # Compute perturbed outputs
            param.data += eps
            outputs_plus = encoder(inputs_sub)

            param.data = original  # Restore
            param.data -= eps
            outputs_minus = encoder(inputs_sub)

            # Restore original value
            param.data = original

            # Approximate Jacobian
            jacobian = (outputs_plus - outputs_minus) / (2 * eps)
            jacobian_flat = jacobian.view(n_samples, -1)

            # Add to NTK
            ntk += jacobian_flat @ jacobian_flat.t()

        return ntk

    def _compute_ntk_for_encoder_text(
        self,
        encoder: nn.Module,
        texts: List[str],
        n_subsample: int = 20
    ) -> torch.Tensor:
        """Compute simplified NTK for text encoder."""
        n_samples = min(len(texts), n_subsample)
        texts_sub = texts[:n_samples]

        # Tokenize once
        encoded = encoder.tokenizer(
            texts_sub, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(self.device)

        # Get a subset of parameters
        params = []
        for i, (name, p) in enumerate(encoder.named_parameters()):
            if len(params) >= 10:  # Limit number of parameter tensors
                break
            if p.requires_grad and len(p.shape) <= 2:  # Focus on dense layers
                params.append(p)

        if not params:
            return torch.eye(n_samples, device=self.device)

        # Compute NTK
        ntk = torch.zeros(n_samples, n_samples, device=self.device)

        for param in params:
            # Store original value
            original = param.data.clone()

            # Use smaller epsilon for text
            eps = 1e-5

            # Perturb parameter and compute change in outputs
            param.data += eps
            with torch.no_grad():
                outputs_plus = encoder.transformer(**encoded).last_hidden_state[:, 0, :]

            param.data = original
            param.data -= eps
            with torch.no_grad():
                outputs_minus = encoder.transformer(**encoded).last_hidden_state[:, 0, :]

            # Restore
            param.data = original

            # Approximate Jacobian contribution
            jacobian = (outputs_plus - outputs_minus) / (2 * eps)

            # Add to NTK
            ntk += jacobian @ jacobian.t()

        return ntk

    def _compare_ntk_matrices(
        self,
        ntk_1: torch.Tensor,
        ntk_2: torch.Tensor
    ) -> float:
        """Compare two NTK matrices using CKA or correlation."""
        # Use CKA to compare the kernel matrices themselves
        return self.compute_cka(ntk_1, ntk_2)

    def _effective_rank(self, matrix: torch.Tensor) -> float:
        """Compute effective rank using entropy of normalized eigenvalues."""
        # Handle MPS device for eigenvalue computation
        if isinstance(matrix.device, torch.device) and matrix.device.type == 'mps':
            matrix_cpu = matrix.to('cpu')
        else:
            matrix_cpu = matrix
        eigenvalues = torch.linalg.eigvalsh(matrix_cpu)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        eigenvalues = eigenvalues / eigenvalues.sum()
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues + 1e-10))
        return torch.exp(entropy).item()

    # =========================================================================
    # AGOP (Average Gradient Outer Product) Analysis
    # =========================================================================

    def accumulate_agop(
        self,
        vision_grads: torch.Tensor,
        text_grads: torch.Tensor
    ) -> None:
        """
        Accumulate gradient outer products for AGOP analysis.

        AGOP = E[g @ g^T] where g is the gradient vector.
        The eigenstructure of AGOP reveals what directions dominate learning.

        Args:
            vision_grads: Flattened gradients for vision encoder
            text_grads: Flattened gradients for text encoder
        """
        # Compute outer products
        vision_outer = torch.outer(vision_grads, vision_grads)
        text_outer = torch.outer(text_grads, text_grads)

        if self.vision_agop_accumulator is None:
            self.vision_agop_accumulator = vision_outer
            self.text_agop_accumulator = text_outer
        else:
            self.vision_agop_accumulator += vision_outer
            self.text_agop_accumulator += text_outer

        self.agop_count += 1

    def compute_agop_metrics(self) -> Dict[str, float]:
        """
        Compute AGOP-based metrics from accumulated gradients.

        Returns:
            Dictionary with AGOP metrics including:
            - effective rank of each encoder's AGOP
            - alignment between principal components
        """
        if self.agop_count == 0:
            return {}

        # Average the accumulated outer products
        vision_agop = self.vision_agop_accumulator / self.agop_count
        text_agop = self.text_agop_accumulator / self.agop_count

        # Handle MPS device for eigenvalue computation
        if isinstance(self.device, torch.device) and self.device.type == 'mps':
            vision_agop_cpu = vision_agop.to('cpu')
            text_agop_cpu = text_agop.to('cpu')
        else:
            vision_agop_cpu = vision_agop
            text_agop_cpu = text_agop

        # Compute eigendecomposition
        vision_eigenvalues, vision_eigenvectors = torch.linalg.eigh(vision_agop_cpu)
        text_eigenvalues, text_eigenvectors = torch.linalg.eigh(text_agop_cpu)

        # Sort by eigenvalue magnitude (descending)
        vision_idx = torch.argsort(vision_eigenvalues, descending=True)
        text_idx = torch.argsort(text_eigenvalues, descending=True)

        vision_eigenvalues = vision_eigenvalues[vision_idx]
        text_eigenvalues = text_eigenvalues[text_idx]
        vision_eigenvectors = vision_eigenvectors[:, vision_idx]
        text_eigenvectors = text_eigenvectors[:, text_idx]

        # Compute metrics
        metrics = {
            'vision_agop_rank': self._effective_rank_from_eigenvalues(vision_eigenvalues),
            'text_agop_rank': self._effective_rank_from_eigenvalues(text_eigenvalues),
            'agop_eigenvalue_correlation': self._eigenvalue_correlation(
                vision_eigenvalues, text_eigenvalues
            ),
        }

        # Compare top principal components (if same dimension)
        if vision_eigenvectors.shape[0] == text_eigenvectors.shape[0]:
            top_k = min(10, vision_eigenvectors.shape[1])
            pc_alignment = self._principal_component_alignment(
                vision_eigenvectors[:, :top_k],
                text_eigenvectors[:, :top_k]
            )
            metrics['top_pc_alignment'] = pc_alignment

        # Reset accumulators
        self.vision_agop_accumulator = None
        self.text_agop_accumulator = None
        self.agop_count = 0

        return metrics

    def _effective_rank_from_eigenvalues(self, eigenvalues: torch.Tensor) -> float:
        """Compute effective rank from eigenvalues."""
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        eigenvalues = eigenvalues / eigenvalues.sum()
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues + 1e-10))
        return torch.exp(entropy).item()

    def _eigenvalue_correlation(
        self,
        ev1: torch.Tensor,
        ev2: torch.Tensor
    ) -> float:
        """Compute correlation between eigenvalue spectra."""
        # Take top-k eigenvalues
        k = min(len(ev1), len(ev2), 100)
        ev1_topk = ev1[:k]
        ev2_topk = ev2[:k]

        # Normalize and compute correlation
        ev1_norm = (ev1_topk - ev1_topk.mean()) / (ev1_topk.std() + 1e-8)
        ev2_norm = (ev2_topk - ev2_topk.mean()) / (ev2_topk.std() + 1e-8)

        correlation = (ev1_norm * ev2_norm).mean()
        return correlation.item()

    def _principal_component_alignment(
        self,
        pcs1: torch.Tensor,
        pcs2: torch.Tensor
    ) -> float:
        """Compute alignment between principal components."""
        # Compute absolute cosine similarities between top PCs
        similarities = torch.abs(pcs1.t() @ pcs2)
        # Average of max alignment for each PC
        alignment = similarities.max(dim=1)[0].mean()
        return alignment.item()

    # =========================================================================
    # High-Level Metric Collection
    # =========================================================================

    @torch.no_grad()
    def compute_representation_metrics(
        self,
        dataloader: DataLoader,
        n_batches: int = 10
    ) -> Dict[str, float]:
        """
        Compute all representation metrics on validation data.

        Args:
            dataloader: Validation dataloader (uncorrupted)
            n_batches: Number of batches to use

        Returns:
            Dictionary of metric values
        """
        self.model.eval()

        vision_embeddings_list = []
        text_embeddings_list = []

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            images = batch['images'].to(self.device)
            texts = batch['captions']

            v_emb = self.model.vision_encoder(images)
            t_emb = self.model.text_encoder(texts, self.device)

            vision_embeddings_list.append(v_emb)
            text_embeddings_list.append(t_emb)

        vision_embeddings = torch.cat(vision_embeddings_list)
        text_embeddings = torch.cat(text_embeddings_list)

        metrics = {
            'cka': self.compute_cka(vision_embeddings, text_embeddings),
            'mutual_knn_5': self.compute_mutual_knn(vision_embeddings, text_embeddings, k=5),
            'mutual_knn_10': self.compute_mutual_knn(vision_embeddings, text_embeddings, k=10),
        }

        return metrics


# =============================================================================
# SECTION 5: TRAINING LOOP WITH INSTRUMENTATION
# =============================================================================

class Trainer:
    """
    Training orchestrator with comprehensive metric instrumentation.

    This class handles the training loop, metric computation scheduling,
    checkpointing, and logging for a single misalignment condition.
    """

    def __init__(
        self,
        model: CLIPModel,
        config: ExperimentConfig,
        misalignment_ratio: float
    ):
        self.model = model
        self.config = config
        self.misalignment_ratio = misalignment_ratio
        self.device = torch.device(config.device)

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        # Metric computer
        self.metric_computer = MetricComputer(model, self.device)

        # Logging
        self.metrics_history = {
            'train_loss': [],
            'cka': [],
            'mutual_knn': [],
            'ntk_similarity': [],
            'retrieval_r1': [],
            'retrieval_r5': [],
        }

        self.global_step = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Run full training with metric instrumentation.

        Returns:
            Complete metrics history
        """
        logging.info(f"Starting training for {self.misalignment_ratio*100:.0f}% misalignment")

        for epoch in range(self.config.num_epochs):
            epoch_loss = self._train_epoch(train_loader, val_loader, epoch)

            # End of epoch metrics
            logging.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {epoch_loss:.4f}")

            # Compute optimization metrics (less frequent)
            if (epoch + 1) % self.config.optim_metric_every_n_epochs == 0:
                optim_metrics = self._compute_optimization_metrics(val_loader)
                logging.info(f"  Optimization metrics: {optim_metrics}")

            # Checkpoint
            if (epoch + 1) in self.config.checkpoint_epochs:
                self._save_checkpoint(epoch + 1)
                retrieval_metrics = self._evaluate_retrieval(val_loader)
                logging.info(f"  Retrieval: R@1={retrieval_metrics['r1']:.3f}, "
                           f"R@5={retrieval_metrics['r5']:.3f}")

            self.scheduler.step()

        return self.metrics_history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epoch: int
    ) -> float:
        """Train for one epoch, computing metrics at scheduled intervals."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            self.global_step += 1

            # Forward pass
            images = batch['images'].to(self.device)
            texts = batch['captions']

            img_emb, txt_emb, logit_scale = self.model(images, texts, self.device)
            loss = self.model.compute_loss(img_emb, txt_emb, logit_scale)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Skip AGOP for quick testing to avoid memory issues
            if self.global_step % 10000 == 0:  # Effectively disabled
                self._accumulate_gradients_for_agop()

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Representation metrics at scheduled intervals
            if self.global_step % self.config.repr_metric_every_n_steps == 0:
                repr_metrics = self.metric_computer.compute_representation_metrics(
                    val_loader, n_batches=5
                )
                self.metrics_history['cka'].append(repr_metrics['cka'])
                self.metrics_history['mutual_knn'].append(repr_metrics['mutual_knn_10'])
                logging.info(f"  Step {self.global_step}: CKA={repr_metrics['cka']:.3f}, "
                           f"MutualKNN={repr_metrics['mutual_knn_10']:.3f}")

        avg_loss = total_loss / n_batches
        self.metrics_history['train_loss'].append(avg_loss)
        return avg_loss

    def _accumulate_gradients_for_agop(self) -> None:
        """Collect gradients for AGOP analysis with strict dimension limiting."""
        # Use a very small subset of gradients to avoid memory issues
        max_params = 500  # Very conservative limit

        # Collect gradients from vision encoder
        vision_grads = []
        for p in self.model.vision_encoder.parameters():
            if p.grad is not None:
                flat_grad = p.grad.flatten()
                if len(vision_grads) + len(flat_grad) <= max_params:
                    vision_grads.append(flat_grad)
                else:
                    # Take only what we need
                    remaining = max_params - len(vision_grads)
                    if remaining > 0:
                        vision_grads.append(flat_grad[:remaining])
                    break
        vision_grads = torch.cat(vision_grads) if vision_grads else torch.zeros(100)

        # Collect gradients from text encoder
        text_grads = []
        for p in self.model.text_encoder.parameters():
            if p.grad is not None:
                flat_grad = p.grad.flatten()
                if len(text_grads) + len(flat_grad) <= max_params:
                    text_grads.append(flat_grad)
                else:
                    # Take only what we need
                    remaining = max_params - len(text_grads)
                    if remaining > 0:
                        text_grads.append(flat_grad[:remaining])
                    break
        text_grads = torch.cat(text_grads) if text_grads else torch.zeros(100)

        # For AGOP, we typically want same-dimensional comparison
        min_dim = min(len(vision_grads), len(text_grads))
        min_dim = max(min_dim, 50)  # Ensure at least 50 dimensions

        self.metric_computer.accumulate_agop(
            vision_grads[:min_dim],
            text_grads[:min_dim]
        )

    def _compute_optimization_metrics(self, val_loader: DataLoader) -> Dict[str, float]:
        """Compute all optimization-based metrics."""
        metrics = {}

        # NTK similarity (expensive)
        try:
            ntk_metrics = self.metric_computer.compute_ntk_similarity(
                val_loader, n_samples=50
            )
            metrics.update(ntk_metrics)
            self.metrics_history['ntk_similarity'].append(
                ntk_metrics.get('ntk_similarity', 0.0)
            )
        except Exception as e:
            logging.warning(f"NTK computation failed: {e}")

        # AGOP metrics
        agop_metrics = self.metric_computer.compute_agop_metrics()
        metrics.update(agop_metrics)

        return metrics

    @torch.no_grad()
    def _evaluate_retrieval(
        self,
        val_loader: DataLoader,
        n_batches: int = 20
    ) -> Dict[str, float]:
        """Evaluate image-text retrieval performance."""
        self.model.eval()

        all_img_emb = []
        all_txt_emb = []

        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break

            images = batch['images'].to(self.device)
            texts = batch['captions']

            img_emb = self.model.vision_encoder(images)
            txt_emb = self.model.text_encoder(texts, self.device)

            all_img_emb.append(img_emb)
            all_txt_emb.append(txt_emb)

        img_emb = torch.cat(all_img_emb)
        txt_emb = torch.cat(all_txt_emb)

        # Compute similarity matrix
        similarity = img_emb @ txt_emb.t()

        # Image-to-text retrieval
        n = similarity.shape[0]
        labels = torch.arange(n, device=self.device)

        # Recall@k
        _, topk_indices = similarity.topk(5, dim=1)
        r1 = (topk_indices[:, 0] == labels).float().mean().item()
        r5 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        metrics = {'r1': r1, 'r5': r5}
        self.metrics_history['retrieval_r1'].append(r1)
        self.metrics_history['retrieval_r5'].append(r5)

        return metrics

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = (
            Path(self.config.output_dir) / "checkpoints" /
            f"misalign_{self.misalignment_ratio:.2f}_epoch_{epoch}.pt"
        )
        torch.save({
            'epoch': epoch,
            'misalignment_ratio': self.misalignment_ratio,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
        }, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")


# =============================================================================
# SECTION 6: MAIN EXPERIMENT RUNNER
# =============================================================================

def run_phase1_experiment(config: ExperimentConfig) -> Dict[str, Dict]:
    """
    Run the complete Phase 1 experiment across all misalignment conditions.

    This is the main entry point that orchestrates training for each
    condition and collects results for analysis.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary mapping misalignment ratios to their metrics histories
    """
    setup_experiment(config)

    all_results = {}

    for misalignment_ratio in config.misalignment_ratios:
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting condition: {misalignment_ratio*100:.0f}% misalignment")
        logging.info(f"{'='*60}\n")

        # Create fresh model for each condition
        model = CLIPModel(
            vision_encoder=config.vision_encoder,
            text_encoder=config.text_encoder,
            embedding_dim=config.embedding_dim,
            temperature=config.temperature
        )

        # Create dataloaders with appropriate corruption
        train_loader, val_loader = create_dataloaders(
            config, misalignment_ratio,
            max_samples=config.max_train_samples
        )

        # Train
        trainer = Trainer(model, config, misalignment_ratio)
        metrics_history = trainer.train(train_loader, val_loader)

        all_results[misalignment_ratio] = metrics_history

        # Save results for this condition
        results_path = Path(config.output_dir) / "metrics" / f"results_{misalignment_ratio:.2f}.json"
        with open(results_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)

    # Final analysis
    analyze_results(all_results, config)

    return all_results


def analyze_results(
    all_results: Dict[float, Dict],
    config: ExperimentConfig
) -> None:
    """
    Analyze and visualize results across all conditions.

    This computes the key comparisons: do representation metrics fail
    to detect misalignment while optimization metrics succeed?
    """
    import matplotlib.pyplot as plt

    misalignment_levels = sorted(all_results.keys())

    # Extract final metrics for each condition
    final_cka = []
    final_mutual_knn = []
    final_ntk = []
    final_r1 = []

    for ratio in misalignment_levels:
        metrics = all_results[ratio]
        final_cka.append(metrics['cka'][-1] if metrics['cka'] else 0)
        final_mutual_knn.append(metrics['mutual_knn'][-1] if metrics['mutual_knn'] else 0)
        final_ntk.append(metrics['ntk_similarity'][-1] if metrics['ntk_similarity'] else 0)
        final_r1.append(metrics['retrieval_r1'][-1] if metrics['retrieval_r1'] else 0)

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: All metrics vs misalignment
    ax = axes[0, 0]
    ax.plot(misalignment_levels, final_cka, 'o-', label='CKA')
    ax.plot(misalignment_levels, final_mutual_knn, 's-', label='Mutual kNN')
    ax.plot(misalignment_levels, final_ntk, '^-', label='NTK Similarity')
    ax.set_xlabel('Misalignment Ratio')
    ax.set_ylabel('Metric Value')
    ax.set_title('Metrics vs. Misalignment Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Retrieval performance vs misalignment
    ax = axes[0, 1]
    ax.plot(misalignment_levels, final_r1, 'o-', color='green')
    ax.set_xlabel('Misalignment Ratio')
    ax.set_ylabel('Retrieval R@1')
    ax.set_title('Downstream Performance vs. Misalignment')
    ax.grid(True, alpha=0.3)

    # Plot 3: CKA vs Retrieval (does CKA predict performance?)
    ax = axes[1, 0]
    ax.scatter(final_cka, final_r1)
    for i, ratio in enumerate(misalignment_levels):
        ax.annotate(f'{ratio:.0%}', (final_cka[i], final_r1[i]))
    ax.set_xlabel('CKA')
    ax.set_ylabel('Retrieval R@1')
    ax.set_title('CKA vs. Performance')
    ax.grid(True, alpha=0.3)

    # Plot 4: NTK vs Retrieval (does NTK predict performance?)
    ax = axes[1, 1]
    ax.scatter(final_ntk, final_r1)
    for i, ratio in enumerate(misalignment_levels):
        ax.annotate(f'{ratio:.0%}', (final_ntk[i], final_r1[i]))
    ax.set_xlabel('NTK Similarity')
    ax.set_ylabel('Retrieval R@1')
    ax.set_title('NTK Similarity vs. Performance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(config.output_dir) / 'figures' / 'phase1_analysis.png', dpi=150)
    plt.close()

    # Compute correlations
    try:
        from scipy import stats
        cka_corr, cka_p = stats.pearsonr(final_cka, final_r1)
        ntk_corr, ntk_p = stats.pearsonr(final_ntk, final_r1)
    except ImportError:
        # Simple correlation computation if scipy not available
        def simple_corr(x, y):
            x_mean, y_mean = np.mean(x), np.mean(y)
            x_std, y_std = np.std(x), np.std(y)
            return np.mean((x - x_mean) * (y - y_mean)) / (x_std * y_std)

        cka_corr = simple_corr(final_cka, final_r1)
        ntk_corr = simple_corr(final_ntk, final_r1)
        cka_p, ntk_p = 0.0, 0.0

    logging.info(f"\n{'='*60}")
    logging.info("PHASE 1 RESULTS SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"CKA-Performance correlation: r={cka_corr:.3f} (p={cka_p:.3f})")
    logging.info(f"NTK-Performance correlation: r={ntk_corr:.3f} (p={ntk_p:.3f})")

    if abs(ntk_corr) > abs(cka_corr):
        logging.info("\n✓ NTK similarity is a better predictor of performance than CKA")
        logging.info("  This supports the hypothesis that optimization metrics")
        logging.info("  detect semantic misalignment that representation metrics miss.")
    else:
        logging.info("\n✗ CKA performed as well or better than NTK")
        logging.info("  This would challenge the hypothesis.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Create default configuration
    config = ExperimentConfig(
        coco_root="/path/to/coco",  # UPDATE THIS
        output_dir="./phase1_results",
        device="mps",  # or "cuda"
        batch_size=128,  # Reduce if memory-constrained
        num_epochs=15,
    )

    # Run experiment
    results = run_phase1_experiment(config)

    print("\nExperiment complete! Results saved to:", config.output_dir)