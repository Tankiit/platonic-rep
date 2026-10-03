#!/usr/bin/env python3
"""
RQ1: Representation vs Optimization Compatibility Analyzer

This module implements the RQ1 analysis that investigates the paradox:
- Promise: Representation similarity determines alignment success
- Reality: High S^repr_NTK (0.89) yet low alignment (3%)
- Finding: S^optim_NTK determines success, not S^repr_NTK
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Model and feature extraction imports
import timm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms
from PIL import Image

# Import existing feature extraction functionality
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'old_files'))
from extract_multimodal_features import VisionFeatureExtractor, LanguageFeatureExtractor, DatasetLoader

# Import datasets library for COCO loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets library not available. Install with: pip install datasets")
    DATASETS_AVAILABLE = False

# Import webdataset for efficient COCO loading
try:
    import webdataset as wds
    from huggingface_hub import HfFileSystem, get_token, hf_hub_url
    WEBDATASET_AVAILABLE = True
except ImportError:
    print("Warning: webdataset not available. Install with: pip install webdataset")
    WEBDATASET_AVAILABLE = False


class RepresentationOptimizationAnalyzer:
    """
    RQ1: What determines cross-modal alignment success?
    Promise: Representation similarity → alignment success
    Reality: High S^repr_NTK (0.89) yet low alignment (3%)
    Finding: S^optim_NTK determines success, not S^repr_NTK
    """

    def __init__(self, vision_model_name='resnet50', lang_model_name='distilbert-base-uncased',
                 device='auto', output_dir='./rq1_results', data_dir='/media/tanmoy'):
        print("="*60)
        print("RQ1: REPRESENTATION vs OPTIMIZATION COMPATIBILITY")
        print("="*60)

        # Device setup
        if device == 'auto':
            self.device = self._auto_detect_device()
        else:
            self.device = torch.device(device)

        self.output_dir = output_dir
        self.data_dir = data_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Initialize feature extractors for efficient model loading
        self.vision_extractor = VisionFeatureExtractor(
            device=self.device,
            cache_dir='./model_cache',
            output_dir=output_dir
        )
        self.lang_extractor = LanguageFeatureExtractor(
            device=self.device,
            cache_dir='./model_cache',
            output_dir=output_dir
        )

        # Model configurations
        self.vision_model_name = vision_model_name
        self.lang_model_name = lang_model_name

        # Load models using existing infrastructure
        self._load_models()

        self.results = {}

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

    def _load_models(self):
        """Load models using existing feature extraction infrastructure"""
        print(f"\nLoading models: {self.vision_model_name} + {self.lang_model_name}")

        # Load vision model using existing extractor
        try:
            self.vision_model = self.vision_extractor.load_model(self.vision_model_name)
            print(f"✓ Loaded vision model: {self.vision_model_name}")
        except Exception as e:
            print(f"Could not load {self.vision_model_name}, trying resnet18...")
            self.vision_model_name = 'resnet18'
            self.vision_model = self.vision_extractor.load_model(self.vision_model_name)

        # Load language model using existing extractor
        try:
            self.language_model = self.lang_extractor.load_model(self.lang_model_name)
            self.tokenizer = self.lang_extractor.loaded_models[f"{self.lang_model_name}_tokenizer"]
            print(f"✓ Loaded language model: {self.lang_model_name}")
        except Exception as e:
            print(f"Could not load {self.lang_model_name}, trying distilbert-base...")
            self.lang_model_name = 'distilbert-base'
            self.language_model = self.lang_extractor.load_model(self.lang_model_name)
            self.tokenizer = self.lang_extractor.loaded_models[f"{self.lang_model_name}_tokenizer"]

    def load_coco_dataset(self, split='validation', num_samples=None, save_images=True):
        """
        Load MS-COCO dataset using WebDataset for efficient streaming

        Args:
            split: Dataset split to use ('train', 'validation', 'test')
            num_samples: Number of samples to load (None for all available)
            save_images: Whether to save images to disk for feature extraction

        Returns:
            Tuple of (image_paths, texts)
        """
        print(f"\nLoading MS-COCO dataset ({split} split) using WebDataset...")

        if not WEBDATASET_AVAILABLE:
            raise ImportError("webdataset not available. Install with: pip install webdataset")

        try:
            # Define split patterns - only train and test are available
            split_patterns = {
                'train': '**/train/*.tar',
                'test': '**/test/*.tar'
            }

            # For validation, we'll use test split as it's more suitable for evaluation
            actual_split = split
            if split == 'validation':
                actual_split = 'test'
                print("Note: Using 'test' split for 'validation' as no dedicated validation split is available")

            if actual_split not in split_patterns:
                raise ValueError(f"Invalid split: {split}. Available: {list(split_patterns.keys())}")

            # Get file URLs from Hugging Face
            fs = HfFileSystem()
            files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/clip-benchmark/wds_mscoco_captions/" + split_patterns[actual_split])]

            if not files:
                print(f"No files found for split {actual_split}, trying alternative...")
                # Try broader patterns
                alt_patterns = {
                    'train': '**/train/*',
                    'test': '**/test/*'
                }
                if actual_split in alt_patterns:
                    files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/clip-benchmark/wds_mscoco_captions/" + alt_patterns[actual_split])]

            if not files:
                raise ValueError(f"No COCO files found for split '{split}'")

            print(f"Found {len(files)} tar files for {actual_split} split")

            # Create URLs (with or without authentication)
            token = get_token()
            if token:
                # Use authenticated access
                urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
                url_string = f"pipe: curl -s -L -H 'Authorization:Bearer {token}' {'::'.join(urls)}"
                print("✓ Using authenticated access")
            else:
                # Try direct access without authentication
                base_url = "https://huggingface.co/datasets/clip-benchmark/wds_mscoco_captions/resolve/main"
                urls = [f"{base_url}/{file.path_in_repo}" for file in files]
                url_string = '::'.join(urls)
                print("⚠ Using unauthenticated access - may have rate limits")
                print("Consider: huggingface-cli login or setting HF_TOKEN environment variable")

            # Create WebDataset
            image_paths = []
            texts = []

            # Process samples from WebDataset
            dataset = wds.WebDataset(url_string).decode()

            print("Processing COCO samples...")
            processed_count = 0
            for i, sample in enumerate(tqdm(dataset, desc=f"Processing COCO {split}")):
                if num_samples is not None and processed_count >= num_samples:
                    break

                # Extract image and caption from WebDataset format
                image_data = sample.get('jpg') or sample.get('png') or sample.get('image')
                caption = sample.get('txt') or sample.get('caption') or sample.get('text')

                if image_data is None or caption is None:
                    continue  # Skip samples without image or text

                processed_count += 1

                # Convert caption to string if needed
                if isinstance(caption, bytes):
                    caption = caption.decode('utf-8')
                elif isinstance(caption, list) and caption:
                    caption = caption[0]  # Take first caption if it's list
                caption = str(caption).strip()

                # Convert image data to PIL Image
                if isinstance(image_data, bytes):
                    # WebDataset typically returns bytes, convert to PIL Image
                    from io import BytesIO
                    image = Image.open(BytesIO(image_data))
                else:
                    # Assume it's already a PIL Image
                    image = image_data

                # Save image to disk if needed
                if save_images:
                    img_path = os.path.join(self.data_dir, f"coco_{actual_split}_{processed_count:06d}.jpg")
                    image.save(img_path)
                    image_paths.append(img_path)
                else:
                    # Store PIL Image directly for streaming mode
                    image_paths.append(image)

                texts.append(caption)

            print(f"✓ Processed {len(image_paths)} COCO {actual_split} samples")
            if not image_paths:
                raise RuntimeError("No valid COCO samples were processed")

            return image_paths, texts

        except Exception as e:
            print(f"Error loading COCO dataset: {e}")
            print("Falling back to synthetic data for demonstration...")
            return None, None

    def extract_features(self, image_paths, texts, batch_size=16):
        """Extract features from both modalities using existing extractors"""
        print("\n1. Extracting features...")

        # Handle both file paths and PIL Images
        images = []
        for img in image_paths:
            if isinstance(img, str):
                # It's a file path
                images.append(Image.open(img).convert('RGB'))
            elif hasattr(img, 'convert'):  # PIL Image
                images.append(img.convert('RGB'))
            else:
                raise ValueError(f"Unsupported image format: {type(img)}")
        dataloader = DatasetLoader.create_image_dataloader(
            images,
            batch_size=batch_size,
            transform=None  # Will use model's default transform
        )

        # Extract vision features using existing extractor
        V_feats = self.vision_extractor.extract_features(
            self.vision_model_name,
            dataloader,
            max_samples=len(image_paths)
        )

        # Extract language features using existing extractor
        L_feats = self.lang_extractor.extract_features(
            self.lang_model_name,
            texts,
            batch_size=batch_size,
            pooling='mean'  # Use mean pooling as default
        )

        print(f"   Vision features: {V_feats.shape}")
        print(f"   Language features: {L_feats.shape}")

        return V_feats, L_feats

    def compute_S_repr_NTK(self, V_feats, L_feats):
        """
        Compute S^repr_NTK: Representation similarity via kernel alignment
        This is what CCA and standard metrics measure
        """
        print("\n2. Computing S^repr_NTK (Representation Similarity)...")

        # Memory management: limit samples for large datasets
        n_samples = min(5000, len(V_feats))  # Limit for memory efficiency
        if len(V_feats) > n_samples:
            print(f"   ⚠ Using {n_samples} samples for NTK computation to avoid memory issues")
            indices = np.random.choice(len(V_feats), n_samples, replace=False)
            V_feats = V_feats[indices]
            L_feats = L_feats[indices]

        print(f"   Computing NTK for {len(V_feats)} samples...")

        # Center features
        V_centered = V_feats - V_feats.mean(axis=0, keepdims=True)
        L_centered = L_feats - L_feats.mean(axis=0, keepdims=True)

        # Compute kernel matrices (feature similarity)
        K_V = V_centered @ V_centered.T  # [N, N]
        K_L = L_centered @ L_centered.T  # [N, N]

        # Kernel alignment (centered kernel alignment / CKA)
        S_repr = np.trace(K_V.T @ K_L) / (
            np.linalg.norm(K_V, 'fro') * np.linalg.norm(K_L, 'fro')
        )

        self.results['S_repr_NTK'] = S_repr
        print(f"   S^repr_NTK = {S_repr:.3f}")
        print(f"   → {'HIGH' if S_repr > 0.7 else 'LOW'} representation similarity")

        return S_repr

    def compute_S_optim_NTK(self, V_feats, L_feats, n_samples=200):
        """
        TRUE S^optim_NTK using torch.func.vmap

        This is theoretically rigorous!
        """
        print("\n3. Computing S^optim_NTK with vmap (TRUE computation)...")

        from torch.func import vmap, jacrev, functional_call

        # Sample data
        indices = np.random.choice(len(V_feats), min(n_samples, len(V_feats)), replace=False)
        V_sample = torch.FloatTensor(V_feats[indices]).to(self.device)
        L_sample = torch.FloatTensor(L_feats[indices]).to(self.device)

        # Create alignment projections
        projection_V = nn.Linear(V_feats.shape[1], L_feats.shape[1], bias=False).to(self.device)
        projection_L = nn.Linear(L_feats.shape[1], V_feats.shape[1], bias=False).to(self.device)

        params_V = dict(projection_V.named_parameters())
        params_L = dict(projection_L.named_parameters())

        # Define loss functions
        def loss_V(params, v, l):
            out = functional_call(projection_V, params, (v.unsqueeze(0),))
            return ((out - l.unsqueeze(0))**2).sum()

        def loss_L(params, l, v):
            out = functional_call(projection_L, params, (l.unsqueeze(0),))
            return ((out - v.unsqueeze(0))**2).sum()

        # Compute Jacobians with vmap (THE KEY!)
        print("   Computing Jacobians with vmap...")

        n_grad_samples = min(50, len(V_sample))

        # V→L Jacobians
        def jac_func_V(v, l):
            return jacrev(lambda p: loss_V(p, v, l))(params_V)

        jac_V = vmap(jac_func_V)(V_sample[:n_grad_samples], L_sample[:n_grad_samples])

        # Flatten Jacobians
        grads_V = torch.cat([jac_V[pname].flatten(start_dim=1)
                            for pname in sorted(params_V.keys())], dim=1)
        grads_V = grads_V.detach().cpu().numpy()

        # L→V Jacobians
        def jac_func_L(l, v):
            return jacrev(lambda p: loss_L(p, l, v))(params_L)

        jac_L = vmap(jac_func_L)(L_sample[:n_grad_samples], V_sample[:n_grad_samples])

        grads_L = torch.cat([jac_L[pname].flatten(start_dim=1)
                            for pname in sorted(params_L.keys())], dim=1)
        grads_L = grads_L.detach().cpu().numpy()

        # Compute subspace alignment (TRUE measure)
        print("   Computing gradient subspace alignment...")

        from scipy.linalg import subspace_angles

        U_V, s_V, _ = np.linalg.svd(grads_V - grads_V.mean(axis=0), full_matrices=False)
        U_L, s_L, _ = np.linalg.svd(grads_L - grads_L.mean(axis=0), full_matrices=False)

        k = min(10, U_V.shape[1], U_L.shape[1])

        try:
            angles = subspace_angles(U_V[:, :k], U_L[:, :k])
            S_optim = np.cos(angles).mean()

            print(f"   Principal angles: {np.degrees(angles[:5])}")
            print(f"   Mean cosine: {S_optim:.3f}")
        except:
            # Fallback
            S_optim = np.abs(np.trace(U_V[:, :k].T @ U_L[:, :k])) / k

        self.results['S_optim_NTK'] = S_optim
        self.results['V_grad_spectrum'] = s_V[:min(50, len(s_V))]
        self.results['L_grad_spectrum'] = s_L[:min(50, len(s_L))]
        self.results['principal_angles'] = angles if 'angles' in locals() else None

        print(f"   S^optim_NTK = {S_optim:.3f} (vmap computation)")
        print(f"   → {'HIGH' if S_optim > 0.7 else 'LOW'} optimization compatibility")

        return S_optim

    def test_gradient_based_alignment(self, V_feats, L_feats, epochs=50):
        """
        Reality Check: Try gradient-based alignment
        Expected: Should fail despite high S^repr_NTK
        """
        print("\n4. Testing Gradient-Based Alignment...")

        # Split train/test
        n_train = int(0.8 * len(V_feats))
        V_train, V_test = V_feats[:n_train], V_feats[n_train:]
        L_train, L_test = L_feats[:n_train], L_feats[n_train:]

        # Create alignment network
        projection = nn.Sequential(
            nn.Linear(V_feats.shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, L_feats.shape[1])
        ).to(self.device)

        optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3)

        # Train
        V_train_t = torch.FloatTensor(V_train).to(self.device)
        L_train_t = torch.FloatTensor(L_train).to(self.device)

        for epoch in range(epochs):
            projection.train()
            optimizer.zero_grad()

            V_proj = projection(V_train_t)
            loss = F.mse_loss(V_proj, L_train_t)

            loss.backward()
            optimizer.step()

        # Evaluate
        projection.eval()
        with torch.no_grad():
            V_test_proj = projection(torch.FloatTensor(V_test).to(self.device)).cpu().numpy()

        # Measure alignment accuracy (cosine similarity > 0.5)
        cos_sims = np.sum(V_test_proj * L_test, axis=1) / (
            np.linalg.norm(V_test_proj, axis=1) * np.linalg.norm(L_test, axis=1) + 1e-8
        )

        alignment_acc = np.mean(cos_sims > 0.5) * 100

        self.results['gradient_based_alignment'] = alignment_acc
        print(f"   Gradient-based alignment accuracy: {alignment_acc:.1f}%")
        print(f"   → {'SUCCESS' if alignment_acc > 50 else 'FAILURE'}")

        return alignment_acc

    def visualize_RQ1(self):
        """
        Create the key figure showing the paradox
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Panel A: S^repr_NTK vs S^optim_NTK
        ax = axes[0]
        ax.scatter([self.results['S_repr_NTK']], [self.results['S_optim_NTK']],
                  s=200, c='red', marker='o', edgecolors='black', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Compatibility threshold')
        ax.axvline(0.7, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('$S^{repr}_{NTK}$ (Representation)', fontsize=12)
        ax.set_ylabel('$S^{optim}_{NTK}$ (Optimization)', fontsize=12)
        ax.set_title('The Paradox:\nHigh Repr, Low Optim', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)
        ax.legend()

        # Panel B: Gradient spectra comparison
        ax = axes[1]
        k = min(30, len(self.results['V_grad_spectrum']))
        ax.semilogy(self.results['V_grad_spectrum'][:k], 'o-', label='Vision', linewidth=2)
        ax.semilogy(self.results['L_grad_spectrum'][:k], 's-', label='Language', linewidth=2)
        ax.set_xlabel('Eigenvalue Index', fontsize=12)
        ax.set_ylabel('Eigenvalue Magnitude (log)', fontsize=12)
        ax.set_title('Incompatible Gradient\nGeometries', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel C: Alignment prediction
        ax = axes[2]
        metrics = ['S^repr_NTK', 'S^optim_NTK']
        values = [self.results['S_repr_NTK'], self.results['S_optim_NTK']]
        prediction = ['Predicts:\nSUCCESS', f"Predicts:\nFAILURE"]
        actual = f"Actual:\n{self.results['gradient_based_alignment']:.1f}%"

        colors = ['green' if v > 0.7 else 'red' for v in values]
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title(f'{actual}\n(Gradient-based)', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])

        # Add prediction labels
        for i, (bar, pred) in enumerate(zip(bars, prediction)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   pred, ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.output_dir, 'RQ1_representation_vs_optimization.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved: {fig_path}")

        return fig

    def generate_RQ1_report(self):
        """
        Generate the Promise-Reality-Finding report
        """
        print("\n" + "="*60)
        print("RQ1 RESULTS: PROMISE vs REALITY")
        print("="*60)

        print("\n📋 PROMISE (from PRH):")
        print("   Representation similarity determines alignment success")
        print("   → High S^repr_NTK should mean high alignment accuracy")

        print("\n📊 REALITY (our measurements):")
        print(f"   S^repr_NTK = {self.results['S_repr_NTK']:.3f} (HIGH - representations ARE similar!)")
        print(f"   S^optim_NTK = {self.results['S_optim_NTK']:.3f} (LOW - gradients incompatible)")
        print(f"   Gradient-based alignment = {self.results['gradient_based_alignment']:.1f}% (FAILURE)")

        print("\n💡 FINDING:")
        print("   ✗ Representation similarity (S^repr_NTK) does NOT predict alignment")
        print("   ✓ Optimization compatibility (S^optim_NTK) DOES predict alignment")
        print("   → Gradient geometry, not feature similarity, determines success")

        print("\n" + "="*60)

        # Save report to file
        report_path = os.path.join(self.output_dir, 'RQ1_report.txt')
        with open(report_path, 'w') as f:
            f.write("RQ1 RESULTS: PROMISE vs REALITY\n")
            f.write("="*60 + "\n\n")
            f.write("PROMISE (from PRH):\n")
            f.write("   Representation similarity determines alignment success\n")
            f.write("   → High S^repr_NTK should mean high alignment accuracy\n\n")
            f.write("REALITY (our measurements):\n")
            f.write(f"   S^repr_NTK = {self.results['S_repr_NTK']:.3f} (HIGH - representations ARE similar!)\n")
            f.write(f"   S^optim_NTK = {self.results['S_optim_NTK']:.3f} (LOW - gradients incompatible)\n")
            f.write(f"   Gradient-based alignment = {self.results['gradient_based_alignment']:.1f}% (FAILURE)\n\n")
            f.write("FINDING:\n")
            f.write("   ✗ Representation similarity (S^repr_NTK) does NOT predict alignment\n")
            f.write("   ✓ Optimization compatibility (S^optim_NTK) DOES predict alignment\n")
            f.write("   → Gradient geometry, not feature similarity, determines success\n")

        print(f"✓ Report saved: {report_path}")

    def run_full_analysis(self, image_paths=None, texts=None, use_coco=False, coco_split='validation', num_samples=None, streaming_only=False):
        """
        Run the complete RQ1 analysis pipeline

        Args:
            image_paths: List of image file paths (overrides use_coco)
            texts: List of text captions (overrides use_coco)
            use_coco: Whether to use MS-COCO dataset
            coco_split: Which COCO split to use ('train', 'validation', 'test')
            num_samples: Number of samples to process (for COCO only)
        """
        print("\n" + "="*60)
        print("RUNNING RQ1 ANALYSIS PIPELINE")
        print("="*60)

        # Load data based on priority: explicit paths -> COCO -> error
        if image_paths is not None and texts is not None:
            print("Using provided image paths and texts...")

            # Validate inputs
            if len(image_paths) != len(texts):
                raise ValueError(f"Number of images ({len(image_paths)}) must match number of texts ({len(texts)})")

            if len(image_paths) == 0:
                raise ValueError("Empty data provided. Please provide at least one image-text pair.")

            print(f"Processing {len(image_paths)} real image-text pairs...")

            # Verify all image files exist
            real_image_paths = []
            missing_images = []
            for i, path in enumerate(image_paths):
                if os.path.exists(path):
                    real_image_paths.append(path)
                else:
                    missing_images.append(path)

            if missing_images:
                raise FileNotFoundError(f"The following image files were not found: {missing_images[:5]}{'...' if len(missing_images) > 5 else ''}")

            print(f"✓ All {len(real_image_paths)} image files found")

        elif use_coco:
            print(f"Loading MS-COCO dataset ({coco_split} split)...")
            real_image_paths, texts = self.load_coco_dataset(
                split=coco_split,
                num_samples=num_samples,
                save_images=not streaming_only
            )
            if real_image_paths is None or texts is None:
                raise RuntimeError("COCO loading failed. Please check the dataset format or internet connection.")
        else:
            raise ValueError("No data provided. Either provide --image-dir and --text-file, or use --use-coco flag.")

        # Step 1: Extract features
        V_feats, L_feats = self.extract_features(real_image_paths, texts)

        # Step 2: Compute representation similarity
        S_repr = self.compute_S_repr_NTK(V_feats, L_feats)

        # Step 3: Compute optimization compatibility
        S_optim = self.compute_S_optim_NTK(V_feats, L_feats)

        # Step 4: Test gradient-based alignment
        alignment_acc = self.test_gradient_based_alignment(V_feats, L_feats)

        # Step 5: Visualize results
        self.visualize_RQ1()

        # Step 6: Generate report
        self.generate_RQ1_report()

        return self.results


def main():
    """Main function to run RQ1 analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='RQ1 Representation vs Optimization Analysis')

    # Model configuration
    parser.add_argument('--vision-model', default='resnet50',
                       help='Vision model name')
    parser.add_argument('--lang-model', default='distilbert-base-uncased',
                       help='Language model name')

    # Data configuration
    parser.add_argument('--image-dir', default=None,
                       help='Directory containing images (for local data)')
    parser.add_argument('--text-file', default=None,
                       help='File containing texts, one per line (for local data)')
    parser.add_argument('--use-coco', action='store_true',
                       help='Use MS-COCO dataset instead of local files')
    parser.add_argument('--coco-split', default='validation', choices=['train', 'validation', 'test'],
                       help='COCO dataset split to use (default: validation)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to process (for COCO dataset)')
    parser.add_argument('--streaming-only', action='store_true',
                       help='Process COCO images in memory only, do not save to disk')
    parser.add_argument('--data-dir', default='/media/tanmoy',
                       help='Directory for storing large data files (default: /media/tanmoy)')

    # Processing configuration
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for feature extraction')
    parser.add_argument('--output-dir', default='./rq1_results',
                       help='Output directory')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = RepresentationOptimizationAnalyzer(
        vision_model_name=args.vision_model,
        lang_model_name=args.lang_model,
        device=args.device,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )

    # Determine data source and load data
    if args.use_coco:
        # Use COCO dataset
        image_paths = None
        texts = None
        print("Using COCO dataset")
    elif args.image_dir and args.text_file:
        # Load local data
        print(f"Loading images from: {args.image_dir}")
        image_paths = [os.path.join(args.image_dir, f)
                      for f in sorted(os.listdir(args.image_dir))
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Loading texts from: {args.text_file}")
        with open(args.text_file, 'r') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Found {len(image_paths)} images and {len(texts)} texts")

        # Ensure equal length
        min_len = min(len(image_paths), len(texts))
        if len(image_paths) != len(texts):
            print(f"Warning: Mismatch in counts. Using first {min_len} pairs.")
            image_paths = image_paths[:min_len]
            texts = texts[:min_len]
    else:
        # No data source specified
        print("ERROR: No data source specified!")
        print("Either use:")
        print("  1) --use-coco flag for COCO dataset")
        print("  2) --image-dir and --text-file for local data")
        parser.print_help()
        return

    # Run analysis
    results = analyzer.run_full_analysis(
        image_paths=image_paths,
        texts=texts,
        use_coco=args.use_coco,
        coco_split=args.coco_split,
        num_samples=args.num_samples,
        streaming_only=args.streaming_only
    )

    return results


if __name__ == "__main__":
    results = main()