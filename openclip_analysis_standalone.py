# openclip_analysis_standalone.py
"""
Standalone OpenCLIP phase analysis combining NTK stability, cross-modal NTK,
and AGOP metrics from mesoscopic analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
import open_clip
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from sklearn.metrics.pairwise import rbf_kernel


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)


class MultiModalDataset(Dataset):
    """Dataset for vision-language pairs using Hugging Face datasets."""

    def __init__(self, dataset_name="minhuh/prh", subset=None, split="train",
                 preprocess=None, max_samples=1000):
        if subset:
            print(f"Loading dataset: {dataset_name}/{subset}")
            self.dataset = load_dataset(dataset_name, subset, split=split)
        else:
            print(f"Loading dataset: {dataset_name}")
            self.dataset = load_dataset(dataset_name, split=split)

        # Limit samples
        if max_samples and len(self.dataset) > max_samples:
            indices = np.random.choice(len(self.dataset), max_samples, replace=False)
            self.dataset = self.dataset.select(indices)

        self.preprocess = preprocess
        print(f"Dataset loaded: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle image
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            image = Image.new('RGB', (224, 224), color='white')

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Handle text
        if 'text' in item:
            text = item['text']
            # Handle if text is a list
            if isinstance(text, list):
                text = text[0] if text else "An image"
        elif 'caption' in item:
            text = item['caption']
            if isinstance(text, list):
                text = text[0] if text else "An image"
        else:
            text = "An image"

        # Preprocess
        if self.preprocess:
            image = self.preprocess(image)

        return {'image': image, 'text': text}


class OpenCLIPAnalyzer:
    """Standalone OpenCLIP phase analyzer."""

    def __init__(self, output_dir="./results/openclip_standalone/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def compute_ntk_stability(self, features: torch.Tensor, num_samples: int = 500) -> Dict:
        """Compute NTK stability metrics from features."""
        n_samples = min(num_samples, features.shape[0])
        features_subset = features[:n_samples]

        # Normalize
        features_norm = features_subset / (features_subset.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute empirical NTK
        ntk = features_norm @ features_norm.T
        ntk_np = ntk.cpu().numpy()

        # Eigenspectrum analysis
        eigenvalues = eigvalsh(ntk_np)
        eigenvalues = eigenvalues[::-1]  # Descending

        # Stability metrics
        top_eigenvalue = float(eigenvalues[0])
        effective_rank = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

        # Spectral decay
        if len(eigenvalues) >= 10:
            log_indices = np.log(np.arange(1, min(50, len(eigenvalues)) + 1))
            log_eigenvals = np.log(eigenvalues[:min(50, len(eigenvalues))] + 1e-10)
            spectral_decay = -np.polyfit(log_indices, log_eigenvals, 1)[0]
        else:
            spectral_decay = 0

        # Concentration
        eigenvalue_concentration = float(eigenvalues[:10].sum() / eigenvalues.sum())

        return {
            'top_eigenvalue': top_eigenvalue,
            'effective_rank': effective_rank,
            'spectral_decay_rate': spectral_decay,
            'eigenvalue_concentration': eigenvalue_concentration,
            'stability_score': 1.0 / (1.0 + spectral_decay)  # Higher is more stable
        }

    def compute_cross_modal_ntk(self, vision_features: torch.Tensor,
                                text_features: torch.Tensor, num_samples: int = 500) -> Dict:
        """Compute cross-modal NTK between vision and text."""
        n_samples = min(num_samples, vision_features.shape[0], text_features.shape[0])

        vision_subset = vision_features[:n_samples]
        text_subset = text_features[:n_samples]

        # Normalize
        vision_norm = vision_subset / (vision_subset.norm(dim=-1, keepdim=True) + 1e-8)
        text_norm = text_subset / (text_subset.norm(dim=-1, keepdim=True) + 1e-8)

        # Cross-modal kernel
        cross_ntk = vision_norm @ text_norm.T

        # Analyze
        singular_values = torch.linalg.svdvals(cross_ntk)

        alignment_strength = float(singular_values[0])
        alignment_rank = float((singular_values.sum() ** 2) / (singular_values ** 2).sum())
        modal_coupling = float(singular_values[:10].sum() / singular_values.sum())

        # CKA
        cka = self._compute_cka(vision_norm, text_norm)

        return {
            'alignment_strength': alignment_strength,
            'alignment_rank': alignment_rank,
            'modal_coupling': modal_coupling,
            'cross_modal_cka': cka,
            'top_singular_values': singular_values[:20].cpu().tolist()
        }

    def _compute_cka(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute CKA between two feature sets."""
        # Center
        features1 = features1 - features1.mean(dim=0, keepdim=True)
        features2 = features2 - features2.mean(dim=0, keepdim=True)

        # Kernels
        K1 = features1 @ features1.T
        K2 = features2 @ features2.T

        # CKA
        hsic_12 = torch.trace(K1 @ K2)
        hsic_11 = torch.trace(K1 @ K1)
        hsic_22 = torch.trace(K2 @ K2)

        cka = hsic_12 / torch.sqrt(hsic_11 * hsic_22)
        return float(cka)

    def compute_agop_metrics(self, features: torch.Tensor, num_samples: int = 500) -> Dict:
        """Compute AGOP-like phase metrics."""
        n_samples = min(num_samples, features.shape[0])
        features_subset = features[:n_samples].cpu().numpy()

        # Create synthetic gradients
        n_features = features_subset.shape[1]
        n_classes = 100

        # Compute gradient approximation
        gradients = []
        for i in range(n_samples):
            grad = features_subset[i:i+1].T @ np.random.randn(1, n_classes)
            gradients.append(grad.flatten())

        # AGOP matrix
        gradient_matrix = np.stack(gradients)
        agop = gradient_matrix.T @ gradient_matrix / len(gradients)

        # Eigenanalysis
        eigenvalues = eigvalsh(agop)
        eigenvalues = eigenvalues[::-1]

        # Phase determination
        eigenvalues_norm = eigenvalues / (eigenvalues.sum() + 1e-10)
        top_10_concentration = eigenvalues_norm[:10].sum()

        if top_10_concentration > 0.9:
            phase = "condensed"
        elif top_10_concentration > 0.7:
            phase = "critical"
        elif top_10_concentration > 0.5:
            phase = "transitional"
        else:
            phase = "diffuse"

        return {
            'phase': phase,
            'eigenvalue_concentration': float(top_10_concentration),
            'effective_rank': float((eigenvalues.sum() ** 2) / ((eigenvalues ** 2).sum() + 1e-10)),
            'top_eigenvalue_ratio': float(eigenvalues[0] / (eigenvalues[1] + 1e-10))
        }

    def extract_features(self, model, images, texts):
        """Extract features from model."""
        vision_features = []
        text_features = []

        model.eval()
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(images), 32):
                batch_images = images[i:i+32].to(self.device)
                batch_texts = texts[i:i+32]

                # Convert batch_texts to list if it's a tuple
                if isinstance(batch_texts, tuple):
                    batch_texts = list(batch_texts)

                # Vision features
                vision_feat = model.encode_image(batch_images)
                vision_features.append(vision_feat.cpu())

                # Text features
                text_tokens = open_clip.tokenize(batch_texts).to(self.device)
                text_feat = model.encode_text(text_tokens)
                text_features.append(text_feat.cpu())

        vision_features = torch.cat(vision_features, dim=0)
        text_features = torch.cat(text_features, dim=0)

        return vision_features, text_features

    def analyze_model(self, model_name: str, pretrained: str, dataset_name="minhuh/prh",
                     subset=None, max_samples=1000):
        """Complete analysis for one OpenCLIP model."""
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_name} ({pretrained})")
        print(f"{'='*60}")

        # Load model
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        model.eval()

        # Load dataset
        dataset = MultiModalDataset(
            dataset_name=dataset_name,
            subset=subset,
            preprocess=preprocess,
            max_samples=max_samples
        )

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

        # Collect all data
        all_images = []
        all_texts = []

        print("Loading data...")
        for batch in tqdm(dataloader):
            all_images.append(batch['image'])
            all_texts.append(batch['text'])

        all_images = torch.cat(all_images, dim=0)
        all_texts = sum([list(t) for t in all_texts], [])  # Flatten text list

        # Extract features
        print("Extracting features...")
        vision_features, text_features = self.extract_features(model, all_images, all_texts)

        # Compute metrics
        print("Computing NTK stability...")
        vision_ntk = self.compute_ntk_stability(vision_features)
        text_ntk = self.compute_ntk_stability(text_features)

        print("Computing cross-modal NTK...")
        cross_ntk = self.compute_cross_modal_ntk(vision_features, text_features)

        print("Computing AGOP metrics...")
        vision_agop = self.compute_agop_metrics(vision_features)
        text_agop = self.compute_agop_metrics(text_features)

        # Compile results
        results = {
            'model': model_name,
            'pretrained': pretrained,
            'num_samples': len(vision_features),
            'vision_ntk': vision_ntk,
            'text_ntk': text_ntk,
            'cross_ntk': cross_ntk,
            'vision_agop': vision_agop,
            'text_agop': text_agop,
            'vision_phase': vision_agop['phase'],
            'text_phase': text_agop['phase'],
            'cross_modal_alignment': cross_ntk['alignment_strength']
        }

        # Save
        output_file = self.output_dir / f"{model_name.replace('/', '_')}_{pretrained}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        print(f"\nResults saved to: {output_file}")

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results):
        """Print analysis summary."""
        print("\n" + "-"*50)
        print("Analysis Summary")
        print("-"*50)
        print(f"Vision Phase: {results['vision_phase']}")
        print(f"Text Phase: {results['text_phase']}")
        print(f"Vision NTK Effective Rank: {results['vision_ntk']['effective_rank']:.2f}")
        print(f"Text NTK Effective Rank: {results['text_ntk']['effective_rank']:.2f}")
        print(f"Cross-Modal Alignment: {results['cross_ntk']['alignment_strength']:.4f}")
        print(f"Cross-Modal CKA: {results['cross_ntk']['cross_modal_cka']:.4f}")
        print(f"Modal Coupling: {results['cross_ntk']['modal_coupling']:.4f}")

    def compare_models(self, model_configs, dataset_name="minhuh/prh",
                      subset=None, max_samples=1000):
        """Compare multiple models."""
        all_results = {}

        for model_name, pretrained in model_configs:
            try:
                results = self.analyze_model(
                    model_name, pretrained, dataset_name, subset, max_samples
                )
                all_results[f"{model_name}_{pretrained}"] = results
            except Exception as e:
                print(f"Error analyzing {model_name}: {e}")

            # Clear cache
            torch.cuda.empty_cache()

        # Generate comparison
        self._generate_comparison(all_results)

        return all_results

    def _generate_comparison(self, all_results):
        """Generate comparison visualization and summary."""
        if len(all_results) < 2:
            return

        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)

        # Create comparison table
        models = list(all_results.keys())

        print(f"\n{'Model':<30} {'V-Phase':<12} {'T-Phase':<12} {'Alignment':<10} {'CKA':<10}")
        print("-"*74)

        for model_key, results in all_results.items():
            model_short = model_key[:28] + '..' if len(model_key) > 30 else model_key
            print(f"{model_short:<30} {results['vision_phase']:<12} "
                  f"{results['text_phase']:<12} "
                  f"{results['cross_modal_alignment']:<10.4f} "
                  f"{results['cross_ntk']['cross_modal_cka']:<10.4f}")

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Alignment comparison
        ax = axes[0, 0]
        alignments = [r['cross_modal_alignment'] for r in all_results.values()]
        model_names = [k.split('_')[0] for k in all_results.keys()]
        ax.bar(range(len(alignments)), alignments)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_ylabel('Alignment Strength')
        ax.set_title('Cross-Modal Alignment')

        # CKA comparison
        ax = axes[0, 1]
        ckas = [r['cross_ntk']['cross_modal_cka'] for r in all_results.values()]
        ax.bar(range(len(ckas)), ckas)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_ylabel('CKA')
        ax.set_title('Cross-Modal CKA')

        # NTK effective rank
        ax = axes[1, 0]
        vision_ranks = [r['vision_ntk']['effective_rank'] for r in all_results.values()]
        text_ranks = [r['text_ntk']['effective_rank'] for r in all_results.values()]
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, vision_ranks, width, label='Vision')
        ax.bar(x + width/2, text_ranks, width, label='Text')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_ylabel('Effective Rank')
        ax.set_title('NTK Effective Rank')
        ax.legend()

        # Phase distribution
        ax = axes[1, 1]
        vision_phases = [r['vision_phase'] for r in all_results.values()]
        text_phases = [r['text_phase'] for r in all_results.values()]
        phase_types = list(set(vision_phases + text_phases))
        phase_counts_v = [vision_phases.count(p) for p in phase_types]
        phase_counts_t = [text_phases.count(p) for p in phase_types]

        x = np.arange(len(phase_types))
        ax.bar(x - width/2, phase_counts_v, width, label='Vision')
        ax.bar(x + width/2, phase_counts_t, width, label='Text')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_types)
        ax.set_ylabel('Count')
        ax.set_title('Phase Distribution')
        ax.legend()

        plt.suptitle('OpenCLIP Model Comparison')
        plt.tight_layout()

        output_file = self.output_dir / 'model_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")
        plt.show()


def main():
    """Main function."""

    analyzer = OpenCLIPAnalyzer()

    # Models to analyze
    model_configs = [
        ("ViT-B-32", "openai"),
        ("ViT-B-32", "laion2b_s34b_b79k"),
        # ("ViT-B-16", "openai"),
        # ("ViT-L-14", "openai"),
        # ("RN50", "openai"),
    ]

    # Run comparison
    results = analyzer.compare_models(
        model_configs,
        dataset_name="minhuh/prh",
        subset=None,  # Use default config
        max_samples=500  # Use smaller sample for testing
    )

    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved in: {analyzer.output_dir}")
    print("="*60)

    return results


if __name__ == "__main__":
    main()