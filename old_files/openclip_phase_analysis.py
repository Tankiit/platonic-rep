# openclip_phase_analysis.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
import open_clip
from mesoscopic import MesoscopicAnalysis
from phase_analyzer import PhaseAnalyzer

class OpenCLIPPhaseAnalyzer:
    """
    Comprehensive phase analysis for OpenCLIP models combining:
    - NTK stability analysis
    - Cross-modal NTK computation
    - AGOP metrics
    - Mesoscopic dynamics
    """

    def __init__(self, output_dir="./results/openclip_analysis/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mesoscopic_analyzer = MesoscopicAnalysis()
        self.phase_analyzer = PhaseAnalyzer()

    def load_openclip_model(self, model_name: str, pretrained: str = 'laion2b_s34b_b79k'):
        """Load OpenCLIP model and preprocessor."""
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        model.eval()
        return model, preprocess

    def extract_features(self, model, dataloader, modality='vision'):
        """Extract features from vision or text encoder."""
        features_by_layer = []

        with torch.no_grad():
            if modality == 'vision':
                # Extract vision features at different depths
                encoder = model.visual

                # Get features from different transformer blocks
                for block_idx in [0, len(encoder.transformer.resblocks)//2, -1]:
                    block_features = []

                    # Hook to extract intermediate features
                    activation = {}
                    def get_activation(name):
                        def hook(model, input, output):
                            activation[name] = output.detach()
                        return hook

                    if block_idx >= 0:
                        hook_handle = encoder.transformer.resblocks[block_idx].register_forward_hook(
                            get_activation(f'block_{block_idx}')
                        )

                    # Process batch
                    for batch in dataloader:
                        if isinstance(batch, dict):
                            images = batch['image'].to(self.device)
                        else:
                            images = batch.to(self.device)

                        # Forward pass
                        _ = encoder(images)

                        # Extract features
                        if block_idx >= 0:
                            feat = activation[f'block_{block_idx}']
                            # Take CLS token
                            feat = feat[:, 0, :]
                        else:
                            feat = encoder(images)

                        block_features.append(feat.cpu())

                    if block_idx >= 0:
                        hook_handle.remove()

                    features_by_layer.append(torch.cat(block_features, dim=0))

            else:  # text modality
                encoder = model.encode_text

                # Extract text features
                text_features = []
                for batch in dataloader:
                    if isinstance(batch, dict):
                        texts = batch['text']
                    else:
                        texts = batch

                    # Tokenize and encode
                    tokens = open_clip.tokenize(texts).to(self.device)
                    features = encoder(tokens)
                    text_features.append(features.cpu())

                features_by_layer.append(torch.cat(text_features, dim=0))

        return torch.stack(features_by_layer, dim=1) if len(features_by_layer) > 1 else features_by_layer[0].unsqueeze(1)

    def compute_ntk_stability(self, features: torch.Tensor, num_samples: int = 1000) -> Dict:
        """
        Compute NTK stability metrics from features.
        Integrates with mesoscopic analysis.
        """
        n_samples = min(num_samples, features.shape[0])
        features_subset = features[:n_samples]

        # Normalize features
        features_norm = features_subset / (features_subset.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute empirical NTK
        ntk = features_norm @ features_norm.T

        # Get eigenspectrum
        eigenvalues = torch.linalg.eigvalsh(ntk)
        eigenvalues = eigenvalues.flip(0)  # Descending order

        # Compute stability metrics
        top_eigenvalue = eigenvalues[0].item()
        effective_rank = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

        # Spectral decay analysis (from mesoscopic)
        spectral_decay = self.mesoscopic_analyzer.compute_spectral_decay(
            eigenvalues.numpy()
        )

        # Kernel alignment
        kernel_alignment = self.mesoscopic_analyzer.compute_kernel_target_alignment(
            ntk.numpy()
        )

        return {
            'top_eigenvalue': top_eigenvalue,
            'effective_rank': effective_rank.item(),
            'spectral_decay_rate': spectral_decay,
            'kernel_alignment': kernel_alignment,
            'eigenvalue_concentration': (eigenvalues[:10].sum() / eigenvalues.sum()).item(),
            'eigenvalue_gap': ((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]).item()
        }

    def compute_cross_modal_ntk(self, vision_features: torch.Tensor,
                                text_features: torch.Tensor,
                                num_samples: int = 1000) -> Dict:
        """
        Compute cross-modal NTK between vision and text features.
        """
        n_samples = min(num_samples, vision_features.shape[0], text_features.shape[0])

        vision_subset = vision_features[:n_samples]
        text_subset = text_features[:n_samples]

        # Normalize
        vision_norm = vision_subset / (vision_subset.norm(dim=-1, keepdim=True) + 1e-8)
        text_norm = text_subset / (text_subset.norm(dim=-1, keepdim=True) + 1e-8)

        # Cross-modal kernel
        cross_ntk = vision_norm @ text_norm.T

        # Analyze cross-modal alignment
        singular_values = torch.linalg.svdvals(cross_ntk)

        # Compute alignment metrics
        alignment_strength = singular_values[0].item()
        alignment_rank = ((singular_values.sum() ** 2) / (singular_values ** 2).sum()).item()

        # Modal coupling (how well modalities predict each other)
        modal_coupling = (singular_values[:10].sum() / singular_values.sum()).item()

        return {
            'alignment_strength': alignment_strength,
            'alignment_rank': alignment_rank,
            'modal_coupling': modal_coupling,
            'top_singular_values': singular_values[:20].tolist(),
            'cross_modal_cka': self._compute_cross_modal_cka(vision_norm, text_norm)
        }

    def _compute_cross_modal_cka(self, features1: torch.Tensor,
                                 features2: torch.Tensor) -> float:
        """Compute CKA between two feature sets."""
        # Center features
        features1 = features1 - features1.mean(dim=0, keepdim=True)
        features2 = features2 - features2.mean(dim=0, keepdim=True)

        # Compute kernels
        K1 = features1 @ features1.T
        K2 = features2 @ features2.T

        # CKA
        hsic_12 = torch.trace(K1 @ K2)
        hsic_11 = torch.trace(K1 @ K1)
        hsic_22 = torch.trace(K2 @ K2)

        cka = hsic_12 / torch.sqrt(hsic_11 * hsic_22)
        return cka.item()

    def compute_agop(self, features: torch.Tensor, num_samples: int = 1000) -> Dict:
        """
        Compute AGOP (Average Gradient Outer Product) metrics.
        """
        # Use phase analyzer's AGOP computation
        agop_results = self.phase_analyzer.compute_agop_from_features(
            features.numpy(),
            n_samples=num_samples
        )
        return agop_results

    def classify_phase(self, ntk_metrics: Dict) -> str:
        """
        Classify the phase based on NTK metrics.
        """
        eigenvalue_concentration = ntk_metrics.get('eigenvalue_concentration', 0)
        effective_rank = ntk_metrics.get('effective_rank', 0)

        # Phase classification based on concentration and rank
        if eigenvalue_concentration > 0.8 and effective_rank < 50:
            return "condensed"
        elif eigenvalue_concentration > 0.6:
            return "critical"
        elif effective_rank > 200:
            return "diffuse"
        else:
            return "transitional"

    def analyze_clip_checkpoint(self, model_name: str, pretrained: str,
                                dataloader: Dict) -> Dict:
        """
        Complete phase analysis for an OpenCLIP checkpoint.

        Args:
            model_name: OpenCLIP model name (e.g., 'ViT-B-32')
            pretrained: Pretrained weights identifier
            dataloader: Dict with 'images', 'texts', and 'paired' dataloaders
        """
        print(f"\nAnalyzing OpenCLIP model: {model_name} ({pretrained})")

        # Load model
        model, preprocess = self.load_openclip_model(model_name, pretrained)

        # Extract features
        print("Extracting vision features...")
        vision_features = self.extract_features(model, dataloader['images'], 'vision')

        print("Extracting text features...")
        text_features = self.extract_features(model, dataloader['texts'], 'text')

        # Ensure same number of samples for cross-modal analysis
        min_samples = min(vision_features.shape[0], text_features.shape[0])
        vision_features = vision_features[:min_samples]
        text_features = text_features[:min_samples]

        # Stack into format for mesoscopic analysis [N, L, D]
        # L = number of layers analyzed
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)

        print("Computing NTK stability...")
        # Compute NTK for each modality (using last layer)
        vision_ntk = self.compute_ntk_stability(vision_features[:, -1, :])
        text_ntk = self.compute_ntk_stability(text_features[:, -1, :])

        print("Computing cross-modal NTK...")
        cross_ntk = self.compute_cross_modal_ntk(
            vision_features[:, -1, :],
            text_features[:, -1, :]
        )

        print("Computing AGOP metrics...")
        vision_agop = self.compute_agop(vision_features[:, -1, :])
        text_agop = self.compute_agop(text_features[:, -1, :])

        # Run mesoscopic analysis on features
        print("Running mesoscopic analysis...")
        vision_mesoscopic = self.mesoscopic_analyzer.analyze_model_from_features(
            {'feats': vision_features},
            f"{model_name}_vision",
            pretrained
        )
        text_mesoscopic = self.mesoscopic_analyzer.analyze_model_from_features(
            {'feats': text_features},
            f"{model_name}_text",
            pretrained
        )

        # Compile results
        results = {
            'model': model_name,
            'pretrained': pretrained,
            'vision_ntk': vision_ntk,
            'text_ntk': text_ntk,
            'cross_ntk': cross_ntk,
            'vision_agop': vision_agop,
            'text_agop': text_agop,
            'agop_ratio': vision_agop['top_eigenvalue_ratio'] / (text_agop['top_eigenvalue_ratio'] + 1e-10),
            'vision_phase': self.classify_phase(vision_ntk),
            'text_phase': self.classify_phase(text_ntk),
            'vision_mesoscopic': vision_mesoscopic,
            'text_mesoscopic': text_mesoscopic,
            'cross_modal_alignment': {
                'ntk_alignment': cross_ntk['alignment_strength'],
                'modal_coupling': cross_ntk['modal_coupling'],
                'cka': cross_ntk['cross_modal_cka']
            }
        }

        # Save results
        output_path = self.output_dir / f"{model_name}_{pretrained}_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        print(f"Results saved to {output_path}")

        return results

    def compare_openclip_models(self, model_configs: List[Tuple[str, str]],
                                dataloader: Dict) -> Dict:
        """
        Compare multiple OpenCLIP models.

        Args:
            model_configs: List of (model_name, pretrained) tuples
            dataloader: Shared dataloader for all models
        """
        all_results = {}

        for model_name, pretrained in tqdm(model_configs, desc="Analyzing models"):
            try:
                results = self.analyze_clip_checkpoint(model_name, pretrained, dataloader)
                all_results[f"{model_name}_{pretrained}"] = results
            except Exception as e:
                print(f"Error analyzing {model_name} ({pretrained}): {e}")
                continue

        # Generate comparison summary
        comparison = self._generate_comparison_summary(all_results)

        # Save comparison
        comparison_path = self.output_dir / "model_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        return comparison

    def _generate_comparison_summary(self, all_results: Dict) -> Dict:
        """Generate summary comparison across models."""
        summary = {
            'models': list(all_results.keys()),
            'ntk_comparison': {},
            'phase_distribution': {},
            'cross_modal_strength': {}
        }

        for model_key, results in all_results.items():
            # NTK comparison
            summary['ntk_comparison'][model_key] = {
                'vision_stability': results['vision_ntk']['effective_rank'],
                'text_stability': results['text_ntk']['effective_rank'],
                'cross_modal_alignment': results['cross_ntk']['alignment_strength']
            }

            # Phase distribution
            summary['phase_distribution'][model_key] = {
                'vision': results['vision_phase'],
                'text': results['text_phase']
            }

            # Cross-modal strength
            summary['cross_modal_strength'][model_key] = results['cross_modal_alignment']

        return summary


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


def create_dummy_dataloader(batch_size=32, num_batches=10, image_size=224, seq_length=77):
    """Create dummy dataloaders for testing."""
    import torchvision.transforms as transforms

    # Vision dataloader (dummy images)
    images = []
    for _ in range(num_batches):
        batch = torch.randn(batch_size, 3, image_size, image_size)
        images.append({'image': batch})

    # Text dataloader (dummy text)
    texts = []
    sample_texts = [
        "A photo of a cat",
        "A dog playing in the park",
        "Beautiful sunset over mountains",
        "Modern architecture building",
        "Fresh fruits and vegetables"
    ] * (batch_size // 5 + 1)

    for _ in range(num_batches):
        batch = sample_texts[:batch_size]
        texts.append({'text': batch})

    # Paired dataloader
    paired = []
    for img_batch, txt_batch in zip(images, texts):
        paired.append({
            'image': img_batch['image'],
            'text': txt_batch['text']
        })

    return {
        'images': images,
        'texts': texts,
        'paired': paired
    }


def main():
    """Example usage of OpenCLIP phase analysis."""

    # Initialize analyzer
    analyzer = OpenCLIPPhaseAnalyzer()

    # Create dummy dataloader (replace with real data)
    dataloader = create_dummy_dataloader()

    # Example 1: Analyze single model
    print("=" * 60)
    print("Analyzing single OpenCLIP model")
    print("=" * 60)

    results = analyzer.analyze_clip_checkpoint(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        dataloader=dataloader
    )

    # Print key results
    print("\n" + "=" * 60)
    print("Analysis Results Summary")
    print("=" * 60)
    print(f"Model: {results['model']} ({results['pretrained']})")
    print(f"Vision Phase: {results['vision_phase']}")
    print(f"Text Phase: {results['text_phase']}")
    print(f"Vision NTK Effective Rank: {results['vision_ntk']['effective_rank']:.2f}")
    print(f"Text NTK Effective Rank: {results['text_ntk']['effective_rank']:.2f}")
    print(f"Cross-Modal Alignment: {results['cross_ntk']['alignment_strength']:.4f}")
    print(f"Modal Coupling: {results['cross_ntk']['modal_coupling']:.4f}")
    print(f"Cross-Modal CKA: {results['cross_ntk']['cross_modal_cka']:.4f}")

    # Example 2: Compare multiple models
    print("\n" + "=" * 60)
    print("Comparing multiple OpenCLIP models")
    print("=" * 60)

    model_configs = [
        ("ViT-B-32", "laion2b_s34b_b79k"),
        ("ViT-B-32", "openai"),
        # Add more models as needed
    ]

    comparison = analyzer.compare_openclip_models(model_configs, dataloader)

    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    for model_key in comparison['models']:
        print(f"\n{model_key}:")
        print(f"  Vision Phase: {comparison['phase_distribution'][model_key]['vision']}")
        print(f"  Text Phase: {comparison['phase_distribution'][model_key]['text']}")
        print(f"  Cross-Modal Alignment: {comparison['ntk_comparison'][model_key]['cross_modal_alignment']:.4f}")


if __name__ == "__main__":
    main()