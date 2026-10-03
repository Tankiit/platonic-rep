# run_openclip_analysis.py
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
import open_clip
from openclip_phase_analysis import OpenCLIPPhaseAnalyzer
from tqdm import tqdm
import json
from pathlib import Path


class MultiModalDataset(Dataset):
    """Dataset for vision-language pairs using Hugging Face datasets."""

    def __init__(self, dataset_name="minhuh/prh", subset="wit_1024", split="train",
                 preprocess=None, max_samples=5000):
        """
        Initialize dataset for OpenCLIP analysis.

        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset/configuration
            split: Train/val/test split
            preprocess: Image preprocessing function from OpenCLIP
            max_samples: Maximum number of samples to load
        """
        print(f"Loading dataset: {dataset_name}/{subset}")
        self.dataset = load_dataset(dataset_name, subset, split=split)

        # Limit samples for memory efficiency
        if max_samples and len(self.dataset) > max_samples:
            indices = np.random.choice(len(self.dataset), max_samples, replace=False)
            self.dataset = self.dataset.select(indices)

        self.preprocess = preprocess
        print(f"Dataset loaded: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle different dataset formats
        if 'image' in item:
            image = item['image']
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
        elif 'img' in item:
            image = item['img']
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
        else:
            # Create dummy image if not available
            image = Image.new('RGB', (224, 224), color='white')

        # Handle text/caption
        if 'text' in item:
            text = item['text']
        elif 'caption' in item:
            text = item['caption']
        elif 'label' in item:
            # For classification datasets, use label as text
            text = f"This is a {item['label']}"
        else:
            text = "An image"

        # Apply preprocessing
        if self.preprocess:
            image = self.preprocess(image)
        else:
            # Default transform if none provided
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)

        return {
            'image': image,
            'text': text
        }


def create_dataloaders(dataset_name="minhuh/prh", subset="wit_1024",
                      batch_size=32, max_samples=5000, preprocess=None):
    """
    Create dataloaders for OpenCLIP analysis.

    Returns:
        Dictionary with 'images', 'texts', and 'paired' dataloaders
    """
    # Create dataset
    dataset = MultiModalDataset(
        dataset_name=dataset_name,
        subset=subset,
        preprocess=preprocess,
        max_samples=max_samples
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Convert to format expected by analyzer
    # We'll create separate iterators for images and texts
    images_list = []
    texts_list = []
    paired_list = []

    print("Preparing dataloaders...")
    for batch in tqdm(dataloader):
        images_list.append(batch['image'])
        texts_list.append(batch['text'])
        paired_list.append(batch)

    return {
        'images': images_list,
        'texts': texts_list,
        'paired': paired_list
    }


def analyze_openclip_models(models_to_analyze=None, dataset_config=None):
    """
    Run comprehensive OpenCLIP analysis on specified models.

    Args:
        models_to_analyze: List of (model_name, pretrained) tuples
        dataset_config: Dict with dataset configuration
    """

    # Default models to analyze
    if models_to_analyze is None:
        models_to_analyze = [
            ("ViT-B-32", "laion2b_s34b_b79k"),
            ("ViT-B-32", "openai"),
            ("ViT-B-16", "laion2b_s34b_b88k"),
            ("ViT-L-14", "laion2b_s32b_b82k"),
            # ("ViT-g-14", "laion2b_s12b_b42k"),  # Very large model
            ("convnext_base_w", "laion2b_s13b_b82k"),
            ("RN50", "openai"),
        ]

    # Default dataset configuration
    if dataset_config is None:
        dataset_config = {
            'dataset_name': 'minhuh/prh',
            'subset': 'wit_1024',
            'batch_size': 32,
            'max_samples': 2000  # Reduced for faster processing
        }

    # Initialize analyzer
    analyzer = OpenCLIPPhaseAnalyzer()

    # Results storage
    all_results = {}

    print("=" * 70)
    print("OpenCLIP Comprehensive Phase Analysis")
    print("=" * 70)
    print(f"Dataset: {dataset_config['dataset_name']}/{dataset_config['subset']}")
    print(f"Models to analyze: {len(models_to_analyze)}")
    print("=" * 70)

    for model_name, pretrained in models_to_analyze:
        print(f"\n{'='*70}")
        print(f"Analyzing: {model_name} ({pretrained})")
        print(f"{'='*70}")

        try:
            # Load model to get preprocessor
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained
            )
            del model  # Free memory

            # Create dataloaders with model-specific preprocessing
            print("Creating dataloaders...")
            dataloaders = create_dataloaders(
                dataset_name=dataset_config['dataset_name'],
                subset=dataset_config['subset'],
                batch_size=dataset_config['batch_size'],
                max_samples=dataset_config['max_samples'],
                preprocess=preprocess
            )

            # Run analysis
            print("Running phase analysis...")
            results = analyzer.analyze_clip_checkpoint(
                model_name=model_name,
                pretrained=pretrained,
                dataloader=dataloaders
            )

            # Store results
            model_key = f"{model_name}_{pretrained}"
            all_results[model_key] = results

            # Print summary
            print_analysis_summary(results)

        except Exception as e:
            print(f"Error analyzing {model_name} ({pretrained}): {e}")
            continue

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Generate comparison report
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("Comparative Analysis")
        print("=" * 70)
        generate_comparison_report(all_results)

    return all_results


def print_analysis_summary(results):
    """Print a formatted summary of analysis results."""
    print("\n" + "-" * 50)
    print("Analysis Summary")
    print("-" * 50)

    # Phase classification
    print(f"Vision Encoder Phase: {results['vision_phase']}")
    print(f"Text Encoder Phase: {results['text_phase']}")

    # NTK metrics
    print(f"\nNTK Stability Metrics:")
    print(f"  Vision Effective Rank: {results['vision_ntk']['effective_rank']:.2f}")
    print(f"  Text Effective Rank: {results['text_ntk']['effective_rank']:.2f}")
    print(f"  Vision Spectral Decay: {results['vision_ntk']['spectral_decay_rate']:.3f}")
    print(f"  Text Spectral Decay: {results['text_ntk']['spectral_decay_rate']:.3f}")

    # Cross-modal alignment
    print(f"\nCross-Modal Alignment:")
    print(f"  Alignment Strength: {results['cross_ntk']['alignment_strength']:.4f}")
    print(f"  Modal Coupling: {results['cross_ntk']['modal_coupling']:.4f}")
    print(f"  Cross-Modal CKA: {results['cross_ntk']['cross_modal_cka']:.4f}")

    # AGOP metrics
    print(f"\nAGOP Phase Indicators:")
    print(f"  Vision AGOP Phase: {results['vision_agop']['phase']}")
    print(f"  Text AGOP Phase: {results['text_agop']['phase']}")
    print(f"  AGOP Ratio: {results['agop_ratio']:.3f}")

    # Mesoscopic summary
    if 'vision_mesoscopic' in results and results['vision_mesoscopic']:
        print(f"\nMesoscopic Analysis:")
        if 'ntk' in results['vision_mesoscopic']:
            vision_conv = results['vision_mesoscopic'].get('evolution', {}).get('convergence_metrics', {})
            text_conv = results['text_mesoscopic'].get('evolution', {}).get('convergence_metrics', {})
            if vision_conv:
                print(f"  Vision Convergence Rate: {vision_conv.get('convergence_rate', 0):.3f}")
            if text_conv:
                print(f"  Text Convergence Rate: {text_conv.get('convergence_rate', 0):.3f}")


def generate_comparison_report(all_results):
    """Generate and print a comparison report across models."""

    # Extract key metrics for comparison
    comparison_data = []

    for model_key, results in all_results.items():
        row = {
            'Model': model_key,
            'V-Phase': results['vision_phase'],
            'T-Phase': results['text_phase'],
            'V-Rank': results['vision_ntk']['effective_rank'],
            'T-Rank': results['text_ntk']['effective_rank'],
            'Alignment': results['cross_ntk']['alignment_strength'],
            'CKA': results['cross_ntk']['cross_modal_cka'],
            'Coupling': results['cross_ntk']['modal_coupling']
        }
        comparison_data.append(row)

    # Sort by alignment strength
    comparison_data.sort(key=lambda x: x['Alignment'], reverse=True)

    # Print table
    print("\nModel Comparison Table:")
    print("-" * 100)
    print(f"{'Model':<30} {'V-Phase':<10} {'T-Phase':<10} {'V-Rank':<8} {'T-Rank':<8} {'Align':<8} {'CKA':<8} {'Coupling':<8}")
    print("-" * 100)

    for row in comparison_data:
        model_short = row['Model'][:28] + '..' if len(row['Model']) > 30 else row['Model']
        print(f"{model_short:<30} {row['V-Phase']:<10} {row['T-Phase']:<10} "
              f"{row['V-Rank']:<8.1f} {row['T-Rank']:<8.1f} "
              f"{row['Alignment']:<8.4f} {row['CKA']:<8.4f} {row['Coupling']:<8.4f}")

    print("-" * 100)

    # Find best models for different criteria
    print("\nBest Models by Metric:")
    print(f"  Highest Cross-Modal Alignment: {comparison_data[0]['Model']} ({comparison_data[0]['Alignment']:.4f})")

    best_cka = max(comparison_data, key=lambda x: x['CKA'])
    print(f"  Highest Cross-Modal CKA: {best_cka['Model']} ({best_cka['CKA']:.4f})")

    best_coupling = max(comparison_data, key=lambda x: x['Coupling'])
    print(f"  Highest Modal Coupling: {best_coupling['Model']} ({best_coupling['Coupling']:.4f})")

    # Phase distribution
    print("\nPhase Distribution:")
    vision_phases = {}
    text_phases = {}
    for row in comparison_data:
        vision_phases[row['V-Phase']] = vision_phases.get(row['V-Phase'], 0) + 1
        text_phases[row['T-Phase']] = text_phases.get(row['T-Phase'], 0) + 1

    print("  Vision Encoder Phases:", vision_phases)
    print("  Text Encoder Phases:", text_phases)

    # Save comparison to file
    output_dir = Path("./results/openclip_analysis/")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_file = output_dir / "model_comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'comparison_table': comparison_data,
            'vision_phase_distribution': vision_phases,
            'text_phase_distribution': text_phases,
            'best_models': {
                'alignment': {'model': comparison_data[0]['Model'],
                            'score': comparison_data[0]['Alignment']},
                'cka': {'model': best_cka['Model'],
                       'score': best_cka['CKA']},
                'coupling': {'model': best_coupling['Model'],
                           'score': best_coupling['Coupling']}
            }
        }, f, indent=2)

    print(f"\nComparison saved to: {comparison_file}")


def main():
    """Main function to run OpenCLIP analysis."""

    # Configure analysis
    models_to_analyze = [
        # Small to medium models for faster testing
        ("ViT-B-32", "openai"),
        ("ViT-B-32", "laion2b_s34b_b79k"),
        ("ViT-B-16", "openai"),
        # ("ViT-B-16", "laion2b_s34b_b88k"),
        # ("ViT-L-14", "openai"),
        # ("RN50", "openai"),
        # ("RN101", "openai"),
    ]

    dataset_config = {
        'dataset_name': 'minhuh/prh',
        'subset': 'wit_1024',
        'batch_size': 32,
        'max_samples': 1000  # Use smaller sample for testing
    }

    # Run analysis
    results = analyze_openclip_models(
        models_to_analyze=models_to_analyze,
        dataset_config=dataset_config
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Analyzed {len(results)} models")
    print("Results saved in: ./results/openclip_analysis/")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()