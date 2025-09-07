#!/usr/bin/env python3
"""
Example script for multi-model analysis across different architectures
"""

from multi_model_analysis import MultiModelAnalyzer

def main():
    """Example usage of multi-model analysis"""
    
    # Initialize analyzer
    analyzer = MultiModelAnalyzer(output_dir="./results/example_analysis/")
    
    # Define models to analyze (from your request)
    models = [
        'resnet18',           # ResNet architecture
        'vit_base_patch16_224',  # Vision Transformer
        'convnext_tiny',      # ConvNeXt (CNN)
        'mlp_mixer_b16_224'   # MLP-Mixer
    ]
    
    # Define datasets
    datasets = ['cifar10', 'cifar100', 'svhn']
    
    print("=== Multi-Model Analysis Example ===")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Output directory: {analyzer.output_dir}")
    
    # Run comprehensive analysis
    print("\nStarting analysis...")
    results = analyzer.run_comprehensive_analysis(
        models=models,
        datasets=datasets,
        pretrained=True,
        device=None  # Will use auto-detected device
    )
    
    print("\n=== Analysis Complete ===")
    print("Results saved to:", analyzer.output_dir)
    
    # Print summary
    for dataset in results:
        print(f"\n{dataset.upper()}:")
        for model in results[dataset]:
            if 'macroscopic' in results[dataset][model]:
                summary = results[dataset][model]['macroscopic'].get('information_flow', {}).get('summary', {})
                compression = summary.get('total_compression', 0)
                task_info = summary.get('total_task_info_gain', 0)
                print(f"  {model}: Compression={compression:.3f}, Task Info={task_info:.3f}")
    
    return results

if __name__ == "__main__":
    main()
