#!/usr/bin/env python3
"""
Comprehensive Multi-Model Analysis across Multiple Datasets
Saves all intermediate files with organized folder structure
"""

import os
import time
from datetime import datetime
from pathlib import Path
import json
from multi_model_analysis import MultiModelAnalyzer

def run_comprehensive_analysis():
    """Run comprehensive analysis across multiple models and datasets"""
    
    # Configuration
    models = [
        'resnet18',           # ResNet architecture
        'vit_base_patch16_224',  # Vision Transformer
        'convnext_tiny',      # ConvNeXt (CNN)
        'mlp_mixer_b16_224'   # MLP-Mixer
    ]
    
    datasets = ['cifar10', 'cifar100', 'svhn']
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"./results/comprehensive_analysis_{timestamp}/"
    
    print("=== Comprehensive Multi-Model Analysis ===")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Output directory: {base_output_dir}")
    print(f"Timestamp: {timestamp}")
    
    # Initialize analyzer with base directory
    analyzer = MultiModelAnalyzer(output_dir=base_output_dir)
    
    # Track all results
    all_results = {}
    experiment_log = {
        'timestamp': timestamp,
        'models': models,
        'datasets': datasets,
        'device': str(analyzer.device),
        'results': {},
        'timing': {},
        'errors': []
    }
    
    start_time = time.time()
    
    # Run analysis for each model-dataset combination
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"ANALYZING DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        all_results[dataset] = {}
        
        for model_name in models:
            print(f"\n--- Analyzing {model_name} on {dataset} ---")
            model_start_time = time.time()
            
            try:
                # Create model-specific output directory
                model_output_dir = Path(base_output_dir) / f"{model_name}_{dataset}"
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run analysis
                result = analyzer.run_analysis(
                    model_name=model_name,
                    dataset=dataset,
                    pretrained=True,
                    save_features=True,
                    device=None  # Use auto-detected device
                )
                
                if result:
                    all_results[dataset][model_name] = result
                    
                    # Save model-specific results
                    model_results_path = model_output_dir / f"analysis_results.json"
                    with open(model_results_path, 'w') as f:
                        json.dump(result, f, indent=2, cls=analyzer.NumpyEncoder)
                    
                    # Save features separately
                    features_path = model_output_dir / "extracted_features.pt"
                    if 'feats' in result.get('metadata', {}):
                        import torch
                        torch.save({
                            'features': result['metadata'],
                            'layer_names': result.get('metadata', {}).get('layer_names', []),
                            'feature_dims': result.get('metadata', {}).get('feature_dims', {}),
                            'targets': result.get('metadata', {}).get('targets', [])
                        }, features_path)
                    
                    # Save individual analysis components
                    if 'macroscopic' in result:
                        macro_path = model_output_dir / "macroscopic_analysis.json"
                        with open(macro_path, 'w') as f:
                            json.dump(result['macroscopic'], f, indent=2, cls=analyzer.NumpyEncoder)
                    
                    if 'mesoscopic' in result:
                        meso_path = model_output_dir / "mesoscopic_analysis.json"
                        with open(meso_path, 'w') as f:
                            json.dump(result['mesoscopic'], f, indent=2, cls=analyzer.NumpyEncoder)
                    
                    # Timing information
                    model_duration = time.time() - model_start_time
                    experiment_log['timing'][f"{model_name}_{dataset}"] = model_duration
                    
                    print(f"✅ {model_name} on {dataset} completed in {model_duration:.2f}s")
                    print(f"   Results saved to: {model_output_dir}")
                    
                else:
                    print(f"❌ {model_name} on {dataset} failed - no results returned")
                    experiment_log['errors'].append(f"{model_name}_{dataset}: No results returned")
                    
            except Exception as e:
                print(f"❌ Error analyzing {model_name} on {dataset}: {e}")
                experiment_log['errors'].append(f"{model_name}_{dataset}: {str(e)}")
                continue
        
        # Generate dataset-specific comparisons
        if len(all_results[dataset]) > 1:
            print(f"\n--- Generating comparisons for {dataset} ---")
            try:
                analyzer.plot_model_comparison({dataset: all_results[dataset]})
                print(f"   Comparison plots saved for {dataset}")
            except Exception as e:
                print(f"   Warning: Could not generate comparison plots for {dataset}: {e}")
    
    # Generate comprehensive comparisons
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE COMPARISONS")
    print(f"{'='*60}")
    
    try:
        analyzer.plot_architecture_comparison(all_results)
        print("✅ Architecture comparison plots generated")
    except Exception as e:
        print(f"❌ Error generating architecture comparison: {e}")
    
    # Save comprehensive experiment log
    total_duration = time.time() - start_time
    experiment_log['total_duration'] = total_duration
    experiment_log['total_models_analyzed'] = sum(len(results) for results in all_results.values())
    experiment_log['successful_analyses'] = sum(len(results) for results in all_results.values())
    experiment_log['failed_analyses'] = len(experiment_log['errors'])
    
    # Save experiment log
    log_path = Path(base_output_dir) / "experiment_log.json"
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2, cls=analyzer.NumpyEncoder)
    
    # Save comprehensive results
    comprehensive_path = Path(base_output_dir) / "comprehensive_results.json"
    with open(comprehensive_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=analyzer.NumpyEncoder)
    
    # Generate summary report
    generate_summary_report(base_output_dir, experiment_log, all_results)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Successful analyses: {experiment_log['successful_analyses']}")
    print(f"Failed analyses: {experiment_log['failed_analyses']}")
    print(f"Output directory: {base_output_dir}")
    print(f"TensorBoard logs: {analyzer.tensorboard_dir}")
    print(f"\nTo view TensorBoard logs:")
    print(f"tensorboard --logdir={analyzer.tensorboard_dir}")
    
    return all_results, experiment_log

def generate_summary_report(output_dir, experiment_log, all_results):
    """Generate a human-readable summary report"""
    report_path = Path(output_dir) / "ANALYSIS_SUMMARY.md"
    
    with open(report_path, 'w') as f:
        f.write("# Multi-Model Analysis Summary Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Experiment ID**: {experiment_log['timestamp']}\n")
        f.write(f"**Device**: {experiment_log['device']}\n")
        f.write(f"**Total Duration**: {experiment_log['total_duration']:.2f} seconds\n\n")
        
        f.write("## Models Analyzed\n")
        for model in experiment_log['models']:
            f.write(f"- {model}\n")
        
        f.write("\n## Datasets Analyzed\n")
        for dataset in experiment_log['datasets']:
            f.write(f"- {dataset}\n")
        
        f.write(f"\n## Results Summary\n")
        f.write(f"- **Total Models**: {len(experiment_log['models'])}\n")
        f.write(f"- **Total Datasets**: {len(experiment_log['datasets'])}\n")
        f.write(f"- **Successful Analyses**: {experiment_log['successful_analyses']}\n")
        f.write(f"- **Failed Analyses**: {experiment_log['failed_analyses']}\n")
        
        f.write("\n## Detailed Results\n")
        for dataset in all_results:
            f.write(f"\n### {dataset.upper()}\n")
            for model in all_results[dataset]:
                f.write(f"- **{model}**: ✅ Success\n")
                if 'macroscopic' in all_results[dataset][model]:
                    summary = all_results[dataset][model]['macroscopic'].get('information_flow', {}).get('summary', {})
                    compression = summary.get('total_compression', 0)
                    task_info = summary.get('total_task_info_gain', 0)
                    f.write(f"  - Compression: {compression:.3f}, Task Info: {task_info:.3f}\n")
        
        if experiment_log['errors']:
            f.write("\n## Errors Encountered\n")
            for error in experiment_log['errors']:
                f.write(f"- {error}\n")
        
        f.write("\n## File Structure\n")
        f.write("```\n")
        f.write(f"{output_dir}/\n")
        f.write("├── experiment_log.json\n")
        f.write("├── comprehensive_results.json\n")
        f.write("├── ANALYSIS_SUMMARY.md\n")
        f.write("├── tensorboard_logs/\n")
        f.write("└── [model]_[dataset]/\n")
        f.write("    ├── analysis_results.json\n")
        f.write("    ├── extracted_features.pt\n")
        f.write("    ├── macroscopic_analysis.json\n")
        f.write("    └── mesoscopic_analysis.json\n")
        f.write("```\n")
        
        f.write("\n## Next Steps\n")
        f.write("1. View TensorBoard logs: `tensorboard --logdir=tensorboard_logs`\n")
        f.write("2. Examine individual model results in their respective folders\n")
        f.write("3. Use the comprehensive_results.json for further analysis\n")
        f.write("4. Check the macroscopic and mesoscopic analysis files for detailed metrics\n")

def main():
    """Main function"""
    print("Starting comprehensive multi-model analysis...")
    
    try:
        results, log = run_comprehensive_analysis()
        print("\n🎉 Analysis completed successfully!")
        return results, log
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return None, None
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()
