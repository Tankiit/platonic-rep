#!/usr/bin/env python3
"""
Extract all I_X_T and I_Y_T (mutual information) values from saved analysis results
"""

import json
import os
import pandas as pd
from pathlib import Path
import numpy as np

def extract_mutual_information_values():
    """Extract all I_X_T and I_Y_T values from saved results"""
    
    all_results = []
    
    # Search for all JSON result files
    result_files = []
    
    # Check various directories
    search_dirs = [
        "results/",
        "experiments/",
        "test_output/",
        "."
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.json') and ('analysis' in file or 'results' in file):
                        result_files.append(os.path.join(root, file))
    
    print(f"Found {len(result_files)} result files to analyze...")
    
    # Extract data from each file
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract mutual information values
            if 'macroscopic' in data and 'information_flow' in data['macroscopic']:
                layers = data['macroscopic']['information_flow'].get('layers', {})
                
                for layer_name, layer_data in layers.items():
                    all_results.append({
                        'file': file_path,
                        'model': data.get('model', 'unknown'),
                        'dataset': data.get('dataset', 'unknown'),
                        'layer': layer_name,
                        'layer_idx': layer_data.get('layer_idx', 0),
                        'I_X_T': layer_data.get('I_X_T', None),
                        'I_Y_T': layer_data.get('I_Y_T', None),
                        'H_T': layer_data.get('H_T', None),
                        'efficiency': layer_data.get('efficiency', None),
                        'compression': layer_data.get('compression', None)
                    })
            
            # Also check for summary information
            if 'macroscopic' in data and 'information_flow' in data['macroscopic']:
                summary = data['macroscopic']['information_flow'].get('summary', {})
                if summary:
                    # Add summary data
                    all_results.append({
                        'file': file_path,
                        'model': data.get('model', 'unknown'),
                        'dataset': data.get('dataset', 'unknown'),
                        'layer': 'summary',
                        'layer_idx': -1,
                        'I_X_T': summary.get('initial_state', {}).get('I_X_T', None),
                        'I_Y_T': summary.get('initial_state', {}).get('I_Y_T', None),
                        'H_T': None,
                        'efficiency': None,
                        'compression': None
                    })
                    
                    all_results.append({
                        'file': file_path,
                        'model': data.get('model', 'unknown'),
                        'dataset': data.get('dataset', 'unknown'),
                        'layer': 'summary_final',
                        'layer_idx': 999,
                        'I_X_T': summary.get('final_state', {}).get('I_X_T', None),
                        'I_Y_T': summary.get('final_state', {}).get('I_Y_T', None),
                        'H_T': None,
                        'efficiency': None,
                        'compression': None
                    })
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return all_results

def create_mutual_information_summary(all_results):
    """Create a summary of mutual information results"""
    
    if not all_results:
        print("No mutual information results found!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Clean up the data
    df = df.dropna(subset=['I_X_T', 'I_Y_T'])
    
    # Add derived metrics
    df['info_efficiency'] = df['I_Y_T'] / (df['I_X_T'] + 1e-6)
    df['compression_ratio'] = 1 - (df['I_X_T'] / df['I_X_T'].max())
    
    return df

def print_mutual_information_summary(df):
    """Print detailed summary of mutual information results"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MUTUAL INFORMATION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal Results Found: {len(df)}")
    print(f"Results with valid I_X_T values: {df['I_X_T'].notna().sum()}")
    print(f"Results with valid I_Y_T values: {df['I_Y_T'].notna().sum()}")
    
    # Summary statistics
    print("\n" + "-"*50)
    print("SUMMARY STATISTICS")
    print("-"*50)
    
    print(f"\nI_X_T (Input-Representation Mutual Information):")
    print(f"  Range: {df['I_X_T'].min():.6f} - {df['I_X_T'].max():.6f}")
    print(f"  Mean: {df['I_X_T'].mean():.6f}")
    print(f"  Std: {df['I_X_T'].std():.6f}")
    
    print(f"\nI_Y_T (Representation-Target Mutual Information):")
    print(f"  Range: {df['I_Y_T'].min():.6f} - {df['I_Y_T'].max():.6f}")
    print(f"  Mean: {df['I_Y_T'].mean():.6f}")
    print(f"  Std: {df['I_Y_T'].std():.6f}")
    
    print(f"\nInformation Efficiency (I_Y_T / I_X_T):")
    print(f"  Range: {df['info_efficiency'].min():.6f} - {df['info_efficiency'].max():.6f}")
    print(f"  Mean: {df['info_efficiency'].mean():.6f}")
    print(f"  Std: {df['info_efficiency'].std():.6f}")
    
    # Model-specific analysis
    print("\n" + "-"*50)
    print("MODEL-SPECIFIC ANALYSIS")
    print("-"*50)
    
    model_stats = df.groupby('model').agg({
        'I_X_T': ['mean', 'std', 'min', 'max'],
        'I_Y_T': ['mean', 'std', 'min', 'max'],
        'info_efficiency': ['mean', 'std', 'min', 'max']
    }).round(6)
    
    print("\nPer-Model Statistics:")
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"\n{model}:")
        print(f"  I_X_T: {model_data['I_X_T'].mean():.6f} ± {model_data['I_X_T'].std():.6f}")
        print(f"  I_Y_T: {model_data['I_Y_T'].mean():.6f} ± {model_data['I_Y_T'].std():.6f}")
        print(f"  Efficiency: {model_data['info_efficiency'].mean():.6f} ± {model_data['info_efficiency'].std():.6f}")
        print(f"  Layers: {len(model_data)}")
    
    # Dataset-specific analysis
    print("\n" + "-"*50)
    print("DATASET-SPECIFIC ANALYSIS")
    print("-"*50)
    
    dataset_stats = df.groupby('dataset').agg({
        'I_X_T': ['mean', 'std'],
        'I_Y_T': ['mean', 'std'],
        'info_efficiency': ['mean', 'std']
    }).round(6)
    
    print("\nPer-Dataset Statistics:")
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        print(f"\n{dataset}:")
        print(f"  I_X_T: {dataset_data['I_X_T'].mean():.6f} ± {dataset_data['I_X_T'].std():.6f}")
        print(f"  I_Y_T: {dataset_data['I_Y_T'].mean():.6f} ± {dataset_data['I_Y_T'].std():.6f}")
        print(f"  Efficiency: {dataset_data['info_efficiency'].mean():.6f} ± {dataset_data['info_efficiency'].std():.6f}")
        print(f"  Models: {len(dataset_data['model'].unique())}")
    
    # Layer-wise analysis
    print("\n" + "-"*50)
    print("LAYER-WISE ANALYSIS")
    print("-"*50)
    
    layer_stats = df[df['layer_idx'] >= 0].groupby('layer_idx').agg({
        'I_X_T': ['mean', 'std'],
        'I_Y_T': ['mean', 'std'],
        'info_efficiency': ['mean', 'std']
    }).round(6)
    
    print("\nPer-Layer Statistics:")
    for layer_idx in sorted(df[df['layer_idx'] >= 0]['layer_idx'].unique()):
        layer_data = df[df['layer_idx'] == layer_idx]
        print(f"\nLayer {layer_idx}:")
        print(f"  I_X_T: {layer_data['I_X_T'].mean():.6f} ± {layer_data['I_X_T'].std():.6f}")
        print(f"  I_Y_T: {layer_data['I_Y_T'].mean():.6f} ± {layer_data['I_Y_T'].std():.6f}")
        print(f"  Efficiency: {layer_data['info_efficiency'].mean():.6f} ± {layer_data['info_efficiency'].std():.6f}")
        print(f"  Models: {len(layer_data['model'].unique())}")
    
    return df

def save_mutual_information_results(df, filename="mutual_information_results.csv"):
    """Save mutual information results to CSV file"""
    
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        # Also save a summary
        summary_filename = filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w') as f:
            f.write("Mutual Information Analysis Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Results: {len(df)}\n")
            f.write(f"I_X_T Range: {df['I_X_T'].min():.6f} - {df['I_X_T'].max():.6f}\n")
            f.write(f"I_Y_T Range: {df['I_Y_T'].min():.6f} - {df['I_Y_T'].max():.6f}\n")
            f.write(f"Efficiency Range: {df['info_efficiency'].min():.6f} - {df['info_efficiency'].max():.6f}\n")
            
            f.write("\nModel Summary:\n")
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                f.write(f"{model}: I_X_T={model_data['I_X_T'].mean():.6f}, I_Y_T={model_data['I_Y_T'].mean():.6f}, Eff={model_data['info_efficiency'].mean():.6f}\n")
        
        print(f"Summary saved to: {summary_filename}")

def create_focused_mutual_information_summary(df):
    """Create a focused summary for specific model pairs"""
    
    print("\n" + "="*80)
    print("FOCUSED MUTUAL INFORMATION SUMMARY")
    print("="*80)
    
    # Your specific representative pairs
    target_models = ["resnet50", "vit_base_patch16_224", "convnext_small"]
    
    print("\n🎯 YOUR SPECIFIC REPRESENTATIVE MODELS:")
    print("-" * 50)
    
    for model in target_models:
        model_data = df[df['model'].str.contains(model.split('_')[0], case=False)]
        
        if len(model_data) > 0:
            print(f"✅ {model}:")
            print(f"   I_X_T: {model_data['I_X_T'].mean():.6f} ± {model_data['I_X_T'].std():.6f}")
            print(f"   I_Y_T: {model_data['I_Y_T'].mean():.6f} ± {model_data['I_Y_T'].std():.6f}")
            print(f"   Efficiency: {model_data['info_efficiency'].mean():.6f} ± {model_data['info_efficiency'].std():.6f}")
            print(f"   Layers: {len(model_data)}")
            print()
        else:
            print(f"❌ {model}: Not found in current data")
            print()
    
    # Show best performers
    print("\n🏆 TOP PERFORMERS BY I_Y_T:")
    print("-" * 50)
    
    top_i_yt = df.nlargest(5, 'I_Y_T')
    for i, (_, row) in enumerate(top_i_yt.iterrows(), 1):
        print(f"{i}. {row['model']} (Layer {row['layer_idx']}):")
        print(f"   I_Y_T: {row['I_Y_T']:.6f}")
        print(f"   I_X_T: {row['I_X_T']:.6f}")
        print(f"   Efficiency: {row['info_efficiency']:.6f}")
        print()
    
    # Show best efficiency
    print("\n🎯 TOP PERFORMERS BY EFFICIENCY:")
    print("-" * 50)
    
    top_efficiency = df.nlargest(5, 'info_efficiency')
    for i, (_, row) in enumerate(top_efficiency.iterrows(), 1):
        print(f"{i}. {row['model']} (Layer {row['layer_idx']}):")
        print(f"   Efficiency: {row['info_efficiency']:.6f}")
        print(f"   I_Y_T: {row['I_Y_T']:.6f}")
        print(f"   I_X_T: {row['I_X_T']:.6f}")
        print()

def main():
    """Main function to extract and summarize all mutual information values"""
    
    print("Extracting all I_X_T and I_Y_T values from saved results...")
    
    # Extract all results
    all_results = extract_mutual_information_values()
    
    if not all_results:
        print("No mutual information results found!")
        return
    
    # Create summary table
    df = create_mutual_information_summary(all_results)
    
    if df is not None:
        # Print detailed summary
        print_mutual_information_summary(df)
        
        # Print focused summary
        create_focused_mutual_information_summary(df)
        
        # Save to files
        save_mutual_information_results(df)
        
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE!")
        print("="*80)
        print("All I_X_T and I_Y_T values have been extracted and summarized.")
        print("Check the CSV file and summary text file for detailed results.")

if __name__ == "__main__":
    main()
