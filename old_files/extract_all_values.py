#!/usr/bin/env python3
"""
Extract all NTK, AGOP, and alignment values from saved analysis results
"""

import json
import os
import pandas as pd
from pathlib import Path
import numpy as np

def extract_all_analysis_values():
    """Extract all NTK, AGOP, and alignment values from saved results"""
    
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
    
    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f}")
    
    # Extract data from each file
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract different types of results
            if 'phase_results' in data:
                # Cross-modal analysis results
                for result in data['phase_results']:
                    all_results.append({
                        'file': file_path,
                        'type': 'cross_modal',
                        'v_model': result.get('v_model', 'unknown'),
                        't_model': result.get('t_model', 'unknown'),
                        'ntk_stability': result.get('ntk_stability', None),
                        'agop_magnitude': result.get('agop_magnitude', None),
                        'alignment': result.get('alignment', None),
                        'phase_region': result.get('phase_region', None),
                        'metadata': result.get('metadata', {})
                    })
            
            elif 'phase_diagram' in data:
                # Phase diagram results
                for i, result in enumerate(data['phase_diagram']):
                    all_results.append({
                        'file': file_path,
                        'type': 'phase_diagram',
                        'v_model': f"width_{result.get('width', 'unknown')}",
                        't_model': f"lr_{result.get('learning_rate', 'unknown')}",
                        'ntk_stability': result.get('ntk_stability', None),
                        'agop_magnitude': result.get('agop_magnitude', None),
                        'alignment': result.get('alignment', None),
                        'phase_region': 'unknown',
                        'metadata': {
                            'width': result.get('width', None),
                            'learning_rate': result.get('learning_rate', None),
                            'init_scale': result.get('init_scale', None)
                        }
                    })
            
            elif 'macroscopic' in data:
                # Individual model analysis results
                model_name = os.path.basename(file_path).replace('_analysis.json', '')
                all_results.append({
                    'file': file_path,
                    'type': 'individual_model',
                    'v_model': model_name,
                    't_model': 'N/A',
                    'ntk_stability': data.get('macroscopic', {}).get('ntk_stability', None),
                    'agop_magnitude': data.get('macroscopic', {}).get('agop_magnitude', None),
                    'alignment': data.get('macroscopic', {}).get('alignment', None),
                    'phase_region': 'unknown',
                    'metadata': data.get('metadata', {})
                })
            
            elif 'ntk_stability' in data and 'agop_magnitude' in data:
                # Direct metrics in file
                model_name = os.path.basename(file_path).replace('.json', '')
                all_results.append({
                    'file': file_path,
                    'type': 'direct_metrics',
                    'v_model': model_name,
                    't_model': 'N/A',
                    'ntk_stability': data.get('ntk_stability', None),
                    'agop_magnitude': data.get('agop_magnitude', None),
                    'alignment': data.get('alignment', None),
                    'phase_region': 'unknown',
                    'metadata': {}
                })
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return all_results

def create_summary_table(all_results):
    """Create a summary table of all results"""
    
    if not all_results:
        print("No results found!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Clean up the data
    df = df.dropna(subset=['ntk_stability', 'agop_magnitude'])
    
    # Add log AGOP for better visualization
    df['agop_log'] = np.log10(df['agop_magnitude'].astype(float) + 1e-10)
    
    # Determine phase regions based on NTK stability
    def determine_phase(ntk):
        if pd.isna(ntk):
            return 'unknown'
        elif ntk >= 0.9:
            return 'lazy'
        elif ntk >= 0.7:
            return 'optimal'
        elif ntk >= 0.5:
            return 'chaotic'
        else:
            return 'chaotic'
    
    df['phase_region'] = df['ntk_stability'].apply(determine_phase)
    
    return df

def print_detailed_summary(df):
    """Print detailed summary of results"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE NTK, AGOP, AND ALIGNMENT ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal Results Found: {len(df)}")
    print(f"Results with valid NTK values: {df['ntk_stability'].notna().sum()}")
    print(f"Results with valid AGOP values: {df['agop_magnitude'].notna().sum()}")
    print(f"Results with valid alignment values: {df['alignment'].notna().sum()}")
    
    # Summary statistics
    print("\n" + "-"*50)
    print("SUMMARY STATISTICS")
    print("-"*50)
    
    print(f"\nNTK Stability:")
    print(f"  Range: {df['ntk_stability'].min():.4f} - {df['ntk_stability'].max():.4f}")
    print(f"  Mean: {df['ntk_stability'].mean():.4f}")
    print(f"  Std: {df['ntk_stability'].std():.4f}")
    
    print(f"\nAGOP Magnitude:")
    print(f"  Range: {df['agop_magnitude'].min():.2e} - {df['agop_magnitude'].max():.2e}")
    print(f"  Mean: {df['agop_magnitude'].mean():.2e}")
    print(f"  Std: {df['agop_magnitude'].std():.2e}")
    
    if df['alignment'].notna().any():
        print(f"\nAlignment:")
        print(f"  Range: {df['alignment'].min():.4f} - {df['alignment'].max():.4f}")
        print(f"  Mean: {df['alignment'].mean():.4f}")
        print(f"  Std: {df['alignment'].std():.4f}")
    
    # Phase distribution
    print("\n" + "-"*50)
    print("PHASE DISTRIBUTION")
    print("-"*50)
    
    phase_counts = df['phase_region'].value_counts()
    for phase, count in phase_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {phase.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Cross-modal pairs
    print("\n" + "-"*50)
    print("CROSS-MODAL MODEL PAIRS")
    print("-"*50)
    
    cross_modal_df = df[df['type'] == 'cross_modal']
    if len(cross_modal_df) > 0:
        print(f"Found {len(cross_modal_df)} cross-modal pairs:")
        for _, row in cross_modal_df.iterrows():
            print(f"  {row['v_model']} + {row['t_model']}:")
            print(f"    NTK: {row['ntk_stability']:.4f}")
            print(f"    AGOP: {row['agop_magnitude']:.2e}")
            print(f"    Alignment: {row['alignment']:.4f}")
            print(f"    Phase: {row['phase_region']}")
            print()
    else:
        print("No cross-modal pairs found in current data")
    
    # Individual models
    print("\n" + "-"*50)
    print("INDIVIDUAL MODEL ANALYSES")
    print("-"*50)
    
    individual_df = df[df['type'] == 'individual_model']
    if len(individual_df) > 0:
        print(f"Found {len(individual_df)} individual model analyses:")
        for _, row in individual_df.iterrows():
            print(f"  {row['v_model']}:")
            print(f"    NTK: {row['ntk_stability']:.4f}")
            print(f"    AGOP: {row['agop_magnitude']:.2e}")
            if pd.notna(row['alignment']):
                print(f"    Alignment: {row['alignment']:.4f}")
            print()
    
    return df

def save_results_to_csv(df, filename="all_analysis_results.csv"):
    """Save results to CSV file"""
    
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        # Also save a summary
        summary_filename = filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w') as f:
            f.write("NTK, AGOP, and Alignment Analysis Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Results: {len(df)}\n")
            f.write(f"NTK Range: {df['ntk_stability'].min():.4f} - {df['ntk_stability'].max():.4f}\n")
            f.write(f"AGOP Range: {df['agop_magnitude'].min():.2e} - {df['agop_magnitude'].max():.2e}\n")
            if df['alignment'].notna().any():
                f.write(f"Alignment Range: {df['alignment'].min():.4f} - {df['alignment'].max():.4f}\n")
            
            f.write("\nCross-Modal Pairs:\n")
            cross_modal_df = df[df['type'] == 'cross_modal']
            for _, row in cross_modal_df.iterrows():
                f.write(f"{row['v_model']} + {row['t_model']}: NTK={row['ntk_stability']:.4f}, AGOP={row['agop_magnitude']:.2e}, Align={row['alignment']:.4f}\n")
        
        print(f"Summary saved to: {summary_filename}")

def main():
    """Main function to extract and summarize all analysis values"""
    
    print("Extracting all NTK, AGOP, and alignment values from saved results...")
    
    # Extract all results
    all_results = extract_all_analysis_values()
    
    if not all_results:
        print("No analysis results found!")
        return
    
    # Create summary table
    df = create_summary_table(all_results)
    
    if df is not None:
        # Print detailed summary
        print_detailed_summary(df)
        
        # Save to files
        save_results_to_csv(df)
        
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE!")
        print("="*80)
        print("All NTK, AGOP, and alignment values have been extracted and summarized.")
        print("Check the CSV file and summary text file for detailed results.")

if __name__ == "__main__":
    main()
