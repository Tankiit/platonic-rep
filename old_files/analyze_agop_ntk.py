#!/usr/bin/env python3
"""
Extract and visualize AGOP and NTK values from the phase analyzer results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_phase_analysis_results():
    """Load the phase analysis results from JSON"""
    results_path = Path('results/complete_embeddings_analysis/complete_embeddings_analysis.json')
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results

def extract_agop_ntk_values(results):
    """Extract AGOP and NTK values from the results"""
    
    # Extract vision model data
    vision_data = []
    for model_name, analysis in results['vision_models'].items():
        ntk_data = analysis['ntk_stability']
        agop_data = analysis['agop_analysis']
        
        vision_data.append({
            'model': model_name,
            'type': 'vision',
            's_ntk': ntk_data['s_ntk'],
            's_ntk_std': ntk_data['s_ntk_std'],
            'phase': ntk_data['phase'],
            'agop_ratio': agop_data['agop_ratio'],
            'top10_concentration': agop_data['top10_concentration'],
            'effective_rank': agop_data['effective_rank'],
            'gradient_anisotropy': agop_data['gradient_anisotropy'],
            'chaotic_indicator': agop_data['chaotic_indicator']
        })
    
    # Extract language model data
    language_data = []
    for model_name, analysis in results['language_models'].items():
        ntk_data = analysis['ntk_stability']
        agop_data = analysis['agop_analysis']
        
        language_data.append({
            'model': model_name,
            'type': 'language',
            's_ntk': ntk_data['s_ntk'],
            's_ntk_std': ntk_data['s_ntk_std'],
            'phase': ntk_data['phase'],
            'agop_ratio': agop_data['agop_ratio'],
            'top10_concentration': agop_data['top10_concentration'],
            'effective_rank': agop_data['effective_rank'],
            'gradient_anisotropy': agop_data['gradient_anisotropy'],
            'chaotic_indicator': agop_data['chaotic_indicator']
        })
    
    # Extract cross-modal data
    cross_modal_data = []
    for pair_name, analysis in results['cross_modal_analysis'].items():
        cross_modal_data.append({
            'pair': pair_name,
            's_ntk_cross': analysis['s_ntk_cross'],
            'condition_number_vision': analysis['condition_number_vision'],
            'condition_number_language': analysis['condition_number_language'],
            'phase_compatible': analysis['phase_compatible'],
            'n_samples': analysis['n_samples']
        })
    
    return vision_data, language_data, cross_modal_data

def create_agop_ntk_visualizations(vision_data, language_data, cross_modal_data):
    """Create comprehensive visualizations of AGOP and NTK values"""
    
    # Set up the plot
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('AGOP and NTK Analysis: ResNet18 vs DistilBERT', fontsize=16, fontweight='bold')
    
    # Convert to DataFrames for easier plotting
    vision_df = pd.DataFrame(vision_data)
    language_df = pd.DataFrame(language_data)
    cross_modal_df = pd.DataFrame(cross_modal_data)
    
    # Plot 1: NTK Stability by Model Type
    ax1 = axes[0, 0]
    all_models = pd.concat([vision_df, language_df], ignore_index=True)
    
    colors = {'chaotic': 'red', 'optimal': 'green', 'lazy': 'blue'}
    for phase in ['chaotic', 'optimal', 'lazy']:
        phase_data = all_models[all_models['phase'] == phase]
        if not phase_data.empty:
            ax1.scatter(phase_data[phase_data['type'] == 'vision']['s_ntk'], 
                       [1] * len(phase_data[phase_data['type'] == 'vision']), 
                       c=colors[phase], label=f'Vision {phase}', s=100, alpha=0.7)
            ax1.scatter(phase_data[phase_data['type'] == 'language']['s_ntk'], 
                       [2] * len(phase_data[phase_data['type'] == 'language']), 
                       c=colors[phase], label=f'Language {phase}', s=100, alpha=0.7, marker='s')
    
    ax1.set_xlabel('S_NTK (NTK Stability)')
    ax1.set_ylabel('Model Type')
    ax1.set_yticks([1, 2])
    ax1.set_yticklabels(['Vision', 'Language'])
    ax1.set_title('NTK Stability by Model Type')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AGOP Ratio vs NTK Stability
    ax2 = axes[0, 1]
    ax2.scatter(vision_df['s_ntk'], np.log10(vision_df['agop_ratio']), 
               c='lightblue', label='ResNet18', s=100, alpha=0.7)
    ax2.scatter(language_df['s_ntk'], np.log10(language_df['agop_ratio']), 
               c='lightcoral', label='DistilBERT', s=100, alpha=0.7, marker='s')
    
    # Add phase boundaries
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Chaotic|Optimal')
    ax2.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='Optimal|Lazy')
    ax2.axhline(y=6, color='orange', linestyle='--', alpha=0.7, label='AGOP Failure Threshold')
    
    ax2.set_xlabel('S_NTK (NTK Stability)')
    ax2.set_ylabel('log₁₀(AGOP Ratio)')
    ax2.set_title('AGOP Ratio vs NTK Stability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cross-Modal NTK Stability Distribution
    ax3 = axes[0, 2]
    cross_modal_scores = cross_modal_df['s_ntk_cross']
    ax3.hist(cross_modal_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0.25, color='red', linestyle='--', linewidth=2, 
               label=f'Compatibility Threshold (0.25)')
    ax3.axvline(x=cross_modal_scores.mean(), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean Compatibility ({cross_modal_scores.mean():.3f})')
    
    ax3.set_xlabel('S_NTK^cross (Cross-Modal Stability)')
    ax3.set_ylabel('Number of Model Pairs')
    ax3.set_title('Cross-Modal Compatibility Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effective Rank Comparison
    ax4 = axes[1, 0]
    ax4.bar(range(len(vision_df)), vision_df['effective_rank'], 
           alpha=0.7, color='lightblue', label='ResNet18')
    ax4.bar(range(len(language_df)), language_df['effective_rank'], 
           alpha=0.7, color='lightcoral', label='DistilBERT')
    
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Effective Rank')
    ax4.set_title('Effective Rank by Layer')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Top-10 Eigenvalue Concentration
    ax5 = axes[1, 1]
    ax5.plot(range(len(vision_df)), vision_df['top10_concentration'], 
            'o-', label='ResNet18', linewidth=2, markersize=6)
    ax5.plot(range(len(language_df)), language_df['top10_concentration'], 
            's-', label='DistilBERT', linewidth=2, markersize=6)
    
    ax5.set_xlabel('Layer Index')
    ax5.set_ylabel('Top-10 Eigenvalue Concentration')
    ax5.set_title('Eigenvalue Concentration Across Layers')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Gradient Anisotropy
    ax6 = axes[1, 2]
    ax6.bar(range(len(vision_df)), vision_df['gradient_anisotropy'], 
           alpha=0.7, color='lightblue', label='ResNet18')
    ax6.bar(range(len(language_df)), language_df['gradient_anisotropy'], 
           alpha=0.7, color='lightcoral', label='DistilBERT')
    
    ax6.set_xlabel('Layer Index')
    ax6.set_ylabel('Gradient Anisotropy')
    ax6.set_title('Gradient Anisotropy by Layer')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Cross-Modal Compatibility Heatmap
    ax7 = axes[2, 0]
    
    # Create compatibility matrix
    vision_models = vision_df['model'].tolist()
    language_models = language_df['model'].tolist()
    
    compat_matrix = np.zeros((len(vision_models), len(language_models)))
    
    for i, v_model in enumerate(vision_models):
        for j, l_model in enumerate(language_models):
            pair_name = f"{v_model}_×_{l_model}"
            pair_data = cross_modal_df[cross_modal_df['pair'] == pair_name]
            if not pair_data.empty:
                compat_matrix[i, j] = pair_data['s_ntk_cross'].iloc[0]
    
    im = ax7.imshow(compat_matrix, cmap='RdYlBu_r', aspect='auto')
    ax7.set_xticks(range(len(language_models)))
    ax7.set_xticklabels([m.replace('distilbert_', '') for m in language_models], rotation=45)
    ax7.set_yticks(range(len(vision_models)))
    ax7.set_yticklabels([m.replace('resnet18_', '') for m in vision_models])
    ax7.set_title('Cross-Modal Compatibility Matrix')
    plt.colorbar(im, ax=ax7, label='S_NTK^cross')
    
    # Plot 8: Phase Distribution
    ax8 = axes[2, 1]
    phase_counts = all_models['phase'].value_counts()
    colors_phase = [colors[phase] for phase in phase_counts.index]
    ax8.pie(phase_counts.values, labels=phase_counts.index, colors=colors_phase, 
           autopct='%1.1f%%', startangle=90)
    ax8.set_title('Phase Distribution')
    
    # Plot 9: Summary Statistics
    ax9 = axes[2, 2]
    
    # Create summary text
    summary_text = f"""
PHASE ANALYSIS SUMMARY

Vision Models: {len(vision_df)}
Language Models: {len(language_df)}
Cross-Modal Pairs: {len(cross_modal_df)}

NTK Stability:
  Vision Mean: {vision_df['s_ntk'].mean():.3f}
  Language Mean: {language_df['s_ntk'].mean():.3f}

Cross-Modal Compatibility:
  Mean: {cross_modal_scores.mean():.3f}
  Min: {cross_modal_scores.min():.3f}
  Max: {cross_modal_scores.max():.3f}
  Below Threshold: {(cross_modal_scores < 0.25).sum()}/{len(cross_modal_scores)}

AGOP Analysis:
  Vision Mean Ratio: {vision_df['agop_ratio'].mean():.2e}
  Language Mean Ratio: {language_df['agop_ratio'].mean():.2e}
  """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax9.set_title('Summary Statistics')
    ax9.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'results/agop_ntk_analysis.png'
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"AGOP/NTK analysis saved to: {output_path}")
    
    plt.show()
    
    return vision_df, language_df, cross_modal_df

def print_detailed_results(vision_df, language_df, cross_modal_df):
    """Print detailed results"""
    
    print("\n" + "="*80)
    print("DETAILED AGOP AND NTK ANALYSIS RESULTS")
    print("="*80)
    
    print("\n=== VISION MODELS (ResNet18) ===")
    for _, row in vision_df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  Phase: {row['phase']}")
        print(f"  S_NTK: {row['s_ntk']:.4f} ± {row['s_ntk_std']:.4f}")
        print(f"  AGOP Ratio: {row['agop_ratio']:.2e}")
        print(f"  Top-10 Concentration: {row['top10_concentration']:.4f}")
        print(f"  Effective Rank: {row['effective_rank']:.2f}")
        print(f"  Gradient Anisotropy: {row['gradient_anisotropy']:.4f}")
        print(f"  Chaotic Indicator: {row['chaotic_indicator']}")
    
    print("\n=== LANGUAGE MODELS (DistilBERT) ===")
    for _, row in language_df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  Phase: {row['phase']}")
        print(f"  S_NTK: {row['s_ntk']:.4f} ± {row['s_ntk_std']:.4f}")
        print(f"  AGOP Ratio: {row['agop_ratio']:.2e}")
        print(f"  Top-10 Concentration: {row['top10_concentration']:.4f}")
        print(f"  Effective Rank: {row['effective_rank']:.2f}")
        print(f"  Gradient Anisotropy: {row['gradient_anisotropy']:.4f}")
        print(f"  Chaotic Indicator: {row['chaotic_indicator']}")
    
    print("\n=== CROSS-MODAL COMPATIBILITY ===")
    print(f"Total pairs analyzed: {len(cross_modal_df)}")
    print(f"Mean S_NTK^cross: {cross_modal_df['s_ntk_cross'].mean():.4f}")
    print(f"Min S_NTK^cross: {cross_modal_df['s_ntk_cross'].min():.4f}")
    print(f"Max S_NTK^cross: {cross_modal_df['s_ntk_cross'].max():.4f}")
    print(f"Pairs below threshold (0.25): {(cross_modal_df['s_ntk_cross'] < 0.25).sum()}")
    print(f"Compatibility rate: {(cross_modal_df['s_ntk_cross'] >= 0.25).mean():.1%}")
    
    print("\n=== TOP 5 MOST COMPATIBLE PAIRS ===")
    top_pairs = cross_modal_df.nlargest(5, 's_ntk_cross')
    for _, row in top_pairs.iterrows():
        print(f"  {row['pair']}: {row['s_ntk_cross']:.4f}")
    
    print("\n=== TOP 5 LEAST COMPATIBLE PAIRS ===")
    bottom_pairs = cross_modal_df.nsmallest(5, 's_ntk_cross')
    for _, row in bottom_pairs.iterrows():
        print(f"  {row['pair']}: {row['s_ntk_cross']:.4f}")

def main():
    """Main function"""
    print("Loading phase analysis results...")
    results = load_phase_analysis_results()
    
    print("Extracting AGOP and NTK values...")
    vision_data, language_data, cross_modal_data = extract_agop_ntk_values(results)
    
    print("Creating visualizations...")
    vision_df, language_df, cross_modal_df = create_agop_ntk_visualizations(
        vision_data, language_data, cross_modal_data)
    
    print("Printing detailed results...")
    print_detailed_results(vision_df, language_df, cross_modal_df)

if __name__ == "__main__":
    main()
