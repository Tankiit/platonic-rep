#!/usr/bin/env python3
"""
Create an enhanced phase plot using actual analysis results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns

# Set style for better plots
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def load_analysis_data():
    """Load the actual analysis data from CSV files"""
    
    # Load cross-modal analysis data
    try:
        df = pd.read_csv('all_analysis_results.csv')
        cross_modal_df = df[df['type'] == 'cross_modal'].copy()
        print(f"Loaded {len(cross_modal_df)} cross-modal pairs")
    except FileNotFoundError:
        print("Cross-modal analysis file not found. Using sample data.")
        cross_modal_df = None
    
    return cross_modal_df

def create_enhanced_phase_plot(df=None):
    """Create the enhanced phase plot"""
    
    if df is None or len(df) == 0:
        # Use sample data if no real data available
        print("Using sample data for demonstration")
        data_points = [
            (0.2265, 1.03e+01, 0.0155),  # resnet18 + bert_base
            (0.4062, 1.25e+02, 0.1070),  # resnet18 + bert_base (different config)
            (0.3404, 1.86e+02, 0.1120),  # vit_base + roberta_base
            (0.0000, 1.61e+15, 0.0167),  # efficientnet_b1 + bert_base
            (0.8056, 1.78e+00, 0.0247),  # efficientnet_b2 + distilroberta
            (0.1585, 1.04e+01, 0.0170),  # mixer_b16 + bert_base
            (0.7301, 1.92e-01, 0.0168),  # swin_small + bert_base
        ]
        
        ntk_values = [point[0] for point in data_points]
        agop_values = [point[1] for point in data_points]
        alignments = [point[2] for point in data_points]
        model_pairs = [f"Model_{i}" for i in range(len(data_points))]
        
    else:
        # Extract actual data
        ntk_values = df['ntk_stability'].tolist()
        agop_values = df['agop_magnitude'].tolist()
        alignments = df['alignment'].tolist()
        model_pairs = [f"{row['v_model']}+{row['t_model']}" for _, row in df.iterrows()]
        
        print(f"Using actual data: {len(ntk_values)} points")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Convert to numpy arrays for easier manipulation
    ntk_values = np.array(ntk_values)
    agop_values = np.array(agop_values)
    alignments = np.array(alignments)
    
    # Plot all points with enhanced styling
    scatter = ax.scatter(ntk_values, agop_values, c=alignments, 
                        cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.8,
                        vmin=0, vmax=max(alignments) if len(alignments) > 0 else 0.12,
                        zorder=5)
    
    # Add phase boundary (using your threshold)
    ax.axvline(x=0.7, color='red', linewidth=3, linestyle='--', 
               label='Optimal Threshold (NTK=0.7)', zorder=10)
    ax.axvline(x=0.5, color='orange', linewidth=2, linestyle=':', 
               label='Chaotic Threshold (NTK=0.5)', zorder=10)
    
    # Add phase regions with better positioning
    chaotic_patch = Rectangle((0, 1e-2), 0.5, 1e16, alpha=0.15, facecolor='red', zorder=1)
    transition_patch = Rectangle((0.5, 1e-2), 0.2, 1e16, alpha=0.15, facecolor='yellow', zorder=1)
    optimal_patch = Rectangle((0.7, 1e-2), 0.3, 1e16, alpha=0.15, facecolor='green', zorder=1)
    
    ax.add_patch(chaotic_patch)
    ax.add_patch(transition_patch)
    ax.add_patch(optimal_patch)
    
    # Count points in each phase
    chaotic_mask = ntk_values < 0.5
    transition_mask = (ntk_values >= 0.5) & (ntk_values < 0.7)
    optimal_mask = ntk_values >= 0.7
    
    chaotic_count = np.sum(chaotic_mask)
    transition_count = np.sum(transition_mask)
    optimal_count = np.sum(optimal_mask)
    total_count = len(ntk_values)
    
    # Calculate average alignments for each phase
    chaotic_avg_align = np.mean(alignments[chaotic_mask]) if chaotic_count > 0 else 0
    transition_avg_align = np.mean(alignments[transition_mask]) if transition_count > 0 else 0
    optimal_avg_align = np.mean(alignments[optimal_mask]) if optimal_count > 0 else 0
    
    # Add phase labels with actual counts and statistics
    ax.text(0.25, 1e10, f'CHAOTIC PHASE\n{chaotic_count}/{total_count} pairs ({chaotic_count/total_count*100:.0f}%)\nAvg align: {chaotic_avg_align:.4f}', 
            fontsize=12, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='red'),
            zorder=15)
    
    ax.text(0.6, 1e6, f'TRANSITIONAL\n{transition_count}/{total_count} pairs ({transition_count/total_count*100:.0f}%)\nAvg align: {transition_avg_align:.4f}', 
            fontsize=12, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='orange'),
            zorder=15)
    
    ax.text(0.85, 1e2, f'OPTIMAL PHASE\n{optimal_count}/{total_count} pairs ({optimal_count/total_count*100:.0f}%)\nAvg align: {optimal_avg_align:.4f}', 
            fontsize=12, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='green'),
            zorder=15)
    
    # Find and highlight best performers in each phase
    if chaotic_count > 0:
        chaotic_indices = np.where(chaotic_mask)[0]
        best_chaotic_idx = chaotic_indices[np.argmax(alignments[chaotic_mask])]
        best_chaotic = (ntk_values[best_chaotic_idx], agop_values[best_chaotic_idx], alignments[best_chaotic_idx])
        
        ax.scatter(*best_chaotic[:2], s=400, facecolors='none', edgecolors='red', linewidths=4, zorder=20)
        ax.annotate(f'Best in Chaotic:\n{model_pairs[best_chaotic_idx]}\n({best_chaotic[2]:.4f})', 
                    xy=best_chaotic[:2], xytext=(0.35, 1e3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    if optimal_count > 0:
        optimal_indices = np.where(optimal_mask)[0]
        best_optimal_idx = optimal_indices[np.argmax(alignments[optimal_mask])]
        best_optimal = (ntk_values[best_optimal_idx], agop_values[best_optimal_idx], alignments[best_optimal_idx])
        
        ax.scatter(*best_optimal[:2], s=400, facecolors='none', edgecolors='green', linewidths=4, zorder=20)
        ax.annotate(f'Best in Optimal:\n{model_pairs[best_optimal_idx]}\n({best_optimal[2]:.4f})', 
                    xy=best_optimal[:2], xytext=(0.6, 1e8),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Highlight extreme outliers (EfficientNet-B1 cases)
    extreme_mask = (ntk_values == 0.0) & (agop_values > 1e10)
    if np.any(extreme_mask):
        extreme_indices = np.where(extreme_mask)[0]
        for idx in extreme_indices[:3]:  # Show first 3 extreme cases
            ax.scatter(ntk_values[idx], agop_values[idx], s=200, 
                      facecolors='yellow', edgecolors='black', linewidths=2, zorder=20)
        
        ax.annotate('EfficientNet-B1 pairs:\nNTK=0, AGOP~1e15\n(Complete phase collapse)', 
                    xy=(0.0, 1e15), xytext=(0.15, 1e13),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Add statistics box
    stats_text = f"""Phase Distribution Summary:
• Total Pairs: {total_count}
• Chaotic: {chaotic_count} ({chaotic_count/total_count*100:.1f}%)
• Transitional: {transition_count} ({transition_count/total_count*100:.1f}%)
• Optimal: {optimal_count} ({optimal_count/total_count*100:.1f}%)

Alignment Statistics:
• Max: {np.max(alignments):.4f}
• Mean: {np.mean(alignments):.4f}
• Min: {np.min(alignments):.4f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightblue", alpha=0.8), zorder=15)
    
    # Set axis properties
    ax.set_xlabel('Cross-Modal NTK Stability', fontsize=14, weight='bold')
    ax.set_ylabel('AGOP Magnitude (log scale)', fontsize=14, weight='bold')
    ax.set_yscale('log')
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(1e-1, 1e16)
    
    # Add colorbar with better styling
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Alignment Score', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Title and grid
    ax.set_title('Enhanced Cross-Modal Phase Diagram\nPhase Incompatibility Prevents Cross-Modal Alignment', 
                 fontsize=16, pad=20, weight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add subtle background grid
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('enhanced_phase_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('enhanced_phase_plot.pdf', bbox_inches='tight', facecolor='white')
    
    print(f"Enhanced phase plot saved as 'enhanced_phase_plot.png' and 'enhanced_phase_plot.pdf'")
    print(f"Plot shows {total_count} model pairs with phase distribution:")
    print(f"  - Chaotic: {chaotic_count} ({chaotic_count/total_count*100:.1f}%)")
    print(f"  - Transitional: {transition_count} ({transition_count/total_count*100:.1f}%)")
    print(f"  - Optimal: {optimal_count} ({optimal_count/total_count*100:.1f}%)")
    
    return fig, ax

def main():
    """Main function to create the enhanced phase plot"""
    
    print("Creating enhanced phase plot using actual analysis results...")
    
    # Load the data
    df = load_analysis_data()
    
    # Create the plot
    fig, ax = create_enhanced_phase_plot(df)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
