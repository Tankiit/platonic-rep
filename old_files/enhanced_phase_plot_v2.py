#!/usr/bin/env python3
"""
Create an enhanced phase plot with additional features and save without display
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

def create_enhanced_phase_plot_v2():
    """Create an enhanced phase plot with additional features"""
    
    # Load the actual data
    try:
        df = pd.read_csv('all_analysis_results.csv')
        cross_modal_df = df[df['type'] == 'cross_modal'].copy()
        print(f"Loaded {len(cross_modal_df)} cross-modal pairs")
    except FileNotFoundError:
        print("Cross-modal analysis file not found. Using sample data.")
        return None
    
    # Extract data
    ntk_values = np.array(cross_modal_df['ntk_stability'].tolist())
    agop_values = np.array(cross_modal_df['agop_magnitude'].tolist())
    alignments = np.array(cross_modal_df['alignment'].tolist())
    model_pairs = [f"{row['v_model']}+{row['t_model']}" for _, row in cross_modal_df.iterrows()]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main phase plot
    ax1 = plt.subplot(2, 2, (1, 3))  # Main plot takes up left side
    
    # Plot all points with enhanced styling
    scatter = ax1.scatter(ntk_values, agop_values, c=alignments, 
                        cmap='RdYlGn', s=120, alpha=0.7, edgecolors='black', linewidth=0.8,
                        vmin=0, vmax=max(alignments),
                        zorder=5)
    
    # Add phase boundaries
    ax1.axvline(x=0.7, color='red', linewidth=3, linestyle='--', 
               label='Optimal Threshold (NTK=0.7)', zorder=10)
    ax1.axvline(x=0.5, color='orange', linewidth=2, linestyle=':', 
               label='Chaotic Threshold (NTK=0.5)', zorder=10)
    
    # Add phase regions
    chaotic_patch = Rectangle((0, 1e-2), 0.5, 1e16, alpha=0.15, facecolor='red', zorder=1)
    transition_patch = Rectangle((0.5, 1e-2), 0.2, 1e16, alpha=0.15, facecolor='yellow', zorder=1)
    optimal_patch = Rectangle((0.7, 1e-2), 0.3, 1e16, alpha=0.15, facecolor='green', zorder=1)
    
    ax1.add_patch(chaotic_patch)
    ax1.add_patch(transition_patch)
    ax1.add_patch(optimal_patch)
    
    # Count points in each phase
    chaotic_mask = ntk_values < 0.5
    transition_mask = (ntk_values >= 0.5) & (ntk_values < 0.7)
    optimal_mask = ntk_values >= 0.7
    
    chaotic_count = np.sum(chaotic_mask)
    transition_count = np.sum(transition_mask)
    optimal_count = np.sum(optimal_mask)
    total_count = len(ntk_values)
    
    # Calculate statistics
    chaotic_avg_align = np.mean(alignments[chaotic_mask]) if chaotic_count > 0 else 0
    transition_avg_align = np.mean(alignments[transition_mask]) if transition_count > 0 else 0
    optimal_avg_align = np.mean(alignments[optimal_mask]) if optimal_count > 0 else 0
    
    # Add phase labels
    ax1.text(0.25, 1e10, f'CHAOTIC PHASE\n{chaotic_count}/{total_count} pairs ({chaotic_count/total_count*100:.0f}%)\nAvg align: {chaotic_avg_align:.4f}', 
            fontsize=12, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='red'),
            zorder=15)
    
    ax1.text(0.6, 1e6, f'TRANSITIONAL\n{transition_count}/{total_count} pairs ({transition_count/total_count*100:.0f}%)\nAvg align: {transition_avg_align:.4f}', 
            fontsize=12, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='orange'),
            zorder=15)
    
    ax1.text(0.85, 1e2, f'OPTIMAL PHASE\n{optimal_count}/{total_count} pairs ({optimal_count/total_count*100:.0f}%)\nAvg align: {optimal_avg_align:.4f}', 
            fontsize=12, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='green'),
            zorder=15)
    
    # Highlight best performers
    if chaotic_count > 0:
        chaotic_indices = np.where(chaotic_mask)[0]
        best_chaotic_idx = chaotic_indices[np.argmax(alignments[chaotic_mask])]
        best_chaotic = (ntk_values[best_chaotic_idx], agop_values[best_chaotic_idx], alignments[best_chaotic_idx])
        
        ax1.scatter(*best_chaotic[:2], s=400, facecolors='none', edgecolors='red', linewidths=4, zorder=20)
        ax1.annotate(f'Best Chaotic:\n{model_pairs[best_chaotic_idx]}\n({best_chaotic[2]:.4f})', 
                    xy=best_chaotic[:2], xytext=(0.35, 1e3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    if optimal_count > 0:
        optimal_indices = np.where(optimal_mask)[0]
        best_optimal_idx = optimal_indices[np.argmax(alignments[optimal_mask])]
        best_optimal = (ntk_values[best_optimal_idx], agop_values[best_optimal_idx], alignments[best_optimal_idx])
        
        ax1.scatter(*best_optimal[:2], s=400, facecolors='none', edgecolors='green', linewidths=4, zorder=20)
        ax1.annotate(f'Best Optimal:\n{model_pairs[best_optimal_idx]}\n({best_optimal[2]:.4f})', 
                    xy=best_optimal[:2], xytext=(0.6, 1e8),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Highlight extreme outliers
    extreme_mask = (ntk_values == 0.0) & (agop_values > 1e10)
    if np.any(extreme_mask):
        extreme_indices = np.where(extreme_mask)[0]
        for idx in extreme_indices[:3]:
            ax1.scatter(ntk_values[idx], agop_values[idx], s=200, 
                      facecolors='yellow', edgecolors='black', linewidths=2, zorder=20)
        
        ax1.annotate('EfficientNet-B1 pairs:\nNTK=0, AGOP~1e15\n(Phase collapse)', 
                    xy=(0.0, 1e15), xytext=(0.15, 1e13),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Set main plot properties
    ax1.set_xlabel('Cross-Modal NTK Stability', fontsize=14, weight='bold')
    ax1.set_ylabel('AGOP Magnitude (log scale)', fontsize=14, weight='bold')
    ax1.set_yscale('log')
    ax1.set_xlim(-0.05, 1.0)
    ax1.set_ylim(1e-1, 1e16)
    ax1.set_title('Cross-Modal Phase Diagram\nPhase Incompatibility Prevents Cross-Modal Alignment', 
                 fontsize=16, pad=20, weight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.set_facecolor('#f8f9fa')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8, aspect=30)
    cbar.set_label('Alignment Score', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Top right: Alignment distribution histogram
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(alignments, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(alignments), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(alignments):.4f}')
    ax2.axvline(np.median(alignments), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(alignments):.4f}')
    ax2.set_xlabel('Alignment Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Alignment Distribution', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom right: NTK vs Alignment scatter
    ax3 = plt.subplot(2, 2, 4)
    scatter2 = ax3.scatter(ntk_values, alignments, c=agop_values, 
                          cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('NTK Stability', fontsize=12)
    ax3.set_ylabel('Alignment Score', fontsize=12)
    ax3.set_title('NTK vs Alignment\n(colored by AGOP)', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(ntk_values, alignments)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax3.transAxes, fontsize=10, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add colorbar for the second scatter plot
    cbar2 = plt.colorbar(scatter2, ax=ax3, shrink=0.8)
    cbar2.set_label('AGOP Magnitude', fontsize=10)
    
    # Add overall statistics
    stats_text = f"""Phase Distribution Summary:
• Total Pairs: {total_count}
• Chaotic: {chaotic_count} ({chaotic_count/total_count*100:.1f}%)
• Transitional: {transition_count} ({transition_count/total_count*100:.1f}%)
• Optimal: {optimal_count} ({optimal_count/total_count*100:.1f}%)

Alignment Statistics:
• Max: {np.max(alignments):.4f}
• Mean: {np.mean(alignments):.4f}
• Min: {np.min(alignments):.4f}
• Std: {np.std(alignments):.4f}

NTK Statistics:
• Max: {np.max(ntk_values):.4f}
• Mean: {np.mean(ntk_values):.4f}
• Min: {np.min(ntk_values):.4f}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightblue", alpha=0.8), zorder=15)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('enhanced_phase_plot_v2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('enhanced_phase_plot_v2.pdf', bbox_inches='tight', facecolor='white')
    
    print(f"Enhanced phase plot v2 saved as 'enhanced_phase_plot_v2.png' and 'enhanced_phase_plot_v2.pdf'")
    print(f"Plot shows {total_count} model pairs with phase distribution:")
    print(f"  - Chaotic: {chaotic_count} ({chaotic_count/total_count*100:.1f}%)")
    print(f"  - Transitional: {transition_count} ({transition_count/total_count*100:.1f}%)")
    print(f"  - Optimal: {optimal_count} ({optimal_count/total_count*100:.1f}%)")
    print(f"  - Correlation (NTK vs Alignment): {correlation:.3f}")
    
    return fig

def main():
    """Main function to create the enhanced phase plot"""
    
    print("Creating enhanced phase plot v2 with additional features...")
    
    # Create the plot
    fig = create_enhanced_phase_plot_v2()
    
    if fig is not None:
        print("Enhanced phase plot created successfully!")
    else:
        print("Failed to create enhanced phase plot.")

if __name__ == "__main__":
    main()
