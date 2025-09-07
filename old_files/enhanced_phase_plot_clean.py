#!/usr/bin/env python3
"""
Create an enhanced phase plot with improved text box styling
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import seaborn as sns

# Set style for better plots
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def create_clean_text_box(ax, x, y, text, boxstyle="round,pad=0.4", 
                         facecolor="white", edgecolor="black", alpha=0.95,
                         fontsize=11, weight='normal', ha='center', va='center'):
    """Create a clean, professional text box"""
    
    # Create fancy box patch
    fancy_box = FancyBboxPatch((x-0.1, y-0.05), 0.2, 0.1,
                              boxstyle=boxstyle,
                              facecolor=facecolor,
                              edgecolor=edgecolor,
                              alpha=alpha,
                              linewidth=1.5,
                              zorder=20)
    
    ax.add_patch(fancy_box)
    
    # Add text
    ax.text(x, y, text, fontsize=fontsize, weight=weight, 
           ha=ha, va=va, zorder=21,
           bbox=dict(boxstyle="round,pad=0.1", facecolor="none", alpha=0))

def create_enhanced_phase_plot_clean():
    """Create an enhanced phase plot with clean text boxes"""
    
    # Load the actual data
    try:
        df = pd.read_csv('all_analysis_results.csv')
        cross_modal_df = df[df['type'] == 'cross_modal'].copy()
        print(f"Loaded {len(cross_modal_df)} cross-modal pairs")
    except FileNotFoundError:
        print("Cross-modal analysis file not found.")
        return None
    
    # Extract data
    ntk_values = np.array(cross_modal_df['ntk_stability'].tolist())
    agop_values = np.array(cross_modal_df['agop_magnitude'].tolist())
    alignments = np.array(cross_modal_df['alignment'].tolist())
    model_pairs = [f"{row['v_model']}+{row['t_model']}" for _, row in cross_modal_df.iterrows()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot all points with enhanced styling
    scatter = ax.scatter(ntk_values, agop_values, c=alignments, 
                        cmap='RdYlGn', s=120, alpha=0.7, edgecolors='black', linewidth=0.8,
                        vmin=0, vmax=max(alignments),
                        zorder=5)
    
    # Add phase boundaries
    ax.axvline(x=0.7, color='red', linewidth=3, linestyle='--', 
               label='Optimal Threshold (NTK=0.7)', zorder=10)
    ax.axvline(x=0.5, color='orange', linewidth=2, linestyle=':', 
               label='Chaotic Threshold (NTK=0.5)', zorder=10)
    
    # Add phase regions with subtle colors
    chaotic_patch = Rectangle((0, 1e-2), 0.5, 1e16, alpha=0.1, facecolor='red', zorder=1)
    transition_patch = Rectangle((0.5, 1e-2), 0.2, 1e16, alpha=0.1, facecolor='yellow', zorder=1)
    optimal_patch = Rectangle((0.7, 1e-2), 0.3, 1e16, alpha=0.1, facecolor='green', zorder=1)
    
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
    
    # Calculate statistics
    chaotic_avg_align = np.mean(alignments[chaotic_mask]) if chaotic_count > 0 else 0
    transition_avg_align = np.mean(alignments[transition_mask]) if transition_count > 0 else 0
    optimal_avg_align = np.mean(alignments[optimal_mask]) if optimal_count > 0 else 0
    
    # Add clean phase labels with better positioning
    create_clean_text_box(ax, 0.25, 1e10, 
                         f'CHAOTIC PHASE\n{chaotic_count}/{total_count} pairs ({chaotic_count/total_count*100:.0f}%)\nAvg alignment: {chaotic_avg_align:.4f}',
                         facecolor='white', edgecolor='red', alpha=0.95, fontsize=12, weight='bold')
    
    create_clean_text_box(ax, 0.6, 1e6, 
                         f'TRANSITIONAL\n{transition_count}/{total_count} pairs ({transition_count/total_count*100:.0f}%)\nAvg alignment: {transition_avg_align:.4f}',
                         facecolor='white', edgecolor='orange', alpha=0.95, fontsize=12, weight='bold')
    
    create_clean_text_box(ax, 0.85, 1e2, 
                         f'OPTIMAL PHASE\n{optimal_count}/{total_count} pairs ({optimal_count/total_count*100:.0f}%)\nAvg alignment: {optimal_avg_align:.4f}',
                         facecolor='white', edgecolor='green', alpha=0.95, fontsize=12, weight='bold')
    
    # Highlight best performers with clean annotations
    if chaotic_count > 0:
        chaotic_indices = np.where(chaotic_mask)[0]
        best_chaotic_idx = chaotic_indices[np.argmax(alignments[chaotic_mask])]
        best_chaotic = (ntk_values[best_chaotic_idx], agop_values[best_chaotic_idx], alignments[best_chaotic_idx])
        
        ax.scatter(*best_chaotic[:2], s=400, facecolors='none', edgecolors='red', linewidths=4, zorder=20)
        
        # Clean annotation for best chaotic
        ax.annotate(f'Best Chaotic\n{model_pairs[best_chaotic_idx]}\nAlignment: {best_chaotic[2]:.4f}', 
                    xy=best_chaotic[:2], xytext=(0.4, 1e3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.8),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95, 
                             edgecolor='red', linewidth=1.5))
    
    if optimal_count > 0:
        optimal_indices = np.where(optimal_mask)[0]
        best_optimal_idx = optimal_indices[np.argmax(alignments[optimal_mask])]
        best_optimal = (ntk_values[best_optimal_idx], agop_values[best_optimal_idx], alignments[best_optimal_idx])
        
        ax.scatter(*best_optimal[:2], s=400, facecolors='none', edgecolors='green', linewidths=4, zorder=20)
        
        # Clean annotation for best optimal
        ax.annotate(f'Best Optimal\n{model_pairs[best_optimal_idx]}\nAlignment: {best_optimal[2]:.4f}', 
                    xy=best_optimal[:2], xytext=(0.6, 1e8),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.8),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95, 
                             edgecolor='green', linewidth=1.5))
    
    # Highlight extreme outliers with clean annotation
    extreme_mask = (ntk_values == 0.0) & (agop_values > 1e10)
    if np.any(extreme_mask):
        extreme_indices = np.where(extreme_mask)[0]
        for idx in extreme_indices[:3]:
            ax.scatter(ntk_values[idx], agop_values[idx], s=200, 
                      facecolors='yellow', edgecolors='black', linewidths=2, zorder=20)
        
        ax.annotate('EfficientNet-B1 Pairs\nNTK = 0, AGOP ~ 1e15\n(Complete Phase Collapse)', 
                    xy=(0.0, 1e15), xytext=(0.2, 1e13),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.8),
                    fontsize=10, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.95, 
                             edgecolor='black', linewidth=1.5))
    
    # Add clean statistics box
    stats_text = f"""Phase Distribution Summary:
• Total Pairs: {total_count}
• Chaotic: {chaotic_count} ({chaotic_count/total_count*100:.1f}%)
• Transitional: {transition_count} ({transition_count/total_count*100:.1f}%)
• Optimal: {optimal_count} ({optimal_count/total_count*100:.1f}%)

Alignment Statistics:
• Maximum: {np.max(alignments):.4f}
• Mean: {np.mean(alignments):.4f}
• Minimum: {np.min(alignments):.4f}
• Std Dev: {np.std(alignments):.4f}

NTK Statistics:
• Maximum: {np.max(ntk_values):.4f}
• Mean: {np.mean(ntk_values):.4f}
• Minimum: {np.min(ntk_values):.4f}"""
    
    # Create a clean statistics box
    stats_box = FancyBboxPatch((0.02, 0.02), 0.35, 0.35,
                              boxstyle="round,pad=0.02",
                              facecolor="lightblue",
                              edgecolor="navy",
                              alpha=0.95,
                              linewidth=1.5,
                              zorder=20)
    
    ax.add_patch(stats_box)
    ax.text(0.195, 0.195, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            weight='normal', zorder=21)
    
    # Set axis properties
    ax.set_xlabel('Cross-Modal NTK Stability', fontsize=14, weight='bold')
    ax.set_ylabel('AGOP Magnitude (log scale)', fontsize=14, weight='bold')
    ax.set_yscale('log')
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(1e-1, 1e16)
    
    # Add colorbar with clean styling
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Alignment Score', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Title and grid
    ax.set_title('Enhanced Cross-Modal Phase Diagram\nPhase Incompatibility Prevents Cross-Modal Alignment', 
                 fontsize=16, pad=20, weight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Clean background
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('enhanced_phase_plot_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('enhanced_phase_plot_clean.pdf', bbox_inches='tight', facecolor='white')
    
    print(f"Clean enhanced phase plot saved as 'enhanced_phase_plot_clean.png' and 'enhanced_phase_plot_clean.pdf'")
    print(f"Plot shows {total_count} model pairs with phase distribution:")
    print(f"  - Chaotic: {chaotic_count} ({chaotic_count/total_count*100:.1f}%)")
    print(f"  - Transitional: {transition_count} ({transition_count/total_count*100:.1f}%)")
    print(f"  - Optimal: {optimal_count} ({optimal_count/total_count*100:.1f}%)")
    
    return fig, ax

def main():
    """Main function to create the clean enhanced phase plot"""
    
    print("Creating clean enhanced phase plot with improved text boxes...")
    
    # Create the plot
    fig, ax = create_enhanced_phase_plot_clean()
    
    if fig is not None:
        print("Clean enhanced phase plot created successfully!")
    else:
        print("Failed to create clean enhanced phase plot.")

if __name__ == "__main__":
    main()
