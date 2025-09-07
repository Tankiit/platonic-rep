#!/usr/bin/env python3
"""
Create an enhanced phase plot with information theoretic analysis
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
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_data():
    """Load both phase and mutual information data"""
    
    # Load phase data
    try:
        phase_df = pd.read_csv('all_analysis_results.csv')
        cross_modal_df = phase_df[phase_df['type'] == 'cross_modal'].copy()
        print(f"Loaded {len(cross_modal_df)} cross-modal pairs")
    except FileNotFoundError:
        print("Phase analysis file not found.")
        return None, None
    
    # Load mutual information data
    try:
        mi_df = pd.read_csv('mutual_information_results.csv')
        print(f"Loaded {len(mi_df)} mutual information measurements")
    except FileNotFoundError:
        print("Mutual information file not found.")
        return cross_modal_df, None
    
    return cross_modal_df, mi_df

def create_clean_text_box(ax, x, y, text, boxstyle="round,pad=0.3", 
                         facecolor="white", edgecolor="black", alpha=0.95,
                         fontsize=10, weight='normal', ha='center', va='center'):
    """Create a clean, professional text box"""
    
    fancy_box = FancyBboxPatch((x-0.08, y-0.04), 0.16, 0.08,
                              boxstyle=boxstyle,
                              facecolor=facecolor,
                              edgecolor=edgecolor,
                              alpha=alpha,
                              linewidth=1.2,
                              zorder=20)
    
    ax.add_patch(fancy_box)
    ax.text(x, y, text, fontsize=fontsize, weight=weight, 
           ha=ha, va=va, zorder=21)

def create_enhanced_phase_plot_with_info():
    """Create enhanced phase plot with information theoretic analysis"""
    
    # Load data
    phase_df, mi_df = load_data()
    
    if phase_df is None:
        print("No phase data available")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    
    # Main phase plot (top left)
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # Extract phase data
    ntk_values = np.array(phase_df['ntk_stability'].tolist())
    agop_values = np.array(phase_df['agop_magnitude'].tolist())
    alignments = np.array(phase_df['alignment'].tolist())
    model_pairs = [f"{row['v_model']}+{row['t_model']}" for _, row in phase_df.iterrows()]
    
    # Plot phase diagram
    scatter = ax1.scatter(ntk_values, agop_values, c=alignments, 
                        cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.6,
                        vmin=0, vmax=max(alignments), zorder=5)
    
    # Add phase boundaries
    ax1.axvline(x=0.7, color='red', linewidth=3, linestyle='--', 
               label='Optimal Threshold (NTK=0.7)', zorder=10)
    ax1.axvline(x=0.5, color='orange', linewidth=2, linestyle=':', 
               label='Chaotic Threshold (NTK=0.5)', zorder=10)
    
    # Add phase regions
    chaotic_patch = Rectangle((0, 1e-2), 0.5, 1e16, alpha=0.1, facecolor='red', zorder=1)
    transition_patch = Rectangle((0.5, 1e-2), 0.2, 1e16, alpha=0.1, facecolor='yellow', zorder=1)
    optimal_patch = Rectangle((0.7, 1e-2), 0.3, 1e16, alpha=0.1, facecolor='green', zorder=1)
    
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
    
    # Add clean phase labels
    create_clean_text_box(ax1, 0.25, 1e10, 
                         f'CHAOTIC\n{chaotic_count}/{total_count} ({chaotic_count/total_count*100:.0f}%)\nAvg: {chaotic_avg_align:.4f}',
                         facecolor='white', edgecolor='red', alpha=0.95, fontsize=11, weight='bold')
    
    create_clean_text_box(ax1, 0.6, 1e6, 
                         f'TRANSITIONAL\n{transition_count}/{total_count} ({transition_count/total_count*100:.0f}%)\nAvg: {transition_avg_align:.4f}',
                         facecolor='white', edgecolor='orange', alpha=0.95, fontsize=11, weight='bold')
    
    create_clean_text_box(ax1, 0.85, 1e2, 
                         f'OPTIMAL\n{optimal_count}/{total_count} ({optimal_count/total_count*100:.0f}%)\nAvg: {optimal_avg_align:.4f}',
                         facecolor='white', edgecolor='green', alpha=0.95, fontsize=11, weight='bold')
    
    # Highlight best performers
    if chaotic_count > 0:
        chaotic_indices = np.where(chaotic_mask)[0]
        best_chaotic_idx = chaotic_indices[np.argmax(alignments[chaotic_mask])]
        best_chaotic = (ntk_values[best_chaotic_idx], agop_values[best_chaotic_idx], alignments[best_chaotic_idx])
        
        ax1.scatter(*best_chaotic[:2], s=300, facecolors='none', edgecolors='red', linewidths=3, zorder=20)
    
    if optimal_count > 0:
        optimal_indices = np.where(optimal_mask)[0]
        best_optimal_idx = optimal_indices[np.argmax(alignments[optimal_mask])]
        best_optimal = (ntk_values[best_optimal_idx], agop_values[best_optimal_idx], alignments[best_optimal_idx])
        
        ax1.scatter(*best_optimal[:2], s=300, facecolors='none', edgecolors='green', linewidths=3, zorder=20)
    
    # Set main plot properties
    ax1.set_xlabel('Cross-Modal NTK Stability', fontsize=12, weight='bold')
    ax1.set_ylabel('AGOP Magnitude (log scale)', fontsize=12, weight='bold')
    ax1.set_yscale('log')
    ax1.set_xlim(-0.05, 1.0)
    ax1.set_ylim(1e-1, 1e16)
    ax1.set_title('Cross-Modal Phase Diagram', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label('Alignment Score', fontsize=10, weight='bold')
    
    # Information Plane Analysis (top right)
    ax2 = plt.subplot(2, 3, 3)
    
    if mi_df is not None:
        # Group by model and create trajectories
        model_groups = mi_df.groupby('model')
        
        # Sample trajectories for different model types
        trajectories = {}
        colors = ['red', 'darkred', 'green', 'blue', 'purple']
        
        for i, (model_name, group) in enumerate(model_groups):
            if i >= 5:  # Limit to 5 models for clarity
                break
                
            # Get layer-wise data
            layer_data = group[group['layer_idx'] >= 0].sort_values('layer_idx')
            
            if len(layer_data) > 1:
                # Determine phase based on average NTK from phase data
                model_phase_data = phase_df[phase_df['v_model'].str.contains(model_name.split('_')[0], case=False)]
                avg_ntk = model_phase_data['ntk_stability'].mean() if len(model_phase_data) > 0 else 0.3
                
                phase_type = 'Optimal' if avg_ntk >= 0.7 else 'Chaotic'
                color = 'green' if phase_type == 'Optimal' else 'red'
                
                trajectories[f'{model_name} ({phase_type})'] = {
                    'I_X_T': layer_data['I_X_T'].tolist(),
                    'I_Y_T': layer_data['I_Y_T'].tolist(),
                    'color': color
                }
        
        # Plot trajectories
        for name, traj in trajectories.items():
            if len(traj['I_X_T']) > 1:
                ax2.plot(traj['I_X_T'], traj['I_Y_T'], 'o-', 
                        color=traj['color'], markersize=6, linewidth=2,
                        label=name, alpha=0.8)
                
                # Add arrows to show direction
                for i in range(len(traj['I_X_T'])-1):
                    ax2.annotate('', xy=(traj['I_X_T'][i+1], traj['I_Y_T'][i+1]), 
                                xytext=(traj['I_X_T'][i], traj['I_Y_T'][i]),
                                arrowprops=dict(arrowstyle='->', color=traj['color'], 
                                              alpha=0.6, lw=1))
        
        # Add compression zones
        ax2.axvspan(0.030, 0.033, alpha=0.2, color='red', label='Over-compression')
        ax2.axvspan(0.033, 0.036, alpha=0.2, color='green', label='Optimal compression')
        
        ax2.set_xlabel('I(X;T) - Input Information', fontsize=11)
        ax2.set_ylabel('I(Y;T) - Task Information', fontsize=11)
        ax2.set_title('Information Compression Patterns', fontsize=12, weight='bold')
        ax2.legend(loc='lower left', fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Mutual Information\nData Not Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Information Compression Patterns', fontsize=12, weight='bold')
    
    # Compression Rate vs NTK (bottom left)
    ax3 = plt.subplot(2, 3, 4)
    
    if mi_df is not None:
        # Calculate compression rates
        compression_data = []
        
        for model_name in mi_df['model'].unique():
            model_data = mi_df[mi_df['model'] == model_name]
            layer_data = model_data[model_data['layer_idx'] >= 0].sort_values('layer_idx')
            
            if len(layer_data) > 1:
                initial_I_X_T = layer_data.iloc[0]['I_X_T']
                final_I_X_T = layer_data.iloc[-1]['I_X_T']
                compression_rate = (initial_I_X_T - final_I_X_T) / initial_I_X_T if initial_I_X_T > 0 else 0
                
                # Get NTK and alignment from phase data
                model_phase_data = phase_df[phase_df['v_model'].str.contains(model_name.split('_')[0], case=False)]
                if len(model_phase_data) > 0:
                    ntk = model_phase_data['ntk_stability'].mean()
                    alignment = model_phase_data['alignment'].mean()
                    compression_data.append((ntk, compression_rate, alignment))
        
        if compression_data:
            ntks, compressions, aligns = zip(*compression_data)
            
            scatter3 = ax3.scatter(ntks, compressions, c=aligns, cmap='RdYlGn', 
                                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax3.axvline(x=0.5, color='red', linestyle='--', lw=2, label='Phase boundary')
            ax3.axvline(x=0.7, color='green', linestyle='--', lw=2, label='Optimal boundary')
            
            # Add trend line
            valid_data = [(n, c) for n, c in zip(ntks, compressions) if n > 0 and not np.isnan(c)]
            if len(valid_data) > 2:
                valid_ntks, valid_comps = zip(*valid_data)
                z = np.polyfit(valid_ntks, valid_comps, 1)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(valid_ntks), max(valid_ntks), 100)
                ax3.plot(x_smooth, p(x_smooth), 'b-', alpha=0.6, lw=2, label='Trend')
            
            # Annotate regions
            ax3.text(0.25, 0.8, 'Chaotic:\nAggressive\ncompression', 
                     fontsize=9, ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2))
            ax3.text(0.85, 0.2, 'Optimal:\nControlled\ncompression', 
                     fontsize=9, ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.2))
            
            ax3.set_xlabel('NTK Stability', fontsize=11)
            ax3.set_ylabel('Compression Rate', fontsize=11)
            ax3.set_title('Phase Controls Information Compression', fontsize=12, weight='bold')
            ax3.legend(loc='upper right', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
            cbar3.set_label('Alignment', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Compression Data\nNot Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=11)
            ax3.set_title('Phase Controls Information Compression', fontsize=12, weight='bold')
    else:
        ax3.text(0.5, 0.5, 'Mutual Information\nData Not Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title('Phase Controls Information Compression', fontsize=12, weight='bold')
    
    # Alignment Distribution (bottom center)
    ax4 = plt.subplot(2, 3, 5)
    
    ax4.hist(alignments, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(np.mean(alignments), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(alignments):.4f}')
    ax4.axvline(np.median(alignments), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(alignments):.4f}')
    ax4.set_xlabel('Alignment Score', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Alignment Distribution', fontsize=12, weight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # NTK vs Alignment (bottom right)
    ax5 = plt.subplot(2, 3, 6)
    
    scatter5 = ax5.scatter(ntk_values, alignments, c=agop_values, 
                          cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax5.set_xlabel('NTK Stability', fontsize=11)
    ax5.set_ylabel('Alignment Score', fontsize=11)
    ax5.set_title('NTK vs Alignment\n(colored by AGOP)', fontsize=12, weight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(ntk_values, alignments)[0, 1]
    ax5.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax5.transAxes, fontsize=10, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add colorbar
    cbar5 = plt.colorbar(scatter5, ax=ax5, shrink=0.8)
    cbar5.set_label('AGOP Magnitude', fontsize=9)
    
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
    
    # Create statistics box
    stats_box = FancyBboxPatch((0.02, 0.02), 0.25, 0.25,
                              boxstyle="round,pad=0.02",
                              facecolor="lightblue",
                              edgecolor="navy",
                              alpha=0.95,
                              linewidth=1.2,
                              zorder=20)
    
    ax1.add_patch(stats_box)
    ax1.text(0.145, 0.145, stats_text, transform=ax1.transAxes, fontsize=8,
            verticalalignment='center', horizontalalignment='center',
            weight='normal', zorder=21)
    
    # Clean background
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('enhanced_phase_plot_with_info.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('enhanced_phase_plot_with_info.pdf', bbox_inches='tight', facecolor='white')
    
    print(f"Enhanced phase plot with information analysis saved as:")
    print(f"  - enhanced_phase_plot_with_info.png")
    print(f"  - enhanced_phase_plot_with_info.pdf")
    print(f"Plot shows {total_count} model pairs with comprehensive analysis")
    
    return fig

def main():
    """Main function to create the enhanced phase plot with information analysis"""
    
    print("Creating enhanced phase plot with information theoretic analysis...")
    
    # Create the plot
    fig = create_enhanced_phase_plot_with_info()
    
    if fig is not None:
        print("Enhanced phase plot with information analysis created successfully!")
    else:
        print("Failed to create enhanced phase plot with information analysis.")

if __name__ == "__main__":
    main()
