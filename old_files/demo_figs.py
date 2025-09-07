import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json

# Set style for publication
plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

def create_cross_modal_phase_figure(json_file='cross_modal_phase_analysis.json'):
    """Create comprehensive cross-modal phase landscape figure"""
    
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract metrics from all pairs
    ntk_values = []
    agop_values = []
    alignment_values = []
    phases = []
    
    for result in data['phase_results']:
        ntk_values.append(result['ntk_stability'])
        agop_values.append(np.log10(result['agop_magnitude']) if result['agop_magnitude'] > 0 else 0)
        alignment_values.append(result['alignment'])
        phases.append(result['phase_region'])
    
    ntk_values = np.array(ntk_values)
    agop_values = np.array(agop_values)
    alignment_values = np.array(alignment_values)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 5))
    
    # ========== Panel A: Main Phase Landscape ==========
    ax1 = plt.subplot(131)
    
    # Phase regions
    ax1.axvspan(0, 0.5, alpha=0.2, color='red', label='Chaotic Phase')
    ax1.axvspan(0.5, 0.9, alpha=0.2, color='green', label='Optimal Phase')
    ax1.axvspan(0.9, 1.0, alpha=0.2, color='blue', label='Lazy Phase')
    
    # Create custom colormap
    colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('alignment', colors, N=n_bins)
    
    # Scatter plot with all 144 points
    scatter = ax1.scatter(ntk_values, agop_values, 
                         c=alignment_values, cmap=cmap, 
                         s=50, alpha=0.7, edgecolors='black', 
                         linewidth=0.5, vmin=0.012, vmax=0.025)
    
    # Add phase annotations
    chaotic_mask = ntk_values < 0.5
    optimal_mask = (ntk_values >= 0.5) & (ntk_values < 0.9)
    
    n_chaotic = np.sum(chaotic_mask)
    n_optimal = np.sum(optimal_mask)
    
    # Add text annotations
    ax1.text(0.25, 14, f'CHAOTIC PHASE\n{n_chaotic} pairs ({n_chaotic/144*100:.0f}%)', 
            ha='center', va='center', fontsize=11, weight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax1.text(0.7, 3, f'"OPTIMAL" PHASE\n{n_optimal} pairs ({n_optimal/144*100:.0f}%)\nStill 36× worse than\nwithin-modal', 
            ha='center', va='center', fontsize=10, weight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Highlight best performers
    best_idx = np.argmax(alignment_values)
    ax1.annotate(f'Best: {alignment_values[best_idx]:.4f}\n(vs >0.900 within-modal)', 
                xy=(ntk_values[best_idx], agop_values[best_idx]),
                xytext=(0.9, 8), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax1.set_xlabel('Cross-Modal NTK Stability ($S_{NTK}$)')
    ax1.set_ylabel('log(AGOP Magnitude)')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-1, 17)
    ax1.set_title('(a) Phase Landscape: 144 Vision-Language Pairs', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label('Cross-Modal Alignment Score', rotation=270, labelpad=20)
    
    # ========== Panel B: Alignment Distribution ==========
    ax2 = plt.subplot(132)
    
    # Create box plots for chaotic vs optimal
    chaotic_align = alignment_values[chaotic_mask]
    optimal_align = alignment_values[optimal_mask]
    
    # Box plot
    bp = ax2.boxplot([chaotic_align, optimal_align], 
                     labels=['Chaotic\n(n=108)', 'Optimal\n(n=36)'],
                     patch_artist=True, showmeans=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    # Add individual points
    x1 = np.random.normal(1, 0.04, size=len(chaotic_align))
    x2 = np.random.normal(2, 0.04, size=len(optimal_align))
    ax2.scatter(x1, chaotic_align, alpha=0.4, s=20, color='darkred')
    ax2.scatter(x2, optimal_align, alpha=0.4, s=20, color='darkgreen')
    
    # Add horizontal line for within-modal performance
    ax2.axhline(y=0.9, color='blue', linestyle='--', linewidth=2, label='Within-modal (>0.900)')
    
    # Statistics
    ax2.text(1, 0.025, f'μ={np.mean(chaotic_align):.4f}\nσ={np.std(chaotic_align):.4f}', 
            ha='center', va='bottom', fontsize=9)
    ax2.text(2, 0.025, f'μ={np.mean(optimal_align):.4f}\nσ={np.std(optimal_align):.4f}', 
            ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Alignment Score')
    ax2.set_ylim(0, 1)
    ax2.set_title('(b) Phase-Dependent Alignment', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========== Panel C: Architecture Analysis ==========
    ax3 = plt.subplot(133)
    
    # Calculate architecture-specific metrics
    arch_data = {}
    vision_models = ['resnet', 'vit', 'convnext', 'efficientnet', 'mixer', 'swin']
    
    for i, result in enumerate(data['phase_results']):
        v_model = result['v_model']
        # Extract architecture type
        for arch in vision_models:
            if arch in v_model:
                if arch not in arch_data:
                    arch_data[arch] = {'ntk': [], 'align': []}
                arch_data[arch]['ntk'].append(ntk_values[i])
                arch_data[arch]['align'].append(alignment_values[i])
                break
    
    # Calculate means
    architectures = []
    mean_ntk = []
    mean_align = []
    
    for arch in ['resnet', 'vit', 'convnext', 'efficientnet', 'mixer', 'swin']:
        if arch in arch_data:
            architectures.append(arch.capitalize())
            mean_ntk.append(np.mean(arch_data[arch]['ntk']))
            mean_align.append(np.mean(arch_data[arch]['align']))
    
    # Create scatter plot
    colors_arch = ['red', 'red', 'orange', 'yellow', 'red', 'green']
    ax3.scatter(mean_ntk, mean_align, s=200, c=colors_arch, 
               alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add architecture labels
    for i, arch in enumerate(architectures):
        ax3.annotate(arch, (mean_ntk[i], mean_align[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add phase boundary
    ax3.axvline(x=0.5, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax3.text(0.5, 0.022, 'Phase Boundary', rotation=90, 
            va='bottom', ha='right', fontsize=9, color='green')
    
    ax3.set_xlabel('Average NTK Stability')
    ax3.set_ylabel('Average Alignment Score')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0.015, 0.020)
    ax3.set_title('(c) Architecture-Phase-Alignment Relationship', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Modal Alignment Fails Universally Across All Phases\n' + 
                'Even "optimal" phase pairs achieve <3% alignment (vs >90% within-modal)',
                fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save figure
    plt.savefig('cross_modal_phase_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cross_modal_phase_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total pairs analyzed: {len(ntk_values)}")
    print(f"Chaotic phase: {n_chaotic} pairs ({n_chaotic/144*100:.1f}%)")
    print(f"Optimal phase: {n_optimal} pairs ({n_optimal/144*100:.1f}%)")
    print(f"\nAlignment scores:")
    print(f"Overall: {np.mean(alignment_values):.4f} ± {np.std(alignment_values):.4f}")
    print(f"Chaotic: {np.mean(chaotic_align):.4f} ± {np.std(chaotic_align):.4f}")
    print(f"Optimal: {np.mean(optimal_align):.4f} ± {np.std(optimal_align):.4f}")
    print(f"Best: {np.max(alignment_values):.4f}")
    print(f"\nWithin-modal baseline: >0.900")
    print(f"Performance gap: {0.9/np.max(alignment_values):.1f}× worse")

# Create the figure
# Uncomment the line below when you have the JSON file
# create_cross_modal_phase_figure('cross_modal_phase_analysis.json')

# For demonstration, here's a simplified version that doesn't require the JSON file
def create_demo_figure():
    """Create a demonstration version using your summary statistics"""
    
    # Generate synthetic data based on your summary
    np.random.seed(42)
    
    # Chaotic phase (108 points)
    chaotic_ntk = np.random.normal(0.20, 0.08, 108)
    chaotic_ntk = np.clip(chaotic_ntk, 0, 0.49)
    chaotic_agop = np.random.normal(8, 3, 108)
    chaotic_align = np.random.normal(0.0173, 0.0018, 108)
    
    # Optimal phase (36 points)
    optimal_ntk = np.random.normal(0.75, 0.1, 36)
    optimal_ntk = np.clip(optimal_ntk, 0.5, 0.89)
    optimal_agop = np.random.normal(2, 1, 36)
    optimal_align = np.random.normal(0.0178, 0.0022, 36)
    
    # Combine
    ntk_values = np.concatenate([chaotic_ntk, optimal_ntk])
    agop_values = np.concatenate([chaotic_agop, optimal_agop])
    alignment_values = np.concatenate([chaotic_align, optimal_align])
    
    # Create main figure
    plt.figure(figsize=(12, 8))
    
    # Phase regions
    plt.axvspan(0, 0.5, alpha=0.2, color='red')
    plt.axvspan(0.5, 0.9, alpha=0.2, color='green')
    plt.axvspan(0.9, 1.0, alpha=0.2, color='blue')
    
    # Scatter plot
    scatter = plt.scatter(ntk_values, agop_values, 
                         c=alignment_values, cmap='RdYlGn', 
                         s=60, alpha=0.7, edgecolors='black', 
                         linewidth=0.5, vmin=0.012, vmax=0.025)
    
    # Annotations
    plt.text(0.25, 14, 'CHAOTIC PHASE\n108 pairs (75%)\nGradient directions ~orthogonal', 
            ha='center', va='center', fontsize=12, weight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.text(0.7, 5, '"OPTIMAL" PHASE\n36 pairs (25%)\nStill 36× worse than within-modal', 
            ha='center', va='center', fontsize=11, weight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Best performer annotation (no arrow to avoid crossing text)
    plt.text(0.85, 12, 'Best: EfficientNet-B2\n+ DistilRoBERTa\n0.0247', 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))
    
    # Reference box
    plt.text(0.95, 15, 'Within-modal:\n>0.900\n(36× better)', 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    plt.xlabel('Cross-Modal NTK Stability ($S_{NTK}$)', fontsize=14)
    plt.ylabel('log(AGOP Magnitude)', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1, 17)
    plt.title('Cross-Modal Alignment Fails Universally Across All Phases\n' + 
             'Even "optimal" phase pairs achieve <3% alignment (vs >90% within-modal)',
             fontsize=16, weight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Cross-Modal Alignment Score', rotation=270, labelpad=25, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('demo_figure_clean.png', dpi=300, bbox_inches='tight')
    plt.savefig('demo_figure_clean.pdf', dpi=300, bbox_inches='tight')
    print("Demo figure saved as: demo_figure_clean.png/pdf")
    plt.show()

# Create the demo figure
create_demo_figure()