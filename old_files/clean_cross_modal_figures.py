import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Set style for cleaner figures
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_clean_cross_modal_analysis():
    """Create a clean cross-modal alignment analysis without overlapping text"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Simplified data - key model pairs only
    model_pairs = [
        {"v": "EffNet-B2", "t": "DistilRoBERTa", "ntk": 0.81, "agop": 1.78, "align": 0.0247, "phase": "optimal"},
        {"v": "EffNet-B2", "t": "XLNet-L", "ntk": 0.82, "agop": 1.66, "align": 0.0212, "phase": "optimal"},
        {"v": "ConvNeXt-S", "t": "DistilRoBERTa", "ntk": 0.18, "agop": 20.4, "align": 0.0210, "phase": "chaotic"},
        {"v": "Swin-B", "t": "GPT2-L", "ntk": 0.81, "agop": 0.27, "align": 0.0201, "phase": "optimal"},
        {"v": "ResNet-50", "t": "XLNet-L", "ntk": 0.20, "agop": 4.00, "align": 0.0200, "phase": "chaotic"},
        {"v": "EffNet-B1", "t": "RoBERTa-L", "ntk": 0.00, "agop": 1e15, "align": 0.0207, "phase": "chaotic"}
    ]
    
    # Extract data
    ntk_values = [p["ntk"] for p in model_pairs]
    agop_values = [p["agop"] for p in model_pairs]
    align_values = [p["align"] for p in model_pairs]
    phases = [p["phase"] for p in model_pairs]
    
    # Convert AGOP to log scale for visualization
    agop_log = []
    for agop in agop_values:
        if agop > 1e6:
            agop_log.append(8)  # Cap extreme values
        else:
            agop_log.append(np.log10(agop + 1))
    
    # Create background regions
    chaotic_region = FancyBboxPatch((0, -0.5), 0.5, 9.5, 
                                   boxstyle="round,pad=0.02",
                                   facecolor='#ffe6e6', 
                                   edgecolor='none',
                                   alpha=0.5)
    optimal_region = FancyBboxPatch((0.5, -0.5), 0.5, 9.5,
                                   boxstyle="round,pad=0.02", 
                                   facecolor='#e6ffe6',
                                   edgecolor='none',
                                   alpha=0.5)
    ax.add_patch(chaotic_region)
    ax.add_patch(optimal_region)
    
    # Add phase boundary
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Plot points by phase
    chaotic_mask = [p == "chaotic" for p in phases]
    optimal_mask = [p == "optimal" for p in phases]
    
    # Plot chaotic points
    sc1 = ax.scatter([ntk_values[i] for i, m in enumerate(chaotic_mask) if m],
                    [agop_log[i] for i, m in enumerate(chaotic_mask) if m],
                    c=[align_values[i] for i, m in enumerate(chaotic_mask) if m],
                    s=200,
                    cmap='Reds',
                    vmin=0.011,
                    vmax=0.025,
                    edgecolors='darkred',
                    linewidth=1.2,
                    alpha=0.8,
                    label='Chaotic phase')
    
    # Plot optimal points
    sc2 = ax.scatter([ntk_values[i] for i, m in enumerate(optimal_mask) if m],
                    [agop_log[i] for i, m in enumerate(optimal_mask) if m],
                    c=[align_values[i] for i, m in enumerate(optimal_mask) if m],
                    s=200,
                    cmap='Greens',
                    vmin=0.011,
                    vmax=0.025,
                    edgecolors='darkgreen',
                    linewidth=1.2,
                    alpha=0.8,
                    label='Optimal phase')
    
    # Add colorbar
    cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label('Cross-Modal Alignment Score', fontsize=12)
    
    # Add only 3 key annotations (spaced out to avoid overlap)
    key_annotations = [
        (0.81, np.log10(1.78 + 1), 'Best: EffNet-B2\n+ DistilRoBERTa\n0.0247', 0.65, 6.5),
        (0.18, np.log10(20.4 + 1), 'Best chaotic:\nConvNeXt + DistilRoBERTa\n0.0210', 0.25, 5),
        (0.02, 8, 'Extreme AGOP:\nEffNet-B1 + RoBERTa\n(NTK=0, AGOP=10¹⁵)', 0.1, 7)
    ]
    
    for x, y, label, x_text, y_text in key_annotations:
        ax.annotate(label, xy=(x, y), xytext=(x_text, y_text),
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5))
    
    # Add simple statistics box (top right)
    stats_text = (
        '144 Model Pairs\n'
        'Chaotic: 75%\n'
        'Optimal: 25%\n'
        'Best: 0.0247\n'
        'Within-modal: >0.900'
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95),
            family='monospace')
    
    # Add phase labels in corners (no overlap)
    ax.text(0.25, 8, 'CHAOTIC PHASE\n(75% of pairs)', fontsize=14, fontweight='bold',
            ha='center', va='center', color='darkred', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6', alpha=0.8))
    
    ax.text(0.75, 8, 'OPTIMAL PHASE\n(25% of pairs)', fontsize=14, fontweight='bold',
            ha='center', va='center', color='darkgreen', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e6ffe6', alpha=0.8))
    
    # Labels and title
    ax.set_xlabel('Cross-Modal NTK Stability', fontsize=14)
    ax.set_ylabel('log₁₀(AGOP Magnitude)', fontsize=14)
    ax.set_title('Cross-Modal Alignment Analysis\n' + 
                'Even "optimal" pairs achieve <3% alignment (vs >90% within-modal)',
                fontsize=16, pad=20)
    
    # Set axis limits
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.5, 8.5)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    return fig

def create_clean_alignment_comparison():
    """Create a clean comparison of cross-modal vs within-modal alignment"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    categories = ['Cross-Modal\n(Best)', 'Cross-Modal\n(Average)', 'Within-Modal\n(Typical)']
    values = [0.0247, 0.017, 0.900]
    colors = ['#27ae60', '#f39c12', '#3498db']
    
    # Create bar chart
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Alignment Score')
    ax.set_title('Cross-Modal vs Within-Modal Alignment')
    ax.set_ylim(0, 1.0)
    
    # Add key insight
    ax.text(0.5, 0.95, 'Cross-modal alignment is 36× worse than within-modal', 
           transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_clean_phase_summary():
    """Create a clean summary of phase distribution"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data
    phases = ['Chaotic', 'Optimal', 'Lazy']
    percentages = [75, 25, 0]
    colors = ['#e74c3c', '#27ae60', '#3498db']
    
    # Create horizontal bar chart
    bars = ax.barh(phases, percentages, color=colors, alpha=0.8)
    
    # Add percentage labels
    for bar, percentage in zip(bars, percentages):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
               f'{percentage}%', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Percentage of Model Pairs')
    ax.set_title('Phase Distribution Across 144 Model Pairs')
    ax.set_xlim(0, 100)
    
    # Add insight
    ax.text(0.5, 0.95, '75% of pairs show unstable cross-modal dynamics', 
           transform=ax.transAxes, ha='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Generate clean cross-modal figures
if __name__ == "__main__":
    print("Generating clean cross-modal figures...")
    
    # Create clean cross-modal analysis
    fig1 = create_clean_cross_modal_analysis()
    fig1.savefig('clean_cross_modal_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Clean cross-modal analysis saved")
    
    # Create alignment comparison
    fig2 = create_clean_alignment_comparison()
    fig2.savefig('clean_alignment_comparison_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ Clean alignment comparison saved")
    
    # Create phase summary
    fig3 = create_clean_phase_summary()
    fig3.savefig('clean_phase_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Clean phase summary saved")
    
    print("\nAll clean cross-modal figures generated!")
    print("Key improvements:")
    print("• No overlapping text")
    print("• Clear, readable labels")
    print("• Strategic positioning")
    print("• Professional appearance")
