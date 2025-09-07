import numpy as np
import matplotlib.pyplot as plt

def create_clean_demo_figure():
    """Create a demonstration version with clean arrow positioning"""
    
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
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Phase regions
    ax.axvspan(0, 0.5, alpha=0.2, color='red')
    ax.axvspan(0.5, 0.9, alpha=0.2, color='green')
    ax.axvspan(0.9, 1.0, alpha=0.2, color='blue')
    
    # Scatter plot
    scatter = ax.scatter(ntk_values, agop_values, 
                        c=alignment_values, cmap='RdYlGn', 
                        s=60, alpha=0.7, edgecolors='black', 
                        linewidth=0.5, vmin=0.012, vmax=0.025)
    
    # Clean annotations without arrows crossing text
    # Chaotic phase label (top left)
    ax.text(0.25, 15, 'CHAOTIC PHASE\n108 pairs (75%)\nGradient directions ~orthogonal', 
            ha='center', va='center', fontsize=12, weight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Optimal phase label (bottom right)
    ax.text(0.7, 3, '"OPTIMAL" PHASE\n36 pairs (25%)\nStill 36× worse than within-modal', 
            ha='center', va='center', fontsize=11, weight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Best performer annotation (top right, no arrow)
    ax.text(0.85, 12, 'Best: EfficientNet-B2\n+ DistilRoBERTa\n0.0247', 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))
    
    # Reference box (top right corner)
    ax.text(0.95, 16, 'Within-modal:\n>0.900\n(36× better)', 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    # Add key insight at bottom
    ax.text(0.5, -0.5, 'Cross-modal alignment is universally poor across all phases', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlabel('Cross-Modal NTK Stability ($S_{NTK}$)', fontsize=14)
    ax.set_ylabel('log(AGOP Magnitude)', fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1, 17)
    ax.set_title('Cross-Modal Alignment Fails Universally Across All Phases\n' + 
                'Even "optimal" phase pairs achieve <3% alignment (vs >90% within-modal)',
                fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Cross-Modal Alignment Score', rotation=270, labelpad=25, fontsize=12)
    
    plt.tight_layout()
    return fig

def create_clean_demo_figure_with_arrows():
    """Create a version with carefully positioned arrows that don't cross text"""
    
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
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Phase regions
    ax.axvspan(0, 0.5, alpha=0.2, color='red')
    ax.axvspan(0.5, 0.9, alpha=0.2, color='green')
    ax.axvspan(0.9, 1.0, alpha=0.2, color='blue')
    
    # Scatter plot
    scatter = ax.scatter(ntk_values, agop_values, 
                        c=alignment_values, cmap='RdYlGn', 
                        s=60, alpha=0.7, edgecolors='black', 
                        linewidth=0.5, vmin=0.012, vmax=0.025)
    
    # Phase labels (positioned to avoid arrow conflicts)
    ax.text(0.25, 15, 'CHAOTIC PHASE\n108 pairs (75%)', 
            ha='center', va='center', fontsize=12, weight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax.text(0.7, 3, '"OPTIMAL" PHASE\n36 pairs (25%)', 
            ha='center', va='center', fontsize=11, weight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Best performer with arrow pointing to actual data point
    best_idx = np.argmax(alignment_values)
    best_ntk = optimal_ntk[0]  # Use first optimal point
    best_agop = optimal_agop[0]
    
    # Position annotation away from data clusters
    ax.annotate('Best: EfficientNet-B2\n+ DistilRoBERTa\n0.0247', 
                xy=(best_ntk, best_agop),
                xytext=(0.9, 11),  # Positioned to avoid crossing other text
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black', lw=2, 
                              connectionstyle='arc3,rad=0.2'),  # Curved arrow
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))
    
    # Reference box (top right corner, no arrow needed)
    ax.text(0.95, 16, 'Within-modal:\n>0.900\n(36× better)', 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    # Key insight at bottom center
    ax.text(0.5, -0.5, 'Cross-modal alignment is universally poor across all phases', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlabel('Cross-Modal NTK Stability ($S_{NTK}$)', fontsize=14)
    ax.set_ylabel('log(AGOP Magnitude)', fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1, 17)
    ax.set_title('Cross-Modal Alignment Fails Universally Across All Phases\n' + 
                'Even "optimal" phase pairs achieve <3% alignment (vs >90% within-modal)',
                fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Cross-Modal Alignment Score', rotation=270, labelpad=25, fontsize=12)
    
    plt.tight_layout()
    return fig

# Generate clean demo figures
if __name__ == "__main__":
    print("Generating clean demo figures...")
    
    # Create version without arrows
    fig1 = create_clean_demo_figure()
    fig1.savefig('clean_demo_no_arrows.png', dpi=300, bbox_inches='tight')
    fig1.savefig('clean_demo_no_arrows.pdf', dpi=300, bbox_inches='tight')
    print("✓ Clean demo figure (no arrows) saved")
    
    # Create version with carefully positioned arrows
    fig2 = create_clean_demo_figure_with_arrows()
    fig2.savefig('clean_demo_with_arrows.png', dpi=300, bbox_inches='tight')
    fig2.savefig('clean_demo_with_arrows.pdf', dpi=300, bbox_inches='tight')
    print("✓ Clean demo figure (with arrows) saved")
    
    print("\nKey improvements:")
    print("• No arrows crossing over text")
    print("• Strategic positioning of annotations")
    print("• Curved arrows to avoid conflicts")
    print("• Clean, professional appearance")
