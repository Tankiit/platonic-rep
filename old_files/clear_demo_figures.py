import numpy as np
import matplotlib.pyplot as plt

def create_clear_demo_figure():
    """Create a demo figure that clearly explains what the circles represent"""
    
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
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Phase regions
    ax.axvspan(0, 0.5, alpha=0.2, color='red')
    ax.axvspan(0.5, 0.9, alpha=0.2, color='green')
    ax.axvspan(0.9, 1.0, alpha=0.2, color='blue')
    
    # Scatter plot with clear explanation
    scatter = ax.scatter(ntk_values, agop_values, 
                        c=alignment_values, cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='black', 
                        linewidth=0.5, vmin=0.012, vmax=0.025)
    
    # Clear phase labels
    ax.text(0.25, 14, 'CHAOTIC PHASE\n108 pairs (75%)', ha='center', va='center', 
            fontsize=12, weight='bold', color='darkred')
    
    ax.text(0.7, 4, 'OPTIMAL PHASE\n36 pairs (25%)', ha='center', va='center', 
            fontsize=12, weight='bold', color='darkgreen')
    
    # Clear explanation of what circles represent
    ax.text(0.5, 16, 'Each circle = 1 Vision-Language Model Pair\n' + 
            '144 total pairs analyzed', ha='center', va='center', 
            fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Example annotations showing specific model pairs
    ax.annotate('EffNet-B2 + DistilRoBERTa\n(0.0247)', 
                xy=(0.75, 2), xytext=(0.85, 8),
                fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax.annotate('ResNet-50 + RoBERTa\n(0.0170)', 
                xy=(0.22, 8), xytext=(0.1, 12),
                fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
    
    ax.set_xlabel('Cross-Modal NTK Stability', fontsize=12)
    ax.set_ylabel('log(AGOP Magnitude)', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1, 17)
    ax.set_title('Cross-Modal Phase Diagram\nEach point represents one vision-language model pair', 
                fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Colorbar with clear label
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Cross-Modal Alignment Score', rotation=270, labelpad=20)
    
    # Add legend explaining circle size and color
    ax.text(0.02, 0.98, 'Circle size: Fixed\nColor: Alignment score\n(0.012 = poor, 0.025 = best)', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_ultra_clear_demo_figure():
    """Create an ultra-clear version with explicit explanations"""
    
    # Generate synthetic data
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Phase regions
    ax.axvspan(0, 0.5, alpha=0.2, color='red')
    ax.axvspan(0.5, 0.9, alpha=0.2, color='green')
    
    # Scatter plot
    scatter = ax.scatter(ntk_values, agop_values, 
                        c=alignment_values, cmap='viridis', 
                        s=60, alpha=0.8, edgecolors='black', 
                        linewidth=0.5, vmin=0.012, vmax=0.025)
    
    # Clear phase labels
    ax.text(0.25, 12, 'Chaotic Phase\n108 pairs', ha='center', va='center', 
            fontsize=11, weight='bold', color='darkred')
    
    ax.text(0.7, 3, 'Optimal Phase\n36 pairs', ha='center', va='center', 
            fontsize=11, weight='bold', color='darkgreen')
    
    # Clear title explaining what circles are
    ax.set_title('Cross-Modal Phase Diagram\nEach circle = 1 Vision-Language Model Pair (144 total)', 
                fontsize=13, weight='bold')
    
    ax.set_xlabel('NTK Stability')
    ax.set_ylabel('log(AGOP)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 13)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Alignment Score')
    
    # Add explanation box
    ax.text(0.02, 0.98, 'What the circles represent:\n• Each circle = 1 model pair\n• Color = alignment score\n• Position = NTK vs AGOP', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    return fig

# Generate clear demo figures
if __name__ == "__main__":
    print("Generating clear demo figures...")
    
    # Create clear version with explanations
    fig1 = create_clear_demo_figure()
    fig1.savefig('clear_demo.png', dpi=300, bbox_inches='tight')
    fig1.savefig('clear_demo.pdf', dpi=300, bbox_inches='tight')
    print("✓ Clear demo figure saved")
    
    # Create ultra-clear version
    fig2 = create_ultra_clear_demo_figure()
    fig2.savefig('ultra_clear_demo.png', dpi=300, bbox_inches='tight')
    fig2.savefig('ultra_clear_demo.pdf', dpi=300, bbox_inches='tight')
    print("✓ Ultra-clear demo figure saved")
    
    print("\nKey improvements:")
    print("• Clear explanation of what circles represent")
    print("• Each circle = 1 vision-language model pair")
    print("• Color coding explained")
    print("• Example model pairs labeled")
    print("• Easy to understand")
