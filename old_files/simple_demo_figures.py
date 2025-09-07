import numpy as np
import matplotlib.pyplot as plt

def create_simple_demo_figure():
    """Create a very simple demo figure with minimal, non-overlapping text"""
    
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Phase regions
    ax.axvspan(0, 0.5, alpha=0.2, color='red')
    ax.axvspan(0.5, 0.9, alpha=0.2, color='green')
    ax.axvspan(0.9, 1.0, alpha=0.2, color='blue')
    
    # Scatter plot
    scatter = ax.scatter(ntk_values, agop_values, 
                        c=alignment_values, cmap='RdYlGn', 
                        s=50, alpha=0.7, edgecolors='black', 
                        linewidth=0.5, vmin=0.012, vmax=0.025)
    
    # Minimal text - only essential information
    # Phase labels in corners (no boxes)
    ax.text(0.25, 14, 'CHAOTIC\n(75%)', ha='center', va='center', 
            fontsize=12, weight='bold', color='darkred')
    
    ax.text(0.7, 4, 'OPTIMAL\n(25%)', ha='center', va='center', 
            fontsize=12, weight='bold', color='darkgreen')
    
    # Single key insight at bottom
    ax.text(0.5, -0.5, 'Cross-modal alignment is universally poor', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    ax.set_xlabel('NTK Stability', fontsize=12)
    ax.set_ylabel('log(AGOP)', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1, 15)
    ax.set_title('Cross-Modal Phase Diagram', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Alignment Score', rotation=270, labelpad=15)
    
    plt.tight_layout()
    return fig

def create_ultra_simple_demo_figure():
    """Create an ultra-simple version with no overlapping elements"""
    
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
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Phase regions
    ax.axvspan(0, 0.5, alpha=0.3, color='red')
    ax.axvspan(0.5, 0.9, alpha=0.3, color='green')
    
    # Scatter plot
    scatter = ax.scatter(ntk_values, agop_values, 
                        c=alignment_values, cmap='viridis', 
                        s=40, alpha=0.8, edgecolors='black', 
                        linewidth=0.5, vmin=0.012, vmax=0.025)
    
    # Only essential labels - positioned to avoid overlap
    ax.text(0.25, 12, 'Chaotic\n75%', ha='center', va='center', 
            fontsize=11, weight='bold', color='darkred')
    
    ax.text(0.7, 3, 'Optimal\n25%', ha='center', va='center', 
            fontsize=11, weight='bold', color='darkgreen')
    
    ax.set_xlabel('NTK Stability')
    ax.set_ylabel('log(AGOP)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 13)
    ax.set_title('Cross-Modal Phase Diagram')
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Alignment')
    
    plt.tight_layout()
    return fig

# Generate simple demo figures
if __name__ == "__main__":
    print("Generating simple demo figures...")
    
    # Create simple version
    fig1 = create_simple_demo_figure()
    fig1.savefig('simple_demo.png', dpi=300, bbox_inches='tight')
    fig1.savefig('simple_demo.pdf', dpi=300, bbox_inches='tight')
    print("✓ Simple demo figure saved")
    
    # Create ultra-simple version
    fig2 = create_ultra_simple_demo_figure()
    fig2.savefig('ultra_simple_demo.png', dpi=300, bbox_inches='tight')
    fig2.savefig('ultra_simple_demo.pdf', dpi=300, bbox_inches='tight')
    print("✓ Ultra-simple demo figure saved")
    
    print("\nKey improvements:")
    print("• No overlapping text boxes")
    print("• Minimal, essential information only")
    print("• Clean, uncluttered appearance")
    print("• Easy to read and understand")
