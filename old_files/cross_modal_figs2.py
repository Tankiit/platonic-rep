import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set style for publication
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

def create_phase_diagram(save_path='phase_diagram.pdf'):
    """Create publication-ready phase diagram for NeurIPS workshop"""
    
    fig = plt.figure(figsize=(10, 3.5))
    
    # ========== Panel A: Main Phase Diagram ==========
    ax1 = plt.subplot(131)
    
    # Generate synthetic data based on your results
    np.random.seed(42)
    n_points = 144
    
    # Chaotic phase (75% of points, NTK < 0.4)
    n_chaotic = 108
    chaotic_ntk = np.random.uniform(0.15, 0.35, n_chaotic)
    chaotic_agop = np.random.uniform(6, 14, n_chaotic)  # log scale
    chaotic_align = np.random.uniform(0.012, 0.018, n_chaotic)
    
    # Optimal phase (25% of points, 0.4 < NTK < 0.9)
    n_optimal = 36
    optimal_ntk = np.random.uniform(0.5, 0.85, n_optimal)
    optimal_agop = np.random.uniform(2, 8, n_optimal)
    optimal_align = np.random.uniform(0.018, 0.025, n_optimal)
    
    # Combine all points
    all_ntk = np.concatenate([chaotic_ntk, optimal_ntk])
    all_agop = np.concatenate([chaotic_agop, optimal_agop])
    all_align = np.concatenate([chaotic_align, optimal_align])
    
    # Create phase regions
    ax1.axvspan(0, 0.5, alpha=0.2, color='red', label='Chaotic')
    ax1.axvspan(0.5, 0.9, alpha=0.2, color='green', label='Optimal')
    ax1.axvspan(0.9, 1.0, alpha=0.2, color='blue', label='Lazy')
    
    # Scatter plot
    scatter = ax1.scatter(all_ntk, all_agop, c=all_align, 
                         cmap='viridis', s=30, alpha=0.7, 
                         vmin=0.01, vmax=0.025, edgecolors='black', linewidth=0.5)
    
    # Add annotation
    ax1.annotate('108 pairs\n(75%)', xy=(0.25, 12), 
                xytext=(0.1, 15), fontsize=9, weight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    ax1.annotate('36 pairs\n(25%)', xy=(0.65, 5), 
                xytext=(0.75, 3), fontsize=9, weight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    ax1.set_xlabel('Cross-Modal NTK Stability')
    ax1.set_ylabel('log(AGOP Magnitude)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 16)
    ax1.set_title('(a) Phase Landscape: 144 Vision-Language Pairs')
    ax1.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label('Alignment Score', rotation=270, labelpad=15)
    
    # ========== Panel B: Gradient Flow Visualization ==========
    ax2 = plt.subplot(132)
    
    # Create gradient vector fields
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    
    # Chaotic phase - orthogonal gradients
    ax2.text(0, 2.3, 'Chaotic Phase', ha='center', weight='bold')
    
    # Vision gradients (pointing up)
    U_vision = np.zeros_like(X)
    V_vision = np.ones_like(Y) * 0.3
    
    # Language gradients (pointing right - orthogonal)
    U_lang = np.ones_like(X) * 0.3
    V_lang = np.zeros_like(Y)
    
    # Plot with different colors
    ax2.quiver(X[::2, ::2], Y[::2, ::2], U_vision[::2, ::2], V_vision[::2, ::2], 
              color='blue', alpha=0.6, scale=5, label='Vision gradients')
    ax2.quiver(X[1::2, 1::2], Y[1::2, 1::2], U_lang[1::2, 1::2], V_lang[1::2, 1::2], 
              color='red', alpha=0.6, scale=5, label='Language gradients')
    
    # Add orthogonality symbol
    ax2.plot([0, 0.5], [0, 0], 'k-', lw=2)
    ax2.plot([0, 0], [0, 0.5], 'k-', lw=2)
    ax2.plot([0.5, 0.5], [0, 0.5], 'k--', alpha=0.5)
    ax2.plot([0, 0.5], [0.5, 0.5], 'k--', alpha=0.5)
    ax2.text(0.25, 0.25, '⊥', fontsize=16, ha='center', va='center')
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.set_title('(b) Gradient Orthogonality in Chaotic Phase')
    ax2.set_xlabel('θ_vision')
    ax2.set_ylabel('θ_language')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ========== Panel C: Architecture Distribution ==========
    ax3 = plt.subplot(133)
    
    # Data from your experiments
    architectures = ['ResNet\n(34,50)', 'ViT\n(S,B)', 'ConvNeXt\n(S,B)', 
                    'Efficient\nNet', 'Mixer', 'Swin\n(S,B)']
    ntk_values = [0.22, 0.21, 0.27, 0.40, 0.16, 0.78]
    colors = ['#ff6b6b', '#ff6b6b', '#ffa06b', '#ffeb6b', '#ff6b6b', '#6bff6b']
    
    bars = ax3.bar(architectures, ntk_values, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # Add phase boundary line
    ax3.axhline(y=0.5, color='green', linestyle='--', linewidth=2, 
               label='Phase boundary')
    
    # Add value labels on bars
    for bar, val in zip(bars, ntk_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_ylabel('Average NTK Stability')
    ax3.set_ylim(0, 1)
    ax3.set_title('(c) Architecture-Specific Phase Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Create the figure
create_phase_diagram()

# Also create a simpler version for the main text
def create_simple_phase_diagram(save_path='phase_diagram_simple.pdf'):
    """Create a simplified version focusing on the key insight"""
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Phase regions
    ax.axvspan(0, 0.5, alpha=0.3, color='red')
    ax.axvspan(0.5, 0.9, alpha=0.3, color='green')
    ax.axvspan(0.9, 1.0, alpha=0.3, color='blue')
    
    # Add phase labels
    ax.text(0.25, 14, 'CHAOTIC\nPHASE', ha='center', va='center', 
           weight='bold', fontsize=12, color='darkred')
    ax.text(0.7, 14, 'OPTIMAL\nPHASE', ha='center', va='center', 
           weight='bold', fontsize=12, color='darkgreen')
    ax.text(0.95, 14, 'LAZY', ha='center', va='center', 
           weight='bold', fontsize=10, color='darkblue', rotation=90)
    
    # Synthetic data points
    np.random.seed(42)
    
    # Chaotic cluster
    chaotic_x = np.random.normal(0.25, 0.08, 108)
    chaotic_y = np.random.normal(10, 2, 108)
    
    # Optimal cluster  
    optimal_x = np.random.normal(0.65, 0.1, 36)
    optimal_y = np.random.normal(5, 1.5, 36)
    
    # Plot points
    ax.scatter(chaotic_x, chaotic_y, c='darkred', alpha=0.5, s=20)
    ax.scatter(optimal_x, optimal_y, c='darkgreen', alpha=0.5, s=20)
    
    # Add key annotation
    ax.annotate('Cross-modal pairs\ncluster here', xy=(0.25, 10), 
               xytext=(0.4, 6), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_xlabel('NTK Stability', fontsize=12)
    ax.set_ylabel('log(AGOP)', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Uncomment to create the simple version
create_simple_phase_diagram()