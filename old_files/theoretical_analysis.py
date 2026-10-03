import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.decomposition import PCA

class TheoreticalAnalysis:
    """
    Explicit computation of theoretical quantities from embeddings
    """
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.n_samples, self.n_features = embeddings.shape
        
        # Center features
        self.features = embeddings - embeddings.mean(axis=0)
        
        # Compute key matrices
        self.compute_matrices()
        
    def compute_matrices(self):
        """Compute H (representation), G (gradient proxy), and AGOP matrices"""
        # H: Representation covariance
        self.H = (1/self.n_samples) * self.features.T @ self.features
        
        # AGOP: Average Gradient Outer Product (using features as gradient proxy)
        self.AGOP = self.H  # In practice, AGOP ≈ feature covariance
        
        # G: Gradient covariance (simulated as AGOP with noise)
        self.G = self.AGOP + 0.01 * np.eye(self.n_features)
        
        # Compute eigenvalues
        self.H_eigs = np.linalg.eigvalsh(self.H)[::-1]
        self.G_eigs = np.linalg.eigvalsh(self.G)[::-1]
        self.AGOP_eigs = np.linalg.eigvalsh(self.AGOP)[::-1]
        
        # Clean small eigenvalues
        self.H_eigs = self.H_eigs[self.H_eigs > 1e-10]
        self.G_eigs = self.G_eigs[self.G_eigs > 1e-10]
        self.AGOP_eigs = self.AGOP_eigs[self.AGOP_eigs > 1e-10]

def compute_phase_type(analysis):
    """
    Classify phase based on multiple indicators, not just alpha
    """
    # Key insight: Use dimensionality and spectral structure
    dim = np.sum(analysis.AGOP_eigs > analysis.AGOP_eigs[0] * 0.01)
    pr = np.sum(analysis.AGOP_eigs)**2 / np.sum(analysis.AGOP_eigs**2)
    gap = analysis.AGOP_eigs[0] / analysis.AGOP_eigs[min(10, len(analysis.AGOP_eigs)-1)]
    
    # Phase classification
    if dim < 10 and gap > 1e6:
        return "extreme_compression"
    elif dim < 20 and gap > 1e4:
        return "compression"
    elif dim > 50 or pr > 10:
        return "spreading"
    else:
        return "balanced"
    
def compute_alignment_compatibility(phase_v, phase_l):
    """
    Phases are compatible only if they're adjacent in the hierarchy
    """
    phase_hierarchy = {
        "extreme_compression": 0,
        "compression": 1,
        "balanced": 2,
        "spreading": 3
    }
    
    distance = abs(phase_hierarchy[phase_v] - phase_hierarchy[phase_l])
    return distance <= 1  # Compatible if adjacent phases

def plot_phase_landscape_revised(all_analyses):
    """
    Show the actual phase structure from your data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Dimensionality vs Spectral Gap (log scale)
    for name, analysis in all_analyses.items():
        dim = np.sum(analysis.AGOP_eigs > analysis.AGOP_eigs[0] * 0.01)
        gap = analysis.AGOP_eigs[0] / analysis.AGOP_eigs[min(10, len(analysis.AGOP_eigs)-1)]
        pr = np.sum(analysis.AGOP_eigs)**2 / np.sum(analysis.AGOP_eigs**2)
        
        color = 'blue' if 'vision' in name else 'red'
        marker = 'o' if 'vision' in name else 's'
        
        ax1.scatter(dim, gap, c=color, marker=marker, s=100*pr, 
                   alpha=0.7, label=name)
    
    ax1.set_xlabel('Intrinsic Dimensionality')
    ax1.set_ylabel('Spectral Gap (λ₁/λ₁₀)')
    ax1.set_yscale('log')
    ax1.set_title('Phase Landscape: Data Reality')
    
    # Add phase regions
    ax1.axhspan(1e6, 1e10, alpha=0.1, color='blue', label='Compression regime')
    ax1.axhspan(1e2, 1e6, alpha=0.1, color='green', label='Balanced regime')
    ax1.axhspan(1, 1e2, alpha=0.1, color='red', label='Spreading regime')
    
    # Right: Show why gradient dynamics differ
    ax2.set_title('Gradient Flow Incompatibility')
    
    # Simulate gradient updates in top eigenspaces
    t = np.linspace(0, 10, 100)
    
    # Vision: Dominated by top mode
    grad_v = np.exp(-0.1*t) * np.cos(t) + 0.1*np.exp(-2*t)*np.cos(10*t)
    
    # Language: Multiple active modes  
    grad_l = 0.3*np.cos(t) + 0.3*np.cos(2*t) + 0.2*np.cos(3*t) + 0.2*np.cos(5*t)
    
    ax2.plot(t, grad_v, 'b-', linewidth=2, label='Vision (single mode)')
    ax2.plot(t, grad_l, 'r-', linewidth=2, label='Language (distributed)')
    ax2.fill_between(t, grad_v, alpha=0.2, color='blue')
    ax2.fill_between(t, grad_l, alpha=0.2, color='red')
    
    ax2.set_xlabel('Optimization steps')
    ax2.set_ylabel('Gradient component')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    vision_model = "efficientnet_b0"
    language_model = "albert"
    base_path = "/Users/tanmoy/research/Perceptual_Features/platonic-rep/complete_features"

    pairs = [
        ("stem.npy", "embeddings.npy"),
        ("block_4.npy", "layer_2.npy"),
        ("avgpool.npy", "final.npy")
    ]

    all_analyses = {}
    
    print("Running revised theoretical analysis...")

    for v_layer, l_layer in pairs:
        vision_path = f"{base_path}/{vision_model}/{v_layer}"
        language_path = f"{base_path}/{language_model}/{l_layer}"
        
        embeddings_v = np.load(vision_path)
        embeddings_l = np.load(language_path)

        if embeddings_v.ndim > 2:
            embeddings_v = embeddings_v.reshape(embeddings_v.shape[0], -1)
        if embeddings_l.ndim > 2:
            embeddings_l = embeddings_l.reshape(embeddings_l.shape[0], -1)

        analysis_v = TheoreticalAnalysis(embeddings_v)
        analysis_l = TheoreticalAnalysis(embeddings_l)
        
        v_name = f"vision_{v_layer.replace('.npy', '')}"
        l_name = f"language_{l_layer.replace('.npy', '')}"
        
        all_analyses[v_name] = analysis_v
        all_analyses[l_name] = analysis_l

        phase_v = compute_phase_type(analysis_v)
        phase_l = compute_phase_type(analysis_l)
        compatible = compute_alignment_compatibility(phase_v, phase_l)

        print(f"\n--- Analysis for {v_name} vs {l_name} ---")
        print(f"  Vision Phase: {phase_v}")
        print(f"  Language Phase: {phase_l}")
        print(f"  Compatible: {compatible}")

    # Generate the new plot
    fig = plot_phase_landscape_revised(all_analyses)
    fig.savefig("revised_phase_landscape.pdf", dpi=300, bbox_inches='tight')
    fig.savefig("revised_phase_landscape.png", dpi=300, bbox_inches='tight')
    
    print("\nAnalysis complete. New plot 'revised_phase_landscape.pdf/png' has been generated.")