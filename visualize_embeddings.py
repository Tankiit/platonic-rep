#!/usr/bin/env python3
"""
Simple script to visualize the actual embeddings from complete_features directory
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_and_visualize_embeddings():
    """Load and visualize embeddings from complete_features directory"""
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Complete Embeddings Visualization: ResNet18 vs DistilBERT', fontsize=16, fontweight='bold')
    
    # Load ResNet18 embeddings
    resnet_path = Path('complete_features/resnet18/')
    resnet_files = [f for f in os.listdir(resnet_path) if f.endswith('.npy')]
    resnet_files.sort()
    
    # Load DistilBERT embeddings  
    distilbert_path = Path('complete_features/distilbert/')
    distilbert_files = [f for f in os.listdir(distilbert_path) if f.endswith('.npy')]
    distilbert_files.sort()
    
    print("=== Loading ResNet18 Embeddings ===")
    resnet_data = {}
    for file in resnet_files:
        data = np.load(resnet_path / file)
        resnet_data[file] = data
        print(f"{file}: shape {data.shape}, range [{data.min():.3f}, {data.max():.3f}], mean {data.mean():.3f}")
    
    print("\n=== Loading DistilBERT Embeddings ===")
    distilbert_data = {}
    for file in distilbert_files:
        data = np.load(distilbert_path / file)
        distilbert_data[file] = data
        print(f"{file}: shape {data.shape}, range [{data.min():.3f}, {data.max():.3f}], mean {data.mean():.3f}")
    
    # Plot 1: Embedding value distributions
    ax1 = axes[0, 0]
    for i, (file, data) in enumerate(resnet_data.items()):
        # Sample some values for visualization
        sample_data = data.flatten()
        if len(sample_data) > 1000:
            sample_data = np.random.choice(sample_data, 1000, replace=False)
        ax1.hist(sample_data, bins=30, alpha=0.6, label=f'ResNet18_{file[:-4]}', density=True)
    ax1.set_xlabel('Embedding Values')
    ax1.set_ylabel('Density')
    ax1.set_title('ResNet18 Embedding Distributions')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: DistilBERT distributions
    ax2 = axes[0, 1]
    for i, (file, data) in enumerate(distilbert_data.items()):
        # Sample some values for visualization
        sample_data = data.flatten()
        if len(sample_data) > 1000:
            sample_data = np.random.choice(sample_data, 1000, replace=False)
        ax2.hist(sample_data, bins=30, alpha=0.6, label=f'DistilBERT_{file[:-4]}', density=True)
    ax2.set_xlabel('Embedding Values')
    ax2.set_ylabel('Density')
    ax2.set_title('DistilBERT Embedding Distributions')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Layer-wise progression (mean values)
    ax3 = axes[0, 2]
    resnet_means = [resnet_data[file].mean() for file in resnet_files]
    distilbert_means = [distilbert_data[file].mean() for file in distilbert_files]
    
    ax3.plot(range(len(resnet_means)), resnet_means, 'o-', label='ResNet18', linewidth=2, markersize=6)
    ax3.plot(range(len(distilbert_means)), distilbert_means, 's-', label='DistilBERT', linewidth=2, markersize=6)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Mean Embedding Value')
    ax3.set_title('Mean Values Across Layers')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Dimensionality comparison
    ax4 = axes[1, 0]
    resnet_dims = [resnet_data[file].shape[1] for file in resnet_files]
    distilbert_dims = [distilbert_data[file].shape[1] for file in distilbert_files]
    
    # Create separate plots for each model since they have different numbers of layers
    x_pos_resnet = range(len(resnet_files))
    x_pos_distilbert = range(len(distilbert_files))
    
    ax4.bar(x_pos_resnet, resnet_dims, alpha=0.7, color='lightblue', label='ResNet18')
    ax4.bar([x + len(resnet_files) + 1 for x in x_pos_distilbert], distilbert_dims, 
            alpha=0.7, color='lightcoral', label='DistilBERT')
    
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Feature Dimensions')
    ax4.set_title('Dimensionality Across Layers')
    
    # Set x-axis labels
    all_labels = [f[:-4] for f in resnet_files] + [''] + [f[:-4] for f in distilbert_files]
    all_positions = list(x_pos_resnet) + [len(resnet_files)] + [x + len(resnet_files) + 1 for x in x_pos_distilbert]
    ax4.set_xticks(all_positions)
    ax4.set_xticklabels(all_labels, rotation=45)
    
    # Add separator line
    ax4.axvline(x=len(resnet_files) + 0.5, color='black', linestyle='--', alpha=0.5)
    
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Similarity matrices (sample)
    ax5 = axes[1, 1]
    
    # Use the largest ResNet18 layer and a DistilBERT layer
    resnet_layer = 'avgpool.npy'  # 512 dimensions
    distilbert_layer = 'final.npy'  # 768 dimensions
    
    resnet_sample = resnet_data[resnet_layer][:20]  # Use first 20 samples
    distilbert_sample = distilbert_data[distilbert_layer][:20]
    
    # Compute cosine similarity matrices
    def cosine_similarity_matrix(data):
        data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)
        return data_norm @ data_norm.T
    
    resnet_sim = cosine_similarity_matrix(resnet_sample)
    distilbert_sim = cosine_similarity_matrix(distilbert_sample)
    
    # Combine matrices
    combined_sim = np.block([[resnet_sim, np.zeros_like(resnet_sim)], 
                           [np.zeros_like(distilbert_sim), distilbert_sim]])
    
    im = ax5.imshow(combined_sim, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_title('Sample Similarity Matrices')
    ax5.axvline(x=19.5, color='black', linewidth=2)
    ax5.axhline(y=19.5, color='black', linewidth=2)
    ax5.text(10, -2, 'ResNet18', ha='center', va='top', fontweight='bold')
    ax5.text(30, -2, 'DistilBERT', ha='center', va='top', fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Cosine Similarity')
    
    # Plot 6: Cross-modal comparison
    ax6 = axes[1, 2]
    
    # Compare the final layers
    resnet_final = resnet_data['avgpool.npy']  # 512 dims
    distilbert_final = distilbert_data['final.npy']  # 768 dims
    
    # Project to common space (use first 512 dims of DistilBERT)
    distilbert_projected = distilbert_final[:, :512]
    
    # Compute cross-modal similarities
    similarities = []
    for i in range(min(30, len(resnet_final))):
        v1 = resnet_final[i] / (np.linalg.norm(resnet_final[i]) + 1e-8)
        v2 = distilbert_projected[i] / (np.linalg.norm(distilbert_projected[i]) + 1e-8)
        sim = np.dot(v1, v2)
        similarities.append(sim)
    
    ax6.hist(similarities, bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax6.axvline(x=np.mean(similarities), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(similarities):.3f}')
    ax6.set_xlabel('Cross-Modal Similarity')
    ax6.set_ylabel('Frequency')
    ax6.set_title('ResNet18 vs DistilBERT Similarities')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'results/complete_embeddings_visualization.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    return resnet_data, distilbert_data

if __name__ == "__main__":
    resnet_data, distilbert_data = load_and_visualize_embeddings()
    
    print("\n=== Summary ===")
    print(f"ResNet18 layers: {len(resnet_data)}")
    print(f"DistilBERT layers: {len(distilbert_data)}")
    print(f"Total samples per layer: {list(resnet_data.values())[0].shape[0]}")
    
    # Show some statistics
    print("\n=== Cross-Modal Statistics ===")
    resnet_final = resnet_data['avgpool.npy']
    distilbert_final = distilbert_data['final.npy']
    
    print(f"ResNet18 final layer: {resnet_final.shape}")
    print(f"DistilBERT final layer: {distilbert_final.shape}")
    print(f"ResNet18 value range: [{resnet_final.min():.3f}, {resnet_final.max():.3f}]")
    print(f"DistilBERT value range: [{distilbert_final.min():.3f}, {distilbert_final.max():.3f}]")
