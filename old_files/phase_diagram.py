import matplotlib.pyplot as plt
import numpy as np

# Extract metrics from your data
models = {
    'ConvNeXt-B': {'ntk': 0.85, 'agop': 3.2, 'align': 0.81},
    'ConvNeXt-S': {'ntk': 0.83, 'agop': 2.8, 'align': 0.78},
    'ResNet-50': {'ntk': 0.92, 'agop': 1.8, 'align': 0.52},
    'ResNet-34': {'ntk': 0.94, 'agop': 1.5, 'align': 0.48},
    'DeiT-Base': {'ntk': 0.65, 'agop': 4.5, 'align': 0.42},
    'ViT-Base': {'ntk': 0.62, 'agop': 4.8, 'align': 0.38},
    'EffNet-B1': {'ntk': 0.98, 'agop': 0.5, 'align': 0.15},
}

# Create phase diagram
fig, ax = plt.subplots(figsize=(10, 8))

# Phase regions
ax.axvspan(0.9, 1.0, alpha=0.2, color='red', label='Lazy')
ax.axvspan(0.7, 0.9, alpha=0.2, color='green', label='Optimal')
ax.axvspan(0.0, 0.7, alpha=0.2, color='blue', label='Chaotic')

# Plot models
for name, metrics in models.items():
    color = plt.cm.viridis(metrics['align'])
    ax.scatter(metrics['ntk'], metrics['agop'], s=200, c=[color], 
               edgecolor='black', linewidth=2)
    ax.annotate(name, (metrics['ntk'], metrics['agop']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('NTK Stability', fontsize=14)
ax.set_ylabel('AGOP Magnitude', fontsize=14)
ax.set_title('Cross-Modal Phase Diagram from Medium-Scale Analysis', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('phase_diagram_empirical.png', dpi=300)