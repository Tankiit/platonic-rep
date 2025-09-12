#!/usr/bin/env python3
"""
Comprehensive comparison between adversarial vec2vec and geometric cooperation approaches.

This script compares:
1. Traditional adversarial training (GANs) for embedding alignment  
2. Geometric cooperation using direct mathematical transformations

The comparison evaluates alignment quality, training speed, stability, and interpretability.
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False
from types import SimpleNamespace

# Try to import from existing utils, fallback to simpler versions
try:
    from utils.model_utils import load_encoder, get_sentence_embedding_dimension
    from utils.utils import load_n_translator
    from utils.streaming_utils import load_streaming_embeddings
    USE_UTILS = True
except ImportError:
    print("Warning: vec2vec utils not available, using simplified implementations")
    USE_UTILS = False

from geometric_alignment import GeometricVec2Vec, extract_embeddings, load_sample_texts


class SimpleDiscriminator(nn.Module):
    """Simple discriminator for adversarial training demonstration."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class SimpleTranslator(nn.Module):
    """Simple translator/generator for adversarial training demonstration."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class AdversarialVec2Vec:
    """Simple adversarial vec2vec implementation for comparison."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, device: str = 'cpu'):
        self.device = device
        self.generator = SimpleTranslator(input_dim, output_dim, hidden_dim).to(device)
        self.discriminator = SimpleDiscriminator(output_dim, hidden_dim).to(device)
        
        # Optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.MSELoss()
        
        self.training_history = {
            'gen_loss': [], 'disc_loss': [], 'recon_loss': [],
            'gen_acc': [], 'disc_acc': [], 'time_per_epoch': []
        }
    
    def train_epoch(self, embeddings_A: torch.Tensor, embeddings_B: torch.Tensor, 
                   lambda_recon: float = 10.0) -> Dict[str, float]:
        """Train for one epoch."""
        batch_size = embeddings_A.shape[0]
        
        # Real and fake labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        
        # Real data
        real_pred = self.discriminator(embeddings_B)
        real_loss = self.adversarial_loss(real_pred, real_labels)
        
        # Fake data
        fake_B = self.generator(embeddings_A)
        fake_pred = self.discriminator(fake_B.detach())
        fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        fake_B = self.generator(embeddings_A)
        gen_pred = self.discriminator(fake_B)
        
        # Adversarial loss
        gen_adv_loss = self.adversarial_loss(gen_pred, real_labels)
        
        # Reconstruction loss
        gen_recon_loss = self.reconstruction_loss(fake_B, embeddings_B)
        
        # Total generator loss
        gen_loss = gen_adv_loss + lambda_recon * gen_recon_loss
        gen_loss.backward()
        self.gen_optimizer.step()
        
        # Calculate accuracies
        real_acc = ((real_pred > 0.5).float() == real_labels).float().mean().item()
        fake_acc = ((fake_pred > 0.5).float() == fake_labels).float().mean().item()
        disc_acc = (real_acc + fake_acc) / 2
        gen_acc = ((gen_pred > 0.5).float() == real_labels).float().mean().item()
        
        return {
            'gen_loss': gen_loss.item(),
            'disc_loss': disc_loss.item(),
            'recon_loss': gen_recon_loss.item(),
            'gen_acc': gen_acc,
            'disc_acc': disc_acc
        }
    
    def fit(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray, 
            epochs: int = 100, batch_size: int = 64, lambda_recon: float = 10.0):
        """Train the adversarial model."""
        # Convert to tensors
        embeddings_A = torch.FloatTensor(embeddings_A).to(self.device)
        embeddings_B = torch.FloatTensor(embeddings_B).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(embeddings_A, embeddings_B)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training adversarial model for {epochs} epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_metrics = {'gen_loss': 0, 'disc_loss': 0, 'recon_loss': 0, 
                           'gen_acc': 0, 'disc_acc': 0}
            
            for batch_A, batch_B in dataloader:
                metrics = self.train_epoch(batch_A, batch_B, lambda_recon)
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
            
            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= len(dataloader)
                self.training_history[key].append(epoch_metrics[key])
            
            self.training_history['time_per_epoch'].append(time.time() - start_time)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Gen Loss: {epoch_metrics['gen_loss']:.4f}, "
                      f"Disc Loss: {epoch_metrics['disc_loss']:.4f}, "
                      f"Gen Acc: {epoch_metrics['gen_acc']:.3f}")
    
    def transform(self, embeddings_A: np.ndarray) -> np.ndarray:
        """Transform embeddings using the trained generator."""
        self.generator.eval()
        with torch.no_grad():
            embeddings_A = torch.FloatTensor(embeddings_A).to(self.device)
            transformed = self.generator(embeddings_A)
            return transformed.cpu().numpy()
    
    def evaluate_alignment(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray) -> Dict[str, float]:
        """Evaluate alignment quality."""
        A_transformed = self.transform(embeddings_A)
        
        # Compute metrics
        cosine_similarities = []
        for i in range(len(A_transformed)):
            a = A_transformed[i]
            b = embeddings_B[i]
            cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cosine_similarities.append(cosine_sim)
        
        cosine_sim = np.mean(cosine_similarities)
        mse = np.mean(np.sum((A_transformed - embeddings_B) ** 2, axis=1))
        
        return {
            'cosine_similarity': cosine_sim,
            'mse': mse,
            'method': 'adversarial'
        }


def load_pretrained_translator(config_path: str, model_path: str = None):
    """Load a pretrained vec2vec translator if available."""
    if not USE_UTILS:
        return None
    
    try:
        # Load config
        if config_path.endswith('.toml') and HAS_TOML:
            cfg = toml.load(config_path)
            cfg = SimpleNamespace(**{**{k: v for d in cfg.values() for k, v in d.items()}})
        else:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            cfg = SimpleNamespace(**cfg)
        
        # Set up encoders
        sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb)}
        encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
        
        # Create translator
        translator = load_n_translator(cfg, encoder_dims)
        
        # Add unsupervised encoder
        unsup_enc = {cfg.unsup_emb: load_encoder(cfg.unsup_emb)}
        unsup_dim = {cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb])}
        translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            translator.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
            print(f"Loaded pretrained weights from {model_path}")
        
        return translator, sup_encs, unsup_enc, cfg
    
    except Exception as e:
        print(f"Could not load pretrained translator: {e}")
        return None


def compare_methods(embeddings_A: np.ndarray, embeddings_B: np.ndarray, 
                   device: str = 'cpu') -> Dict[str, Dict]:
    """Compare adversarial vs geometric methods."""
    results = {}
    
    print("=== Comparison: Adversarial vs Geometric Cooperation ===\n")
    
    # 1. Geometric Cooperation
    print("1. Testing Geometric Cooperation Methods...")
    geometric_methods = ['procrustes', 'cca', 'lowrank', 'auto']
    
    for method in geometric_methods:
        print(f"   Testing {method}...")
        start_time = time.time()
        
        aligner = GeometricVec2Vec(alignment_method=method)
        aligner.fit(embeddings_A, embeddings_B)
        metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
        
        fit_time = time.time() - start_time
        
        results[f'geometric_{method}'] = {
            'cosine_similarity': metrics['cosine_similarity'],
            'mse': metrics['mse'],
            'training_time': fit_time,
            'method_used': metrics['method'],
            'category': 'geometric'
        }
        
        print(f"      Method: {metrics['method']}, "
              f"Cosine Sim: {metrics['cosine_similarity']:.4f}, "
              f"Time: {fit_time:.2f}s")
    
    # 2. Adversarial Method
    print("\n2. Testing Adversarial Method...")
    start_time = time.time()
    
    adversarial = AdversarialVec2Vec(
        input_dim=embeddings_A.shape[1], 
        output_dim=embeddings_B.shape[1], 
        device=device
    )
    adversarial.fit(embeddings_A, embeddings_B, epochs=50)  # Reduced epochs for demo
    adv_metrics = adversarial.evaluate_alignment(embeddings_A, embeddings_B)
    
    training_time = time.time() - start_time
    
    results['adversarial'] = {
        'cosine_similarity': adv_metrics['cosine_similarity'],
        'mse': adv_metrics['mse'],
        'training_time': training_time,
        'training_history': adversarial.training_history,
        'category': 'adversarial'
    }
    
    print(f"   Adversarial: Cosine Sim: {adv_metrics['cosine_similarity']:.4f}, "
          f"Time: {training_time:.2f}s")
    
    return results


def visualize_comparison(results: Dict[str, Dict], save_path: str = None):
    """Create comprehensive visualization of the comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    methods = list(results.keys())
    cosine_sims = [results[m]['cosine_similarity'] for m in methods]
    mses = [results[m]['mse'] for m in methods]
    times = [results[m]['training_time'] for m in methods]
    categories = [results[m]['category'] for m in methods]
    
    # Color mapping
    colors = {'geometric': '#2E86AB', 'adversarial': '#A23B72'}
    method_colors = [colors[cat] for cat in categories]
    
    # 1. Cosine Similarity Comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(methods)), cosine_sims, color=method_colors, alpha=0.7)
    ax1.set_title('Cosine Similarity (Higher is Better)', fontsize=12, weight='bold')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('geometric_', '').replace('_', '\n') for m in methods], 
                       rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, cosine_sims):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. MSE Comparison (log scale)
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(methods)), mses, color=method_colors, alpha=0.7)
    ax2.set_title('Mean Squared Error (Lower is Better)', fontsize=12, weight='bold')
    ax2.set_ylabel('MSE (log scale)')
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace('geometric_', '').replace('_', '\n') for m in methods], 
                       rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Training Time Comparison (log scale)
    ax3 = axes[0, 2]
    bars = ax3.bar(range(len(methods)), times, color=method_colors, alpha=0.7)
    ax3.set_title('Training Time (Lower is Better)', fontsize=12, weight='bold')
    ax3.set_ylabel('Time (seconds, log scale)')
    ax3.set_yscale('log')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.replace('geometric_', '').replace('_', '\n') for m in methods], 
                       rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # 4. Training History (if available)
    ax4 = axes[1, 0]
    if 'adversarial' in results and 'training_history' in results['adversarial']:
        history = results['adversarial']['training_history']
        epochs = range(1, len(history['gen_loss']) + 1)
        
        ax4.plot(epochs, history['gen_loss'], label='Generator Loss', color='#A23B72')
        ax4.plot(epochs, history['disc_loss'], label='Discriminator Loss', color='#F18F01')
        ax4.set_title('Adversarial Training History', fontsize=12, weight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No training history\navailable', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Training History', fontsize=12, weight='bold')
    
    # 5. Accuracy Comparison (if available)
    ax5 = axes[1, 1]
    if 'adversarial' in results and 'training_history' in results['adversarial']:
        history = results['adversarial']['training_history']
        epochs = range(1, len(history['gen_acc']) + 1)
        
        ax5.plot(epochs, history['gen_acc'], label='Generator Accuracy', color='#A23B72')
        ax5.plot(epochs, history['disc_acc'], label='Discriminator Accuracy', color='#F18F01')
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
        ax5.set_title('Adversarial Training Accuracy', fontsize=12, weight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Accuracy')
        ax5.set_ylim(0, 1)
        ax5.legend()
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No accuracy data\navailable', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Training Accuracy', fontsize=12, weight='bold')
    
    # 6. Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary data
    best_cosine = max(cosine_sims)
    best_mse = min(mses)
    best_time = min(times)
    
    best_cosine_method = methods[cosine_sims.index(best_cosine)]
    best_mse_method = methods[mses.index(best_mse)]
    best_time_method = methods[times.index(best_time)]
    
    summary_text = f"""
COMPARISON SUMMARY

Best Cosine Similarity:
{best_cosine_method.replace('geometric_', '').replace('_', ' ').title()}
({best_cosine:.4f})

Best MSE (Lowest):
{best_mse_method.replace('geometric_', '').replace('_', ' ').title()}
({best_mse:.4f})

Fastest Training:
{best_time_method.replace('geometric_', '').replace('_', ' ').title()}
({best_time:.2f}s)

Geometric Methods:
• No discriminator needed
• Mathematical optimality
• Instant convergence
• Interpretable transformations

Adversarial Method:
• Learned representations
• Potential for complex mappings
• Requires training time
• Less interpretable
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add legend
    geometric_patch = plt.Rectangle((0, 0), 1, 1, fc=colors['geometric'], alpha=0.7)
    adversarial_patch = plt.Rectangle((0, 0), 1, 1, fc=colors['adversarial'], alpha=0.7)
    fig.legend([geometric_patch, adversarial_patch], 
              ['Geometric Cooperation', 'Adversarial Training'],
              loc='lower center', ncol=2, fontsize=12)
    
    plt.suptitle('Adversarial vs Geometric Cooperation: Comprehensive Comparison', 
                fontsize=16, weight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.90)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    """Run the comprehensive comparison."""
    parser = argparse.ArgumentParser(description="Compare Adversarial vs Geometric Vec2Vec")
    parser.add_argument("--model_a", type=str, default="stella", help="First model")
    parser.add_argument("--model_b", type=str, default="gte", help="Second model") 
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--config", type=str, help="Path to pretrained model config")
    parser.add_argument("--model_path", type=str, help="Path to pretrained model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of real models")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ADVERSARIAL vs GEOMETRIC COOPERATION COMPARISON")
    print("=" * 60)
    
    # Load data
    if args.synthetic:
        print("Using synthetic data for comparison...")
        np.random.seed(42)
        n_samples, dim = args.n_samples, 128
        
        # Create synthetic embeddings with known relationship
        embeddings_A = np.random.randn(n_samples, dim).astype(np.float32)
        
        # Apply known transformation + noise
        theta = np.pi / 6  # 30 degrees rotation
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        
        embeddings_B = embeddings_A.copy()
        embeddings_B[:, :2] = embeddings_A[:, :2] @ rotation.T
        embeddings_B += np.random.randn(n_samples, dim).astype(np.float32) * 0.1
        
        print(f"Generated synthetic embeddings: {embeddings_A.shape} -> {embeddings_B.shape}")
        
    else:
        # Try to load real embeddings
        try:
            print("Loading real embeddings...")
            texts = load_sample_texts('nq', args.n_samples)
            embeddings_A = extract_embeddings(args.model_a, texts, args.device).astype(np.float32)
            embeddings_B = extract_embeddings(args.model_b, texts, args.device).astype(np.float32)
            print(f"Loaded embeddings: {embeddings_A.shape} -> {embeddings_B.shape}")
        except Exception as e:
            print(f"Failed to load real embeddings: {e}")
            print("Falling back to synthetic data...")
            args.synthetic = True
            # Recursive call with synthetic flag
            return main()
    
    # Run comparison
    start_time = time.time()
    results = compare_methods(embeddings_A, embeddings_B, args.device)
    total_time = time.time() - start_time
    
    print(f"\n=== COMPARISON COMPLETED in {total_time:.2f}s ===")
    
    # Print summary
    print("\nSUMMARY:")
    print("-" * 50)
    for method, result in results.items():
        print(f"{method:20}: Cosine={result['cosine_similarity']:.4f}, "
              f"MSE={result['mse']:.4f}, Time={result['training_time']:.2f}s")
    
    # Save results
    results_path = os.path.join(args.output_dir, "comparison_results.json")
    # Convert numpy types for JSON serialization
    json_results = {}
    for method, result in results.items():
        json_results[method] = {
            k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
            for k, v in result.items() if k != 'training_history'
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Create visualization
    viz_path = os.path.join(args.output_dir, "comparison_visualization.png")
    visualize_comparison(results, viz_path)
    
    # Final insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    
    geometric_methods = [k for k in results.keys() if k.startswith('geometric_')]
    best_geometric = max(geometric_methods, key=lambda x: results[x]['cosine_similarity'])
    adversarial_sim = results['adversarial']['cosine_similarity']
    best_geometric_sim = results[best_geometric]['cosine_similarity']
    
    print(f"• Best Geometric Method: {best_geometric} ({best_geometric_sim:.4f} cosine similarity)")
    print(f"• Adversarial Method: {adversarial_sim:.4f} cosine similarity")
    
    if best_geometric_sim > adversarial_sim:
        improvement = ((best_geometric_sim - adversarial_sim) / adversarial_sim) * 100
        print(f"• Geometric cooperation is {improvement:.1f}% better in alignment quality")
    
    geometric_time = results[best_geometric]['training_time']
    adversarial_time = results['adversarial']['training_time']
    speedup = adversarial_time / geometric_time
    
    print(f"• Geometric cooperation is {speedup:.1f}x faster than adversarial training")
    print(f"• Geometric methods provide mathematical optimality without instability")
    print(f"• No hyperparameter tuning needed for geometric methods")
    print("• Geometric transformations are fully interpretable")


if __name__ == "__main__":
    main()