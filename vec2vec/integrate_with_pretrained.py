#!/usr/bin/env python3
"""
Integration script for working with existing pretrained vec2vec models.

This script shows how to:
1. Load pretrained vec2vec translators (if available)
2. Compare their performance with geometric cooperation
3. Use geometric methods as initialization or fallback
4. Hybrid approaches combining both methods
"""

import os
import argparse
import json
import numpy as np
import torch
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from geometric_alignment import GeometricVec2Vec, extract_embeddings, load_sample_texts


class HybridVec2Vec:
    """
    Hybrid approach combining geometric cooperation with learned refinements.
    
    This approach:
    1. Uses geometric alignment as a strong initialization
    2. Optionally applies learned refinements for complex non-linear mappings
    3. Falls back to geometric methods when adversarial training fails
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.geometric_aligner = None
        self.learned_refinement = None
        self.use_refinement = False
        
    def fit(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray, 
            use_learned_refinement: bool = False, refinement_epochs: int = 20):
        """
        Fit hybrid model using geometric initialization + optional learned refinement.
        """
        print("Fitting hybrid model...")
        
        # 1. Geometric initialization (always done)
        print("  Step 1: Geometric initialization...")
        self.geometric_aligner = GeometricVec2Vec(alignment_method='auto')
        self.geometric_aligner.fit(embeddings_A, embeddings_B)
        
        geometric_metrics = self.geometric_aligner.evaluate_alignment(embeddings_A, embeddings_B)
        print(f"  Geometric baseline: {geometric_metrics['cosine_similarity']:.4f} cosine similarity")
        
        # 2. Optional learned refinement
        if use_learned_refinement:
            print("  Step 2: Learning refinement...")
            try:
                # Get geometric transformation as initialization
                geometric_transformed = self.geometric_aligner.transform(embeddings_A, 'A_to_B')
                
                # Train a small neural network to refine the transformation
                self.learned_refinement = self._train_refinement(
                    embeddings_A, embeddings_B, geometric_transformed, refinement_epochs
                )
                self.use_refinement = True
                
                # Evaluate hybrid performance
                hybrid_transformed = self._apply_hybrid_transform(embeddings_A)
                hybrid_cosine = np.mean([
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                    for a, b in zip(hybrid_transformed, embeddings_B)
                ])
                
                print(f"  Hybrid performance: {hybrid_cosine:.4f} cosine similarity")
                
                if hybrid_cosine > geometric_metrics['cosine_similarity']:
                    print("  ✓ Learned refinement improved performance")
                else:
                    print("  ✗ Learned refinement did not improve - using geometric only")
                    self.use_refinement = False
                    
            except Exception as e:
                print(f"  ✗ Refinement training failed: {e}")
                print("  Falling back to geometric method only")
                self.use_refinement = False
    
    def _train_refinement(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray, 
                         geometric_init: np.ndarray, epochs: int):
        """Train a small refinement network on top of geometric initialization."""
        import torch.nn as nn
        import torch.optim as optim
        
        class RefinementNet(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.ReLU(),
                    nn.Linear(dim // 2, dim),
                    nn.Tanh()  # Small adjustments
                )
            
            def forward(self, x):
                return x + 0.1 * self.net(x)  # Residual connection with small adjustments
        
        dim = embeddings_B.shape[1]
        refinement_net = RefinementNet(dim).to(self.device)
        optimizer = optim.Adam(refinement_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        geom_tensor = torch.FloatTensor(geometric_init).to(self.device)
        target_tensor = torch.FloatTensor(embeddings_B).to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            refined = refinement_net(geom_tensor)
            loss = criterion(refined, target_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"    Refinement epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        return refinement_net
    
    def _apply_hybrid_transform(self, embeddings_A: np.ndarray) -> np.ndarray:
        """Apply hybrid transformation: geometric + learned refinement."""
        # First apply geometric transformation
        geometric_result = self.geometric_aligner.transform(embeddings_A, 'A_to_B')
        
        if not self.use_refinement or self.learned_refinement is None:
            return geometric_result
        
        # Apply learned refinement
        with torch.no_grad():
            self.learned_refinement.eval()
            geom_tensor = torch.FloatTensor(geometric_result).to(self.device)
            refined_tensor = self.learned_refinement(geom_tensor)
            return refined_tensor.cpu().numpy()
    
    def transform(self, embeddings_A: np.ndarray) -> np.ndarray:
        """Transform embeddings using the hybrid approach."""
        return self._apply_hybrid_transform(embeddings_A)
    
    def evaluate_alignment(self, embeddings_A: np.ndarray, embeddings_B: np.ndarray) -> Dict[str, float]:
        """Evaluate alignment quality."""
        transformed = self.transform(embeddings_A)
        
        cosine_similarities = []
        for i in range(len(transformed)):
            a = transformed[i]
            b = embeddings_B[i]
            cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cosine_similarities.append(cosine_sim)
        
        cosine_sim = np.mean(cosine_similarities)
        mse = np.mean(np.sum((transformed - embeddings_B) ** 2, axis=1))
        
        return {
            'cosine_similarity': cosine_sim,
            'mse': mse,
            'method': 'hybrid' if self.use_refinement else 'geometric_only',
            'uses_refinement': self.use_refinement
        }


def demonstrate_integration_patterns(embeddings_A: np.ndarray, embeddings_B: np.ndarray, 
                                   device: str = 'cpu') -> Dict[str, Dict]:
    """
    Demonstrate different integration patterns between geometric and learned methods.
    """
    results = {}
    
    print("=== Integration Pattern Demonstrations ===\n")
    
    # 1. Pure Geometric (baseline)
    print("1. Pure Geometric Baseline...")
    geometric = GeometricVec2Vec(alignment_method='auto')
    geometric.fit(embeddings_A, embeddings_B)
    geom_metrics = geometric.evaluate_alignment(embeddings_A, embeddings_B)
    
    results['pure_geometric'] = {
        'cosine_similarity': geom_metrics['cosine_similarity'],
        'mse': geom_metrics['mse'],
        'method': geom_metrics['method'],
        'pattern': 'baseline'
    }
    print(f"   Result: {geom_metrics['cosine_similarity']:.4f} cosine similarity")
    
    # 2. Hybrid: Geometric + Learned Refinement
    print("\n2. Hybrid: Geometric + Learned Refinement...")
    hybrid = HybridVec2Vec(device)
    hybrid.fit(embeddings_A, embeddings_B, use_learned_refinement=True, refinement_epochs=20)
    hybrid_metrics = hybrid.evaluate_alignment(embeddings_A, embeddings_B)
    
    results['hybrid_refined'] = {
        'cosine_similarity': hybrid_metrics['cosine_similarity'],
        'mse': hybrid_metrics['mse'],
        'method': hybrid_metrics['method'],
        'uses_refinement': hybrid_metrics['uses_refinement'],
        'pattern': 'geometric_initialization_plus_learning'
    }
    print(f"   Result: {hybrid_metrics['cosine_similarity']:.4f} cosine similarity")
    
    # 3. Ensemble: Multiple Geometric Methods
    print("\n3. Ensemble: Multiple Geometric Methods...")
    methods = ['procrustes', 'lowrank', 'cca']
    ensemble_results = []
    
    for method in methods:
        aligner = GeometricVec2Vec(alignment_method=method)
        try:
            aligner.fit(embeddings_A, embeddings_B)
            transformed = aligner.transform(embeddings_A, 'A_to_B')
            ensemble_results.append(transformed)
        except:
            # Skip if method fails
            continue
    
    if len(ensemble_results) > 1:
        # Average the transformations
        ensemble_transformed = np.mean(ensemble_results, axis=0)
        
        # Evaluate ensemble
        ensemble_cosine = np.mean([
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            for a, b in zip(ensemble_transformed, embeddings_B)
        ])
        ensemble_mse = np.mean(np.sum((ensemble_transformed - embeddings_B) ** 2, axis=1))
        
        results['ensemble_geometric'] = {
            'cosine_similarity': ensemble_cosine,
            'mse': ensemble_mse,
            'method': f'ensemble_of_{len(ensemble_results)}_methods',
            'pattern': 'ensemble'
        }
        print(f"   Result: {ensemble_cosine:.4f} cosine similarity (ensemble of {len(ensemble_results)} methods)")
    else:
        print("   Skipped (insufficient methods for ensemble)")
    
    # 4. Adaptive: Choose Best Method Per Sample
    print("\n4. Adaptive: Choose Best Method Per Sample...")
    # This is more complex - for demo, we'll just report that it's possible
    print("   (This would choose the best geometric method for each individual sample)")
    print("   Pattern: Per-sample adaptive method selection")
    
    return results


def load_and_compare_with_pretrained(config_path: Optional[str] = None, 
                                   model_path: Optional[str] = None,
                                   model_A: str = 'stella', model_B: str = 'gte',
                                   n_samples: int = 500, device: str = 'cpu'):
    """
    Main function to load pretrained models and compare with geometric methods.
    """
    print("=" * 70)
    print("INTEGRATION WITH PRETRAINED VEC2VEC MODELS")
    print("=" * 70)
    
    # Load data
    texts = load_sample_texts('nq', n_samples)
    embeddings_A = extract_embeddings(model_A, texts, device)
    embeddings_B = extract_embeddings(model_B, texts, device)
    
    print(f"Data: {model_A}({embeddings_A.shape}) → {model_B}({embeddings_B.shape})")
    
    # Try to load pretrained translator
    pretrained_available = False
    if config_path and model_path and os.path.exists(config_path) and os.path.exists(model_path):
        print("\nAttempting to load pretrained translator...")
        try:
            # This would require the actual vec2vec utilities
            # For now, we'll simulate what would happen
            print("Note: Would load pretrained translator here if vec2vec utils available")
            # pretrained_translator = load_pretrained_translator(config_path, model_path)
            # pretrained_available = True
        except Exception as e:
            print(f"Could not load pretrained translator: {e}")
    
    if not pretrained_available:
        print("No pretrained translator available - demonstrating geometric methods")
    
    # Demonstrate integration patterns
    results = demonstrate_integration_patterns(embeddings_A, embeddings_B, device)
    
    # Analysis and recommendations
    print("\n" + "=" * 70)
    print("INTEGRATION ANALYSIS AND RECOMMENDATIONS")
    print("=" * 70)
    
    best_method = max(results.keys(), key=lambda x: results[x]['cosine_similarity'])
    best_score = results[best_method]['cosine_similarity']
    
    print(f"\nBest performing approach: {best_method}")
    print(f"Best score: {best_score:.4f} cosine similarity")
    
    print("\n📋 INTEGRATION RECOMMENDATIONS:")
    print("=" * 40)
    
    if best_score > 0.95:
        print("✅ HIGH QUALITY ALIGNMENT ACHIEVED")
        print("   • Geometric methods are sufficient for this task")
        print("   • Consider using geometric methods as production solution")
        print("   • Fast inference, no training required")
        
    elif best_score > 0.85:
        print("✅ GOOD ALIGNMENT WITH ROOM FOR IMPROVEMENT")  
        print("   • Geometric provides strong baseline")
        print("   • Consider hybrid approach for critical applications")
        print("   • Ensemble methods may provide further gains")
        
    else:
        print("⚠️  MODERATE ALIGNMENT - COMPLEX MAPPING NEEDED")
        print("   • Models may have very different representations")
        print("   • Consider advanced adversarial training")
        print("   • Use geometric as initialization for learned methods")
    
    print("\n🔧 PRODUCTION DEPLOYMENT PATTERNS:")
    print("=" * 40)
    print("1. 🚀 FAST INFERENCE: Use pure geometric methods")
    print("   • Instant alignment, no model loading")
    print("   • Perfect for real-time applications")
    
    print("\n2. 🎯 HIGH ACCURACY: Use hybrid approach")  
    print("   • Geometric initialization + learned refinement")
    print("   • Best of both worlds")
    
    print("\n3. 🛡️  ROBUST DEPLOYMENT: Use ensemble")
    print("   • Multiple geometric methods combined")
    print("   • Fallback mechanisms built-in")
    
    print("\n4. 🔄 ADAPTIVE: Dynamic method selection")
    print("   • Choose best method based on input characteristics")
    print("   • Use geometric for most cases, learned for exceptions")
    
    return results


def main():
    """Run the integration demonstration."""
    parser = argparse.ArgumentParser(description="Integration with Pretrained Vec2Vec Models")
    parser.add_argument("--model_a", type=str, default="stella", help="Source model")
    parser.add_argument("--model_b", type=str, default="gte", help="Target model")  
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--config", type=str, help="Path to pretrained config")
    parser.add_argument("--model_path", type=str, help="Path to pretrained weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./integration_results")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run integration demo
    results = load_and_compare_with_pretrained(
        args.config, args.model_path, args.model_a, args.model_b, 
        args.n_samples, args.device
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "integration_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    print(f"\n📁 Results saved to {results_path}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR YOUR RESEARCH")
    print("=" * 70)
    print("1. 📊 Compare with your existing adversarial vec2vec models")
    print("2. 🧪 Test on your specific embedding model pairs")
    print("3. 📈 Evaluate on downstream tasks (retrieval, classification)")
    print("4. 🔬 Analyze platonic representation insights integration")
    print("5. 📝 Consider publishing geometric cooperation as alternative to adversarial training")
    

if __name__ == "__main__":
    main()