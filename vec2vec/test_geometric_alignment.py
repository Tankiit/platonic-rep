#!/usr/bin/env python3
"""
Test script for geometric alignment functionality.
"""

import numpy as np
import torch
from geometric_alignment import GeometricVec2Vec


def test_with_synthetic_data():
    """Test the geometric alignment with synthetic data."""
    print("=== Testing with synthetic data ===")
    
    # Create synthetic embeddings with known transformation
    n_samples = 500
    dim_A, dim_B = 128, 128
    
    # Generate random embeddings for space A
    np.random.seed(42)
    embeddings_A = np.random.randn(n_samples, dim_A)
    
    # Create embeddings_B by applying a known transformation to A
    # Add some rotation and translation
    theta = np.pi / 6  # 30 degrees
    rotation_2d = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    
    # Create a higher-dimensional rotation matrix
    R_true = np.eye(dim_A)
    R_true[:2, :2] = rotation_2d
    
    # Apply transformation: rotate + translate + add noise
    embeddings_B = embeddings_A @ R_true.T + np.random.randn(1, dim_B) * 0.1
    embeddings_B += np.random.randn(n_samples, dim_B) * 0.05  # Small noise
    
    # Test different alignment methods
    methods = ['procrustes', 'cca', 'lowrank', 'auto']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} alignment...")
        aligner = GeometricVec2Vec(alignment_method=method)
        aligner.fit(embeddings_A, embeddings_B)
        
        # Evaluate
        metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
        results[method] = metrics
        
        print(f"  Method: {metrics['method']}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
    
    return results


def test_with_different_dimensions():
    """Test alignment with different dimensional spaces."""
    print("\n=== Testing with different dimensions ===")
    
    n_samples = 300
    dim_A, dim_B = 256, 128  # Different dimensions
    
    np.random.seed(42)
    embeddings_A = np.random.randn(n_samples, dim_A)
    
    # Project A to lower dimensional space B with some linear transformation
    projection_matrix = np.random.randn(dim_A, dim_B) * 0.1
    embeddings_B = embeddings_A @ projection_matrix
    embeddings_B += np.random.randn(n_samples, dim_B) * 0.1  # Add noise
    
    # Test alignment (should automatically choose CCA due to dimension mismatch)
    aligner = GeometricVec2Vec(alignment_method='auto')
    aligner.fit(embeddings_A, embeddings_B)
    
    metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
    print(f"Method chosen: {metrics['method']}")
    print(f"Cosine similarity: {metrics['cosine_similarity']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    
    return metrics


def test_low_rank_data():
    """Test with low-rank data structure."""
    print("\n=== Testing with low-rank data ===")
    
    n_samples = 400
    latent_dim = 20
    dim_A, dim_B = 128, 128
    
    np.random.seed(42)
    # Generate low-rank data
    latent_factors = np.random.randn(n_samples, latent_dim)
    basis_A = np.random.randn(latent_dim, dim_A)
    basis_B = np.random.randn(latent_dim, dim_B)
    
    embeddings_A = latent_factors @ basis_A + np.random.randn(n_samples, dim_A) * 0.1
    embeddings_B = latent_factors @ basis_B + np.random.randn(n_samples, dim_B) * 0.1
    
    # Test alignment (should detect low-rank structure)
    aligner = GeometricVec2Vec(alignment_method='auto')
    aligner.fit(embeddings_A, embeddings_B)
    
    metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
    print(f"Method chosen: {metrics['method']}")
    print(f"Cosine similarity: {metrics['cosine_similarity']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    
    if hasattr(aligner.alignment_info, 'explained_variance_A'):
        print(f"Explained variance A: {aligner.alignment_info.get('explained_variance_A', 'N/A')}")
        print(f"Explained variance B: {aligner.alignment_info.get('explained_variance_B', 'N/A')}")
    
    return metrics


def main():
    """Run all tests."""
    print("Testing Geometric Vec2Vec Alignment")
    print("=" * 50)
    
    try:
        # Test 1: Synthetic data with known transformation
        synthetic_results = test_with_synthetic_data()
        
        # Test 2: Different dimensions
        diff_dim_results = test_with_different_dimensions()
        
        # Test 3: Low-rank structure
        lowrank_results = test_low_rank_data()
        
        print("\n" + "=" * 50)
        print("SUMMARY OF RESULTS")
        print("=" * 50)
        
        print("\nSynthetic data (known transformation):")
        for method, results in synthetic_results.items():
            print(f"  {method:12}: cosine={results['cosine_similarity']:.4f}, mse={results['mse']:.6f}")
        
        print(f"\nDifferent dimensions: cosine={diff_dim_results['cosine_similarity']:.4f}")
        print(f"Low-rank data:       cosine={lowrank_results['cosine_similarity']:.4f}")
        
        # Find best method for synthetic data
        best_method = max(synthetic_results.keys(), 
                         key=lambda k: synthetic_results[k]['cosine_similarity'])
        print(f"\nBest method for synthetic data: {best_method}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()