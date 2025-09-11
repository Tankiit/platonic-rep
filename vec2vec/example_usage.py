#!/usr/bin/env python3
"""
Example usage of the Geometric Vec2Vec alignment system.

This demonstrates how to use the geometric alignment approach to align
embedding spaces without adversarial training.
"""

import numpy as np
from geometric_alignment import GeometricVec2Vec, extract_embeddings, load_sample_texts


def example_with_real_models():
    """Example using real embedding models (requires sentence-transformers)."""
    print("=== Example with Real Models ===")
    
    # Create some sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns in data.",
        "Embeddings capture semantic meaning in vector representations."
    ]
    
    print(f"Using {len(texts)} sample texts")
    
    try:
        # Extract embeddings from two different models
        embeddings_A = extract_embeddings('sbert', texts, device='cpu')  # Fallback model
        embeddings_B = extract_embeddings('sbert', texts, device='cpu')  # Same model for demo
        
        # Add some noise to make them different
        embeddings_B = embeddings_B + np.random.randn(*embeddings_B.shape) * 0.1
        
        print(f"Embeddings A shape: {embeddings_A.shape}")
        print(f"Embeddings B shape: {embeddings_B.shape}")
        
        # Test different alignment methods
        methods = ['procrustes', 'lowrank', 'auto']
        for method in methods:
            print(f"\n--- Testing {method} alignment ---")
            
            # Initialize aligner
            aligner = GeometricVec2Vec(alignment_method=method)
            
            # Fit the alignment
            aligner.fit(embeddings_A, embeddings_B)
            
            # Evaluate
            metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
            
            print(f"Method used: {metrics['method']}")
            print(f"Cosine similarity: {metrics['cosine_similarity']:.4f}")
            print(f"MSE: {metrics['mse']:.4f}")
            
            # Transform some embeddings
            A_to_B = aligner.transform(embeddings_A[:2], 'A_to_B')
            print(f"Transformed shape: {A_to_B.shape}")
        
    except Exception as e:
        print(f"Real model example failed: {e}")
        print("This is likely because sentence-transformers is not available")
        example_with_synthetic_data()


def example_with_synthetic_data():
    """Example using synthetic data to demonstrate the concepts."""
    print("\n=== Example with Synthetic Data ===")
    
    # Generate synthetic embeddings
    np.random.seed(42)
    n_samples = 100
    dim_A, dim_B = 64, 64
    
    # Create embeddings A
    embeddings_A = np.random.randn(n_samples, dim_A)
    
    # Create embeddings B with a known relationship to A
    # Apply rotation + translation + noise
    theta = np.pi / 4  # 45 degrees
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    
    # Apply rotation to first 2 dimensions, keep others similar
    embeddings_B = embeddings_A.copy()
    embeddings_B[:, :2] = embeddings_A[:, :2] @ rotation
    embeddings_B += np.random.randn(n_samples, dim_B) * 0.1  # Add noise
    
    print(f"Generated synthetic embeddings:")
    print(f"  Space A: {embeddings_A.shape}")
    print(f"  Space B: {embeddings_B.shape}")
    print(f"  Known transformation: 45° rotation + noise")
    
    # Test alignment
    aligner = GeometricVec2Vec(alignment_method='auto')
    aligner.fit(embeddings_A, embeddings_B)
    
    # Evaluate
    metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
    
    print(f"\nAlignment Results:")
    print(f"  Method chosen: {metrics['method']}")
    print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    
    # Show alignment quality
    A_transformed = aligner.transform(embeddings_A, 'A_to_B')
    
    # Compute alignment error
    alignment_error = np.mean(np.linalg.norm(A_transformed - embeddings_B, axis=1))
    print(f"  Mean alignment error: {alignment_error:.4f}")
    
    # Show that reverse transformation works
    B_to_A = aligner.transform(embeddings_B, 'B_to_A')
    reverse_error = np.mean(np.linalg.norm(B_to_A - embeddings_A, axis=1))
    print(f"  Reverse alignment error: {reverse_error:.4f}")


def demonstrate_method_selection():
    """Demonstrate how different methods are chosen automatically."""
    print("\n=== Method Selection Demonstration ===")
    
    np.random.seed(42)
    scenarios = [
        {
            'name': 'Same dimensions',
            'dim_A': 64, 'dim_B': 64,
            'expected': 'procrustes'
        },
        {
            'name': 'Different dimensions', 
            'dim_A': 128, 'dim_B': 64,
            'expected': 'cca'
        },
        {
            'name': 'Low-rank structure',
            'dim_A': 64, 'dim_B': 64,
            'low_rank': True,
            'expected': 'lowrank'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        n_samples = 200
        dim_A, dim_B = scenario['dim_A'], scenario['dim_B']
        
        if scenario.get('low_rank', False):
            # Create low-rank data
            latent_dim = 10
            latent = np.random.randn(n_samples, latent_dim)
            basis_A = np.random.randn(latent_dim, dim_A)
            basis_B = np.random.randn(latent_dim, dim_B)
            
            embeddings_A = latent @ basis_A + np.random.randn(n_samples, dim_A) * 0.1
            embeddings_B = latent @ basis_B + np.random.randn(n_samples, dim_B) * 0.1
        else:
            # Create regular random data
            embeddings_A = np.random.randn(n_samples, dim_A)
            if dim_A == dim_B:
                embeddings_B = embeddings_A + np.random.randn(n_samples, dim_B) * 0.2
            else:
                # Project to different dimension
                projection = np.random.randn(dim_A, dim_B) * 0.1
                embeddings_B = embeddings_A @ projection + np.random.randn(n_samples, dim_B) * 0.1
        
        # Test automatic method selection
        aligner = GeometricVec2Vec(alignment_method='auto')
        aligner.fit(embeddings_A, embeddings_B)
        
        metrics = aligner.evaluate_alignment(embeddings_A, embeddings_B)
        
        print(f"  Expected method: {scenario['expected']}")
        print(f"  Chosen method: {metrics['method']}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
        
        match = "✓" if metrics['method'] == scenario['expected'] else "✗"
        print(f"  Selection correct: {match}")


def main():
    """Run all examples."""
    print("Geometric Vec2Vec Alignment Examples")
    print("=" * 50)
    
    # Example 1: Try with real models
    example_with_real_models()
    
    # Example 2: Synthetic data
    example_with_synthetic_data()
    
    # Example 3: Method selection
    demonstrate_method_selection()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nKey insights from geometric alignment:")
    print("• Procrustes works best for same-dimensional spaces with similar structure")
    print("• CCA handles different dimensions by finding shared semantic directions")
    print("• Low-rank alignment leverages universal structural patterns")
    print("• Auto selection chooses the best method based on data characteristics")
    print("• No adversarial training needed - direct mathematical optimization!")


if __name__ == "__main__":
    main()