
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def _flatten_normalize(features):
    """Flatten and normalize feature numpy arrays"""
    flat = features.reshape(features.shape[0], -1)
    
    # Normalize
    flat = (flat - np.mean(flat, axis=0)) / (np.std(flat, axis=0) + 1e-8)
    return flat

def compute_representation_compatibility(vision_features, language_features):
    """
    Test static feature alignment using linear probe.
    This is based on the _test_static_alignment function from the provided script.
    """
    
    # Flatten features
    v_flat = _flatten_normalize(vision_features)
    l_flat = _flatten_normalize(language_features)

    # Ensure the number of samples is the same
    n_samples = min(v_flat.shape[0], l_flat.shape[0])
    v_flat = v_flat[:n_samples]
    l_flat = l_flat[:n_samples]
    
    # Train linear mapping
    reg = LinearRegression()
    reg.fit(v_flat, l_flat)
    
    # Test alignment success
    predicted = reg.predict(v_flat)
    r2 = r2_score(l_flat, predicted)
    
    return max(0, r2)  # Clip negative R2

if __name__ == "__main__":
    # Load pre-extracted features
    # Using final layer features as done in the original script
    vision_features_path = "/Users/tanmoy/research/Perceptual_Features/platonic-rep/complete_features/resnet18/avgpool.npy"
    language_features_path = "/Users/tanmoy/research/Perceptual_Features/platonic-rep/complete_features/distilbert/final.npy"
    
    try:
        vision_features = np.load(vision_features_path)
        language_features = np.load(language_features_path)
        
        print(f"Loaded vision features from {vision_features_path} with shape: {vision_features.shape}")
        print(f"Loaded language features from {language_features_path} with shape: {language_features.shape}")

        # Compute representation compatibility
        repr_compat = compute_representation_compatibility(vision_features, language_features)
        
        print("\n--- Analysis Results ---")
        print(f"Representation Compatibility (S^repr) between resnet18 and distilbert: {repr_compat:.4f}")
        
        print("\nNote: Optimization Compatibility (S^optim) could not be computed as it requires access to the original models and data for gradient calculation.")

    except FileNotFoundError as e:
        print(f"Error: Feature file not found. {e}")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
