import numpy as np
import os
import pandas as pd

def analyze_embeddings(embedding_path):
    """
    Analyzes embeddings by computing norm, standard deviation, correlation, AGOP, and rank.
    """
    try:
        embeddings = np.load(embedding_path)
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        
        if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            return np.nan, np.nan, np.nan, np.nan, False, np.nan

        # --- Previous Metrics ---
        mean_norm = np.mean(np.linalg.norm(embeddings, axis=-1))
        feature_std = np.mean(np.std(embeddings, axis=0))
        if embeddings.shape[1] > 1:
            corr_matrix = np.corrcoef(embeddings.T)
            np.fill_diagonal(corr_matrix, 0)
            mean_corr = np.mean(np.abs(corr_matrix))
        else:
            mean_corr = np.nan

        # --- New Metrics (AGOP, Chaotic Indicator, Rank) ---
        # Center the features
        features = embeddings - embeddings.mean(axis=0)
        n_samples, n_features = features.shape

        # 1. AGOP Matrix (approximated by covariance matrix)
        agop_matrix = (1 / n_samples) * features.T @ features
        
        # 2. Eigenvalues
        eigenvalues = np.linalg.eigvalsh(agop_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1] # Sort descending

        # 3. AGOP Ratio
        agop_ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10) if len(eigenvalues) > 1 else 1.0

        # 4. Chaotic Indicator
        chaotic_indicator = agop_ratio > 1e6

        # 5. Effective Rank
        effective_rank = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2) if np.sum(eigenvalues**2) > 0 else 0

        return mean_norm, feature_std, mean_corr, agop_ratio, chaotic_indicator, effective_rank

    except Exception as e:
        print(f"Could not process {embedding_path}: {e}")
        return np.nan, np.nan, np.nan, np.nan, False, np.nan

def main():
    """
    Main function to analyze all embeddings.
    """
    base_dir = "/Users/tanmoy/research/Perceptual_Features/platonic-rep/complete_features"
    
    vision_models = ["efficientnet_b0", "mobilenet_v2", "resnet18", "squeezenet"]
    language_models = ["albert", "distilbert"]
    
    results = []

    print("Analyzing Vision Models...")
    for model in vision_models:
        model_path = os.path.join(base_dir, model)
        if os.path.isdir(model_path):
            for layer_file in sorted(os.listdir(model_path)):
                if layer_file.endswith(".npy"):
                    layer_path = os.path.join(model_path, layer_file)
                    mean_norm, feature_std, mean_corr, agop_ratio, chaotic_indicator, effective_rank = analyze_embeddings(layer_path)
                    results.append({
                        "Modality": "Vision",
                        "Model": model,
                        "Layer": layer_file.replace(".npy", ""),
                        "Mean Norm": mean_norm,
                        "Feature Std": feature_std,
                        "Mean Correlation": mean_corr,
                        "AGOP Ratio": agop_ratio,
                        "Chaotic": chaotic_indicator,
                        "Eff. Rank": effective_rank
                    })

    print("Analyzing Language Models...")
    for model in language_models:
        model_path = os.path.join(base_dir, model)
        if os.path.isdir(model_path):
            for layer_file in sorted(os.listdir(model_path)):
                if layer_file.endswith(".npy"):
                    layer_path = os.path.join(model_path, layer_file)
                    mean_norm, feature_std, mean_corr, agop_ratio, chaotic_indicator, effective_rank = analyze_embeddings(layer_path)
                    results.append({
                        "Modality": "Language",
                        "Model": model,
                        "Layer": layer_file.replace(".npy", ""),
                        "Mean Norm": mean_norm,
                        "Feature Std": feature_std,
                        "Mean Correlation": mean_corr,
                        "AGOP Ratio": agop_ratio,
                        "Chaotic": chaotic_indicator,
                        "Eff. Rank": effective_rank
                    })

    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    print("\n--- Vision Models Analysis ---")
    print(df[df["Modality"] == "Vision"].to_string())
    
    print("\n--- Language Models Analysis ---")
    print(df[df["Modality"] == "Language"].to_string())

if __name__ == "__main__":
    main()