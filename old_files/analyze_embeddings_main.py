import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def analyze_embeddings(embedding_path):
    """
    Analyzes embeddings by computing norm, standard deviation, correlation, AGOP, and rank.
    """
    try:
        embeddings = np.load(embedding_path)
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        
        if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            return {}

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
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero eigenvalues

        # 3. AGOP Ratio
        agop_ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10) if len(eigenvalues) > 1 else 1.0

        # 4. Chaotic Indicator
        chaotic_indicator = agop_ratio > 1e6

        # 5. Effective Rank
        effective_rank = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2) if np.sum(eigenvalues**2) > 0 else 0

        # 6. Spectral decay rate (fit power law)
        if len(eigenvalues) > 10:
            k_values = np.arange(1, min(len(eigenvalues), 50) + 1)
            log_k = np.log(k_values)
            log_lambda = np.log(eigenvalues[:len(k_values)])
            slope, _ = np.polyfit(log_k, log_lambda, 1)
            spectral_decay = -slope
        else:
            spectral_decay = np.nan

        return {
            "mean_norm": mean_norm,
            "feature_std": feature_std,
            "mean_corr": mean_corr,
            "agop_ratio": agop_ratio,
            "chaotic": chaotic_indicator,
            "effective_rank": effective_rank,
            "eigenvalues": eigenvalues,
            "spectral_decay": spectral_decay,
            "agop_matrix": agop_matrix
        }

    except Exception as e:
        print(f"Could not process {embedding_path}: {e}")
        return {}

def compute_phase_alpha(agop_matrix, embeddings):
    """
    Compute the phase exponent alpha by analyzing H^alpha ∝ G relationship
    """
    try:
        # Compute representation covariance
        H = np.cov(embeddings.T)
        
        # Use AGOP as proxy for gradient covariance G
        G = agop_matrix
        
        # Get eigenvalues
        h_eigs = np.linalg.eigvalsh(H)
        g_eigs = np.linalg.eigvalsh(G)
        
        # Sort and take top eigenvalues
        h_eigs = np.sort(h_eigs)[::-1][:20]
        g_eigs = np.sort(g_eigs)[::-1][:20]
        
        # Fit log(H) = (1/alpha) * log(G) + c
        log_h = np.log(h_eigs[h_eigs > 1e-10])
        log_g = np.log(g_eigs[g_eigs > 1e-10])
        
        if len(log_h) > 5 and len(log_g) > 5:
            # Use regression to find alpha
            model = LinearRegression()
            model.fit(log_g.reshape(-1, 1), log_h)
            alpha = 1.0 / model.coef_[0] if model.coef_[0] != 0 else np.nan
            return alpha
        else:
            return np.nan
    except:
        return np.nan

def analyze_all_models():
    """
    Analyze all models and compute cross-modal metrics
    """
    base_dir = "/Users/tanmoy/research/Perceptual_Features/platonic-rep/complete_features"
    
    vision_models = ["efficientnet_b0", "mobilenet_v2", "resnet18", "squeezenet"]
    language_models = ["albert", "distilbert"]
    
    results = []
    detailed_results = {}

    print("Analyzing Vision Models...")
    for model in vision_models:
        model_path = os.path.join(base_dir, model)
        if os.path.isdir(model_path):
            model_data = []
            for layer_file in sorted(os.listdir(model_path)):
                if layer_file.endswith(".npy"):
                    layer_path = os.path.join(model_path, layer_file)
                    embeddings = np.load(layer_path)
                    if embeddings.ndim > 2:
                        embeddings = embeddings.reshape(embeddings.shape[0], -1)
                    
                    analysis = analyze_embeddings(layer_path)
                    if analysis:
                        alpha = compute_phase_alpha(analysis.get('agop_matrix', None), embeddings)
                        
                        result = {
                            "Modality": "Vision",
                            "Model": model,
                            "Layer": layer_file.replace(".npy", ""),
                            "Mean Norm": analysis['mean_norm'],
                            "Feature Std": analysis['feature_std'],
                            "Mean Correlation": analysis['mean_corr'],
                            "AGOP Ratio": analysis['agop_ratio'],
                            "Chaotic": analysis['chaotic'],
                            "Eff. Rank": analysis['effective_rank'],
                            "Spectral Decay": analysis['spectral_decay'],
                            "Alpha": alpha
                        }
                        results.append(result)
                        model_data.append(analysis)
            
            detailed_results[model] = model_data

    print("Analyzing Language Models...")
    for model in language_models:
        model_path = os.path.join(base_dir, model)
        if os.path.isdir(model_path):
            model_data = []
            for layer_file in sorted(os.listdir(model_path)):
                if layer_file.endswith(".npy"):
                    layer_path = os.path.join(model_path, layer_file)
                    embeddings = np.load(layer_path)
                    if embeddings.ndim > 2:
                        embeddings = embeddings.reshape(embeddings.shape[0], -1)
                    
                    analysis = analyze_embeddings(layer_path)
                    if analysis:
                        alpha = compute_phase_alpha(analysis.get('agop_matrix', None), embeddings)
                        
                        result = {
                            "Modality": "Language",
                            "Model": model,
                            "Layer": layer_file.replace(".npy", ""),
                            "Mean Norm": analysis['mean_norm'],
                            "Feature Std": analysis['feature_std'],
                            "Mean Correlation": analysis['mean_corr'],
                            "AGOP Ratio": analysis['agop_ratio'],
                            "Chaotic": analysis['chaotic'],
                            "Eff. Rank": analysis['effective_rank'],
                            "Spectral Decay": analysis['spectral_decay'],
                            "Alpha": alpha
                        }
                        results.append(result)
                        model_data.append(analysis)
            
            detailed_results[model] = model_data

    df = pd.DataFrame(results)
    return df, detailed_results

def plot_agop_spectral_decay(df, detailed_results):
    """
    Plot AGOP eigenvalue decay for vision vs language models
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get representative eigenvalues from each modality
    vision_eigs = []
    language_eigs = []
    
    for _, row in df.iterrows():
        model = row['Model']
        if model in detailed_results and detailed_results[model]:
            # Take eigenvalues from middle layer
            mid_idx = len(detailed_results[model]) // 2
            if 'eigenvalues' in detailed_results[model][mid_idx]:
                eigs = detailed_results[model][mid_idx]['eigenvalues']
                if row['Modality'] == 'Vision':
                    vision_eigs.append(eigs)
                else:
                    language_eigs.append(eigs)
    
    # Average eigenvalues
    if vision_eigs:
        max_len = max(len(e) for e in vision_eigs)
        vision_avg = np.zeros(max_len)
        for e in vision_eigs:
            vision_avg[:len(e)] += e / len(vision_eigs)
        
        k = np.arange(1, min(len(vision_avg), 100) + 1)
        ax.loglog(k, vision_avg[:len(k)], 'b-', linewidth=2, label='Vision (avg)')
    
    if language_eigs:
        max_len = max(len(e) for e in language_eigs)
        language_avg = np.zeros(max_len)
        for e in language_eigs:
            language_avg[:len(e)] += e / len(language_eigs)
        
        k = np.arange(1, min(len(language_avg), 100) + 1)
        ax.loglog(k, language_avg[:len(k)], 'r-', linewidth=2, label='Language (avg)')
    
    # Add theoretical slopes
    k_theory = np.logspace(0.5, 2, 50)
    ax.loglog(k_theory, 0.1 * k_theory**(-2), 'b--', alpha=0.5, label='k⁻² (theory)')
    ax.loglog(k_theory, 0.5 * k_theory**(-1), 'r--', alpha=0.5, label='k⁻¹ (theory)')
    
    ax.set_xlabel('Eigenvalue Index k')
    ax.set_ylabel('λₖ(AGOP)')
    ax.set_title('AGOP Spectral Decay: Vision vs Language')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_phase_space_real(df):
    """
    Plot phase space with real alpha values
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get alpha values for each model pair
    vision_alphas = df[df['Modality'] == 'Vision'].groupby('Model')['Alpha'].mean()
    language_alphas = df[df['Modality'] == 'Language'].groupby('Model')['Alpha'].mean()
    
    # Create all pairs
    pairs = []
    for v_model, v_alpha in vision_alphas.items():
        for l_model, l_alpha in language_alphas.items():
            if not np.isnan(v_alpha) and not np.isnan(l_alpha):
                phase_diff = abs(v_alpha - l_alpha)
                # Simulate alignment score based on phase difference
                alignment = 0.9 * np.exp(-3 * phase_diff) + 0.02
                pairs.append({
                    'vision_model': v_model,
                    'language_model': l_model,
                    'alpha_v': v_alpha,
                    'alpha_l': l_alpha,
                    'phase_diff': phase_diff,
                    'alignment': alignment
                })
    
    pairs_df = pd.DataFrame(pairs)

    if pairs_df.empty:
        print("Could not generate phase space plot because no valid alpha pairs were found.")
        # Draw an empty plot with a message
        ax.text(0.5, 0.5, "No valid data to plot Phase Space",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red')
        ax.set_xlabel('Phase Difference |αᵥ - αₗ|', fontsize=12)
        ax.set_ylabel('Predicted Alignment Success', fontsize=12)
        ax.set_title('Phase Compatibility Analysis: Real Model Pairs', fontsize=14)
        return fig, pd.DataFrame()

    # Create scatter plot
    scatter = ax.scatter(pairs_df['phase_diff'], pairs_df['alignment'], 
                        c=pairs_df['alignment'], s=100, alpha=0.7,
                        cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add annotations for some points
    for _, row in pairs_df.iterrows():
        if row['alignment'] > 0.5 or row['alignment'] < 0.1:
            ax.annotate(f"{row['vision_model'][:4]}-{row['language_model'][:4]}", 
                       (row['phase_diff'], row['alignment']),
                       fontsize=8, alpha=0.7)
    
    # Add threshold line
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
               label='Critical threshold |Δα| = 0.5')
    
    ax.set_xlabel('Phase Difference |αᵥ - αₗ|', fontsize=12)
    ax.set_ylabel('Predicted Alignment Success', fontsize=12)
    ax.set_title('Phase Compatibility Analysis: Real Model Pairs', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Alignment Score', fontsize=10)
    
    return fig, pairs_df

def plot_model_characteristics(df):
    """
    Plot model characteristics by modality
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. AGOP Ratio distribution
    ax = axes[0, 0]
    vision_agop = df[df['Modality'] == 'Vision']['AGOP Ratio'].dropna()
    language_agop = df[df['Modality'] == 'Language']['AGOP Ratio'].dropna()
    
    ax.hist(np.log10(vision_agop), bins=20, alpha=0.5, label='Vision', color='blue')
    ax.hist(np.log10(language_agop), bins=20, alpha=0.5, label='Language', color='red')
    ax.axvline(x=6, color='black', linestyle='--', label='Chaos threshold (10⁶)')
    ax.set_xlabel('log₁₀(AGOP Ratio)')
    ax.set_ylabel('Count')
    ax.set_title('AGOP Ratio Distribution')
    ax.legend()
    
    # 2. Effective Rank
    ax = axes[0, 1]
    vision_rank = df[df['Modality'] == 'Vision']['Eff. Rank'].dropna()
    language_rank = df[df['Modality'] == 'Language']['Eff. Rank'].dropna()
    
    ax.boxplot([vision_rank, language_rank], labels=['Vision', 'Language'])
    ax.set_ylabel('Effective Rank')
    ax.set_title('Representation Dimensionality')
    
    # 3. Alpha distribution
    ax = axes[1, 0]
    vision_alpha = df[df['Modality'] == 'Vision']['Alpha'].dropna()
    language_alpha = df[df['Modality'] == 'Language']['Alpha'].dropna()
    
    ax.hist(vision_alpha, bins=15, alpha=0.5, label='Vision', color='blue')
    ax.hist(language_alpha, bins=15, alpha=0.5, label='Language', color='red')
    ax.set_xlabel('Phase Exponent α')
    ax.set_ylabel('Count')
    ax.set_title('Phase Distribution')
    ax.legend()
    
    # 4. Spectral Decay
    ax = axes[1, 1]
    vision_decay = df[df['Modality'] == 'Vision']['Spectral Decay'].dropna()
    language_decay = df[df['Modality'] == 'Language']['Spectral Decay'].dropna()
    
    ax.scatter(['Vision']*len(vision_decay), vision_decay, alpha=0.5, color='blue')
    ax.scatter(['Language']*len(language_decay), language_decay, alpha=0.5, color='red')
    ax.axhline(y=2, color='blue', linestyle='--', alpha=0.5, label='k⁻² (vision)')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='k⁻¹ (language)')
    ax.set_ylabel('Spectral Decay Rate')
    ax.set_title('AGOP Eigenvalue Decay')
    ax.legend()
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function to analyze embeddings and create plots
    """
    # Analyze all models
    df, detailed_results = analyze_all_models()
    
    # Save analysis results
    df.to_csv('embedding_analysis_results.csv', index=False)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    
    print("\nVision Models:")
    print(df[df['Modality'] == 'Vision'].groupby('Model')[['AGOP Ratio', 'Eff. Rank', 'Alpha']].mean())
    
    print("\nLanguage Models:")
    print(df[df['Modality'] == 'Language'].groupby('Model')[['AGOP Ratio', 'Eff. Rank', 'Alpha']].mean())
    
    # Create plots
    print("\nGenerating plots...")
    
    # 1. AGOP Spectral Decay
    fig1 = plot_agop_spectral_decay(df, detailed_results)
    fig1.savefig('figure_agop_decay.pdf', bbox_inches='tight', dpi=300)
    
    # 2. Phase Space
    fig2, pairs_df = plot_phase_space_real(df)
    fig2.savefig('figure_phase_space.pdf', bbox_inches='tight', dpi=300)
    
    # 3. Model Characteristics
    fig3 = plot_model_characteristics(df)
    fig3.savefig('figure_model_characteristics.pdf', bbox_inches='tight', dpi=300)
    
    # Save cross-modal predictions
    if not pairs_df.empty:
        pairs_df.to_csv('cross_modal_predictions.csv', index=False)
    
    print("\nAnalysis complete! Files saved:")
    print("- embedding_analysis_results.csv")
    if not pairs_df.empty:
        print("- cross_modal_predictions.csv")
    print("- figure_agop_decay.pdf")
    print("- figure_phase_space.pdf")
    print("- figure_model_characteristics.pdf")
    
    # Print key findings
    print("\n=== Key Findings ===")
    chaotic_vision = (df[df['Modality'] == 'Vision']['AGOP Ratio'] > 1e6).mean()
    chaotic_language = (df[df['Modality'] == 'Language']['AGOP Ratio'] > 1e6).mean()
    print(f"Chaotic layers - Vision: {chaotic_vision:.1%}, Language: {chaotic_language:.1%}")
    
    if not pairs_df.empty:
        compatible_pairs = (pairs_df['phase_diff'] < 0.5).mean()
        print(f"Compatible model pairs (|Δα| < 0.5): {compatible_pairs:.1%}")

if __name__ == "__main__":
    main()
