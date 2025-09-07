#!/usr/bin/env python3
"""
Create a simplified information-theoretic analysis focusing on compression patterns
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns

# Set style for better plots
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_mutual_information_data():
    """Load mutual information data and organize by model type"""
    
    try:
        mi_df = pd.read_csv('mutual_information_results.csv')
        print(f"Loaded {len(mi_df)} mutual information measurements")
        return mi_df
    except FileNotFoundError:
        print("Mutual information file not found.")
        return None

def create_compression_patterns_analysis():
    """Create simplified compression patterns analysis"""
    
    # Load data
    mi_df = load_mutual_information_data()
    
    if mi_df is None:
        print("No mutual information data available")
        return None
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Information Plane Trajectories (Focus on Compression)
    ax1.set_title('Information Compression Patterns by Model Type', fontsize=14, weight='bold', pad=20)
    
    # Group models by architecture type and analyze trajectories
    model_groups = mi_df.groupby('model')
    
    # Define model categories based on typical behavior
    chaotic_models = ['convnext_tiny', 'convnext_small', 'convnext_base', 'resnet18', 'resnet34', 'resnet50']
    optimal_models = ['swin_small_patch4_window7_224', 'swin_base_patch4_window7_224', 'efficientnet_b2']
    extreme_models = ['efficientnet_b1']
    
    # Colors for different model types
    colors = {
        'chaotic': {'color': 'red', 'alpha': 0.8, 'linewidth': 2.5},
        'optimal': {'color': 'green', 'alpha': 0.8, 'linewidth': 2.5},
        'extreme': {'color': 'orange', 'alpha': 0.8, 'linewidth': 2.5}
    }
    
    trajectories_plotted = 0
    max_trajectories = 6  # Limit for clarity
    
    for model_name, group in model_groups:
        if trajectories_plotted >= max_trajectories:
            break
            
        # Get layer-wise data
        layer_data = group[group['layer_idx'] >= 0].sort_values('layer_idx')
        
        if len(layer_data) < 2:  # Need at least 2 layers for trajectory
            continue
        
        # Determine model category
        model_category = 'chaotic'
        if any(opt in model_name.lower() for opt in optimal_models):
            model_category = 'optimal'
        elif any(ext in model_name.lower() for ext in extreme_models):
            model_category = 'extreme'
        elif any(chaos in model_name.lower() for chaos in chaotic_models):
            model_category = 'chaotic'
        
        # Extract trajectory data
        i_xt_values = layer_data['I_X_T'].tolist()
        i_yt_values = layer_data['I_Y_T'].tolist()
        
        # Plot trajectory
        style = colors[model_category]
        ax1.plot(i_xt_values, i_yt_values, 'o-', 
                color=style['color'], markersize=6, linewidth=style['linewidth'],
                alpha=style['alpha'], label=f'{model_name} ({model_category.title()})')
        
        # Add arrows to show compression direction
        for i in range(len(i_xt_values)-1):
            ax1.annotate('', xy=(i_xt_values[i+1], i_yt_values[i+1]), 
                        xytext=(i_xt_values[i], i_yt_values[i]),
                        arrowprops=dict(arrowstyle='->', color=style['color'], 
                                      alpha=0.6, lw=1.5))
        
        trajectories_plotted += 1
    
    # Add compression zones
    ax1.axvspan(0.030, 0.033, alpha=0.15, color='red', label='Over-compression zone')
    ax1.axvspan(0.033, 0.036, alpha=0.15, color='green', label='Optimal compression zone')
    
    # Add horizontal line at I(Y;T) = 0.1 to show stability
    ax1.axhline(y=0.1, color='black', linestyle='--', alpha=0.7, linewidth=2, 
               label='Stable Task Information (I(Y;T) = 0.1)')
    
    ax1.set_xlabel('I(X;T) - Input Information', fontsize=12, weight='bold')
    ax1.set_ylabel('I(Y;T) - Task Information', fontsize=12, weight='bold')
    ax1.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.025, 0.040)
    ax1.set_ylim(0.095, 0.105)
    
    # Panel B: I(X;T) Trajectory Comparison
    ax2.set_title('Input Information Compression Trajectories', fontsize=14, weight='bold', pad=20)
    
    # Plot I(X;T) trajectories over layers
    trajectories_plotted = 0
    
    for model_name, group in model_groups:
        if trajectories_plotted >= max_trajectories:
            break
            
        layer_data = group[group['layer_idx'] >= 0].sort_values('layer_idx')
        
        if len(layer_data) < 2:
            continue
        
        # Determine model category
        model_category = 'chaotic'
        if any(opt in model_name.lower() for opt in optimal_models):
            model_category = 'optimal'
        elif any(ext in model_name.lower() for ext in extreme_models):
            model_category = 'extreme'
        elif any(chaos in model_name.lower() for chaos in chaotic_models):
            model_category = 'chaotic'
        
        # Extract layer indices and I(X;T) values
        layer_indices = layer_data['layer_idx'].tolist()
        i_xt_values = layer_data['I_X_T'].tolist()
        
        # Plot trajectory
        style = colors[model_category]
        ax2.plot(layer_indices, i_xt_values, 'o-', 
                color=style['color'], markersize=6, linewidth=style['linewidth'],
                alpha=style['alpha'], label=f'{model_name} ({model_category.title()})')
        
        trajectories_plotted += 1
    
    # Add compression threshold lines
    ax2.axhline(y=0.033, color='green', linestyle='--', alpha=0.7, linewidth=2, 
               label='Optimal Compression Threshold')
    ax2.axhline(y=0.030, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='Over-compression Threshold')
    
    ax2.set_xlabel('Layer Index', fontsize=12, weight='bold')
    ax2.set_ylabel('I(X;T) - Input Information', fontsize=12, weight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Compression Rate Analysis
    ax3.set_title('Compression Rate vs Model Type', fontsize=14, weight='bold', pad=20)
    
    # Calculate compression rates for each model
    compression_data = []
    
    for model_name, group in model_groups:
        layer_data = group[group['layer_idx'] >= 0].sort_values('layer_idx')
        
        if len(layer_data) > 1:
            initial_I_X_T = layer_data.iloc[0]['I_X_T']
            final_I_X_T = layer_data.iloc[-1]['I_X_T']
            compression_rate = (initial_I_X_T - final_I_X_T) / initial_I_X_T if initial_I_X_T > 0 else 0
            
            # Determine model category
            model_category = 'chaotic'
            if any(opt in model_name.lower() for opt in optimal_models):
                model_category = 'optimal'
            elif any(ext in model_name.lower() for ext in extreme_models):
                model_category = 'extreme'
            elif any(chaos in model_name.lower() for chaos in chaotic_models):
                model_category = 'chaotic'
            
            compression_data.append({
                'model': model_name,
                'compression_rate': compression_rate,
                'category': model_category,
                'initial_I_X_T': initial_I_X_T,
                'final_I_X_T': final_I_X_T
            })
    
    # Create box plot of compression rates by category
    compression_df = pd.DataFrame(compression_data)
    
    if len(compression_df) > 0:
        # Create box plot
        categories = compression_df['category'].unique()
        box_data = [compression_df[compression_df['category'] == cat]['compression_rate'].tolist() 
                   for cat in categories]
        
        bp = ax3.boxplot(box_data, labels=[cat.title() for cat in categories], 
                        patch_artist=True, notch=True)
        
        # Color the boxes
        box_colors = ['red', 'green', 'orange']
        for patch, color in zip(bp['boxes'], box_colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for i, cat in enumerate(categories):
            cat_data = compression_df[compression_df['category'] == cat]['compression_rate']
            y_pos = np.random.normal(i+1, 0.04, size=len(cat_data))
            ax3.scatter(y_pos, cat_data, alpha=0.6, s=50, color=box_colors[i])
        
        ax3.set_ylabel('Compression Rate', fontsize=12, weight='bold')
        ax3.set_xlabel('Model Category', fontsize=12, weight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add threshold line
        ax3.axhline(y=0.1, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                   label='Moderate Compression Threshold')
    
    # Panel D: Information Efficiency Analysis
    ax4.set_title('Information Efficiency by Model Type', fontsize=14, weight='bold', pad=20)
    
    # Calculate information efficiency (I(Y;T) / I(X;T)) for each model
    efficiency_data = []
    
    for model_name, group in model_groups:
        layer_data = group[group['layer_idx'] >= 0].sort_values('layer_idx')
        
        if len(layer_data) > 0:
            # Use average values across layers
            avg_i_xt = layer_data['I_X_T'].mean()
            avg_i_yt = layer_data['I_Y_T'].mean()
            efficiency = avg_i_yt / avg_i_xt if avg_i_xt > 0 else 0
            
            # Determine model category
            model_category = 'chaotic'
            if any(opt in model_name.lower() for opt in optimal_models):
                model_category = 'optimal'
            elif any(ext in model_name.lower() for ext in extreme_models):
                model_category = 'extreme'
            elif any(chaos in model_name.lower() for chaos in chaotic_models):
                model_category = 'chaotic'
            
            efficiency_data.append({
                'model': model_name,
                'efficiency': efficiency,
                'category': model_category,
                'avg_I_X_T': avg_i_xt,
                'avg_I_Y_T': avg_i_yt
            })
    
    # Create scatter plot of efficiency
    efficiency_df = pd.DataFrame(efficiency_data)
    
    if len(efficiency_df) > 0:
        for category in efficiency_df['category'].unique():
            cat_data = efficiency_df[efficiency_df['category'] == category]
            color = colors[category]['color']
            ax4.scatter(cat_data['avg_I_X_T'], cat_data['efficiency'], 
                      c=color, s=100, alpha=0.7, label=f'{category.title()} Models',
                      edgecolors='black', linewidth=0.5)
        
        ax4.set_xlabel('Average I(X;T)', fontsize=12, weight='bold')
        ax4.set_ylabel('Information Efficiency (I(Y;T)/I(X;T))', fontsize=12, weight='bold')
        ax4.legend(fontsize=10, framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        
        # Add efficiency threshold
        ax4.axhline(y=3.0, color='green', linestyle='--', alpha=0.7, linewidth=2, 
                   label='High Efficiency Threshold')
    
    # Add overall insights
    insights_text = """Key Insights:
• Chaotic models show erratic I(X;T) trajectories
• All models maintain stable I(Y;T) = 0.1
• ConvNeXt demonstrates over-compression
• Swin models show controlled compression
• EfficientNet-B1 shows extreme compression"""
    
    # Create insights box
    insights_box = FancyBboxPatch((0.02, 0.02), 0.25, 0.25,
                                 boxstyle="round,pad=0.02",
                                 facecolor="lightyellow",
                                 edgecolor="darkorange",
                                 alpha=0.95,
                                 linewidth=1.5,
                                 zorder=20)
    
    ax1.add_patch(insights_box)
    ax1.text(0.145, 0.145, insights_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            weight='normal', zorder=21)
    
    # Clean backgrounds
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('compression_patterns_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('compression_patterns_analysis.pdf', bbox_inches='tight', facecolor='white')
    
    print(f"Compression patterns analysis saved as:")
    print(f"  - compression_patterns_analysis.png")
    print(f"  - compression_patterns_analysis.pdf")
    
    # Print summary statistics
    if len(compression_df) > 0:
        print(f"\nCompression Rate Summary:")
        for category in compression_df['category'].unique():
            cat_data = compression_df[compression_df['category'] == category]
            avg_compression = cat_data['compression_rate'].mean()
            print(f"  {category.title()}: {avg_compression:.4f} ± {cat_data['compression_rate'].std():.4f}")
    
    if len(efficiency_df) > 0:
        print(f"\nInformation Efficiency Summary:")
        for category in efficiency_df['category'].unique():
            cat_data = efficiency_df[efficiency_df['category'] == category]
            avg_efficiency = cat_data['efficiency'].mean()
            print(f"  {category.title()}: {avg_efficiency:.4f} ± {cat_data['efficiency'].std():.4f}")
    
    return fig

def main():
    """Main function to create compression patterns analysis"""
    
    print("Creating simplified information-theoretic analysis focusing on compression patterns...")
    
    # Create the analysis
    fig = create_compression_patterns_analysis()
    
    if fig is not None:
        print("Compression patterns analysis created successfully!")
    else:
        print("Failed to create compression patterns analysis.")

if __name__ == "__main__":
    main()
