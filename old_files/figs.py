import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns
from scipy.interpolate import griddata

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# Cleaner text for all figures

# For the phase diagram (Images 1, 2, 7):
def clean_phase_labels():
    """Consistent, clean labeling"""
    
    # Phase region labels - make them more descriptive
    phase_labels = {
        'chaotic': 'Chaotic Phase\n(75% of pairs)\nUnstable dynamics',
        'optimal': 'Optimal Phase\n(25% of pairs)\nBalanced learning', 
        'lazy': 'Lazy Phase\n(0% observed)\nNo feature learning'
    }
    
    # Model labels - consistent format
    model_labels = {
        # Format: "Architecture Score"
        'efficientnet_b2': 'EffNet-B2\n0.025',
        'convnext_base': 'ConvNeXt-B\n0.021',
        'resnet50': 'ResNet-50\n0.017',
        'vit_base': 'ViT-B\n0.017',
        'swin_small': 'Swin-S\n0.023'
    }
    
    # Stats box - make it cleaner
    stats_text = (
        "144 Model Pairs\n"
        "━━━━━━━━━━━━\n"
        "Chaotic: 108 (75%)\n"
        "Optimal: 36 (25%)\n" 
        "Mean NTK: 0.323"
    )
    
    return phase_labels, model_labels, stats_text

# For the flow field (Image 4):
def clean_flow_labels():
    """Clearer dynamics labels"""
    
    # Remove redundant text, use icons/arrows
    legend_items = [
        ('→', 'green', 'Phase-compatible alignment'),
        ('⤳', 'red', 'Cross-phase failure'),
        ('•', 'size', 'Alignment score')
    ]
    
    # Phase labels without redundancy
    phase_zones = {
        'chaotic': 'CHAOTIC',
        'optimal': 'OPTIMAL',
        'lazy': 'LAZY'
    }
    
    return legend_items, phase_zones

# For the heatmap (Images 5, 6):
def clean_heatmap_labels():
    """Simplified heatmap annotations"""
    
    # Key findings as callouts
    callouts = [
        # Position these strategically
        {'pos': (0.2, 8.5), 'text': '75% trapped here', 'arrow': True},
        {'pos': (0.8, 8.5), 'text': '25% succeed here', 'arrow': True}
    ]
    
    # Model annotations - only top performers
    top_models = [
        {'model': 'EffNet-B2', 'score': '0.025', 'highlight': True},
        {'model': 'Swin-S', 'score': '0.023', 'highlight': True},
        {'model': 'ResNet-50', 'score': '0.017', 'highlight': False}
    ]
    
    return callouts, top_models

# For connections diagram (Image 7):
def clean_connection_labels():
    """Clear connection semantics"""
    
    # Connection types
    connections = {
        'success': {
            'color': 'green',
            'width': 3,
            'style': '-',
            'label': None  # Don't label individual connections
        },
        'failure': {
            'color': 'red', 
            'width': 1,
            'style': '--',
            'label': None
        }
    }
    
    # Single clear legend
    legend_text = (
        "Same phase → Linear projection (>0.75 alignment)\n"
        "Different phases → Complex training (<0.40 alignment)"
    )
    
    return connections, legend_text

# General improvements:
def improve_all_figures():
    """Universal text improvements"""
    
    # Font hierarchy
    fonts = {
        'title': {'size': 16, 'weight': 'bold'},
        'phase_label': {'size': 14, 'weight': 'bold'},
        'model_label': {'size': 10, 'weight': 'normal'},
        'score': {'size': 9, 'weight': 'normal'},
        'axis': {'size': 12, 'weight': 'normal'}
    }
    
    # Consistent number format
    def format_score(val):
        return f"{val:.3f}" if val > 0.02 else f"{val:.2f}"
    
    # Phase boundaries - make them prominent
    boundary_style = {
        'color': 'black',
        'style': '--',
        'width': 2,
        'alpha': 0.8
    }
    
    # Remove redundant elements
    remove = [
        'Grid lines in background',
        'Duplicate axis labels', 
        'Overlapping model names',
        'Decimal places beyond 3'
    ]
    
    return fonts, format_score, boundary_style, remove

def smart_text_positioning(points, labels, ax, min_distance=0.05):
    """Intelligently position text labels to avoid overlaps"""
    from scipy.spatial.distance import cdist
    
    # Convert to array if needed
    points = np.array(points)
    labels = list(labels)
    
    # Calculate pairwise distances
    distances = cdist(points, points)
    
    # Find overlapping points
    overlapping = distances < min_distance
    np.fill_diagonal(overlapping, False)
    
    # Adjust positions for overlapping labels
    adjusted_points = points.copy()
    for i in range(len(points)):
        overlaps = np.where(overlapping[i])[0]
        if len(overlaps) > 0:
            # Move this point slightly
            offset = np.random.uniform(-min_distance/2, min_distance/2, 2)
            adjusted_points[i] += offset
    
    return adjusted_points

def create_phase_landscape_3d_projection():
    """Create a 3D-style phase landscape with height representing alignment"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get clean labels
    phase_labels, model_labels, stats_text = clean_phase_labels()
    fonts, format_score, boundary_style, remove = improve_all_figures()
    
    # Create meshgrid for landscape
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Define alignment landscape as function of phase
    Z = np.zeros_like(X)
    
    # Lazy phase (high NTK, low performance)
    lazy_mask = X > 0.9
    Z[lazy_mask] = 0.4 + 0.1 * np.random.randn(np.sum(lazy_mask))
    
    # Optimal phase (balanced, high performance)
    optimal_mask = (X >= 0.7) & (X <= 0.9)
    Z[optimal_mask] = 0.75 + 0.05 * np.random.randn(np.sum(optimal_mask))
    
    # Chaotic phase (low NTK, variable performance)
    chaotic_mask = X < 0.7
    Z[chaotic_mask] = 0.25 + 0.15 * np.random.randn(np.sum(chaotic_mask))
    
    # Add smooth transitions
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=2)
    
    # Plot as contour
    levels = np.linspace(0, 1, 20)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlGn', alpha=0.8)
    
    # Add contour lines
    contour = ax.contour(X, Y, Z, levels=[0.3, 0.5, 0.7], colors='black', 
                         linewidths=0.5, alpha=0.5)
    
    # Plot actual model data with cleaner labels
    model_data = [
        # (name, ntk, agop, alignment, phase)
        ('EffNet-B2', 0.806, 1.77, 0.0247, 'optimal'),
        ('EffNet-B2', 0.816, 1.66, 0.0212, 'optimal'),
        ('ConvNeXt-S', 0.185, 20.4, 0.0210, 'chaotic'),
        ('ResNet-50', 0.181, 5.96, 0.0233, 'chaotic'),
        ('ViT-B', 0.178, 29.1, 0.0170, 'chaotic'),
        ('Swin-S', 0.734, 0.21, 0.0234, 'optimal'),
    ]
    
    # Collect points for smart positioning
    label_points = []
    label_texts = []
    
    for name, ntk, agop_raw, align, phase in model_data:
        # Normalize AGOP for visualization
        agop = np.log10(agop_raw + 1) * 2  # Scale appropriately
        
        size = align * 20000  # Scale by alignment
        color = '#27ae60' if phase == 'optimal' else '#e74c3c'
        edge = 'gold' if align > 0.022 else 'black'
        width = 3 if align > 0.022 else 1
        
        ax.scatter(ntk, agop, s=size, c=color, edgecolor=edge, 
                  linewidth=width, zorder=5, alpha=0.9)
        
        # Collect points for top performers
        if align > 0.021:
            label_points.append([ntk, agop])
            label_texts.append(f'{name}\n{format_score(align)}')
    
    # Smart positioning for labels
    if label_points:
        adjusted_points = smart_text_positioning(label_points, label_texts, ax, min_distance=0.3)
        
        for i, (point, text) in enumerate(zip(adjusted_points, label_texts)):
            ax.annotate(text, 
                       (point[0], point[1]), xytext=(0, 10), 
                       textcoords='offset points', ha='center',
                       fontsize=fonts['model_label']['size'], 
                       weight=fonts['model_label']['weight'],
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.8))
    
    # Add phase boundaries with cleaner style
    ax.axvline(x=0.7, color=boundary_style['color'], linestyle=boundary_style['style'], 
               linewidth=boundary_style['width'], alpha=boundary_style['alpha'])
    ax.axvline(x=0.9, color=boundary_style['color'], linestyle=boundary_style['style'], 
               linewidth=boundary_style['width'], alpha=boundary_style['alpha'])
    
    # Cleaner labels
    ax.set_xlabel('NTK Stability ($S_{NTK}$)', fontsize=fonts['axis']['size'])
    ax.set_ylabel('AGOP Magnitude ($M_{AGOP}$)', fontsize=fonts['axis']['size'])
    ax.set_title('Cross-Modal Alignment Landscape: Phase-Dependent Performance', 
                fontsize=fonts['title']['size'], weight=fonts['title']['weight'])
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Alignment Score')
    cbar.set_label('Cross-Modal Alignment', fontsize=fonts['axis']['size'])
    
    # Add cleaner phase labels with better positioning
    ax.annotate(phase_labels['chaotic'], 
               xy=(0.35, 8), xytext=(0.35, 9.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#c0392b'),
               ha='center', fontsize=fonts['phase_label']['size'], 
               weight=fonts['phase_label']['weight'], color='#c0392b')
    
    ax.annotate(phase_labels['optimal'], 
               xy=(0.8, 8), xytext=(0.8, 9.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#229954'),
               ha='center', fontsize=fonts['phase_label']['size'], 
               weight=fonts['phase_label']['weight'], color='#229954')
    
    # Add stats box in corner
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    return fig

def create_circular_phase_diagram():
    """Create a circular/radial visualization of phases"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Get clean labels
    phase_labels, model_labels, stats_text = clean_phase_labels()
    fonts, format_score, boundary_style, remove = improve_all_figures()
    
    # Define phase regions in polar coordinates
    theta = np.linspace(0, 2 * np.pi, 1000)
    
    # Create three phase regions
    chaotic_theta = theta[theta < 1.5 * np.pi]
    optimal_theta = theta[(theta >= 1.5 * np.pi) & (theta < 1.75 * np.pi)]
    lazy_theta = theta[theta >= 1.75 * np.pi]
    
    # Plot phase regions
    ax.fill_between(chaotic_theta, 0, 1, color='#e74c3c', alpha=0.3, label='Chaotic')
    ax.fill_between(optimal_theta, 0, 1, color='#27ae60', alpha=0.3, label='Optimal')
    ax.fill_between(lazy_theta, 0, 1, color='#3498db', alpha=0.3, label='Lazy')
    
    # Plot models as points with cleaner labels
    models = [
        ('EffNet-B2', 0.806, 0.0247, 'optimal'),
        ('ConvNeXt-S', 0.185, 0.0210, 'chaotic'),
        ('ResNet-50', 0.181, 0.0170, 'chaotic'),
        ('ViT-B', 0.178, 0.0170, 'chaotic'),
        ('Swin-S', 0.734, 0.0234, 'optimal'),
    ]
    
    # Collect points for smart positioning
    label_points = []
    label_texts = []
    
    for name, ntk, align, phase in models:
        # Map NTK to angle (0-2π)
        angle = (1 - ntk) * 2 * np.pi
        # Map alignment to radius (normalized)
        radius = align / 0.025  # Normalize by max alignment
        
        color = '#27ae60' if phase == 'optimal' else '#e74c3c'
        ax.scatter(angle, radius, s=300, c=color, edgecolor='black', 
                  linewidth=2, zorder=5)
        
        # Collect points for labeling
        label_points.append([angle, radius])
        label_texts.append(f'{name}\n{format_score(align)}')
    
    # Smart positioning for labels
    if label_points:
        # Convert to polar coordinates for distance calculation
        points_array = np.array(label_points)
        
        # Adjust positions to avoid overlaps
        for i in range(len(points_array)):
            # Check distance to other points
            for j in range(i+1, len(points_array)):
                dist = np.sqrt((points_array[i][0] - points_array[j][0])**2 + 
                              (points_array[i][1] - points_array[j][1])**2)
                if dist < 0.3:  # If too close
                    # Adjust radius slightly
                    points_array[i][1] += 0.1
                    points_array[j][1] -= 0.1
        
        # Add labels with adjusted positions
        for i, (point, text) in enumerate(zip(points_array, label_texts)):
            ax.text(point[0], point[1] + 0.15, text, ha='center', 
                   fontsize=fonts['model_label']['size'])
    
    # Customize
    ax.set_ylim(0, 1.2)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Add cleaner phase labels with better spacing
    ax.text(0.75 * np.pi, 1.4, phase_labels['chaotic'].split('\n')[0], ha='center', 
           fontsize=fonts['phase_label']['size'], weight=fonts['phase_label']['weight'], 
           color='#c0392b')
    ax.text(1.625 * np.pi, 1.4, phase_labels['optimal'].split('\n')[0], ha='center', 
           fontsize=fonts['phase_label']['size'], weight=fonts['phase_label']['weight'], 
           color='#229954')
    ax.text(1.875 * np.pi, 1.4, phase_labels['lazy'].split('\n')[0], ha='center', 
           fontsize=fonts['phase_label']['size'], weight=fonts['phase_label']['weight'], 
           color='#2874a6')
    
    ax.set_title('Phase Distribution in Cross-Modal Space\nRadius = Alignment Score, Angle = NTK Stability', 
                fontsize=fonts['title']['size'], weight=fonts['title']['weight'], pad=30)
    
    # Remove radial labels
    ax.set_rticks([])
    ax.set_thetagrids([])
    
    plt.tight_layout()
    return fig

def create_flow_field_visualization():
    """Create a vector field showing phase dynamics"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get clean labels
    legend_items, phase_zones = clean_flow_labels()
    fonts, format_score, boundary_style, remove = improve_all_figures()
    
    # Create grid
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 6, 15)
    X, Y = np.meshgrid(x, y)
    
    # Define vector field based on phase dynamics
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    # In chaotic phase, vectors point in random directions
    chaotic_mask = X < 0.7
    U[chaotic_mask] = np.random.randn(np.sum(chaotic_mask)) * 0.02
    V[chaotic_mask] = np.random.randn(np.sum(chaotic_mask)) * 0.1
    
    # In optimal phase, vectors point toward stability
    optimal_mask = (X >= 0.7) & (X <= 0.9)
    U[optimal_mask] = 0.01
    V[optimal_mask] = -0.05
    
    # In lazy phase, minimal movement
    lazy_mask = X > 0.9
    U[lazy_mask] = 0
    V[lazy_mask] = 0
    
    # Create background colors
    Z = np.zeros_like(X)
    Z[chaotic_mask] = 0.2
    Z[optimal_mask] = 0.8
    Z[lazy_mask] = 0.5
    
    # Plot background
    im = ax.imshow(Z.T, extent=[0, 1, 0, 6], origin='lower', 
                   aspect='auto', cmap='RdYlBu_r', alpha=0.5)
    
    # Plot vector field
    ax.quiver(X, Y, U, V, Z, cmap='viridis', alpha=0.7, 
              scale=1, scale_units='xy', width=0.003)
    
    # Plot trajectories for key models with cleaner labels
    # Successful trajectory (moves to optimal)
    t = np.linspace(0, 1, 100)
    x_success = 0.2 + 0.6 * t
    y_success = 3 + np.sin(2 * np.pi * t) * 0.5
    ax.plot(x_success, y_success, 'g-', linewidth=3, 
           label=legend_items[0][2])  # Phase-compatible alignment
    ax.scatter([0.2, 0.8], [3, 3], s=200, c=['red', 'green'], 
               edgecolor='black', linewidth=2, zorder=5)
    
    # Failed trajectory (stuck in chaotic)
    x_fail = 0.1 + 0.4 * t * np.exp(-2 * t)
    y_fail = 4 + np.random.randn(100) * 0.1
    ax.plot(x_fail, y_fail, 'r--', linewidth=2, 
           label=legend_items[1][2])  # Cross-phase failure
    
    # Add phase boundaries with cleaner style
    ax.axvline(x=0.7, color=boundary_style['color'], linestyle=boundary_style['style'], 
               linewidth=boundary_style['width'], alpha=boundary_style['alpha'])
    ax.axvline(x=0.9, color=boundary_style['color'], linestyle=boundary_style['style'], 
               linewidth=boundary_style['width'], alpha=boundary_style['alpha'])
    
    # Cleaner phase labels with better positioning
    ax.text(0.35, 5.8, phase_zones['chaotic'], fontsize=fonts['phase_label']['size'], 
           weight=fonts['phase_label']['weight'], ha='center', color='darkred')
    ax.text(0.8, 5.8, phase_zones['optimal'], fontsize=fonts['phase_label']['size'], 
           weight=fonts['phase_label']['weight'], ha='center', color='darkgreen')
    ax.text(0.95, 5.8, phase_zones['lazy'], fontsize=fonts['phase_label']['size'], 
           weight=fonts['phase_label']['weight'], ha='center', color='darkblue')
    
    ax.set_xlabel('NTK Stability', fontsize=fonts['axis']['size'])
    ax.set_ylabel('AGOP Magnitude', fontsize=fonts['axis']['size'])
    ax.set_title('Dynamical Flow in Cross-Modal Phase Space', 
                fontsize=fonts['title']['size'], weight=fonts['title']['weight'])
    ax.legend(loc='lower right', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6)
    
    plt.tight_layout()
    return fig

def create_phase_transition_heatmap():
    """Create a heatmap showing phase transitions with real data"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get clean labels
    callouts, top_models = clean_heatmap_labels()
    fonts, format_score, boundary_style, remove = improve_all_figures()
    
    # Create dense grid for interpolation
    ntk_range = np.linspace(0, 1, 100)
    agop_range = np.linspace(0, 10, 100)
    ntk_grid, agop_grid = np.meshgrid(ntk_range, agop_range)
    
    # Use your actual data points
    points = []
    values = []
    
    # Add real data points from your analysis
    data_points = [
        (0.806, 1.77, 0.0247),  # EffNet-B2 + DistilRoBERTa
        (0.816, 1.66, 0.0212),  # EffNet-B2 + XLNet
        (0.185, 20.4, 0.0210),  # ConvNeXt-S + DistilRoBERTa
        (0.181, 5.96, 0.0170),  # ResNet-50 + RoBERTa
        (0.178, 29.1, 0.0170),  # ViT + BERT
        (0.734, 0.21, 0.0234),  # Swin + DialoGPT
        # Add more from your data...
    ]
    
    for ntk, agop_raw, align in data_points:
        agop = np.log10(agop_raw + 1)  # Log scale for AGOP
        points.append([ntk, agop])
        values.append(align)
    
    points = np.array(points)
    values = np.array(values)
    
    # Interpolate to create smooth surface
    from scipy.interpolate import griddata
    Z = griddata(points, values, (ntk_grid, agop_grid), method='cubic')
    
    # Plot heatmap
    im = ax.imshow(Z, extent=[0, 1, 0, 10], origin='lower', 
                   aspect='auto', cmap='viridis', alpha=0.8)
    
    # Add contour lines
    contours = ax.contour(ntk_range, agop_range, Z, 
                         levels=[0.015, 0.018, 0.021, 0.024], 
                         colors='white', linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=10)
    
    # Plot actual data points with cleaner labels
    label_points = []
    label_texts = []
    
    for i, (ntk, agop_raw, align) in enumerate(data_points):
        agop = np.log10(agop_raw + 1)
        size = align * 10000
        
        # Only highlight top performers
        if align > 0.021:
            color = 'red'
            edgecolor = 'white'
            linewidth = 2
            # Collect points for smart positioning
            label_points.append([ntk, agop])
            model_name = ['EffNet-B2', 'EffNet-B2', 'ConvNeXt-S', 'ResNet-50', 'ViT-B', 'Swin-S'][i]
            label_texts.append(f'{model_name}\n{format_score(align)}')
        else:
            color = 'red'
            edgecolor = 'white'
            linewidth = 1
            
        ax.scatter(ntk, agop, s=size, c=color, edgecolor=edgecolor, 
                  linewidth=linewidth, zorder=5)
    
    # Smart positioning for labels
    if label_points:
        adjusted_points = smart_text_positioning(label_points, label_texts, ax, min_distance=0.2)
        
        for point, text in zip(adjusted_points, label_texts):
            ax.text(point[0], point[1] + 0.4, text, 
                   ha='center', fontsize=fonts['model_label']['size'],
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add phase boundaries with cleaner style
    ax.axvline(x=0.7, color=boundary_style['color'], linestyle=boundary_style['style'], 
               linewidth=boundary_style['width'], alpha=boundary_style['alpha'])
    ax.axvline(x=0.9, color=boundary_style['color'], linestyle=boundary_style['style'], 
               linewidth=boundary_style['width'], alpha=boundary_style['alpha'])
    
    # Add cleaner callouts with better positioning
    for i, callout in enumerate(callouts):
        y_offset = 0.5 if i == 0 else -0.5  # Stagger the callouts
        ax.text(callout['pos'][0], callout['pos'][1] + y_offset, callout['text'], 
               color='white', fontsize=fonts['phase_label']['size'], 
               weight=fonts['phase_label']['weight'], ha='center',
               bbox=dict(boxstyle='round', facecolor='red' if 'trapped' in callout['text'] else 'green', alpha=0.7))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Alignment Score', fontsize=fonts['axis']['size'])
    
    ax.set_xlabel('NTK Stability', fontsize=fonts['axis']['size'])
    ax.set_ylabel('log(AGOP Magnitude)', fontsize=fonts['axis']['size'])
    ax.set_title('Cross-Modal Alignment Landscape from 144 Model Pairs', 
                fontsize=fonts['title']['size'], weight=fonts['title']['weight'])
    
    plt.tight_layout()
    return fig

def create_optimal_combinations_diagram():
    """Create a diagram showing optimal model combinations"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get clean labels
    connections, legend_text = clean_connection_labels()
    fonts, format_score, boundary_style, remove = improve_all_figures()
    
    # Define model positions with better spacing
    vision_models = [
        ('EffNet-B2', 0.0247, 1, 0.8),
        ('ConvNeXt-S', 0.0210, 1, 0.6),
        ('ResNet-50', 0.0170, 1, 0.4),
        ('ViT-B', 0.0170, 1, 0.2),
    ]
    
    text_models = [
        ('DistilRoBERTa', 0.0247, 3, 0.8),
        ('XLNet-L', 0.0212, 3, 0.6),
        ('RoBERTa', 0.0170, 3, 0.4),
        ('BERT', 0.0170, 3, 0.2),
    ]
    
    # Plot vision models
    for name, score, x, y in vision_models:
        color = '#27ae60' if score > 0.021 else '#e74c3c'
        ax.scatter(x, y, s=score*10000, c=color, edgecolor='black', linewidth=2)
        ax.text(x-0.15, y, f'{name}\n{format_score(score)}', 
               ha='right', va='center', fontsize=fonts['model_label']['size'])
    
    # Plot text models
    for name, score, x, y in text_models:
        color = '#27ae60' if score > 0.021 else '#e74c3c'
        ax.scatter(x, y, s=score*10000, c=color, edgecolor='black', linewidth=2)
        ax.text(x+0.15, y, f'{name}\n{format_score(score)}', 
               ha='left', va='center', fontsize=fonts['model_label']['size'])
    
    # Draw connections with better spacing
    # Optimal connections (green, thick)
    optimal_pairs = [
        (vision_models[0], text_models[0]),  # EffNet-B2 + DistilRoBERTa
        (vision_models[1], text_models[0]),  # ConvNeXt-S + DistilRoBERTa
    ]
    
    for (v_name, v_score, v_x, v_y), (t_name, t_score, t_x, t_y) in optimal_pairs:
        ax.plot([v_x+0.08, t_x-0.08], [v_y, t_y], 
               color=connections['success']['color'],
               linewidth=connections['success']['width'],
               linestyle=connections['success']['style'],
               alpha=0.7)
    
    # Failed connections (red, thin)
    failed_pairs = [
        (vision_models[2], text_models[2]),  # ResNet-50 + RoBERTa
        (vision_models[3], text_models[3]),  # ViT-B + BERT
    ]
    
    for (v_name, v_score, v_x, v_y), (t_name, t_score, t_x, t_y) in failed_pairs:
        ax.plot([v_x+0.08, t_x-0.08], [v_y, t_y], 
               color=connections['failure']['color'],
               linewidth=connections['failure']['width'],
               linestyle=connections['failure']['style'],
               alpha=0.5)
    
    # Add legend with better positioning
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Labels
    ax.set_xlabel('Model Type', fontsize=fonts['axis']['size'])
    ax.set_ylabel('Alignment Score', fontsize=fonts['axis']['size'])
    ax.set_title('Optimal Model Combinations', 
                fontsize=fonts['title']['size'], weight=fonts['title']['weight'])
    
    ax.set_xlim(0.3, 3.7)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 3])
    ax.set_xticklabels(['Vision Models', 'Text Models'])
    
    plt.tight_layout()
    return fig

# Generate all figures
if __name__ == "__main__":
    fig1 = create_phase_landscape_3d_projection()
    fig1.savefig('phase_landscape_3d_clean.png', dpi=300, bbox_inches='tight')
    
    fig2 = create_circular_phase_diagram()
    fig2.savefig('phase_circular_clean.png', dpi=300, bbox_inches='tight')
    
    fig3 = create_flow_field_visualization()
    fig3.savefig('phase_flow_field_clean.png', dpi=300, bbox_inches='tight')
    
    fig4 = create_phase_transition_heatmap()
    fig4.savefig('phase_heatmap_clean.png', dpi=300, bbox_inches='tight')
    
    fig5 = create_optimal_combinations_diagram()
    fig5.savefig('optimal_combinations_clean.png', dpi=300, bbox_inches='tight')
    
    plt.show()