#!/usr/bin/env python3
"""
Create a focused summary for specific model pairs with I_X_T and I_Y_T values
"""

import pandas as pd
import json

def create_focused_mutual_information_summary():
    """Create a focused summary for the specific model pairs mentioned"""
    
    # Load the CSV data
    df = pd.read_csv('mutual_information_results.csv')
    
    print("="*80)
    print("FOCUSED SUMMARY: I_X_T AND I_Y_T VALUES")
    print("="*80)
    
    # Your specific representative pairs
    target_models = [
        "resnet50",      # ResNet-BERT
        "vit_base_patch16_224",  # ViT-GPT2  
        "convnext_small"  # ConvNeXt-RoBERTa
    ]
    
    print("\n🎯 YOUR SPECIFIC REPRESENTATIVE MODELS:")
    print("-" * 50)
    
    found_models = []
    for model in target_models:
        # Look for exact matches or close matches
        matches = df[df['model'].str.contains(model.split('_')[0], case=False)]
        
        if len(matches) > 0:
            # Take the first match
            match = matches.iloc[0]
            found_models.append({
                'model': str(match['model']),
                'dataset': str(match['dataset']),
                'layer': str(match['layer']),
                'layer_idx': int(match['layer_idx']),
                'I_X_T': float(match['I_X_T']),
                'I_Y_T': float(match['I_Y_T']),
                'efficiency': float(match['info_efficiency'])
            })
            
            print(f"✅ {match['model']} ({match['dataset']}):")
            print(f"   I_X_T: {match['I_X_T']:.6f}")
            print(f"   I_Y_T: {match['I_Y_T']:.6f}")
            print(f"   Efficiency: {match['info_efficiency']:.6f}")
            print(f"   Layer: {match['layer']} (idx: {match['layer_idx']})")
            print()
        else:
            print(f"❌ {model}: Not found in current data")
            print()
    
    # Show best performers by I_X_T
    print("\n🏆 TOP PERFORMERS BY I_X_T:")
    print("-" * 50)
    
    top_i_xt = df.nlargest(5, 'I_X_T')
    for i, (_, row) in enumerate(top_i_xt.iterrows(), 1):
        print(f"{i}. {row['model']} (Layer {row['layer_idx']}):")
        print(f"   I_X_T: {row['I_X_T']:.6f}")
        print(f"   I_Y_T: {row['I_Y_T']:.6f}")
        print(f"   Efficiency: {row['info_efficiency']:.6f}")
        print()
    
    # Show best performers by efficiency
    print("\n🎯 TOP PERFORMERS BY EFFICIENCY:")
    print("-" * 50)
    
    top_efficiency = df.nlargest(5, 'info_efficiency')
    for i, (_, row) in enumerate(top_efficiency.iterrows(), 1):
        print(f"{i}. {row['model']} (Layer {row['layer_idx']}):")
        print(f"   Efficiency: {row['info_efficiency']:.6f}")
        print(f"   I_X_T: {row['I_X_T']:.6f}")
        print(f"   I_Y_T: {row['I_Y_T']:.6f}")
        print()
    
    # Model comparison
    print("\n📊 MODEL COMPARISON:")
    print("-" * 50)
    
    model_stats = df.groupby('model').agg({
        'I_X_T': ['mean', 'std', 'min', 'max'],
        'I_Y_T': ['mean', 'std', 'min', 'max'],
        'info_efficiency': ['mean', 'std', 'min', 'max']
    }).round(6)
    
    print("\nPer-Model Statistics:")
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"\n{model}:")
        print(f"  I_X_T: {model_data['I_X_T'].mean():.6f} ± {model_data['I_X_T'].std():.6f}")
        print(f"  I_Y_T: {model_data['I_Y_T'].mean():.6f} ± {model_data['I_Y_T'].std():.6f}")
        print(f"  Efficiency: {model_data['info_efficiency'].mean():.6f} ± {model_data['info_efficiency'].std():.6f}")
        print(f"  Layers: {len(model_data)}")
    
    # Overall statistics
    print("\n📈 OVERALL STATISTICS:")
    print("-" * 50)
    
    print(f"Total Results: {len(df)}")
    print(f"I_X_T Range: {df['I_X_T'].min():.6f} - {df['I_X_T'].max():.6f}")
    print(f"I_X_T Mean: {df['I_X_T'].mean():.6f}")
    print(f"I_Y_T Range: {df['I_Y_T'].min():.6f} - {df['I_Y_T'].max():.6f}")
    print(f"I_Y_T Mean: {df['I_Y_T'].mean():.6f}")
    print(f"Efficiency Range: {df['info_efficiency'].min():.6f} - {df['info_efficiency'].max():.6f}")
    print(f"Efficiency Mean: {df['info_efficiency'].mean():.6f}")
    
    # Save focused results
    if found_models:
        with open('focused_mutual_information.json', 'w') as f:
            json.dump(found_models, f, indent=2)
        print(f"\n💾 Focused results saved to: focused_mutual_information.json")
    
    # Create a simple table
    print("\n📋 SUMMARY TABLE:")
    print("-" * 80)
    print(f"{'Model':<30} {'Dataset':<10} {'Layer':<10} {'I_X_T':<10} {'I_Y_T':<10} {'Efficiency':<12}")
    print("-" * 80)
    
    for model in found_models:
        print(f"{model['model']:<30} {model['dataset']:<10} {model['layer']:<10} {model['I_X_T']:<10.6f} {model['I_Y_T']:<10.6f} {model['efficiency']:<12.6f}")

if __name__ == "__main__":
    create_focused_mutual_information_summary()
