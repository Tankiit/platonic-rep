#!/usr/bin/env python3
"""
Create a focused summary for specific model pairs
"""

import pandas as pd
import json

def create_focused_summary():
    """Create a focused summary for the specific model pairs mentioned"""
    
    # Load the CSV data
    df = pd.read_csv('all_analysis_results.csv')
    
    # Filter for cross-modal pairs only
    cross_modal_df = df[df['type'] == 'cross_modal'].copy()
    
    print("="*80)
    print("FOCUSED SUMMARY: NTK, AGOP, AND ALIGNMENT VALUES")
    print("="*80)
    
    # Your specific representative pairs
    target_pairs = [
        ("resnet50", "bert_base"),      # ResNet-BERT
        ("vit_base_patch16_224", "gpt2_medium"),  # ViT-GPT2  
        ("convnext_small", "roberta_base")  # ConvNeXt-RoBERTa
    ]
    
    print("\n🎯 YOUR SPECIFIC REPRESENTATIVE PAIRS:")
    print("-" * 50)
    
    found_pairs = []
    for v_model, t_model in target_pairs:
        # Look for exact matches or close matches
        matches = cross_modal_df[
            (cross_modal_df['v_model'].str.contains(v_model.split('_')[0], case=False)) &
            (cross_modal_df['t_model'].str.contains(t_model.split('_')[0], case=False))
        ]
        
        if len(matches) > 0:
            # Take the first match
            match = matches.iloc[0]
            found_pairs.append({
                'pair': f"{match['v_model']} + {match['t_model']}",
                'ntk': match['ntk_stability'],
                'agop': match['agop_magnitude'],
                'alignment': match['alignment'],
                'phase': match['phase_region']
            })
            
            print(f"✅ {match['v_model']} + {match['t_model']}:")
            print(f"   NTK Stability: {match['ntk_stability']:.4f}")
            print(f"   AGOP Magnitude: {match['agop_magnitude']:.2e}")
            print(f"   Alignment: {match['alignment']:.4f}")
            print(f"   Phase: {match['phase_region']}")
            print()
        else:
            print(f"❌ {v_model} + {t_model}: Not found in current data")
            print()
    
    # Show best performers
    print("\n🏆 TOP PERFORMERS BY ALIGNMENT:")
    print("-" * 50)
    
    top_alignment = cross_modal_df.nlargest(5, 'alignment')
    for i, (_, row) in enumerate(top_alignment.iterrows(), 1):
        print(f"{i}. {row['v_model']} + {row['t_model']}:")
        print(f"   Alignment: {row['alignment']:.4f}")
        print(f"   NTK: {row['ntk_stability']:.4f}")
        print(f"   AGOP: {row['agop_magnitude']:.2e}")
        print(f"   Phase: {row['phase_region']}")
        print()
    
    # Show best NTK stability
    print("\n🎯 TOP PERFORMERS BY NTK STABILITY:")
    print("-" * 50)
    
    top_ntk = cross_modal_df.nlargest(5, 'ntk_stability')
    for i, (_, row) in enumerate(top_ntk.iterrows(), 1):
        print(f"{i}. {row['v_model']} + {row['t_model']}:")
        print(f"   NTK: {row['ntk_stability']:.4f}")
        print(f"   Alignment: {row['alignment']:.4f}")
        print(f"   AGOP: {row['agop_magnitude']:.2e}")
        print(f"   Phase: {row['phase_region']}")
        print()
    
    # Phase distribution
    print("\n📊 PHASE DISTRIBUTION:")
    print("-" * 50)
    
    phase_counts = cross_modal_df['phase_region'].value_counts()
    total = len(cross_modal_df)
    
    for phase, count in phase_counts.items():
        percentage = (count / total) * 100
        print(f"{phase.capitalize()}: {count} pairs ({percentage:.1f}%)")
    
    # Overall statistics
    print("\n📈 OVERALL STATISTICS:")
    print("-" * 50)
    
    print(f"Total Cross-Modal Pairs: {len(cross_modal_df)}")
    print(f"NTK Range: {cross_modal_df['ntk_stability'].min():.4f} - {cross_modal_df['ntk_stability'].max():.4f}")
    print(f"NTK Mean: {cross_modal_df['ntk_stability'].mean():.4f}")
    print(f"AGOP Range: {cross_modal_df['agop_magnitude'].min():.2e} - {cross_modal_df['agop_magnitude'].max():.2e}")
    print(f"Alignment Range: {cross_modal_df['alignment'].min():.4f} - {cross_modal_df['alignment'].max():.4f}")
    print(f"Alignment Mean: {cross_modal_df['alignment'].mean():.4f}")
    
    # Save focused results
    if found_pairs:
        with open('focused_model_pairs.json', 'w') as f:
            json.dump(found_pairs, f, indent=2)
        print(f"\n💾 Focused results saved to: focused_model_pairs.json")
    
    # Create a simple table
    print("\n📋 SUMMARY TABLE:")
    print("-" * 80)
    print(f"{'Model Pair':<40} {'NTK':<8} {'AGOP':<12} {'Align':<8} {'Phase':<10}")
    print("-" * 80)
    
    for pair in found_pairs:
        print(f"{pair['pair']:<40} {pair['ntk']:<8.4f} {pair['agop']:<12.2e} {pair['alignment']:<8.4f} {pair['phase']:<10}")

if __name__ == "__main__":
    create_focused_summary()
