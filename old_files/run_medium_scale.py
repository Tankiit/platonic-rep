#!/usr/bin/env python3
"""
Medium-Scale Multiscale Analysis Runner
Runs analysis on a subset of medium-sized models for faster experiments
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
import time
from multi_scale import MultiscaleInformationAnalysis
from microscopic_analysis import microscopic_analysis
import torch

def run_medium_scale():
    """Run medium-scale multiscale analysis with selected models"""
    
    print("=" * 80)
    print("MEDIUM-SCALE MULTISCALE ANALYSIS PIPELINE")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"./results/medium_scale_{timestamp}/"
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Medium-scale vision models (reduced set)
    vision_models = [
        # Core ResNet
        'resnet34',
        
        # Core Vision Transformer
        'vit_small_patch16_224',
        
        # Core ConvNeXt
        'convnext_small',
        
        # Core EfficientNet
        'efficientnet_b1',
        
        # Core Swin
        'swin_small_patch4_window7_224'
    ]
    
    # Medium-scale language models (reduced set)
    language_models = [
        # Core BERT family
        'bert_base',
        'roberta_base',
        
        # Core GPT family
        'gpt2_medium',
        
        # Core alternatives
        'distilbert_base',
        'albert_base_v2'
    ]
    
    # Medium-scale speech models (reduced set)
    speech_models = [
        'wav2vec2_base',
        'hubert_base',
        'whisper_small'
    ]
    
    # Configuration for feature extraction
    feature_config = {
        'vision': {
            'dataset': 'CIFAR10',
            'models': vision_models,
            'output_dir': f"{base_output_dir}/features/vision/"
        },
        'language': {
            'dataset': 'WIKITEXT',
            'models': language_models,
            'output_dir': f"{base_output_dir}/features/language/"
        },
        'speech': {
            'dataset': 'LIBRISPEECH',
            'models': speech_models,
            'output_dir': f"{base_output_dir}/features/speech/"
        }
    }
    
    # Step 1: Feature Extraction
    print("\n" + "=" * 60)
    print("STEP 1: FEATURE EXTRACTION (MEDIUM-SCALE)")
    print("=" * 60)
    
    for modality, config in feature_config.items():
        dataset = config['dataset']
        models = config['models']
        output_dir = config['output_dir']
        
        print(f"\nExtracting features for {dataset}...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if modality == 'vision':
            print(f"  Extracting vision model features for {dataset.lower()}...")
            for model in models:
                print(f"    Extracting {model} features...")
                extract_vision_features(model, dataset.lower(), output_dir)
                print(f"      ✓ {model} feature extraction complete")
                
        elif modality == 'language':
            print(f"  Extracting language model features for {dataset.lower()}...")
            for model in models:
                print(f"    Extracting {model} features...")
                extract_language_features(model, dataset.lower(), output_dir)
                print(f"      ✓ {model} feature extraction complete")
                
        elif modality == 'speech':
            print(f"  Extracting speech model features for {dataset.lower()}...")
            for model in models:
                print(f"    Extracting {model} features...")
                extract_speech_features(model, dataset.lower(), output_dir)
                print(f"      ✓ {model} feature extraction complete")
    
    # Step 2: Multiscale Analysis
    print("\n" + "=" * 60)
    print("STEP 2: MULTISCALE ANALYSIS (MEDIUM-SCALE)")
    print("=" * 60)
    
    for modality, config in feature_config.items():
        dataset = config['dataset']
        feature_dir = config['output_dir']
        
        print(f"\nRunning multiscale analysis for {dataset}...")
        
        if modality == 'vision':
            print(f"  Analyzing vision models...")
            analyzer = MultiscaleInformationAnalysis(feature_dir=feature_dir)
            analyzer.output_dir = Path(f"{base_output_dir}/multiscale_analysis/vision/")
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
            analyzer.analyze_all_models()
            
        elif modality == 'language':
            print(f"  Analyzing language models...")
            analyzer = MultiscaleInformationAnalysis(feature_dir=feature_dir)
            analyzer.output_dir = Path(f"{base_output_dir}/multiscale_analysis/language/")
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
            analyzer.analyze_all_models()
            
        elif modality == 'speech':
            print(f"  Analyzing speech models...")
            analyzer = MultiscaleInformationAnalysis(feature_dir=feature_dir)
            analyzer.output_dir = Path(f"{base_output_dir}/multiscale_analysis/speech/")
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
            analyzer.analyze_all_models()
    
    # Step 3: Microscopic Analysis
    print("\n" + "=" * 60)
    print("STEP 3: MICROSCOPIC ANALYSIS (MEDIUM-SCALE)")
    print("=" * 60)
    
    for modality, config in feature_config.items():
        print(f"\nRunning microscopic analysis for {modality}...")
        feature_dir = config['output_dir']
        output_dir = f"{base_output_dir}/microscopic_analysis/{modality}/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        microscopic_analysis(
            feature_dir=feature_dir,
            output_dir=output_dir,
            modality=modality
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("MEDIUM-SCALE ANALYSIS COMPLETE!")
    print(f"Results saved to: {base_output_dir}")
    print("=" * 80)
    
    # Generate summary report
    generate_summary_report(base_output_dir)

def extract_vision_features(model_name, dataset, output_dir):
    """Extract features for vision models"""
    cmd = [
        "python", "extract_features.py",
        "--model", model_name,
        "--dataset", dataset,
        "--output_dir", output_dir,
        "--modality", "vision"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"      Warning: Feature extraction failed for {model_name}")
        print(f"      Error: {e.stderr}")

def extract_language_features(model_name, dataset, output_dir):
    """Extract features for language models"""
    cmd = [
        "python", "extract_features.py",
        "--model", model_name,
        "--dataset", dataset,
        "--output_dir", output_dir,
        "--modality", "language"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"      Warning: Feature extraction failed for {model_name}")
        print(f"      Error: {e.stderr}")

def extract_speech_features(model_name, dataset, output_dir):
    """Extract features for speech models"""
    cmd = [
        "python", "extract_features.py",
        "--model", model_name,
        "--dataset", dataset,
        "--output_dir", output_dir,
        "--modality", "speech"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"      Warning: Feature extraction failed for {model_name}")
        print(f"      Error: {e.stderr}")

def generate_summary_report(output_dir):
    """Generate a summary report of the analysis"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'output_directory': output_dir,
        'analysis_type': 'medium-scale',
        'modalities': ['vision', 'language', 'speech'],
        'model_counts': {
            'vision': 5,
            'language': 5,
            'speech': 3
        }
    }
    
    summary_path = Path(output_dir) / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary report saved to: {summary_path}")

if __name__ == "__main__":
    run_medium_scale()