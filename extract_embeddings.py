#!/usr/bin/env python3
"""
Comprehensive Embedding Extraction Script for MLX Models
Extracts embeddings from small and medium datasets using various MLX models
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as load_lm
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import time
import psutil
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
import requests
from PIL import Image
import io
import h5py
import logging
from tqdm import tqdm

# Import from our MLX verifier
from mlx_verifier import (
    MLXCrossModalVerifier, 
    ModelConfig, 
    DatasetConfig, 
    RECOMMENDED_MODELS, 
    RECOMMENDED_DATASETS,
    DATASETS_AVAILABLE,
    VLM_AVAILABLE
)

# Try to import MLX-VLM for vision-language models
try:
    from mlx_vlm import load as load_vlm, generate
    VLM_AVAILABLE = True
except ImportError:
    print("Warning: mlx-vlm not installed. Install with: pip install mlx-vlm")
    VLM_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    """
    Comprehensive embedding extraction for various MLX models and datasets
    """
    
    def __init__(self, memory_limit_gb: Optional[float] = None, output_dir: str = "representations"):
        self.verifier = MLXCrossModalVerifier(memory_limit_gb)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different model categories
        self.model_dirs = {
            'small': self.output_dir / "small_models",
            'medium': self.output_dir / "medium_models"
        }
        
        for model_dir in self.model_dirs.values():
            model_dir.mkdir(exist_ok=True)
        
        logger.info(f"EmbeddingExtractor initialized. Output directory: {self.output_dir}")
        logger.info(f"System memory: {self.verifier.memory_monitor.total_memory:.1f}GB")
        logger.info(f"Available memory: {self.verifier.memory_monitor.get_available_memory_gb():.1f}GB")

    def extract_vision_embeddings(self, model_info: Dict, images: List[Image.Image]) -> np.ndarray:
        """Extract vision embeddings from images using a vision model"""
        model = model_info['model']
        processor = model_info['processor']
        
        embeddings = []
        batch_size = min(4, model_info['config'].max_batch_size)
        
        logger.info(f"Extracting vision embeddings from {len(images)} images with batch size {batch_size}")
        
        for i in tqdm(range(0, len(images), batch_size), desc="Processing images"):
            batch_images = images[i:i+batch_size]
            
            try:
                with mx.no_grad():
                    # Process images through the model
                    if hasattr(processor, 'preprocess'):
                        # For CLIP-like models
                        inputs = processor.preprocess(batch_images)
                    else:
                        # For other vision models
                        inputs = processor(images=batch_images, return_tensors="np")
                    
                    # Extract features
                    if hasattr(model, 'encode_image'):
                        features = model.encode_image(inputs)
                    elif hasattr(model, 'get_image_features'):
                        features = model.get_image_features(inputs)
                    else:
                        # Generic feature extraction
                        outputs = model(**inputs, output_hidden_states=True)
                        if hasattr(outputs, 'last_hidden_state'):
                            features = outputs.last_hidden_state.mean(axis=1)  # Pool over sequence
                        else:
                            features = outputs.hidden_states[-1].mean(axis=1)
                    
                    # Convert to numpy
                    if hasattr(features, 'numpy'):
                        features = features.numpy()
                    else:
                        features = np.array(features)
                    
                    embeddings.append(features)
                    
            except Exception as e:
                logger.warning(f"Failed to process batch {i//batch_size}: {e}")
                # Add zero embeddings as fallback
                dummy_features = np.zeros((len(batch_images), 512))  # Default dimension
                embeddings.append(dummy_features)
            
            # Clear cache
            mx.metal.clear_cache()
        
        return np.vstack(embeddings) if embeddings else np.array([])

    def extract_text_embeddings(self, model_info: Dict, texts: List[str]) -> np.ndarray:
        """Extract text embeddings from texts using a language model"""
        model = model_info['model']
        tokenizer = model_info['processor']
        
        embeddings = []
        batch_size = min(8, model_info['config'].max_batch_size)
        
        logger.info(f"Extracting text embeddings from {len(texts)} texts with batch size {batch_size}")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
            batch_texts = texts[i:i+batch_size]
            
            try:
                with mx.no_grad():
                    # Tokenize texts
                    inputs = tokenizer(batch_texts, return_tensors="np", padding=True, truncation=True)
                    
                    # Extract features
                    if hasattr(model, 'encode_text'):
                        features = model.encode_text(inputs)
                    elif hasattr(model, 'get_text_features'):
                        features = model.get_text_features(inputs)
                    else:
                        # Generic feature extraction
                        outputs = model(**inputs, output_hidden_states=True)
                        if hasattr(outputs, 'last_hidden_state'):
                            # Use [CLS] token or mean pooling
                            features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                        else:
                            features = outputs.hidden_states[-1].mean(axis=1)  # Mean pooling
                    
                    # Convert to numpy
                    if hasattr(features, 'numpy'):
                        features = features.numpy()
                    else:
                        features = np.array(features)
                    
                    embeddings.append(features)
                    
            except Exception as e:
                logger.warning(f"Failed to process batch {i//batch_size}: {e}")
                # Add zero embeddings as fallback
                dummy_features = np.zeros((len(batch_texts), 512))  # Default dimension
                embeddings.append(dummy_features)
            
            # Clear cache
            mx.metal.clear_cache()
        
        return np.vstack(embeddings) if embeddings else np.array([])

    def extract_multimodal_embeddings(self, model_info: Dict, images: List[Image.Image], texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both vision and text embeddings from a multimodal model"""
        model = model_info['model']
        processor = model_info['processor']
        
        vision_embeddings = []
        text_embeddings = []
        batch_size = min(2, model_info['config'].max_batch_size)  # Smaller batch for multimodal
        
        logger.info(f"Extracting multimodal embeddings from {len(images)} image-text pairs with batch size {batch_size}")
        
        for i in tqdm(range(0, len(images), batch_size), desc="Processing multimodal"):
            batch_images = images[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            try:
                with mx.no_grad():
                    # Process multimodal inputs
                    inputs = processor(images=batch_images, text=batch_texts, return_tensors="np")
                    
                    # Extract features
                    outputs = model(**inputs, output_hidden_states=True)
                    
                    # Extract vision features (first part of sequence)
                    if hasattr(outputs, 'vision_hidden_states'):
                        v_features = outputs.vision_hidden_states[-1].mean(axis=1)
                    else:
                        # Use first tokens as vision proxy
                        v_features = outputs.hidden_states[-1][:, :10, :].mean(axis=1)
                    
                    # Extract text features (remaining part of sequence)
                    if hasattr(outputs, 'text_hidden_states'):
                        t_features = outputs.text_hidden_states[-1].mean(axis=1)
                    else:
                        # Use remaining tokens as text proxy
                        t_features = outputs.hidden_states[-1][:, 10:, :].mean(axis=1)
                    
                    # Convert to numpy
                    if hasattr(v_features, 'numpy'):
                        v_features = v_features.numpy()
                    else:
                        v_features = np.array(v_features)
                    
                    if hasattr(t_features, 'numpy'):
                        t_features = t_features.numpy()
                    else:
                        t_features = np.array(t_features)
                    
                    vision_embeddings.append(v_features)
                    text_embeddings.append(t_features)
                    
            except Exception as e:
                logger.warning(f"Failed to process multimodal batch {i//batch_size}: {e}")
                # Add zero embeddings as fallback
                dummy_v = np.zeros((len(batch_images), 512))
                dummy_t = np.zeros((len(batch_texts), 512))
                vision_embeddings.append(dummy_v)
                text_embeddings.append(dummy_t)
            
            # Clear cache
            mx.metal.clear_cache()
        
        vision_emb = np.vstack(vision_embeddings) if vision_embeddings else np.array([])
        text_emb = np.vstack(text_embeddings) if text_embeddings else np.array([])
        
        return vision_emb, text_emb

    def save_embeddings(self, embeddings: Dict, model_name: str, dataset_name: str, 
                       extraction_time: float, metadata: Dict = None) -> str:
        """Save embeddings to HDF5 file following the established format"""
        
        # Clean model name for filename
        clean_model_name = model_name.replace('-', '_').replace('/', '_').replace(' ', '_')
        clean_dataset_name = dataset_name.replace('-', '_').replace('/', '_').replace(' ', '_')
        
        filename = f"{clean_dataset_name}_{clean_model_name}_features.h5"
        
        # Determine output directory based on model size
        if 'small' in model_name.lower() or any(small_model in model_name for small_model in ['nano', 'smol', 'phi-3', 'qwen2.5', 'gemma-2']):
            output_path = self.model_dirs['small'] / filename
        else:
            output_path = self.model_dirs['medium'] / filename
        
        logger.info(f"Saving embeddings to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Save embeddings
            if 'vision_features' in embeddings:
                f.create_dataset('vision_features', data=embeddings['vision_features'])
            
            if 'text_features' in embeddings:
                f.create_dataset('text_features', data=embeddings['text_features'])
            
            # Save attributes
            f.attrs['model_name'] = model_name
            f.attrs['dataset_name'] = dataset_name
            f.attrs['extraction_time'] = extraction_time
            f.attrs['n_samples'] = embeddings.get('n_samples', 0)
            f.attrs['feature_dim'] = embeddings.get('feature_dim', 0)
            
            # Save metadata
            if metadata:
                metadata_group = f.create_group('dataset_metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        metadata_group.attrs[key] = value
                    else:
                        metadata_group.create_dataset(key, data=value)
        
        return str(output_path)

    def process_dataset_model_pair(self, dataset_info: Dict, model_config: ModelConfig) -> bool:
        """Process a single dataset-model pair and extract embeddings"""
        dataset_name = dataset_info['config'].name
        model_name = model_config.name
        
        logger.info(f"Processing {dataset_name} with {model_name}")
        
        start_time = time.time()
        
        try:
            # Load model
            model_info = self.verifier.load_model(model_config)
            if not model_info:
                logger.error(f"Failed to load model {model_name}")
                return False
            
            embeddings = {}
            processed_data = dataset_info['processed_data']
            
            # Extract embeddings based on dataset type
            if dataset_info['config'].dataset_type == 'vision':
                images = processed_data['images']
                vision_emb = self.extract_vision_embeddings(model_info, images)
                embeddings['vision_features'] = vision_emb
                embeddings['n_samples'] = len(images)
                embeddings['feature_dim'] = vision_emb.shape[1] if len(vision_emb) > 0 else 0
                
            elif dataset_info['config'].dataset_type == 'text':
                texts = processed_data['texts']
                text_emb = self.extract_text_embeddings(model_info, texts)
                embeddings['text_features'] = text_emb
                embeddings['n_samples'] = len(texts)
                embeddings['feature_dim'] = text_emb.shape[1] if len(text_emb) > 0 else 0
                
            elif dataset_info['config'].dataset_type == 'multimodal':
                images = processed_data['images']
                texts = processed_data['texts']
                vision_emb, text_emb = self.extract_multimodal_embeddings(model_info, images, texts)
                embeddings['vision_features'] = vision_emb
                embeddings['text_features'] = text_emb
                embeddings['n_samples'] = min(len(images), len(texts))
                embeddings['feature_dim'] = vision_emb.shape[1] if len(vision_emb) > 0 else 0
            
            # Save embeddings
            extraction_time = time.time() - start_time
            metadata = {
                'dataset_type': dataset_info['config'].dataset_type,
                'model_type': model_config.model_type,
                'load_time': dataset_info['load_time'],
                'num_samples_original': dataset_info['num_samples']
            }
            
            output_path = self.save_embeddings(
                embeddings, model_name, dataset_name, extraction_time, metadata
            )
            
            logger.info(f"Successfully processed {dataset_name} with {model_name} in {extraction_time:.1f}s")
            logger.info(f"Saved to: {output_path}")
            
            # Clean up
            del model_info
            mx.metal.clear_cache()
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_name} with {model_name}: {e}")
            return False

    def run_comprehensive_extraction(self, dataset_sizes: List[str] = ['small', 'medium'], 
                                   model_sizes: List[str] = ['small', 'medium'],
                                   dataset_types: List[str] = ['vision', 'text', 'multimodal']) -> Dict:
        """Run comprehensive embedding extraction across datasets and models"""
        
        logger.info("Starting comprehensive embedding extraction")
        logger.info(f"Dataset sizes: {dataset_sizes}")
        logger.info(f"Model sizes: {model_sizes}")
        logger.info(f"Dataset types: {dataset_types}")
        
        results = {
            'successful': [],
            'failed': [],
            'total_processed': 0,
            'total_time': 0
        }
        
        start_time = time.time()
        
        # Load datasets
        datasets = {}
        for size in dataset_sizes:
            logger.info(f"Loading {size} datasets...")
            if size == 'small':
                datasets[size] = self.verifier.load_small_datasets(dataset_types)
            else:
                datasets[size] = self.verifier.load_medium_datasets(dataset_types)
            
            logger.info(f"Loaded {len(datasets[size])} {size} datasets")
        
        # Process each dataset-model combination
        for dataset_size in dataset_sizes:
            if dataset_size not in datasets:
                continue
                
            for dataset_name, dataset_info in datasets[dataset_size].items():
                logger.info(f"Processing dataset: {dataset_name}")
                
                # Select appropriate models based on dataset type and size
                if dataset_info['config'].dataset_type == 'vision':
                    model_categories = ['vision_encoders']
                elif dataset_info['config'].dataset_type == 'text':
                    model_categories = ['language_small', 'language_medium']
                else:  # multimodal
                    model_categories = ['vlm_small', 'vlm_medium']
                
                for model_category in model_categories:
                    if model_category not in RECOMMENDED_MODELS:
                        continue
                    
                    # Filter models by size - use category matching instead of name matching
                    models_to_use = []
                    for model_config in RECOMMENDED_MODELS[model_category]:
                        # For small datasets, use small/medium models based on category
                        if 'small' in model_sizes and ('small' in model_category or 'vision_encoders' in model_category):
                            models_to_use.append(model_config)
                        elif 'medium' in model_sizes and 'medium' in model_category:
                            models_to_use.append(model_config)
                    
                    # Process each model
                    for model_config in models_to_use:
                        results['total_processed'] += 1
                        
                        # Check memory before processing
                        if self.verifier.memory_monitor.check_memory_pressure():
                            logger.warning("High memory pressure detected, clearing caches...")
                            mx.metal.clear_cache()
                            gc.collect()
                        
                        success = self.process_dataset_model_pair(dataset_info, model_config)
                        
                        if success:
                            results['successful'].append(f"{dataset_name}_{model_config.name}")
                        else:
                            results['failed'].append(f"{dataset_name}_{model_config.name}")
        
        results['total_time'] = time.time() - start_time
        
        # Save results summary
        results_path = self.output_dir / "extraction_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("="*60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total processed: {results['total_processed']}")
        logger.info(f"Successful: {len(results['successful'])}")
        logger.info(f"Failed: {len(results['failed'])}")
        logger.info(f"Total time: {results['total_time']:.1f}s")
        logger.info(f"Results saved to: {results_path}")
        
        return results

def main():
    """Main function to run embedding extraction"""
    
    # Initialize extractor
    extractor = EmbeddingExtractor(memory_limit_gb=8)  # Limit to 8GB for MacBook
    
    # Run comprehensive extraction
    # Start with small datasets and small models for testing
    results = extractor.run_comprehensive_extraction(
        dataset_sizes=['small'],  # Start with small datasets
        model_sizes=['small'],    # Start with small models
        dataset_types=['vision', 'text', 'multimodal']
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("="*60)
    print(f"Successful extractions: {len(results['successful'])}")
    print(f"Failed extractions: {len(results['failed'])}")
    print(f"Total time: {results['total_time']:.1f}s")
    
    if results['failed']:
        print("\nFailed extractions:")
        for failure in results['failed']:
            print(f"  - {failure}")

if __name__ == "__main__":
    main()
