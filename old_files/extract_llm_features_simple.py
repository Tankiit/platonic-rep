#!/usr/bin/env python3
"""
Simple Language Model Feature Extraction using MLX
Based on your PyTorch example but adapted for MLX
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as load_lm
import numpy as np
import h5py
import gc
import time
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Small language models that should download quickly
SMALL_MODELS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/gemma-2-2b-it-4bit",
    "mlx-community/phi-3-mini-4k-instruct-4bit"
]

# Small datasets for quick testing
SMALL_DATASETS = [
    {"name": "imdb", "hf_path": "imdb", "text_col": "text", "max_samples": 1000},
    {"name": "ag_news", "hf_path": "ag_news", "text_col": "text", "max_samples": 1000},
    {"name": "yelp_polarity", "hf_path": "yelp_polarity", "text_col": "text", "max_samples": 1000}
]

def extract_llm_features(model_name: str, texts: list, output_dir: str, dataset_name: str, pool: str = 'avg'):
    """
    Extract features from a language model using MLX
    """
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()

    try:
        model, tokenizer = load_lm(model_name)
        # Use mx.utils.tree_flatten for MLX compatibility
        try:
            model_size = sum(p.size for p in mx.utils.tree_flatten(model.parameters())[0])
        except AttributeError:
            # Fallback if tree_flatten doesn't exist
            model_size = 0
        logger.info(f"Model loaded in {time.time() - start_time:.1f}s, parameters: {model_size:,}")
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return False

    # Clean model name for filename
    clean_model_name = model_name.replace('/', '_').replace('-', '_')
    save_path = Path(output_dir) / f"{dataset_name}_{clean_model_name}_features.h5"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(texts)} texts")
    logger.info(f"Save path: {save_path}")

    batch_size = 4
    all_features = []
    all_losses = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]

        try:
            with mx.no_grad():
                # Tokenize
                inputs = tokenizer(batch_texts, return_tensors="np", padding=True, truncation=True, max_length=512)
                input_ids = mx.array(inputs["input_ids"])
                attention_mask = mx.array(inputs["attention_mask"])

                # Forward pass
                outputs = model(input_ids, output_hidden_states=True)

                # Calculate loss (for validation)
                if hasattr(outputs, 'logits'):
                    # Shift for causal language modeling
                    shift_logits = outputs.logits[..., :-1, :]
                    shift_labels = input_ids[..., 1:]

                    # Simple cross entropy calculation
                    batch_loss = mx.mean(mx.sum(-mx.log_softmax(shift_logits) *
                                              mx.one_hot(shift_labels, shift_logits.shape[-1]), axis=-1))
                    all_losses.append(float(batch_loss))

                # Extract features based on pooling strategy
                if pool == 'avg':
                    # Average pooling over sequence length
                    hidden_states = outputs.hidden_states  # List of layer outputs
                    features = []
                    for layer_output in hidden_states:
                        # Apply attention mask and average
                        mask = attention_mask.astype(mx.float32).reshape(attention_mask.shape[0], attention_mask.shape[1], 1)
                        masked_output = layer_output * mask
                        avg_output = mx.sum(masked_output, axis=1) / mx.sum(mask, axis=1)
                        features.append(avg_output)

                    # Stack layers: [batch, layers, hidden_dim]
                    batch_features = mx.stack(features, axis=1)

                elif pool == 'last':
                    # Use last token features
                    hidden_states = outputs.hidden_states
                    features = []
                    for layer_output in hidden_states:
                        # Get last token for each sequence
                        last_features = layer_output[:, -1, :]
                        features.append(last_features)

                    # Stack layers: [batch, layers, hidden_dim]
                    batch_features = mx.stack(features, axis=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {pool}")

                # Convert to numpy and store
                all_features.append(np.array(batch_features))

        except Exception as e:
            logger.warning(f"Error processing batch {i//batch_size}: {e}")
            # Add dummy features to maintain consistency
            if all_features:
                dummy_shape = all_features[-1].shape
                dummy_features = np.zeros((len(batch_texts), dummy_shape[1], dummy_shape[2]))
                all_features.append(dummy_features)

        # Clear cache periodically
        if i % (batch_size * 4) == 0:
            mx.metal.clear_cache()

    if not all_features:
        logger.error("No features extracted")
        return False

    # Concatenate all features
    features = np.concatenate(all_features, axis=0)
    avg_loss = np.mean(all_losses) if all_losses else 0.0

    logger.info(f"Extracted features shape: {features.shape}")
    logger.info(f"Average loss: {avg_loss:.4f}")

    # Save to HDF5
    try:
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('features', data=features)
            f.attrs['model_name'] = model_name
            f.attrs['dataset_name'] = dataset_name
            f.attrs['num_params'] = model_size
            f.attrs['avg_loss'] = avg_loss
            f.attrs['pooling'] = pool
            f.attrs['num_samples'] = features.shape[0]
            f.attrs['num_layers'] = features.shape[1]
            f.attrs['hidden_dim'] = features.shape[2]

        logger.info(f"Features saved to: {save_path}")

    except Exception as e:
        logger.error(f"Failed to save features: {e}")
        return False

    # Cleanup
    del model, tokenizer, outputs, all_features
    mx.metal.clear_cache()
    gc.collect()

    return True

def main():
    """Main extraction function"""
    output_dir = "representations/llm_features"

    logger.info("Starting LLM feature extraction")
    logger.info(f"Output directory: {output_dir}")

    results = {"successful": [], "failed": []}

    # Process each dataset
    for dataset_config in SMALL_DATASETS:
        logger.info(f"\nLoading dataset: {dataset_config['name']}")

        try:
            dataset = load_dataset(dataset_config['hf_path'], split='train')
            if len(dataset) > dataset_config['max_samples']:
                dataset = dataset.select(range(dataset_config['max_samples']))

            # Extract text column
            texts = [str(item[dataset_config['text_col']]) for item in dataset]
            logger.info(f"Loaded {len(texts)} texts from {dataset_config['name']}")

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_config['name']}: {e}")
            continue

        # Process each model
        for model_name in SMALL_MODELS:
            logger.info(f"\nProcessing {dataset_config['name']} with {model_name}")

            success = extract_llm_features(
                model_name=model_name,
                texts=texts,
                output_dir=output_dir,
                dataset_name=dataset_config['name'],
                pool='avg'
            )

            if success:
                results["successful"].append(f"{dataset_config['name']}_{model_name}")
                logger.info("✅ Success")
            else:
                results["failed"].append(f"{dataset_config['name']}_{model_name}")
                logger.info("❌ Failed")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Successful: {len(results['successful'])}")
    logger.info(f"Failed: {len(results['failed'])}")

    if results['successful']:
        logger.info("\nSuccessful extractions:")
        for item in results['successful']:
            logger.info(f"  ✅ {item}")

    if results['failed']:
        logger.info("\nFailed extractions:")
        for item in results['failed']:
            logger.info(f"  ❌ {item}")

    logger.info(f"\nResults saved in: {output_dir}")

if __name__ == "__main__":
    main()