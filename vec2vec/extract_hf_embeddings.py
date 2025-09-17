#!/usr/bin/env python3
"""
Comprehensive Hugging Face embedding extractor for systematic experiments.
Extracts embeddings from multiple model families across diverse datasets.
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from tqdm import tqdm
import argparse


class HuggingFaceEmbeddingExtractor:
    """
    Extract embeddings from any model on Hugging Face.
    This handles the complexity of different model architectures
    while providing a uniform interface.
    """

    def __init__(self, cache_dir="./hf_cache"):
        self.cache_dir = cache_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_model_list_for_experiments(self):
        """
        Curated list of models that are good for testing representation similarity.
        I've organized these by family to test your hypothesis systematically.
        """
        return {
            # BERT family - same architecture, different training
            'bert_variants': [
                'bert-base-uncased',
                'bert-large-uncased',
                'roberta-base',
                'roberta-large',
                'distilbert-base-uncased',
                'microsoft/deberta-v3-base'
            ],

            # GPT family - autoregressive models
            'gpt_variants': [
                'gpt2',
                'gpt2-medium',
                'gpt2-large',
                'EleutherAI/gpt-neo-1.3B',
                'EleutherAI/gpt-j-6B'  # Large but useful for comparison
            ],

            # T5 family - encoder-decoder
            't5_variants': [
                'google/t5-v1_1-small',
                'google/t5-v1_1-base',
                'google/t5-v1_1-large',
                'google/flan-t5-base'  # Instruction-tuned variant
            ],

            # Multilingual models - test cross-lingual alignment
            'multilingual': [
                'xlm-roberta-base',
                'google/mt5-base',
                'microsoft/mdeberta-v3-base'
            ],

            # Specialized models - test domain-specific representations
            'specialized': [
                'allenai/scibert_scivocab_uncased',  # Scientific text
                'microsoft/codebert-base',  # Code understanding
                'emilyalsentzer/Bio_ClinicalBERT',  # Biomedical
                'nlpaueb/legal-bert-base-uncased'  # Legal text
            ]
        }

    def extract_embeddings(self, model_name: str, texts: List[str],
                          layer: int = -1, pooling: str = 'mean',
                          batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings with fine-grained control over which representations to use.

        The layer parameter is crucial for your experiments - different layers
        capture different levels of abstraction, which affects alignment quality.
        """

        print(f"Loading model: {model_name}")
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            output_hidden_states=True  # Important for layer selection
        ).to(self.device)
        model.eval()

        # Handle models without pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting from {model_name}"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)

                # Get representations from specified layer
                hidden_states = outputs.hidden_states[layer]

                # Apply pooling strategy
                if pooling == 'cls':
                    # Use [CLS] token (first token)
                    batch_embeddings = hidden_states[:, 0, :]
                elif pooling == 'mean':
                    # Mean pooling with attention mask
                    attention_mask = inputs.attention_mask.unsqueeze(-1)
                    sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
                    sum_mask = torch.sum(attention_mask, dim=1)
                    batch_embeddings = sum_embeddings / sum_mask
                elif pooling == 'max':
                    # Max pooling
                    batch_embeddings = torch.max(hidden_states, dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling}")

                embeddings.append(batch_embeddings.cpu().numpy())

        # Clear GPU memory
        if self.device.type == 'cuda':
            del model
            torch.cuda.empty_cache()

        return np.vstack(embeddings)


def load_diverse_datasets_for_testing(max_samples_per_dataset: int = 1000):
    """
    Load different types of datasets from Hugging Face to test if
    representation similarity depends on the type of text.

    This directly addresses your question about whether data matters.
    """
    datasets_dict = {}

    try:
        # General knowledge - Wikipedia-like text
        print("Loading Wikipedia dataset...")
        wiki = load_dataset('wikipedia', '20220301.simple', split=f'train[:{max_samples_per_dataset}]')
        datasets_dict['general'] = wiki['text']
    except Exception as e:
        print(f"Failed to load Wikipedia: {e}")
        # Fallback dataset
        datasets_dict['general'] = ["This is a general text sample."] * min(100, max_samples_per_dataset)

    try:
        # Question-answering - tests semantic understanding
        print("Loading Natural Questions dataset...")
        nq = load_dataset('natural_questions', split=f'train[:{max_samples_per_dataset}]')
        datasets_dict['questions'] = [item['question']['text'] for item in nq]
    except Exception as e:
        print(f"Failed to load Natural Questions: {e}")
        datasets_dict['questions'] = ["What is this question?"] * min(100, max_samples_per_dataset)

    try:
        # Scientific text - specialized domain
        print("Loading PubMed QA dataset...")
        pubmed = load_dataset('pubmed_qa', 'pqa_labeled', split=f'train[:{max_samples_per_dataset}]')
        datasets_dict['scientific'] = [item['context'] for item in pubmed]
    except Exception as e:
        print(f"Failed to load PubMed QA: {e}")
        datasets_dict['scientific'] = ["Scientific research shows that..."] * min(100, max_samples_per_dataset)

    try:
        # Code - very different structure from natural language
        print("Loading GitHub Code dataset...")
        code = load_dataset('codeparrot/github-code', streaming=True)
        code_samples = []
        for i, item in enumerate(code):
            if i >= max_samples_per_dataset:
                break
            code_samples.append(item['code'][:1000])  # Limit code length
        datasets_dict['code'] = code_samples
    except Exception as e:
        print(f"Failed to load GitHub Code: {e}")
        datasets_dict['code'] = ["def function(): pass"] * min(100, max_samples_per_dataset)

    try:
        # Conversational - different style
        print("Loading Daily Dialog dataset...")
        conv = load_dataset('daily_dialog', split=f'train[:{max_samples_per_dataset}]')
        datasets_dict['conversation'] = [' '.join(dialog['dialog']) for dialog in conv]
    except Exception as e:
        print(f"Failed to load Daily Dialog: {e}")
        datasets_dict['conversation'] = ["Hello, how are you?"] * min(100, max_samples_per_dataset)

    return datasets_dict


def save_embeddings(embeddings: np.ndarray, output_dir: Path, model_name: str,
                   dataset_name: str, layer: int, pooling: str):
    """
    Save embeddings with descriptive filenames for easy organization.
    """
    # Clean model name for filename
    model_clean = model_name.replace('/', '_').replace('-', '_')

    # Create filename
    filename = f"{dataset_name}_{model_clean}_layer{layer}_{pooling}.npz"
    filepath = output_dir / filename

    # Save embeddings with metadata
    np.savez_compressed(
        filepath,
        embeddings=embeddings,
        model_name=model_name,
        dataset_name=dataset_name,
        layer=layer,
        pooling=pooling
    )

    print(f"Saved: {filepath}")
    return filepath


def extract_all_embeddings(
    models: Optional[List[str]] = None,
    datasets: Optional[Dict[str, List[str]]] = None,
    output_dir: str = "./embeddings",
    layers: List[int] = [-1],
    pooling_strategies: List[str] = ['mean'],
    max_samples: int = 1000,
    batch_size: int = 32
):
    """
    Main function to extract embeddings from multiple models and datasets.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = HuggingFaceEmbeddingExtractor()

    # Get models if not specified
    if models is None:
        model_groups = extractor.get_model_list_for_experiments()
        models = []
        for family, model_list in model_groups.items():
            models.extend(model_list[:2])  # Take first 2 from each family for testing

    # Load datasets if not specified
    if datasets is None:
        print("Loading diverse datasets...")
        datasets = load_diverse_datasets_for_testing(max_samples)

    # Track results
    results = []
    metadata = {
        'models': models,
        'datasets': list(datasets.keys()),
        'layers': layers,
        'pooling_strategies': pooling_strategies,
        'max_samples': max_samples
    }

    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Extract embeddings for each combination
    for model_name in models:
        for dataset_name, texts in datasets.items():
            # Limit number of samples
            texts = texts[:max_samples]

            for layer in layers:
                for pooling in pooling_strategies:
                    try:
                        print(f"\n{'='*60}")
                        print(f"Model: {model_name}")
                        print(f"Dataset: {dataset_name}")
                        print(f"Layer: {layer}, Pooling: {pooling}")
                        print(f"{'='*60}")

                        # Extract embeddings
                        embeddings = extractor.extract_embeddings(
                            model_name=model_name,
                            texts=texts,
                            layer=layer,
                            pooling=pooling,
                            batch_size=batch_size
                        )

                        # Save embeddings
                        filepath = save_embeddings(
                            embeddings=embeddings,
                            output_dir=output_path,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            layer=layer,
                            pooling=pooling
                        )

                        # Record result
                        results.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'layer': layer,
                            'pooling': pooling,
                            'shape': embeddings.shape,
                            'filepath': str(filepath)
                        })

                    except Exception as e:
                        print(f"Failed to extract: {e}")
                        results.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'layer': layer,
                            'pooling': pooling,
                            'error': str(e)
                        })

    # Save results summary
    with open(output_path / 'extraction_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Extraction complete! Results saved to {output_path}")
    print(f"Total extractions: {len(results)}")
    print(f"Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"Failed: {sum(1 for r in results if 'error' in r)}")

    return results


def load_embeddings(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load embeddings and metadata from saved file.
    """
    data = np.load(filepath, allow_pickle=True)
    embeddings = data['embeddings']
    metadata = {
        'model_name': str(data['model_name']),
        'dataset_name': str(data['dataset_name']),
        'layer': int(data['layer']),
        'pooling': str(data['pooling'])
    }
    return embeddings, metadata


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from HuggingFace models")

    # Model selection
    parser.add_argument('--models', nargs='+',
                       help='Specific model names to use')
    parser.add_argument('--model-family', choices=['bert_variants', 'gpt_variants',
                                                   't5_variants', 'multilingual',
                                                   'specialized', 'all'],
                       help='Use predefined model family')

    # Dataset selection
    parser.add_argument('--datasets', nargs='+',
                       choices=['general', 'questions', 'scientific',
                               'code', 'conversation', 'all'],
                       default='all',
                       help='Which datasets to use')
    parser.add_argument('--custom-texts', type=str,
                       help='Path to custom text file')

    # Extraction parameters
    parser.add_argument('--layers', nargs='+', type=int, default=[-1],
                       help='Which layers to extract from')
    parser.add_argument('--pooling', nargs='+',
                       choices=['mean', 'cls', 'max'],
                       default=['mean'],
                       help='Pooling strategies to use')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples per dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for extraction')

    # Output
    parser.add_argument('--output-dir', type=str, default='./embeddings',
                       help='Output directory for embeddings')

    args = parser.parse_args()

    # Prepare models list
    models = args.models
    if args.model_family:
        extractor = HuggingFaceEmbeddingExtractor()
        model_groups = extractor.get_model_list_for_experiments()
        if args.model_family == 'all':
            models = []
            for family_models in model_groups.values():
                models.extend(family_models)
        else:
            models = model_groups[args.model_family]

    # Prepare datasets
    if args.custom_texts:
        with open(args.custom_texts, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        datasets = {'custom': texts}
    elif args.datasets == ['all']:
        datasets = None  # Will load all default datasets
    else:
        all_datasets = load_diverse_datasets_for_testing(args.max_samples)
        datasets = {k: v for k, v in all_datasets.items() if k in args.datasets}

    # Extract embeddings
    results = extract_all_embeddings(
        models=models,
        datasets=datasets,
        output_dir=args.output_dir,
        layers=args.layers,
        pooling_strategies=args.pooling,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )

    return results


if __name__ == '__main__':
    main()