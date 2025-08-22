import gc
import os
import argparse
import warnings
from tqdm import trange

import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset
from tasks import get_models
from models import load_llm, load_tokenizer
import utils 


def check_model_availability(model_name, model_type="llm"):
    """
    Check if a model is available before attempting to load it.
    """
    try:
        if model_type == "llm":
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            return True
        elif model_type == "lvm":
            available_models = timm.list_models()
            return model_name in available_models
    except Exception as e:
        print(f"Model {model_name} not available: {e}")
        return False
    return False


def safe_batch_processing(dataset, batch_size, start_idx=0):
    """
    Safely handle batch processing with proper bounds checking.
    """
    total_samples = len(dataset)
    for i in range(start_idx, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        yield i, batch_end, list(range(i, batch_end))
    

def extract_llm_features(filenames, dataset, args):
    """
    Extracts features from language models.
    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """

    texts = [str(x['text'][args.caption_idx]) for x in dataset]
        
    for llm_model_name in filenames[::-1]:
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, llm_model_name,
            pool=args.pool, prompt=args.prompt, caption_idx=args.caption_idx,
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{llm_model_name}")
        print(f'save_path: \t{save_path}')
        
        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue
        
        try:
            language_model = load_llm(llm_model_name, qlora=args.qlora, force_download=args.force_download)
            llm_param_count = sum([p.numel() for p in language_model.parameters()])
            tokenizer = load_tokenizer(llm_model_name)
        except Exception as e:
            print(f"Failed to load model {llm_model_name}: {e}")
            print("Continuing with next model...")
            continue
    
        tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
        llm_feats, losses, bpb_losses = [], [], []

        # hack to get around HF mapping data incorrectly when using model-parallel
        device = next(language_model.parameters()).device

        for batch_info in trange(0, len(dataset), args.batch_size, desc="Processing batches"):
            batch_start = batch_info
            batch_end = min(batch_start + args.batch_size, len(dataset))
            # get embedding cuda device
            token_inputs = {k: v[batch_start:batch_end].to(device).long() for (k, v) in tokens.items()}

            with torch.no_grad():
                # Handle different model architectures
                if "olmo" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                elif "gpt2" in llm_model_name.lower() or "gpt-neo" in llm_model_name.lower() or "gpt-j" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                elif "pythia" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                elif "codegen" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                elif "falcon" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                elif "mpt" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                elif "opt" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    # Default case for most models (BLOOM, LLaMA, Gemma, Mistral, etc.)
                    try:
                        llm_output = language_model(
                            input_ids=token_inputs["input_ids"],
                            attention_mask=token_inputs["attention_mask"],
                            output_hidden_states=True,
                        )
                    except Exception as e:
                        print(f"Warning: Failed to get hidden states for {llm_model_name}, trying without output_hidden_states")
                        llm_output = language_model(
                            input_ids=token_inputs["input_ids"],
                            attention_mask=token_inputs["attention_mask"],
                        )
                        # If no hidden states, we'll skip feature extraction for this model
                        if not hasattr(llm_output, 'hidden_states') or llm_output.hidden_states is None:
                            print(f"Skipping feature extraction for {llm_model_name} - no hidden states available")
                            continue

                loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
                losses.extend(avg_loss.cpu())
                
                bpb = utils.cross_entropy_to_bits_per_unit(loss.cpu(), texts[batch_start:batch_end], unit="byte")
                bpb_losses.extend(bpb)
                
                # make sure to do all the processing in cpu to avoid memory problems
                if args.pool == 'avg':
                    feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    feats = (feats * mask).sum(2) / mask.sum(2)
                elif args.pool == 'last':
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2) 
                else:
                    raise NotImplementedError(f"unknown pooling {args.pool}")
                llm_feats.append(feats.cpu())

        print(f"average loss:\t{torch.stack(losses).mean().item()}")
        save_dict = {
            "feats": torch.cat(llm_feats).cpu(),
            "num_params": llm_param_count,
            "mask": tokens["attention_mask"].cpu(),
            "loss": torch.stack(losses).mean(),
            "bpb": torch.stack(bpb_losses).mean(),
        }

        torch.save(save_dict, save_path)

        del language_model, tokenizer, llm_feats, llm_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    return
    
        
def extract_lvm_features(filenames, dataset, args):
    """
    Extracts features from vision models.
    Args:
        filenames: list of vision model names by timm identifiers
        dataset: huggingface dataset with images
        args: argparse arguments
    """
    # Support different pooling strategies for different model types
    supported_pools = ['cls', 'avg', 'max'] if 'vit' in str(filenames) else ['cls']
    if args.pool not in supported_pools:
        print(f"Warning: {args.pool} pooling may not be optimal for some vision models. Supported: {supported_pools}")
    
    for lvm_model_name in filenames:
        # Support different vision transformer architectures
        is_vit = any(arch in lvm_model_name.lower() for arch in ['vit', 'deit'])
        
        if not is_vit:
            print(f"Warning: {lvm_model_name} may not be a Vision Transformer. Proceeding with caution...")
        
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, lvm_model_name,
            pool=args.pool, prompt=None, caption_idx=None,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{lvm_model_name}")
        print(f'save_path: \t{save_path}')

        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            vision_model = timm.create_model(lvm_model_name, pretrained=True).to(device).eval()
            lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

            transform = create_transform(
                **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
            )

            # Determine return nodes based on model architecture
            if "vit" in lvm_model_name or "deit" in lvm_model_name:
                # Handle different ViT variants
                if hasattr(vision_model, 'blocks'):
                    return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
                elif hasattr(vision_model, 'layers'):
                    return_nodes = [f"layers.{i}" for i in range(len(vision_model.layers))]
                else:
                    # Fallback for unknown ViT structure
                    print(f"Warning: Unknown ViT structure for {lvm_model_name}, using default extraction")
                    return_nodes = None
            else:
                print(f"Warning: Unsupported model architecture for {lvm_model_name}")
                return_nodes = None

            if return_nodes:
                vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
            
            lvm_feats = []

            for i, batch_end, indices in safe_batch_processing(dataset, args.batch_size):
                with torch.no_grad():
                    try:
                        ims = torch.stack([transform(dataset[j]['image']) for j in indices]).to(device)
                        
                        if return_nodes:
                            lvm_output = vision_model(ims)
                            
                            if args.pool == "cls":
                                # Use CLS token for ViTs
                                feats = [v[:, 0, :] for v in lvm_output.values()]
                                feats = torch.stack(feats).permute(1, 0, 2)
                            elif args.pool == "avg":
                                # Average pool over spatial dimensions
                                feats = [v.mean(dim=1) for v in lvm_output.values()]
                                feats = torch.stack(feats).permute(1, 0, 2)
                            elif args.pool == "max":
                                # Max pool over spatial dimensions
                                feats = [v.max(dim=1)[0] for v in lvm_output.values()]
                                feats = torch.stack(feats).permute(1, 0, 2)
                            else:
                                raise NotImplementedError(f"Unknown pooling strategy: {args.pool}")
                        else:
                            # Fallback: use model's final output
                            lvm_output = vision_model(ims)
                            if isinstance(lvm_output, torch.Tensor):
                                feats = lvm_output.unsqueeze(1)  # Add layer dimension
                            else:
                                # Handle tuple/dict outputs
                                feats = lvm_output[0].unsqueeze(1) if isinstance(lvm_output, (tuple, list)) else lvm_output
                                
                        lvm_feats.append(feats.cpu())
                        
                    except Exception as e:
                        print(f"Error processing batch {i}-{batch_end} for {lvm_model_name}: {e}")
                        print("Skipping this batch...")
                        continue

            if lvm_feats:
                torch.save({"feats": torch.cat(lvm_feats), "num_params": lvm_param_count}, save_path)
                print(f"Successfully saved features for {lvm_model_name}")
            else:
                print(f"No features extracted for {lvm_model_name}")

        except Exception as e:
            print(f"Failed to process {lvm_model_name}: {e}")
            print("Continuing with next model...")
            continue

        finally:
            # Cleanup
            if 'vision_model' in locals():
                del vision_model
            if 'transform' in locals():
                del transform  
            if 'lvm_feats' in locals():
                del lvm_feats
            if 'lvm_output' in locals():
                del lvm_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Extract features from language and vision models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features using validation model set
  python extract_features.py --modelset val --modality all
  
  # Extract features using exhaustive model set (language only)
  python extract_features.py --modelset exhaustive --modality language --pool avg
  
  # Extract features using exhaustive model set (vision only) 
  python extract_features.py --modelset exhaustive --modality vision --pool cls
  
  # Run exhaustive extraction with custom settings
  python extract_features.py --modelset exhaustive --batch_size 8 --num_samples 2048
  
  # Use the convenience script for full exhaustive extraction
  ./run_exhaustive_extraction.sh prh wit_1024 4 1024
  
Model Sets:
  - val: Validation set with balanced language and vision models
  - test: Test set with modern large models  
  - exhaustive: Comprehensive set covering major model families
  - custom: Small set optimized for CPU/testing
  
Supported Model Families:
  Language: BLOOM, LLaMA, Gemma, Mistral, GPT-2/Neo/J, Pythia, CodeGen, Falcon, MPT, OPT
  Vision: ViT (various sizes), DeiT, CLIP, MAE, DINOv2, DINO
        """)
    
    parser.add_argument("--force_download", action="store_true", help="Force download of models")
    parser.add_argument("--force_remake",   action="store_true", help="Force remake of existing feature files")
    parser.add_argument("--num_samples",    type=int, default=1024, help="Number of samples to process")
    parser.add_argument("--batch_size",     type=int, default=4, help="Batch size for processing")
    parser.add_argument("--pool",           type=str, default='avg', choices=['avg', 'cls', 'last', 'max'], 
                       help="Pooling strategy: avg=average, cls=CLS token, last=last token, max=max pool")
    parser.add_argument("--prompt",         action="store_true", help="Use prompting for language models")
    parser.add_argument("--dataset",        type=str, default="prh", help="Dataset to use")
    parser.add_argument("--subset",         type=str, default="wit_1024", help="Dataset subset")
    parser.add_argument("--caption_idx",    type=int, default=0, help="Caption index for multi-caption datasets")
    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test", "exhaustive", "custom"],
                       help="Model set to use")
    parser.add_argument("--modality",       type=str, default="all", choices=["vision", "language", "all"],
                       help="Modality to extract features for")
    parser.add_argument("--output_dir",     type=str, default="./results/features", help="Output directory")
    parser.add_argument("--qlora",          action="store_true", help="Use QLoRA quantization")
    args = parser.parse_args()

    if args.qlora:
        print(f"QLoRA is set to True. The alignment score will be slightly off.")

    llm_models, lvm_models = get_models(args.modelset, modality=args.modality)
    
    # load dataset once outside    
    dataset = load_dataset(args.dataset, revision=args.subset, split='train')

    if args.modality in ["all", "language"]:
        # extract all language model features
        extract_llm_features(llm_models, dataset, args)
    
    if args.modality in ["all", "vision"]:
        # extract all vision model features
        extract_lvm_features(lvm_models, dataset, args)
