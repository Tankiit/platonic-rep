def get_model_size_estimate(model_name, model_type="llm"):
    """
    Estimate model size for sorting purposes.
    Returns size in millions of parameters.
    """
    model_name_lower = model_name.lower()
    
    if model_type == "llm":
        # Extract size from model name
        if "70m" in model_name_lower: return 70
        elif "125m" in model_name_lower: return 125  
        elif "160m" in model_name_lower: return 160
        elif "350m" in model_name_lower: return 350
        elif "410m" in model_name_lower: return 410
        elif "560m" in model_name_lower: return 560
        elif "1b" in model_name_lower or "1.3b" in model_name_lower: return 1300
        elif "1b1" in model_name_lower: return 1100
        elif "1b7" in model_name_lower: return 1700
        elif "1.4b" in model_name_lower: return 1400
        elif "2b" in model_name_lower: return 2000
        elif "2.7b" in model_name_lower: return 2700
        elif "2.8b" in model_name_lower: return 2800
        elif "3b" in model_name_lower: return 3000
        elif "6b" in model_name_lower or "6.7b" in model_name_lower: return 6700
        elif "6.9b" in model_name_lower: return 6900
        elif "7b" in model_name_lower: return 7000
        elif "8b" in model_name_lower: return 8000
        elif "12b" in model_name_lower: return 12000
        elif "13b" in model_name_lower: return 13000
        elif "16b" in model_name_lower: return 16000
        elif "30b" in model_name_lower: return 30000
        elif "40b" in model_name_lower: return 40000
        elif "65b" in model_name_lower: return 65000
        elif "66b" in model_name_lower: return 66000
        elif "70b" in model_name_lower: return 70000
        elif "8x7b" in model_name_lower: return 45000  # Mixtral effective size
        # GPT models
        elif "gpt2" == model_name_lower: return 117
        elif "gpt2-medium" in model_name_lower: return 345
        elif "gpt2-large" in model_name_lower: return 762
        elif "gpt2-xl" in model_name_lower: return 1542
        elif "medium" in model_name_lower and "dialo" in model_name_lower: return 345
        elif "large" in model_name_lower and "dialo" in model_name_lower: return 762
        else: return 7000  # Default fallback
            
    elif model_type == "lvm":
        # Vision model sizes
        if "tiny" in model_name_lower: return 5
        elif "small" in model_name_lower: return 22
        elif "base" in model_name_lower: return 86
        elif "large" in model_name_lower: return 307
        elif "huge" in model_name_lower: return 632
        elif "giant" in model_name_lower: return 1137
        else: return 86  # Default to base size
    
    return 7000  # Default fallback


def sort_models_by_size(models, model_type="llm"):
    """
    Sort models from smallest to largest based on estimated parameter count.
    """
    model_sizes = [(model, get_model_size_estimate(model, model_type)) for model in models]
    sorted_models = sorted(model_sizes, key=lambda x: x[1])
    return [model for model, size in sorted_models]


def get_models(modelset, modality='all'):
    
    assert modality in ['all', 'vision', 'language']
    
    if modelset == 'val':
        llm_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
        ]

        lvm_models = [
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_small_patch16_224.augreg_in21k",
            "vit_base_patch16_224.augreg_in21k",
            "vit_large_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
            "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_large_patch14_dinov2.lvd142m",
            "vit_giant_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b",
            "vit_base_patch16_clip_224.laion2b_ft_in12k",
            "vit_large_patch14_clip_224.laion2b_ft_in12k",
            "vit_huge_patch14_clip_224.laion2b_ft_in12k",
        ]
        
    elif modelset == 'test':
        llm_models = [
            "allenai/OLMo-1B-hf",
            "allenai/OLMo-7B-hf", 
            "google/gemma-2b",
            "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
            # "mistralai/Mixtral-8x22B-v0.1",
            "NousResearch/Meta-Llama-3-8B",
            "NousResearch/Meta-Llama-3-70B",
        ]
        
        lvm_models = []
        
    elif modelset == 'exhaustive':
        # Comprehensive list of language models
        llm_models = [
            # BLOOM models
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "bigscience/bloom-1b7", 
            "bigscience/bloom-3b",
            "bigscience/bloom-7b1",
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            
            # LLaMA models
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
            "NousResearch/Meta-Llama-3-8B",
            "NousResearch/Meta-Llama-3-70B",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            
            # OpenLLaMA models
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b", 
            "openlm-research/open_llama_13b",
            
            # OLMo models
            "allenai/OLMo-1B-hf",
            "allenai/OLMo-7B-hf",
            
            # Gemma models
            "google/gemma-2b",
            "google/gemma-7b", 
            "google/gemma-2b-it",
            "google/gemma-7b-it",
            
            # Mistral models
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-v0.3",
            "mistralai/Mixtral-8x7B-v0.1",
            
            # GPT models
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            
            # GPT-Neo and GPT-J models
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/gpt-j-6B",
            
            # Pythia models
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
            "EleutherAI/pythia-12b",
            
            # CodeGen models
            "Salesforce/codegen-350M-mono",
            "Salesforce/codegen-2B-mono",
            "Salesforce/codegen-6B-mono",
            "Salesforce/codegen-16B-mono",
            
            # Falcon models
            "tiiuae/falcon-7b",
            "tiiuae/falcon-40b",
            
            # MPT models
            "mosaicml/mpt-7b",
            "mosaicml/mpt-30b",
            
            # Additional modern models
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "facebook/opt-13b",
            "facebook/opt-30b",
            "facebook/opt-66b",
        ]

        # Comprehensive list of vision models
        lvm_models = [
            # Standard ViT models
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_small_patch16_224.augreg_in21k", 
            "vit_base_patch16_224.augreg_in21k",
            "vit_large_patch16_224.augreg_in21k",
            "vit_huge_patch14_224.orig_in21k",
            
            # MAE pretrained models
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
            
            # DINOv2 models
            "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_large_patch14_dinov2.lvd142m",
            "vit_giant_patch14_dinov2.lvd142m",
            
            # CLIP models
            "vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b",
            "vit_base_patch16_clip_224.laion2b_ft_in12k",
            "vit_large_patch14_clip_224.laion2b_ft_in12k", 
            "vit_huge_patch14_clip_224.laion2b_ft_in12k",
            "vit_base_patch16_clip_224.openai",
            "vit_large_patch14_clip_224.openai",
            
            # Different patch sizes
            "vit_base_patch32_224.augreg_in21k",
            "vit_large_patch32_224.orig_in21k",
            "vit_base_patch8_224.augreg_in21k",
            
            # Different input resolutions
            "vit_base_patch16_384.augreg_in21k",
            "vit_large_patch16_384.augreg_in21k",
            
            # DEIT models
            "deit_tiny_patch16_224.fb_in1k",
            "deit_small_patch16_224.fb_in1k",
            "deit_base_patch16_224.fb_in1k",
            "deit3_small_patch16_224.fb_in1k",
            "deit3_base_patch16_224.fb_in1k",
            "deit3_large_patch16_224.fb_in1k",
            
            # Additional ViT variants
            "vit_small_patch16_224.dino",
            "vit_base_patch16_224.dino",
            "vit_base_patch8_224.dino",
        ]
        
    elif modelset == 'custom':
        # Keep only small, CPU-friendly models for macOS
        llm_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
        ]
        lvm_models = [
            "vit_giant_patch14_dinov2.lvd142m",
        ]
    else:
        raise ValueError(f"Unknown modelset: {modelset}. Available options: 'val', 'test', 'exhaustive', 'custom'")
    
    if modality == "vision":
        llm_models = []
    elif modality == "language":
        lvm_models = []

    return llm_models, lvm_models
