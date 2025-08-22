

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
