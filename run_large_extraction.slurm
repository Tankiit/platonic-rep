#!/bin/bash
#SBATCH --job-name=large_feat_extract
#SBATCH --output=logs/extraction_%A_%a.out
#SBATCH --error=logs/extraction_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1  # Request 1 A100 GPU
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9  # Process 10 models in parallel

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate platonic

# Define large models to extract
VISION_MODELS=(
    "eva_giant_patch14_560"
    "vit_huge_patch14_clip_336"
    "convnext_xxlarge"
    "dinov2_vitg14"
    "beit_large_patch16_384"
)

LANGUAGE_MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "mistralai/Mistral-7B-v0.1"
    "tiiuae/falcon-40b"
    "EleutherAI/gpt-neox-20b"
)

# Combine all models
ALL_MODELS=("${VISION_MODELS[@]}" "${LANGUAGE_MODELS[@]}")

# Get current model based on array task ID
MODEL=${ALL_MODELS[$SLURM_ARRAY_TASK_ID]}

# Determine modality
if [[ " ${VISION_MODELS[@]} " =~ " ${MODEL} " ]]; then
    MODALITY="vision"
else
    MODALITY="language"
fi

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Modality: $MODALITY"
echo "========================================="

# Set dataset parameters
DATASET="minhuh/prh"
SUBSET="wit_1024"
NUM_SAMPLES=1024
BATCH_SIZE=4
OUTPUT_DIR="./results/large_features"

# Run extraction
python extract_large_features.py \
    --models "$MODEL" \
    --modality "$MODALITY" \
    --dataset "$DATASET" \
    --subset "$SUBSET" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --pool avg

echo "Extraction completed for $MODEL"