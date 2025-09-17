#!/bin/bash

# Exhaustive Feature Extraction Script
# This script runs feature extraction on comprehensive model sets

echo "Starting exhaustive feature extraction..."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --save_features)
            SAVE_FEATURES=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            shift
            ;;
    esac
done

# Set default parameters
DATASET=${DATASET:-"cifar10"}
MODEL=${MODEL:-"resnet34"}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES=${NUM_SAMPLES:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-"./results/features"}
DATA_DIR=${DATA_DIR:-"/home/mukherjee/research/data"}
MODEL_TYPE=${MODEL_TYPE:-"vision"}

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Data Directory: $DATA_DIR"
echo "  Model Type: $MODEL_TYPE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run feature extraction based on model type
if [ "$MODEL_TYPE" = "language" ]; then
    echo "Running language model extraction for $MODEL..."
    python extract_features.py \
        --models "$MODEL" \
        --modality language \
        --dataset "$DATASET" \
        --subset "" \
        --batch_size "$BATCH_SIZE" \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
        --pool avg
else
    echo "Running vision model extraction for $MODEL..."
    python extract_features.py \
        --models "$MODEL" \
        --modality vision \
        --dataset "$DATASET" \
        --subset "" \
        --batch_size "$BATCH_SIZE" \
        --num_samples "$NUM_SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --data_dir "$DATA_DIR" \
        --pool cls
fi

echo "Feature extraction completed!"
echo "Results saved to: $OUTPUT_DIR"
