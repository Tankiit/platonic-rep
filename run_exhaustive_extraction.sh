#!/bin/bash

# Exhaustive Feature Extraction Script
# This script runs feature extraction on comprehensive model sets

echo "Starting exhaustive feature extraction..."

# Set default parameters
DATASET=${1:-"prh"}
SUBSET=${2:-"wit_1024"}
BATCH_SIZE=${3:-4}
NUM_SAMPLES=${4:-1024}
OUTPUT_DIR=${5:-"./results/features"}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Subset: $SUBSET"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Output Directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run exhaustive extraction for both language and vision models
echo "Running exhaustive language model extraction..."
python extract_features.py \
    --modelset exhaustive \
    --modality language \
    --dataset "$DATASET" \
    --subset "$SUBSET" \
    --batch_size "$BATCH_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --pool avg

echo "Running exhaustive vision model extraction..."
python extract_features.py \
    --modelset exhaustive \
    --modality vision \
    --dataset "$DATASET" \
    --subset "$SUBSET" \
    --batch_size "$BATCH_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --pool cls

echo "Running exhaustive vision model extraction with average pooling..."
python extract_features.py \
    --modelset exhaustive \
    --modality vision \
    --dataset "$DATASET" \
    --subset "$SUBSET" \
    --batch_size "$BATCH_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --pool avg

echo "Exhaustive feature extraction completed!"
echo "Results saved to: $OUTPUT_DIR"
