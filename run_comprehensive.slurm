#!/bin/bash
#SBATCH --job-name=platonic_analysis
#SBATCH --partition=h100                # GPU partition (h100, quad_rtx_8000, or tesla_p100)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32             # H100 default is 32 CPUs per GPU
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=32G                      # Adjust memory as needed
#SBATCH --output=logs/platonic_%j.out
#SBATCH --error=logs/platonic_%j.err

# Set environment variables
export PYTHONPATH="/home/cril/mukherjee/projects:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"

# Navigate to project directory
cd /home/cril/mukherjee/projects/platonic-rep

# Set data directory and results folder paths
export DATA_DIR="/home/cril/mukherjee/projects/data"
export RESULTS_DIR="/home/cril/mukherjee/projects/platonic-rep"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8

# Activate your conda environment (adjust the name)
source activate platonic_env

# Run the comprehensive analysis
python run_comprehensive_multiscale.py