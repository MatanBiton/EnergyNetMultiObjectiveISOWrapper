#!/bin/bash

#SBATCH --job-name=test_gpu
#SBATCH --output=slurm_gpu_test_%j.out
#SBATCH --error=slurm_gpu_test_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:05:00

echo "GPU Test Job - Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load CUDA module if available
if command -v module &> /dev/null; then
    echo "Loading CUDA modules..."
    module load cuda/12.6 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || echo "No CUDA module found"
fi

# Set CUDA environment variables
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export CUDA_PATH="/usr/local/cuda"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    echo "Set CUDA environment variables"
fi

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate IsoMOEnergyNet
    echo "Activated conda environment: IsoMOEnergyNet"
fi

# Run GPU test
echo ""
echo "Running GPU availability test..."
python3 test_gpu_availability.py

echo ""
echo "GPU test completed - Job ID: $SLURM_JOB_ID"
