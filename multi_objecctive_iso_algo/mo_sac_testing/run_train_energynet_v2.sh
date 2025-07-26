#!/bin/bash

#SBATCH --job-name=mo_sac_energynet_v2
#SBATCH --output=slurm_energynet_%j.out
#SBATCH --error=slurm_energynet_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Multi-Objective SAC EnergyNet Training Script for SLURM (Version 2 - Fixed)
# This script handles permission and path issues on SLURM clusters
#
# Usage: 
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh                # Default training
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 200000         # Custom episodes

echo "=========================================="
echo "Multi-Objective SAC EnergyNet Training v2 (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Get the original script location (where the files actually are)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ORIG_DIR="$SLURM_SUBMIT_DIR"
else
    ORIG_DIR="$(dirname "${BASH_SOURCE[0]}")"
    ORIG_DIR="$(cd "$ORIG_DIR" && pwd)"
fi

echo "Original directory (where files are): $ORIG_DIR"

# Check if Python files exist in original directory
if [ ! -f "$ORIG_DIR/train_energynet.py" ]; then
    echo "Error: train_energynet.py not found in $ORIG_DIR"
    echo "Files in original directory:"
    ls -la "$ORIG_DIR"
    exit 1
fi

# Load necessary modules (uncomment as needed for your cluster)
# module load python/3.8
# module load cuda/11.8
# module load gcc/9.3.0

# Try different Python commands
PYTHON_CMD=""
for cmd in python3 python python3.8 python3.9 python3.10; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        echo "Found Python: $PYTHON_CMD ($(which $cmd))"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: No Python interpreter found"
    exit 1
fi

echo "Python version: $($PYTHON_CMD --version)"

# Set up working directory - use SLURM_TMPDIR if available, otherwise create in /tmp
if [ -n "$SLURM_TMPDIR" ]; then
    WORK_DIR="$SLURM_TMPDIR/energynet_training"
else
    WORK_DIR="/tmp/energynet_training_${USER}_$$"
fi

echo "Creating working directory: $WORK_DIR"
mkdir -p "$WORK_DIR"/{models,plots,logs,checkpoints}

# Copy all necessary files to working directory
echo "Copying files to working directory..."
cp "$ORIG_DIR"/*.py "$WORK_DIR/" 2>/dev/null || true
cp "$ORIG_DIR"/requirements.txt "$WORK_DIR/" 2>/dev/null || true

# Copy parent directory files (multi_objective_sac.py is one level up)
if [ -f "$ORIG_DIR/../multi_objective_sac.py" ]; then
    cp "$ORIG_DIR/../multi_objective_sac.py" "$WORK_DIR/"
    echo "Copied multi_objective_sac.py from parent directory"
fi

# Copy EnergyNet module if it exists
if [ -d "$ORIG_DIR/../EnergyNetMoISO" ]; then
    cp -r "$ORIG_DIR/../EnergyNetMoISO" "$WORK_DIR/"
    echo "Copied EnergyNetMoISO directory"
fi

# Copy any existing best model
if [ -f "$ORIG_DIR/../../best_model.zip" ]; then
    cp "$ORIG_DIR/../../best_model.zip" "$WORK_DIR/"
    echo "Copied best_model.zip"
fi

# Change to working directory
cd "$WORK_DIR"
echo "Changed to working directory: $(pwd)"
echo "Files in working directory:"
ls -la

# SLURM environment setup
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM job manager"
    
    if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        echo "Set OMP_NUM_THREADS to $SLURM_CPUS_PER_TASK"
    else
        export OMP_NUM_THREADS=4
    fi
    
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "GPU devices: $CUDA_VISIBLE_DEVICES"
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
        fi
    else
        echo "No GPU devices allocated"
    fi
else
    export OMP_NUM_THREADS=4
fi

# Training parameters
EPISODES=${1:-100000}  # Default 100k episodes, or use first argument
LEARNING_RATE=${2:-0.0003}  # Default learning rate
BATCH_SIZE=${3:-256}  # Default batch size

echo "Training Parameters:"
echo "  Episodes: $EPISODES"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Create Python arguments array
PYTHON_ARGS=(
    "--episodes" "$EPISODES"
    "--lr" "$LEARNING_RATE"
    "--batch_size" "$BATCH_SIZE"
    "--save_dir" "./models"
    "--log_dir" "./logs"
    "--checkpoint_dir" "./checkpoints"
    "--save_interval" "10000"
    "--eval_interval" "5000"
    "--verbose"
)

# Add GPU argument if available
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    PYTHON_ARGS+=("--device" "cuda")
else
    PYTHON_ARGS+=("--device" "cpu")
fi

echo "Running EnergyNet training..."
echo "Command: $PYTHON_CMD train_energynet.py ${PYTHON_ARGS[*]}"
echo ""

# Start the training
$PYTHON_CMD train_energynet.py "${PYTHON_ARGS[@]}"

# Store the exit code
EXIT_CODE=$?

# Copy results back to original directory
echo "Copying results back to original directory..."

# Copy models
if [ -d "$WORK_DIR/models" ] && [ -n "$(ls -A "$WORK_DIR/models" 2>/dev/null)" ]; then
    mkdir -p "$ORIG_DIR/models"
    cp -r "$WORK_DIR/models"/* "$ORIG_DIR/models/" 2>/dev/null || echo "Warning: Could not copy models back"
    echo "Models copied to: $ORIG_DIR/models/"
fi

# Copy plots
if [ -d "$WORK_DIR/plots" ] && [ -n "$(ls -A "$WORK_DIR/plots" 2>/dev/null)" ]; then
    mkdir -p "$ORIG_DIR/plots"
    cp -r "$WORK_DIR/plots"/* "$ORIG_DIR/plots/" 2>/dev/null || echo "Warning: Could not copy plots back"
    echo "Plots copied to: $ORIG_DIR/plots/"
fi

# Copy logs
if [ -d "$WORK_DIR/logs" ] && [ -n "$(ls -A "$WORK_DIR/logs" 2>/dev/null)" ]; then
    mkdir -p "$ORIG_DIR/logs"
    cp -r "$WORK_DIR/logs"/* "$ORIG_DIR/logs/" 2>/dev/null || echo "Warning: Could not copy logs back"
    echo "Logs copied to: $ORIG_DIR/logs/"
fi

# Copy tensorboard runs
if [ -d "$WORK_DIR/runs" ] && [ -n "$(ls -A "$WORK_DIR/runs" 2>/dev/null)" ]; then
    mkdir -p "$ORIG_DIR/runs"
    cp -r "$WORK_DIR/runs"/* "$ORIG_DIR/runs/" 2>/dev/null || echo "Warning: Could not copy runs back"
    echo "TensorBoard logs copied to: $ORIG_DIR/runs/"
fi

# Copy checkpoints
if [ -d "$WORK_DIR/checkpoints" ] && [ -n "$(ls -A "$WORK_DIR/checkpoints" 2>/dev/null)" ]; then
    mkdir -p "$ORIG_DIR/checkpoints"
    cp -r "$WORK_DIR/checkpoints"/* "$ORIG_DIR/checkpoints/" 2>/dev/null || echo "Warning: Could not copy checkpoints back"
    echo "Checkpoints copied to: $ORIG_DIR/checkpoints/"
fi

# Copy best model to root if it exists
if [ -f "$WORK_DIR/best_model.zip" ]; then
    cp "$WORK_DIR/best_model.zip" "$ORIG_DIR/../../best_model.zip" 2>/dev/null || echo "Warning: Could not copy best model back"
    echo "Best model copied to: $ORIG_DIR/../../best_model.zip"
fi

# Check if the training completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "EnergyNet training completed successfully!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Training Episodes: $EPISODES"
    echo "Results copied back to: $ORIG_DIR"
    echo ""
    echo "Results should be in:"
    echo "  - $ORIG_DIR/models/         (trained model files)"
    echo "  - $ORIG_DIR/plots/          (training plots and analysis)" 
    echo "  - $ORIG_DIR/logs/           (training logs)"
    echo "  - $ORIG_DIR/runs/           (tensorboard logs)"
    echo "  - $ORIG_DIR/checkpoints/    (training checkpoints)"
    echo ""
    echo "To view results:"
    echo "  ls -la $ORIG_DIR/models/ $ORIG_DIR/plots/ $ORIG_DIR/logs/"
    echo ""
    echo "To view tensorboard logs:"
    echo "  tensorboard --logdir $ORIG_DIR/runs/ --host 0.0.0.0 --port 6006"
    echo ""
    echo "To test the trained model:"
    echo "  sbatch -c 2 --gres=gpu:1 ./run_test_trained_model.sh"
    echo ""
    echo "SLURM output files:"
    echo "  - slurm_energynet_${SLURM_JOB_ID}.out (standard output)"
    echo "  - slurm_energynet_${SLURM_JOB_ID}.err (error output)"
else
    echo ""
    echo "=========================================="
    echo "EnergyNet training failed with errors!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Exit code: $EXIT_CODE"
    echo "Check slurm_energynet_${SLURM_JOB_ID}.err for error details"
    echo ""
    echo "Working directory contents at failure:"
    ls -la "$WORK_DIR"
    echo ""
    echo "Common SLURM troubleshooting:"
    echo "  1. Check job status: squeue -j $SLURM_JOB_ID"
    echo "  2. Check available modules: module avail"
    echo "  3. Check Python packages: $PYTHON_CMD -c 'import torch, numpy, gymnasium'"
    echo "  4. Check EnergyNet import: $PYTHON_CMD -c 'from EnergyNetMoISO.MoISOEnv import MoISOEnv'"
    echo "  5. Check file permissions in: $ORIG_DIR"
    echo ""
    echo "If training was interrupted, you can resume from checkpoint:"
    echo "  Look for checkpoint files in: $ORIG_DIR/checkpoints/"
fi

# Cleanup temporary directory if we created it
if [[ "$WORK_DIR" == "/tmp/energynet_training_${USER}_"* ]]; then
    echo "Cleaning up temporary directory: $WORK_DIR"
    rm -rf "$WORK_DIR"
fi

exit $EXIT_CODE
