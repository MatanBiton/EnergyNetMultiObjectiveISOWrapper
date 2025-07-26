#!/bin/bash

#SBATCH --job-name=mo_sac_test
#SBATCH --output=slurm_test_%j.out
#SBATCH --error=slurm_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Multi-Objective SAC Testing Script for SLURM
# This script runs comprehensive tests on all available test environments
# 
# Usage: 
#   sbatch -c 4 --gres=gpu:1 ./run_test_mo_sac.sh                    # Run comprehensive tests
#   sbatch -c 4 --gres=gpu:1 ./run_test_mo_sac.sh CartPole           # Test specific environment
#   sbatch -c 4 --gres=gpu:1 ./run_test_mo_sac.sh Pendulum           # Test specific environment

echo "=========================================="
echo "Multi-Objective SAC Testing Script (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Load necessary modules (uncomment as needed for your cluster)
# module load python/3.8
# module load cuda/11.8
# module load gcc/9.3.0

# Set the working directory to the script location
SCRIPT_DIR="$(dirname "$0")"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"  # Get absolute path
echo "Script directory: $SCRIPT_DIR"

# Make sure we can find the Python files
if [ ! -f "$SCRIPT_DIR/test_mo_sac.py" ]; then
    echo "Error: test_mo_sac.py not found in $SCRIPT_DIR"
    echo "Directory contents:"
    ls -la "$SCRIPT_DIR"
    exit 1
fi

# Try different Python commands (clusters may have different setups)
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
    echo "Tried: python3, python, python3.8, python3.9, python3.10"
    exit 1
fi

# Check Python version
echo "Python version: $($PYTHON_CMD --version)"

# Check if required packages are installed (commented out for cluster - assume pre-installed)
# echo "Checking dependencies..."
# $PYTHON_CMD -c "import torch, numpy, gymnasium, matplotlib" 2>/dev/null
# if [ $? -ne 0 ]; then
#     echo "Warning: Some required packages might be missing."
#     echo "On clusters, packages should be pre-installed or loaded via modules."
#     echo "Contact your system administrator if packages are missing."
# fi

# Create necessary directories with proper permissions in a writable location
echo "Creating output directories..."

# Use SLURM_TMPDIR if available (typically writable), otherwise use current directory
if [ -n "$SLURM_TMPDIR" ]; then
    WORK_DIR="$SLURM_TMPDIR/mo_sac_results"
    echo "Using SLURM temporary directory: $WORK_DIR"
    mkdir -p "$WORK_DIR"/{models,plots,logs}
    cd "$WORK_DIR"
    
    # Copy necessary files to working directory
    cp "$SCRIPT_DIR"/*.py "$WORK_DIR/" 2>/dev/null || true
    
    # Set up symbolic links back to original location for results
    ln -sf "$WORK_DIR/models" "$SCRIPT_DIR/models" 2>/dev/null || true
    ln -sf "$WORK_DIR/plots" "$SCRIPT_DIR/plots" 2>/dev/null || true
    ln -sf "$WORK_DIR/logs" "$SCRIPT_DIR/logs" 2>/dev/null || true
    
else
    # Try to create in current directory, with fallback to /tmp
    if ! mkdir -p models plots logs 2>/dev/null; then
        echo "Cannot create directories in current location, using /tmp"
        WORK_DIR="/tmp/mo_sac_results_${USER}_$$"
        mkdir -p "$WORK_DIR"/{models,plots,logs}
        cd "$WORK_DIR"
        
        # Copy necessary files
        cp "$SCRIPT_DIR"/*.py "$WORK_DIR/" 2>/dev/null || true
    else
        WORK_DIR="$(pwd)"
    fi
fi

echo "Working directory: $WORK_DIR"
echo "Contents: $(ls -la)"

# SLURM environment setup
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM job manager"
    
    # Use SLURM allocated resources
    if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        echo "Set OMP_NUM_THREADS to $SLURM_CPUS_PER_TASK"
    else
        export OMP_NUM_THREADS=4
    fi
    
    # GPU setup - SLURM should set CUDA_VISIBLE_DEVICES automatically
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "GPU devices: $CUDA_VISIBLE_DEVICES"
    else
        echo "No GPU devices allocated"
    fi
    
    # Set memory limits if available
    if [ -n "$SLURM_MEM_PER_NODE" ]; then
        echo "Memory limit: $SLURM_MEM_PER_NODE MB"
    fi
else
    echo "Not running under SLURM"
    # Fallback for non-SLURM environments
    export OMP_NUM_THREADS=4
    # Don't set CUDA_VISIBLE_DEVICES - let system handle it
fi

# Environment selection
if [ $# -eq 0 ]; then
    echo "Running comprehensive tests on all environments..."
    echo "This will take a while (30-60 minutes depending on hardware)"
    echo ""
    
    # Run comprehensive tests with default parameters
    $PYTHON_CMD "$SCRIPT_DIR/test_mo_sac.py"
    
else
    # Run test on specific environment (partial name matching)
    ENV_NAME=$1
    echo "Running test on environment containing: $ENV_NAME"
    echo ""
    
    # Available environments:
    # - MultiObjectiveContinuousCartPole-v0
    # - MultiObjectiveMountainCarContinuous-v0  
    # - MultiObjectivePendulum-v0
    # - MultiObjectiveLunarLander-v0
    
    # Find matching environment
    case $ENV_NAME in
        *CartPole*|*cartpole*|*cart*)
            FULL_ENV_NAME="MultiObjectiveContinuousCartPole-v0"
            ;;
        *Mountain*|*mountain*|*car*)
            FULL_ENV_NAME="MultiObjectiveMountainCarContinuous-v0"
            ;;
        *Pendulum*|*pendulum*)
            FULL_ENV_NAME="MultiObjectivePendulum-v0"
            ;;
        *Lunar*|*lunar*|*lander*)
            FULL_ENV_NAME="MultiObjectiveLunarLander-v0"
            ;;
        *)
            echo "Unknown environment: $ENV_NAME"
            echo "Available environments:"
            echo "  - CartPole (MultiObjectiveContinuousCartPole-v0)"
            echo "  - Mountain (MultiObjectiveMountainCarContinuous-v0)"
            echo "  - Pendulum (MultiObjectivePendulum-v0)"
            echo "  - Lunar (MultiObjectiveLunarLander-v0)"
            exit 1
            ;;
    esac
    
    echo "Testing environment: $FULL_ENV_NAME"
    $PYTHON_CMD "$SCRIPT_DIR/test_mo_sac.py" "$FULL_ENV_NAME"
fi

# Store the exit code
EXIT_CODE=$?

# Copy results back to original directory if we used a temporary location
if [ "$WORK_DIR" != "$SCRIPT_DIR" ] && [ -n "$WORK_DIR" ]; then
    echo "Copying results back to original directory..."
    cp -r "$WORK_DIR"/{models,plots,logs} "$SCRIPT_DIR/" 2>/dev/null || true
    echo "Results copied to: $SCRIPT_DIR"
fi

# Check if the script completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Testing completed successfully!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Results saved in:"
    echo "  - models/     (trained model files)"
    echo "  - plots/      (training plots and analysis)"
    echo "  - runs/       (tensorboard logs)"
    echo ""
    echo "To view results after job completion:"
    echo "  ls -la models/ plots/ runs/"
    echo ""
    echo "To view tensorboard logs:"
    echo "  tensorboard --logdir runs/ --host 0.0.0.0 --port 6006"
    echo ""
    echo "SLURM output files:"
    echo "  - slurm_test_${SLURM_JOB_ID}.out (standard output)"
    echo "  - slurm_test_${SLURM_JOB_ID}.err (error output)"
else
    echo ""
    echo "=========================================="
    echo "Testing failed with errors!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Exit code: $EXIT_CODE"
    echo "Check slurm_test_${SLURM_JOB_ID}.err for error details"
    echo ""
    echo "Common SLURM troubleshooting:"
    echo "  1. Check job status: squeue -j $SLURM_JOB_ID"
    echo "  2. Check job details: scontrol show job $SLURM_JOB_ID"
    echo "  3. Check node resources: sinfo -N -l"
    echo "  4. Check available GPUs: nvidia-smi"
    echo "  5. Verify module loads and Python environment"
fi

exit $EXIT_CODE
