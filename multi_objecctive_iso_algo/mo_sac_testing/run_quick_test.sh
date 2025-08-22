#!/bin/bash

#SBATCH --job-name=mo_sac_quick_test
#SBATCH --output=slurm_quick_%j.out
#SBATCH --error=slurm_quick_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Multi-Objective SAC Quick Testing Script for SLURM
# This script runs a short test to verify everything works
#
# Usage: 
#   sbatch -c 4 --gres=gpu:1 ./run_quick_test.sh                    # Quick test
#   sbatch -c 4 --gres=gpu:1 ./run_quick_test.sh test_imports       # Test imports only

echo "=========================================="
echo "Multi-Objective SAC Quick Test (SLURM)"
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
if [ ! -f "$ORIG_DIR/test_mo_sac.py" ]; then
    echo "Error: test_mo_sac.py not found in $ORIG_DIR"
    echo "Files in original directory:"
    ls -la "$ORIG_DIR"
    exit 1
fi

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
    WORK_DIR="$SLURM_TMPDIR/mo_sac_quick"
else
    WORK_DIR="/tmp/mo_sac_quick_${USER}_$$"
fi

echo "Creating working directory: $WORK_DIR"
mkdir -p "$WORK_DIR"/{models,plots,logs}

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
else
    echo "Warning: EnergyNetMoISO directory not found at $ORIG_DIR/../EnergyNetMoISO"
fi

# Copy any other required directories
if [ -d "$ORIG_DIR/../../EnergyNetMoISO" ]; then
    cp -r "$ORIG_DIR/../../EnergyNetMoISO" "$WORK_DIR/"
    echo "Copied EnergyNetMoISO directory from root level"
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
    
    if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "" ]; then
        echo "GPU devices: $CUDA_VISIBLE_DEVICES"
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
        fi
        echo "✓ GPU acceleration available"
    else
        echo "No GPU devices allocated - will use CPU"
        echo "Note: Training will be significantly slower on CPU"
    fi
else
    export OMP_NUM_THREADS=4
fi

# Test selection and execution
if [ $# -eq 0 ]; then
    echo "Running quick test (CartPole environment only)..."
    echo "This will take about 5-10 minutes with reduced timesteps"
    echo ""
    
    # Run quick test with reduced timesteps
    $PYTHON_CMD test_mo_sac_quick.py "MultiObjectiveContinuousCartPole-v0"
    
elif [ "$1" = "test_imports" ]; then
    echo "Running import tests only..."
    echo ""
    $PYTHON_CMD test_imports.py
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Import tests completed successfully!"
    else
        echo "Import tests failed - check the output above for missing packages"
    fi
    
    # Cleanup and exit early for import test
    if [[ "$WORK_DIR" == "/tmp/mo_sac_quick_${USER}_"* ]]; then
        echo "Cleaning up temporary directory: $WORK_DIR"
        rm -rf "$WORK_DIR"
    fi
    exit $EXIT_CODE
else
    ENV_NAME=$1
    echo "Running quick test on environment containing: $ENV_NAME"
    echo ""
    
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
            echo "  - test_imports (test imports only)"
            exit 1
            ;;
    esac
    
    echo "Testing environment: $FULL_ENV_NAME"
    $PYTHON_CMD test_mo_sac_quick.py "$FULL_ENV_NAME"
fi

# Store the exit code
EXIT_CODE=$?

# Copy results back to original directory
echo "Copying results back to original directory..."
if [ -d "$WORK_DIR/models" ] && [ -n "$(ls -A "$WORK_DIR/models" 2>/dev/null)" ]; then
    cp -r "$WORK_DIR/models" "$ORIG_DIR/" 2>/dev/null || echo "Warning: Could not copy models back"
fi

if [ -d "$WORK_DIR/plots" ] && [ -n "$(ls -A "$WORK_DIR/plots" 2>/dev/null)" ]; then
    cp -r "$WORK_DIR/plots" "$ORIG_DIR/" 2>/dev/null || echo "Warning: Could not copy plots back"
fi

if [ -d "$WORK_DIR/runs" ] && [ -n "$(ls -A "$WORK_DIR/runs" 2>/dev/null)" ]; then
    cp -r "$WORK_DIR/runs" "$ORIG_DIR/" 2>/dev/null || echo "Warning: Could not copy runs back"
fi

# Check if the script completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Quick test completed successfully!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Results copied back to: $ORIG_DIR"
    echo ""
    echo "Results should be in:"
    echo "  - $ORIG_DIR/models/     (trained model files)"
    echo "  - $ORIG_DIR/plots/      (training plots and analysis)" 
    echo "  - $ORIG_DIR/runs/       (tensorboard logs)"
    echo ""
    echo "To run full tests (will take 1-2 hours):"
    echo "  sbatch --time=04:00:00 -c 4 --gres=gpu:1 ./run_test_mo_sac_v2.sh"
    echo ""
    echo "SLURM output files:"
    echo "  - slurm_quick_${SLURM_JOB_ID}.out (standard output)"
    echo "  - slurm_quick_${SLURM_JOB_ID}.err (error output)"
else
    echo ""
    echo "=========================================="
    echo "Quick test failed with errors!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Exit code: $EXIT_CODE"
    echo "Check slurm_quick_${SLURM_JOB_ID}.err for error details"
    echo ""
    echo "Working directory contents at failure:"
    ls -la "$WORK_DIR"
    echo ""
    echo "Try running import test first:"
    echo "  sbatch ./run_quick_test.sh test_imports"
fi

# Cleanup temporary directory if we created it
if [[ "$WORK_DIR" == "/tmp/mo_sac_quick_${USER}_"* ]]; then
    echo "Cleaning up temporary directory: $WORK_DIR"
    rm -rf "$WORK_DIR"
fi

exit $EXIT_CODE
