#!/bin/bash

#SBATCH --job-name=mo_sac_gpu_test
#SBATCH --output=slurm_gpu_test_%j.out
#SBATCH --error=slurm_gpu_test_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Quick GPU test for MO-SAC (should complete in under 10 minutes)
# This tests if GPU is being utilized properly before running long training

echo "=========================================="
echo "MO-SAC Quick GPU Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Get the original script location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ORIG_DIR="$SLURM_SUBMIT_DIR"
else
    ORIG_DIR="$(dirname "${BASH_SOURCE[0]}")"
    ORIG_DIR="$(cd "$ORIG_DIR" && pwd)"
fi

echo "Original directory: $ORIG_DIR"

# Find Python
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

# Check GPU status
echo ""
echo "GPU Status Check:"
if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    if command -v nvidia-smi &> /dev/null; then
        echo "Available GPUs:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
        echo ""
        echo "Detailed GPU info:"
        nvidia-smi
    fi
else
    echo "WARNING: No GPU allocated (CUDA_VISIBLE_DEVICES is empty)"
    echo "This test requires a GPU to be meaningful"
fi

# Set up working directory
WORK_DIR="/tmp/mo_sac_gpu_test_${USER}_$$"
echo ""
echo "Creating working directory: $WORK_DIR"
mkdir -p "$WORK_DIR"

# Copy files
echo "Copying files..."
cp "$ORIG_DIR"/*.py "$WORK_DIR/" 2>/dev/null || true
cp "$ORIG_DIR/../multi_objective_sac.py" "$WORK_DIR/" 2>/dev/null || true

# Copy EnergyNet if available
if [ -d "$ORIG_DIR/../../EnergyNetMoISO" ]; then
    cp -r "$ORIG_DIR/../../EnergyNetMoISO" "$WORK_DIR/"
fi

cd "$WORK_DIR"
echo "Working in: $(pwd)"

# Set environment variables for optimal GPU usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=1  # For debugging

echo ""
echo "Running GPU test..."
echo "========================================"

# Run the quick GPU test
$PYTHON_CMD test_gpu_quick.py

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GPU test PASSED!"
    echo "GPU is available and working properly."
    echo "You can now run the full training with confidence."
    echo ""
    echo "To run full training:"
    echo "  sbatch run_test_mo_sac_v2.sh"
    echo "  sbatch run_test_mo_sac_v2.sh CartPole"
else
    echo "❌ GPU test FAILED!"
    echo "There are issues with GPU utilization."
    echo "Check the output above for problems."
    echo ""
    echo "Common issues:"
    echo "  - CUDA not available in environment"
    echo "  - PyTorch not compiled with CUDA support"
    echo "  - GPU not allocated by SLURM"
    echo "  - Memory issues"
fi

# Cleanup
echo ""
echo "Cleaning up: $WORK_DIR"
rm -rf "$WORK_DIR"

echo "=========================================="
echo "GPU test completed!"
echo "Check slurm_gpu_test_${SLURM_JOB_ID}.out for full output"
echo "=========================================="

exit $EXIT_CODE
