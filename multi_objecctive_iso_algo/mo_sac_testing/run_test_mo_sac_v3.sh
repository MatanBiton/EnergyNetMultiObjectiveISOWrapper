#!/bin/bash

#SBATCH --job-name=mo_sac_test_v3_optimized
#SBATCH --output=slurm_test_v3_%j.out
#SBATCH --error=slurm_test_v3_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Multi-Objective SAC Testing Script v3 - Optimized for GPU
# This version includes optimizations for GPU utilization and faster training
#
# Usage: 
#   sbatch run_test_mo_sac_v3.sh                    # Run quick tests (30-60 minutes)
#   sbatch run_test_mo_sac_v3.sh CartPole           # Test specific environment quickly
#   sbatch run_test_mo_sac_v3.sh test_imports       # Test imports only
#   sbatch run_test_mo_sac_v3.sh full               # Run comprehensive tests (slower)

echo "=========================================="
echo "Multi-Objective SAC Testing Script v3 (Optimized)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
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

# Enhanced GPU verification
echo ""
echo "Enhanced GPU Status Check:"
if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
        echo ""
        echo "Detailed GPU info:"
        nvidia-smi
        echo ""
    fi
    
    # Verify PyTorch can see the GPU
    echo "Verifying PyTorch GPU access..."
    $PYTHON_CMD -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'  Compute capability: {props.major}.{props.minor}')
    
    # Test tensor creation on GPU
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)
    print(f'✓ Successfully created tensor on GPU: {x.device}')
    
    # Test matrix multiplication performance
    import time
    start = time.time()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f'✓ GPU matrix multiplication test: {gpu_time:.4f}s')
    
    # Clear GPU memory
    del x, y
    torch.cuda.empty_cache()
    print(f'✓ GPU memory cleared')
else:
    print('✗ CUDA not available to PyTorch')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "❌ GPU verification failed!"
        echo "PyTorch cannot access the GPU properly."
        exit 1
    fi
    
    echo "✅ GPU verification successful!"
    
else
    echo "❌ No GPU devices allocated"
    echo "This script requires a GPU for optimal performance"
    echo "Use: sbatch --gres=gpu:1 $0"
    exit 1
fi

# Set up working directory
if [ -n "$SLURM_TMPDIR" ]; then
    WORK_DIR="$SLURM_TMPDIR/mo_sac_work"
else
    WORK_DIR="/tmp/mo_sac_work_${USER}_$$"
fi

echo ""
echo "Creating working directory: $WORK_DIR"
mkdir -p "$WORK_DIR"/{models,plots,logs}

# Copy files
echo "Copying files..."
cp "$ORIG_DIR"/*.py "$WORK_DIR/" 2>/dev/null || true
cp "$ORIG_DIR"/requirements.txt "$WORK_DIR/" 2>/dev/null || true

# Copy parent directory files
if [ -f "$ORIG_DIR/../multi_objective_sac.py" ]; then
    cp "$ORIG_DIR/../multi_objective_sac.py" "$WORK_DIR/"
    echo "Copied multi_objective_sac.py"
fi

# Copy EnergyNet module
if [ -d "$ORIG_DIR/../../EnergyNetMoISO" ]; then
    cp -r "$ORIG_DIR/../../EnergyNetMoISO" "$WORK_DIR/"
    echo "Copied EnergyNetMoISO directory"
fi

cd "$WORK_DIR"
echo "Working in: $(pwd)"

# Set environment variables for optimal GPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=0  # Allow async CUDA operations for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Prevent memory fragmentation

echo ""
echo "Environment variables set for GPU optimization:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Determine test type
if [ $# -eq 0 ]; then
    TEST_TYPE="quick"
    echo ""
    echo "Running QUICK tests (optimized for speed)..."
    echo "This should complete in 30-60 minutes with GPU acceleration"
    echo ""
    
    # Use the quick test script
    $PYTHON_CMD test_mo_sac_quick.py
    
else
    ENV_NAME=$1
    
    case $ENV_NAME in
        test_imports)
            echo "Running import tests only..."
            $PYTHON_CMD test_imports.py
            EXIT_CODE=$?
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ Import tests completed successfully!"
            else
                echo "❌ Import tests failed"
            fi
            
            # Cleanup and exit early
            rm -rf "$WORK_DIR"
            exit $EXIT_CODE
            ;;
            
        full)
            echo "Running COMPREHENSIVE tests (full training)..."
            echo "This will take 2-4 hours even with GPU acceleration"
            $PYTHON_CMD test_mo_sac.py
            ;;
            
        *)
            # Test specific environment
            if [[ "$ENV_NAME" == *CartPole* || "$ENV_NAME" == *cartpole* || "$ENV_NAME" == *cart* ]]; then
                FULL_ENV_NAME="MultiObjectiveContinuousCartPole-v0"
            elif [[ "$ENV_NAME" == *Mountain* || "$ENV_NAME" == *mountain* || "$ENV_NAME" == *car* ]]; then
                FULL_ENV_NAME="MultiObjectiveMountainCarContinuous-v0"
            elif [[ "$ENV_NAME" == *Pendulum* || "$ENV_NAME" == *pendulum* ]]; then
                FULL_ENV_NAME="MultiObjectivePendulum-v0"
            elif [[ "$ENV_NAME" == *Lunar* || "$ENV_NAME" == *lunar* || "$ENV_NAME" == *lander* ]]; then
                FULL_ENV_NAME="MultiObjectiveLunarLander-v0"
            else
                echo "Testing environment containing: $ENV_NAME"
                FULL_ENV_NAME="$ENV_NAME"
            fi
            
            echo "Running quick test on: $FULL_ENV_NAME"
            $PYTHON_CMD test_mo_sac_quick.py "$FULL_ENV_NAME"
            ;;
    esac
fi

EXIT_CODE=$?

# Copy results back
echo ""
echo "Copying results back to original directory..."
for dir in models plots logs runs; do
    if [ -d "$WORK_DIR/$dir" ] && [ -n "$(ls -A "$WORK_DIR/$dir" 2>/dev/null)" ]; then
        cp -r "$WORK_DIR/$dir" "$ORIG_DIR/" 2>/dev/null || echo "Warning: Could not copy $dir back"
        echo "✓ Copied $dir back to $ORIG_DIR"
    fi
done

# Final GPU status
echo ""
echo "Final GPU status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
fi

# Results summary
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Testing completed successfully!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Results copied back to: $ORIG_DIR"
    echo ""
    echo "Generated files:"
    for dir in models plots logs runs; do
        if [ -d "$ORIG_DIR/$dir" ]; then
            file_count=$(find "$ORIG_DIR/$dir" -type f | wc -l)
            echo "  📁 $dir/ ($file_count files)"
        fi
    done
    echo ""
    echo "View results:"
    echo "  ls -la $ORIG_DIR/{models,plots,logs,runs}/"
    echo ""
    echo "View tensorboard logs:"
    echo "  tensorboard --logdir $ORIG_DIR/runs/ --host 0.0.0.0 --port 6006"
    
else
    echo ""
    echo "=========================================="
    echo "❌ Testing failed with errors!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Check the error output above for details."
    echo "Common issues:"
    echo "  - GPU memory exhausted"
    echo "  - Environment setup problems" 
    echo "  - Package dependency issues"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Run: sbatch run_gpu_test.sh"
    echo "  2. Check: squeue -j $SLURM_JOB_ID"
    echo "  3. View: cat slurm_test_v3_${SLURM_JOB_ID}.err"
fi

# Cleanup
echo ""
echo "Cleaning up temporary directory: $WORK_DIR"
rm -rf "$WORK_DIR"

echo "=========================================="
echo "Script completed!"
echo "=========================================="

exit $EXIT_CODE
