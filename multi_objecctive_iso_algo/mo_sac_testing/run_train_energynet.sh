#!/bin/bash

#SBATCH --job-name=mo_sac_energynet
#SBATCH --output=slurm_train_%j.out
#SBATCH --error=slurm_train_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Multi-Objective SAC Training Script for EnergyNet MoISO Environment (SLURM)
# This script trains the MO-SAC algorithm on the EnergyNet Multi-Objective ISO environment
#
# Usage: 
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet.sh                    # Default parameters
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet.sh --quick-test       # Quick test run
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet.sh --cost-priority    # Prioritize cost
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet.sh --stability-priority # Prioritize stability

echo "=========================================="
echo "MO-SAC EnergyNet Training Script (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time Limit: $SLURM_TIMELIMIT"
echo "=========================================="

# Load necessary modules (uncomment and modify as needed for your cluster)
# module load python/3.8
# module load cuda/11.8
# module load gcc/9.3.0
# module load pytorch/1.12.0

# Set the working directory to the script location
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

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

# Check Python version and key packages
echo "Python version: $($PYTHON_CMD --version)"
echo "Checking key packages..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not available"

# Check if required packages are installed (commented out for cluster - assume pre-installed)
# echo "Checking dependencies..."
# $PYTHON_CMD -c "import torch, numpy, gymnasium, matplotlib, tensorboard" 2>/dev/null
# if [ $? -ne 0 ]; then
#     echo "Warning: Some required packages might be missing."
#     echo "On clusters, packages should be pre-installed or loaded via modules."
#     echo "Contact your system administrator if packages are missing."
# fi

# Create necessary directories with proper permissions
echo "Creating output directories..."
mkdir -p energynet_experiments
mkdir -p energynet_experiments/models
mkdir -p energynet_experiments/logs
mkdir -p energynet_experiments/plots
chmod 755 energynet_experiments energynet_experiments/models energynet_experiments/logs energynet_experiments/plots 2>/dev/null || true

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
        # Check GPU availability
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
        fi
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
fi

# Parse command line arguments for quick configurations
QUICK_TEST=false
COST_PRIORITY=false
STABILITY_PRIORITY=false

for arg in "$@"; do
    case $arg in
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --cost-priority)
            COST_PRIORITY=true
            shift
            ;;
        --stability-priority)
            STABILITY_PRIORITY=true
            shift
            ;;
    esac
done

# Set base experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_EXP_NAME="mo_sac_energynet_${TIMESTAMP}"

# Configure training parameters based on flags
if [ "$QUICK_TEST" = true ]; then
    echo "Running quick test configuration..."
    TOTAL_TIMESTEPS=50000
    LEARNING_STARTS=2000
    EVAL_FREQ=10000
    SAVE_FREQ=25000
    EXP_NAME="${BASE_EXP_NAME}_quick"
    
elif [ "$COST_PRIORITY" = true ]; then
    echo "Running cost-priority configuration..."
    TOTAL_TIMESTEPS=1000000
    LEARNING_STARTS=10000
    EVAL_FREQ=20000
    SAVE_FREQ=100000
    WEIGHTS="0.8 0.2"  # Prioritize cost reduction
    EXP_NAME="${BASE_EXP_NAME}_cost_priority"
    
elif [ "$STABILITY_PRIORITY" = true ]; then
    echo "Running stability-priority configuration..."
    TOTAL_TIMESTEPS=1000000
    LEARNING_STARTS=10000
    EVAL_FREQ=20000
    SAVE_FREQ=100000
    WEIGHTS="0.3 0.7"  # Prioritize stability
    EXP_NAME="${BASE_EXP_NAME}_stability_priority"
    
else
    echo "Running default configuration..."
    TOTAL_TIMESTEPS=1000000
    LEARNING_STARTS=10000
    EVAL_FREQ=20000
    SAVE_FREQ=100000
    WEIGHTS="0.6 0.4"  # Default: slightly prefer cost
    EXP_NAME="${BASE_EXP_NAME}_default"
fi

echo "Training configuration:"
echo "  Experiment name: $EXP_NAME"
echo "  Total timesteps: $TOTAL_TIMESTEPS"
echo "  Weights: $WEIGHTS"
echo ""

# Run the training with all configurable parameters
$PYTHON_CMD train_energynet.py \
    --experiment-name "$EXP_NAME" \
    --save-dir "energynet_experiments" \
    \
    --total-timesteps $TOTAL_TIMESTEPS \
    --learning-starts $LEARNING_STARTS \
    --eval-freq $EVAL_FREQ \
    --save-freq $SAVE_FREQ \
    \
    --weights $WEIGHTS \
    \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --alpha-lr 3e-4 \
    \
    --gamma 0.99 \
    --tau 0.005 \
    \
    --buffer-size 1000000 \
    --batch-size 256 \
    \
    --dispatch-strategy "PROPORTIONAL" \
    \
    --verbose

# Store the exit code
EXIT_CODE=$?

# Report results
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Results saved in: energynet_experiments/"
    echo ""
    echo "Files created:"
    echo "  - energynet_experiments/${EXP_NAME}_config.json    (configuration)"
    echo "  - energynet_experiments/${EXP_NAME}_results.json   (training results)"
    echo "  - energynet_experiments/models/${EXP_NAME}_final.pth (trained model)"
    echo "  - energynet_experiments/logs/                      (tensorboard logs)"
    echo ""
    echo "SLURM output files:"
    echo "  - slurm_train_${SLURM_JOB_ID}.out (standard output)"
    echo "  - slurm_train_${SLURM_JOB_ID}.err (error output)"
    echo ""
    echo "To view training progress (after job completion):"
    echo "  tensorboard --logdir energynet_experiments/logs/ --host 0.0.0.0 --port 6006"
    echo ""
    echo "To download results from cluster:"
    echo "  scp -r username@cluster:path/to/energynet_experiments ."
    echo ""
    echo "Parameter Tuning Tips:"
    echo "  - If learning is slow: increase learning rates (--actor-lr, --critic-lr)"
    echo "  - If training is unstable: decrease learning rates or increase --tau"
    echo "  - For different objectives: adjust --weights [cost_weight stability_weight]"
    echo "  - For longer training: increase --total-timesteps and SLURM time limit"
    echo "  - For more evaluation: decrease --eval-freq"
    
else
    echo ""
    echo "=========================================="
    echo "Training failed with errors!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Exit code: $EXIT_CODE"
    echo "Check slurm_train_${SLURM_JOB_ID}.err for error details"
    echo ""
    echo "Common SLURM troubleshooting:"
    echo "  1. Check job status: squeue -j $SLURM_JOB_ID"
    echo "  2. Check job details: scontrol show job $SLURM_JOB_ID"
    echo "  3. Check node resources: sinfo -N -l"
    echo "  4. Check available GPUs: nvidia-smi"
    echo "  5. Verify module loads and Python environment"
    echo "  6. Check disk space: df -h"
    echo "  7. Check memory usage: free -h"
    echo ""
    echo "Try running with --quick-test flag for faster debugging"
fi

exit $EXIT_CODE

# Additional analysis suggestions for post-processing
echo ""
echo "Post-Job Analysis Commands:"
echo "==========================="
echo "1. Check job efficiency:"
echo "   seff $SLURM_JOB_ID"
echo ""
echo "2. View job accounting:"
echo "   sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,CPUTime,AveCPU"
echo ""
echo "3. Monitor training progress:"
echo "   tensorboard --logdir energynet_experiments/logs/ --host 0.0.0.0"
echo ""
echo "4. Compare different runs:"
echo "   python -c \""
echo "   import json"
echo "   with open('energynet_experiments/${EXP_NAME}_results.json') as f:"
echo "       results = json.load(f)"
echo "   print('Final performance:', results['final_evaluation'])"
echo "   \""
