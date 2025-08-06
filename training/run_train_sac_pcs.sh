#!/bin/bash
#SBATCH --job-name=sac_pcs_training
#SBATCH --output=slurm_sac_pcs_%j.out
#SBATCH --error=slurm_sac_pcs_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Usage examples:
#   # Basic training (no ISO model)
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs.sh
#
#   # Training with custom experiment name and timesteps
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs.sh baseline_pcs 500000
#
#   # Training with trained ISO model
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs.sh pcs_with_iso 500000 3e-4 256 /path/to/iso_model.pth
#
#   # With optimizations (use environment variables)
#   ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs.sh optimized_pcs 1000000
#
#   # Full optimization suite
#   ENABLE_ALL_OPTIMIZATIONS=true \
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs.sh full_opt_pcs 1000000 3e-4 256 /path/to/iso_model.pth 42

echo "=========================================="
echo "SAC PCS Training (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Parse command line arguments
EXPERIMENT_NAME=${1:-"sac_pcs_training"}
TOTAL_TIMESTEPS=${2:-500000}
LEARNING_RATE=${3:-3e-4}
# Check if 4th argument is a file path (ISO model) or a number (batch size)
if [ -n "$4" ] && [ -f "$4" ]; then
    # 4th argument is a file path, so it's the ISO model
    ISO_MODEL_PATH="$4"
    BATCH_SIZE=${5:-256}
    SEED=${6:-""}
elif [ -n "$4" ] && [[ "$4" =~ ^[0-9]+$ ]]; then
    # 4th argument is a number, so it's the batch size
    BATCH_SIZE="$4"
    ISO_MODEL_PATH=${5:-""}
    SEED=${6:-""}
else
    # Default values
    BATCH_SIZE=${4:-256}
    ISO_MODEL_PATH=${5:-""}
    SEED=${6:-""}
fi

echo "Arguments:"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  ISO Model Path: $ISO_MODEL_PATH"
echo "  Seed: $SEED"
echo ""

# Environment variable optimizations
echo "Environment variables for optimizations:"
echo "  ENABLE_LR_ANNEALING: ${ENABLE_LR_ANNEALING:-false}"
echo "  LR_ANNEALING_TYPE: ${LR_ANNEALING_TYPE:-cosine}"
echo "  LR_ANNEALING_STEPS: ${LR_ANNEALING_STEPS:-auto}"
echo "  LR_MIN_FACTOR: ${LR_MIN_FACTOR:-0.1}"
echo "  LR_DECAY_RATE: ${LR_DECAY_RATE:-0.95}"
echo "  ENABLE_REWARD_SCALING: ${ENABLE_REWARD_SCALING:-false}"
echo "  REWARD_SCALE_EPSILON: ${REWARD_SCALE_EPSILON:-1e-4}"
echo "  ENABLE_ORTHOGONAL_INIT: ${ENABLE_ORTHOGONAL_INIT:-false}"
echo "  ORTHOGONAL_GAIN: ${ORTHOGONAL_GAIN:-1.0}"
echo "  ACTOR_ORTHOGONAL_GAIN: ${ACTOR_ORTHOGONAL_GAIN:-0.01}"
echo "  CRITIC_ORTHOGONAL_GAIN: ${CRITIC_ORTHOGONAL_GAIN:-1.0}"
echo "  ENABLE_VALUE_CLIPPING: ${ENABLE_VALUE_CLIPPING:-false}"
echo "  VALUE_CLIP_RANGE: ${VALUE_CLIP_RANGE:-200.0}"
echo "  ENABLE_ALL_OPTIMIZATIONS: ${ENABLE_ALL_OPTIMIZATIONS:-false}"
echo ""

# Handle ENABLE_ALL_OPTIMIZATIONS
if [ "$ENABLE_ALL_OPTIMIZATIONS" = "true" ]; then
    echo "Enabling all optimizations with good defaults..."
    export ENABLE_LR_ANNEALING=true
    export LR_ANNEALING_TYPE=${LR_ANNEALING_TYPE:-cosine}
    export LR_MIN_FACTOR=${LR_MIN_FACTOR:-0.05}
    export ENABLE_REWARD_SCALING=true
    export ENABLE_ORTHOGONAL_INIT=true
    export ENABLE_VALUE_CLIPPING=true
    export VALUE_CLIP_RANGE=${VALUE_CLIP_RANGE:-150.0}
    echo "All optimizations enabled."
fi

# Determine Python command
if command -v python3 > /dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python > /dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found!"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version)"
echo ""

# Check for GPU
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU available: $CUDA_VISIBLE_DEVICES"
    $PYTHON_CMD -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
else
    echo "No GPU detected - training will be slow!"
fi
echo ""

# Get current directory and set up paths
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # Use the directory from which the job was submitted
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
    echo "Running under SLURM - using submit directory: $SLURM_SUBMIT_DIR"
else
    # Use the script's directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

ORIGINAL_DIR="$PWD"
cd "$SCRIPT_DIR"

echo "Script directory: $SCRIPT_DIR"
echo "Current working directory: $(pwd)"
echo ""

# Check for required modules
echo "Checking imports..."

# First test basic imports
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "   ✓ PyTorch available"
else
    echo "   ✗ PyTorch import failed"
    exit 1
fi

if $PYTHON_CMD -c "import numpy" 2>/dev/null; then
    echo "   ✓ NumPy available"
else
    echo "   ✗ NumPy import failed"
    exit 1
fi

# Test environment import
if $PYTHON_CMD -c "from energy_net.env.pcs_unit_v0 import PCSUnitEnv" 2>/dev/null; then
    echo "   ✓ PCSUnitEnv import successful"
else
    echo "   ✗ PCSUnitEnv import failed"
    echo "   Make sure energy_net is properly installed"
    exit 1
fi

# Test agent import with proper path setup
if $PYTHON_CMD -c "import sys; sys.path.insert(0, '..'); from EnergyNetMoISO.pcs_models.sac_pcs_agent import SACPCSAgent" 2>/dev/null; then
    echo "   ✓ SACPCSAgent import successful"
else
    echo "   ✗ SACPCSAgent import failed"
    echo "   Debugging path setup..."
    $PYTHON_CMD -c "import sys, os; print(f'Current dir: {os.getcwd()}'); print(f'Python path: {sys.path[:3]}'); sys.path.insert(0, '..'); print(f'After adding parent: {sys.path[:3]}'); import EnergyNetMoISO.pcs_models.sac_pcs_agent"
    exit 1
fi

echo ""

# SLURM environment setup
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM job manager"
    
    if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        echo "Set OMP_NUM_THREADS to $SLURM_CPUS_PER_TASK"
    else
        export OMP_NUM_THREADS=4
    fi
    
    # Set memory limits
    if [ -n "$SLURM_MEM_PER_NODE" ]; then
        echo "Memory limit: ${SLURM_MEM_PER_NODE}MB"
    fi
    
    # Set time limit
    if [ -n "$SLURM_JOB_END_TIME" ]; then
        echo "Job end time: $SLURM_JOB_END_TIME"
    fi
else
    export OMP_NUM_THREADS=4
    echo "Not running under SLURM (OMP_NUM_THREADS=4)"
fi

# Build training command
TRAINING_CMD="$PYTHON_CMD sac_pcs_training.py"
TRAINING_CMD="$TRAINING_CMD --experiment-name $EXPERIMENT_NAME"
TRAINING_CMD="$TRAINING_CMD --total-timesteps $TOTAL_TIMESTEPS"
TRAINING_CMD="$TRAINING_CMD --actor-lr $LEARNING_RATE"
TRAINING_CMD="$TRAINING_CMD --critic-lr $LEARNING_RATE"
TRAINING_CMD="$TRAINING_CMD --alpha-lr $LEARNING_RATE"
TRAINING_CMD="$TRAINING_CMD --batch-size $BATCH_SIZE"
TRAINING_CMD="$TRAINING_CMD --verbose"

# Add ISO model path if provided
if [ -n "$ISO_MODEL_PATH" ] && [ "$ISO_MODEL_PATH" != "" ]; then
    echo "Using ISO model: $ISO_MODEL_PATH"
    TRAINING_CMD="$TRAINING_CMD --trained-iso-model-path $ISO_MODEL_PATH"
fi

# Add seed if provided
if [ -n "$SEED" ] && [ "$SEED" != "" ]; then
    echo "Using seed: $SEED"
    TRAINING_CMD="$TRAINING_CMD --seed $SEED"
fi

# Add optimization flags based on environment variables
if [ "$ENABLE_LR_ANNEALING" = "true" ]; then
    TRAINING_CMD="$TRAINING_CMD --use-lr-annealing"
    TRAINING_CMD="$TRAINING_CMD --lr-annealing-type ${LR_ANNEALING_TYPE:-cosine}"
    if [ -n "$LR_ANNEALING_STEPS" ] && [ "$LR_ANNEALING_STEPS" != "auto" ]; then
        TRAINING_CMD="$TRAINING_CMD --lr-annealing-steps $LR_ANNEALING_STEPS"
    fi
    TRAINING_CMD="$TRAINING_CMD --lr-min-factor ${LR_MIN_FACTOR:-0.1}"
    TRAINING_CMD="$TRAINING_CMD --lr-decay-rate ${LR_DECAY_RATE:-0.95}"
fi

if [ "$ENABLE_REWARD_SCALING" = "true" ]; then
    TRAINING_CMD="$TRAINING_CMD --use-reward-scaling"
    TRAINING_CMD="$TRAINING_CMD --reward-scale-epsilon ${REWARD_SCALE_EPSILON:-1e-4}"
fi

if [ "$ENABLE_ORTHOGONAL_INIT" = "true" ]; then
    TRAINING_CMD="$TRAINING_CMD --use-orthogonal-init"
    TRAINING_CMD="$TRAINING_CMD --orthogonal-gain ${ORTHOGONAL_GAIN:-1.0}"
    TRAINING_CMD="$TRAINING_CMD --actor-orthogonal-gain ${ACTOR_ORTHOGONAL_GAIN:-0.01}"
    TRAINING_CMD="$TRAINING_CMD --critic-orthogonal-gain ${CRITIC_ORTHOGONAL_GAIN:-1.0}"
fi

if [ "$ENABLE_VALUE_CLIPPING" = "true" ]; then
    TRAINING_CMD="$TRAINING_CMD --use-value-clipping"
    TRAINING_CMD="$TRAINING_CMD --value-clip-range ${VALUE_CLIP_RANGE:-200.0}"
fi

echo "=========================================="
echo "Final training command:"
echo "$TRAINING_CMD"
echo "=========================================="
echo ""

# Set up experiment directory
EXPERIMENT_DIR="pcs_experiments"
mkdir -p "$EXPERIMENT_DIR"
echo "Experiment directory: $EXPERIMENT_DIR"

# Record start time
START_TIME=$(date)
echo "Training started at: $START_TIME"
echo ""

# Run training
echo "Starting SAC PCS training..."
eval $TRAINING_CMD

TRAINING_EXIT_CODE=$?

# Record end time
END_TIME=$(date)
echo ""
echo "Training finished at: $END_TIME"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    
    # Show results summary if available
    if [ -f "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_results.json" ]; then
        echo ""
        echo "Results summary:"
        if command -v jq > /dev/null 2>&1; then
            echo "  Final evaluation: $(jq -r '.final_evaluation.mean' "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_results.json") ± $(jq -r '.final_evaluation.std' "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_results.json")"
            echo "  Best evaluation: $(jq -r '.best_evaluation' "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_results.json")"
            echo "  Total episodes: $(jq -r '.total_episodes' "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_results.json")"
            echo "  Training time: $(jq -r '.training_time' "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_results.json") seconds"
        else
            echo "  Results file: ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}_results.json"
        fi
    fi
    
    # Show model locations
    echo ""
    echo "Model files:"
    if [ -f "${EXPERIMENT_DIR}/models/${EXPERIMENT_NAME}_final.pth" ]; then
        echo "  Final model: ${EXPERIMENT_DIR}/models/${EXPERIMENT_NAME}_final.pth"
    fi
    if [ -f "${EXPERIMENT_DIR}/models/${EXPERIMENT_NAME}_best.pth" ]; then
        echo "  Best model: ${EXPERIMENT_DIR}/models/${EXPERIMENT_NAME}_best.pth"
    fi
    
    # TensorBoard logs
    if [ -d "${EXPERIMENT_DIR}/logs/${EXPERIMENT_NAME}" ]; then
        echo "  TensorBoard logs: ${EXPERIMENT_DIR}/logs/${EXPERIMENT_NAME}"
        echo "  View with: tensorboard --logdir ${EXPERIMENT_DIR}/logs/${EXPERIMENT_NAME}"
    fi
    
else
    echo "✗ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check the error output above for details."
fi

echo ""
echo "=========================================="
echo "SLURM Job Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $START_TIME"
echo "End time: $END_TIME"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "=========================================="

exit $TRAINING_EXIT_CODE
