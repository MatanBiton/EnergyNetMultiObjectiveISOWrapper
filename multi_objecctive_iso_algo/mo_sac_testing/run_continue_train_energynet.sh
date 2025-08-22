#!/bin/bash

#SBATCH --job-name=mo_sac_continue_energynet
#SBATCH --output=slurm_continue_energynet_%j.out
#SBATCH --error=slurm_continue_energynet_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Multi-Objective SAC EnergyNet Continue Training Script for SLURM
# This script loads an existing trained MO-SAC agent and continues training
# Supports all optimization parameters and trained PCS model integration
#
# Usage Examples:
#   # Basic continue training with required model path
#   sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/model.pth
#
#   # With additional timesteps and experiment name
#   sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/model.pth 500000 continued_experiment
#
#   # With trained PCS model
#   TRAINED_PCS_MODEL=/path/to/pcs_model.pth \
#   sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/model.pth 500000 with_pcs
#
#   # With environment configuration
#   USE_DISPATCH_ACTION=true DISPATCH_STRATEGY=UNIFORM TRAINED_PCS_MODEL=/path/to/pcs.pth \
#   sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/model.pth
#
#   # With optimization overrides and seed
#   ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine RANDOM_SEED=123 \
#   sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/model.pth 1000000 optimized_continue

echo "=========================================="
echo "Multi-Objective SAC EnergyNet Continue Training (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Error: Model path is required as first argument"
    echo ""
    echo "Usage: $0 <model_path> [timesteps] [experiment_name]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/trained_model.pth"
    echo "  $0 /path/to/trained_model.pth 500000"
    echo "  $0 /path/to/trained_model.pth 500000 continued_exp"
    echo ""
    echo "Environment variables:"
    echo "  TRAINED_PCS_MODEL      - Path to trained PCS model (optional)"
    echo "  USE_DISPATCH_ACTION    - Enable dispatch action (true/false)"
    echo "  DISPATCH_STRATEGY      - Dispatch strategy (PROPORTIONAL/UNIFORM/etc)"
    echo "  ENABLE_LR_ANNEALING    - Enable learning rate annealing (true/false)"
    echo "  LR_ANNEALING_TYPE      - Type of annealing (cosine/linear/exponential)"
    echo "  ENABLE_REWARD_SCALING  - Enable reward scaling (true/false)"
    echo "  ENABLE_VALUE_CLIPPING  - Enable value clipping (true/false)"
    echo "  RANDOM_SEED            - Random seed for reproducibility"
    exit 1
fi

MODEL_PATH="$1"

# Get the original script location (where the files actually are)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ORIG_DIR="$SLURM_SUBMIT_DIR"
else
    ORIG_DIR="$(dirname "${BASH_SOURCE[0]}")"
    ORIG_DIR="$(cd "$ORIG_DIR" && pwd)"
fi

echo "Original directory (where files are): $ORIG_DIR"
echo "Model path: $MODEL_PATH"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check if Python files exist in original directory
if [ ! -f "$ORIG_DIR/continue_train_energynet.py" ]; then
    echo "Error: continue_train_energynet.py not found in $ORIG_DIR"
    echo "Files in original directory:"
    ls -la "$ORIG_DIR"
    exit 1
fi

# Load necessary modules (uncomment as needed for your cluster)
# module load python/3.8
# module load cuda/11.8
# module load gcc/9.3.0

# Load CUDA module if available (common on SLURM clusters)
if command -v module &> /dev/null; then
    echo "Loading CUDA modules..."
    module load cuda/12.6 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || echo "No CUDA module found"
    module list 2>&1 | grep -i cuda || echo "No CUDA modules loaded"
fi

# Set CUDA environment variables
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export CUDA_PATH="/usr/local/cuda"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    echo "✓ Set CUDA environment variables"
elif [ -d "/opt/cuda" ]; then
    export CUDA_HOME="/opt/cuda"
    export CUDA_PATH="/opt/cuda"
    export PATH="/opt/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
    echo "✓ Set CUDA environment variables (opt)"
else
    echo "⚠ Warning: CUDA installation not found in standard locations"
fi

# Activate conda environment
echo "Activating conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate IsoMOEnergyNet
    echo "✓ Activated conda environment: IsoMOEnergyNet"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate IsoMOEnergyNet
    echo "✓ Activated conda environment: IsoMOEnergyNet"
else
    echo "⚠ Warning: Conda not found, using system Python"
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
echo "Python location: $(which $PYTHON_CMD)"

# Check and fix CUDA availability
echo ""
echo "Checking CUDA availability..."
echo "CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "CUDA_PATH: ${CUDA_PATH:-Not set}"

# Check if nvidia-smi is available (indicates GPU drivers are present)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
    
    # Check PyTorch CUDA availability
    echo "Checking PyTorch CUDA support..."
    TORCH_CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Unknown")
    
    echo "PyTorch version: $TORCH_VERSION"
    echo "PyTorch CUDA available: $TORCH_CUDA_AVAILABLE"
    
    if [ "$TORCH_CUDA_AVAILABLE" = "False" ]; then
        echo "⚠ Warning: PyTorch reports CUDA not available despite GPU presence"
        echo "Attempting to reinstall PyTorch with CUDA support..."
        
        # Try to reinstall PyTorch with CUDA support
        $PYTHON_CMD -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --quiet || {
            echo "Failed to reinstall PyTorch with CUDA 12.6, trying CUDA 11.8..."
            $PYTHON_CMD -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet || {
                echo "Failed to reinstall PyTorch with CUDA support"
                echo "Continuing with CPU training..."
            }
        }
        
        # Check again after reinstall
        TORCH_CUDA_AVAILABLE_AFTER=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
        if [ "$TORCH_CUDA_AVAILABLE_AFTER" = "True" ]; then
            echo "✓ Successfully fixed PyTorch CUDA support"
        else
            echo "⚠ PyTorch CUDA still not available, training will use CPU"
        fi
    else
        echo "✓ PyTorch CUDA support is working"
    fi
else
    echo "No GPU drivers detected (nvidia-smi not available)"
    echo "Training will use CPU"
fi
echo ""

# Set up working directory - use SLURM_TMPDIR if available, otherwise create in /tmp
if [ -n "$SLURM_TMPDIR" ]; then
    WORK_DIR="$SLURM_TMPDIR/energynet_continue_training"
else
    WORK_DIR="/tmp/energynet_continue_training_${USER}_$$"
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
else
    echo "Warning: multi_objective_sac.py not found at $ORIG_DIR/../multi_objective_sac.py"
fi

# Copy EnergyNet module if it exists
if [ -d "$ORIG_DIR/../../EnergyNetMoISO" ]; then
    cp -r "$ORIG_DIR/../../EnergyNetMoISO" "$WORK_DIR/"
    echo "Copied EnergyNetMoISO directory from $ORIG_DIR/../../EnergyNetMoISO"
elif [ -d "$ORIG_DIR/../EnergyNetMoISO" ]; then
    cp -r "$ORIG_DIR/../EnergyNetMoISO" "$WORK_DIR/"
    echo "Copied EnergyNetMoISO directory from $ORIG_DIR/../EnergyNetMoISO"
else
    echo "Warning: EnergyNetMoISO directory not found in expected locations"
    echo "Searched in: $ORIG_DIR/../../EnergyNetMoISO and $ORIG_DIR/../EnergyNetMoISO"
fi

# Change to working directory
cd "$WORK_DIR"
echo "Changed to working directory: $(pwd)"
echo "Files in working directory:"
ls -la

# Check Python imports before running training
echo ""
echo "Testing Python imports..."
echo "Python path will include:"
echo "  - Current directory: $(pwd)"
echo "  - Parent directories will be added by the script"

# Test basic packages first
echo "Testing basic dependencies..."
if $PYTHON_CMD -c "import torch, numpy, gymnasium, matplotlib" 2>/dev/null; then
    echo "✓ Basic dependencies (torch, numpy, gymnasium, matplotlib) available"
else
    echo "✗ Basic dependencies missing"
    echo "Available packages:"
    $PYTHON_CMD -c "import sys; print([p for p in sys.path if 'site-packages' in p][:3])"
    exit 1
fi

# Test critical imports
echo "Testing multi_objective_sac import..."
if $PYTHON_CMD -c "import multi_objective_sac" 2>/dev/null; then
    echo "✓ multi_objective_sac import successful"
else
    echo "✗ multi_objective_sac import failed"
    echo "Available Python files:"
    ls -la *.py
    exit 1
fi

echo "Testing EnergyNetMoISO import..."
if $PYTHON_CMD -c "from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv" 2>/dev/null; then
    echo "✓ EnergyNetMoISO import successful"
else
    echo "✗ EnergyNetMoISO import failed"
    echo "EnergyNetMoISO directory contents:"
    if [ -d "EnergyNetMoISO" ]; then
        ls -la EnergyNetMoISO/
    else
        echo "EnergyNetMoISO directory not found!"
    fi
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
    
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "GPU devices: $CUDA_VISIBLE_DEVICES"
        
        # Final check for CUDA availability before training
        FINAL_CUDA_CHECK=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
        if [ "$FINAL_CUDA_CHECK" = "True" ]; then
            echo "✓ GPU training enabled"
        else
            echo "⚠ GPU devices allocated but PyTorch CUDA not available - using CPU"
        fi
    else
        echo "No GPU devices allocated"
    fi
else
    export OMP_NUM_THREADS=4
fi

# Training parameters
TIMESTEPS=${2:-500000}  # Second argument: additional timesteps (default: 500k)
EXPERIMENT_NAME=${3:-"mo_sac_continued"}  # Third argument: experiment name
SEED=${4:-${RANDOM_SEED:-}}  # Fourth argument: random seed (default: random or from env var)

# Environment configuration
USE_DISPATCH_ACTION=${USE_DISPATCH_ACTION:-false}
DISPATCH_STRATEGY=${DISPATCH_STRATEGY:-PROPORTIONAL}
TRAINED_PCS_MODEL=${TRAINED_PCS_MODEL:-}

# Optimization parameters (controlled via environment variables)
ENABLE_LR_ANNEALING=${ENABLE_LR_ANNEALING:-false}
LR_ANNEALING_TYPE=${LR_ANNEALING_TYPE:-cosine}
LR_ANNEALING_STEPS=${LR_ANNEALING_STEPS:-}
LR_MIN_FACTOR=${LR_MIN_FACTOR:-0.1}
LR_DECAY_RATE=${LR_DECAY_RATE:-0.95}
ENABLE_REWARD_SCALING=${ENABLE_REWARD_SCALING:-false}
REWARD_SCALE_EPSILON=${REWARD_SCALE_EPSILON:-1e-4}
ENABLE_VALUE_CLIPPING=${ENABLE_VALUE_CLIPPING:-false}
VALUE_CLIP_RANGE=${VALUE_CLIP_RANGE:-200.0}

echo "Continue Training Parameters:"
echo "  Model Path: $MODEL_PATH"
echo "  Additional Timesteps: $TIMESTEPS"
echo "  Experiment Name: $EXPERIMENT_NAME"
if [ -n "$SEED" ]; then
    echo "  Random Seed: $SEED"
else
    echo "  Random Seed: Not specified (will use random)"
fi
echo ""
echo "Environment Configuration:"
echo "  Use Dispatch Action: $USE_DISPATCH_ACTION"
echo "  Dispatch Strategy: $DISPATCH_STRATEGY"
if [ -n "$TRAINED_PCS_MODEL" ]; then
    echo "  Trained PCS Model: $TRAINED_PCS_MODEL"
    # Check if PCS model exists
    if [ ! -f "$TRAINED_PCS_MODEL" ]; then
        echo "  ⚠ Warning: PCS model file not found: $TRAINED_PCS_MODEL"
    else
        echo "  ✓ PCS model file found"
    fi
else
    echo "  Trained PCS Model: None (using default PCS)"
fi
echo ""
echo "Optimization Parameter Overrides:"
echo "  LR Annealing: $ENABLE_LR_ANNEALING"
if [ "$ENABLE_LR_ANNEALING" = "true" ]; then
    echo "    Type: $LR_ANNEALING_TYPE"
    echo "    Min Factor: $LR_MIN_FACTOR"
    [ -n "$LR_ANNEALING_STEPS" ] && echo "    Steps: $LR_ANNEALING_STEPS"
    [ "$LR_ANNEALING_TYPE" = "exponential" ] && echo "    Decay Rate: $LR_DECAY_RATE"
fi
echo "  Reward Scaling: $ENABLE_REWARD_SCALING"
[ "$ENABLE_REWARD_SCALING" = "true" ] && echo "    Epsilon: $REWARD_SCALE_EPSILON"
echo "  Value Clipping: $ENABLE_VALUE_CLIPPING"
[ "$ENABLE_VALUE_CLIPPING" = "true" ] && echo "    Clip Range: $VALUE_CLIP_RANGE"
echo ""

# Create Python arguments array
PYTHON_ARGS=(
    "--model-path" "$MODEL_PATH"
    "--total-timesteps" "$TIMESTEPS"
    "--save-dir" "."
    "--experiment-name" "continue_train_energynet_${EXPERIMENT_NAME}"
    "--eval-freq" "5000"
    "--save-freq" "10000"
    "--verbose"
)

# Add seed if specified
if [ -n "$SEED" ]; then
    PYTHON_ARGS+=("--seed" "$SEED")
fi

# Add environment configuration arguments
if [ "$USE_DISPATCH_ACTION" = "true" ]; then
    PYTHON_ARGS+=("--use-dispatch-action")
fi

PYTHON_ARGS+=("--dispatch-strategy" "$DISPATCH_STRATEGY")

if [ -n "$TRAINED_PCS_MODEL" ]; then
    PYTHON_ARGS+=("--trained-pcs-model" "$TRAINED_PCS_MODEL")
fi

# Add optimization parameter overrides
if [ "$ENABLE_LR_ANNEALING" = "true" ]; then
    PYTHON_ARGS+=("--use-lr-annealing")
    PYTHON_ARGS+=("--lr-annealing-type" "$LR_ANNEALING_TYPE")
    PYTHON_ARGS+=("--lr-min-factor" "$LR_MIN_FACTOR")
    [ -n "$LR_ANNEALING_STEPS" ] && PYTHON_ARGS+=("--lr-annealing-steps" "$LR_ANNEALING_STEPS")
    [ "$LR_ANNEALING_TYPE" = "exponential" ] && PYTHON_ARGS+=("--lr-decay-rate" "$LR_DECAY_RATE")
fi

if [ "$ENABLE_REWARD_SCALING" = "true" ]; then
    PYTHON_ARGS+=("--use-reward-scaling")
    PYTHON_ARGS+=("--reward-scale-epsilon" "$REWARD_SCALE_EPSILON")
fi

if [ "$ENABLE_VALUE_CLIPPING" = "true" ]; then
    PYTHON_ARGS+=("--use-value-clipping")
    PYTHON_ARGS+=("--value-clip-range" "$VALUE_CLIP_RANGE")
fi

echo "Running EnergyNet continue training..."
echo "Command: $PYTHON_CMD continue_train_energynet.py ${PYTHON_ARGS[*]}"
echo ""

# Start the training
$PYTHON_CMD continue_train_energynet.py "${PYTHON_ARGS[@]}"

# Store the exit code
EXIT_CODE=$?

# Copy results back to original directory
echo "Copying results back to original directory..."

# Copy config and results files
cp "$WORK_DIR"/*_config.json "$ORIG_DIR/" 2>/dev/null || echo "Warning: Could not copy config files back"
cp "$WORK_DIR"/*_results.json "$ORIG_DIR/" 2>/dev/null || echo "Warning: Could not copy results files back"

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

# Copy logs (includes tensorboard logs)
if [ -d "$WORK_DIR/logs" ] && [ -n "$(ls -A "$WORK_DIR/logs" 2>/dev/null)" ]; then
    mkdir -p "$ORIG_DIR/logs"
    cp -r "$WORK_DIR/logs"/* "$ORIG_DIR/logs/" 2>/dev/null || echo "Warning: Could not copy logs back"
    echo "Logs (including TensorBoard) copied to: $ORIG_DIR/logs/"
fi

# Copy checkpoints
if [ -d "$WORK_DIR/checkpoints" ] && [ -n "$(ls -A "$WORK_DIR/checkpoints" 2>/dev/null)" ]; then
    mkdir -p "$ORIG_DIR/checkpoints"
    cp -r "$WORK_DIR/checkpoints"/* "$ORIG_DIR/checkpoints/" 2>/dev/null || echo "Warning: Could not copy checkpoints back"
    echo "Checkpoints copied to: $ORIG_DIR/checkpoints/"
fi

# Copy final model to root if it exists
if [ -f "$WORK_DIR/models"/*final.pth ]; then
    cp "$WORK_DIR/models"/*final.pth "$ORIG_DIR/../../continue_train_energynet_${EXPERIMENT_NAME}_final_model.pth" 2>/dev/null || echo "Warning: Could not copy final model back"
    echo "Final model copied to root directory"
fi

# Check if the training completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "EnergyNet continue training completed successfully!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Original Model: $MODEL_PATH"
    echo "Experiment Name: continue_train_energynet_${EXPERIMENT_NAME}"
    echo "Additional Training Timesteps: $TIMESTEPS"
    echo "Results copied back to: $ORIG_DIR"
    echo ""
    echo "Results should be in:"
    echo "  - $ORIG_DIR/                (config and results files: continue_train_energynet_${EXPERIMENT_NAME}_*)"
    echo "  - $ORIG_DIR/models/         (continued trained model files)"
    echo "  - $ORIG_DIR/logs/           (tensorboard logs: continue_train_energynet_${EXPERIMENT_NAME})"
    echo "  - $ORIG_DIR/plots/          (training plots and analysis)" 
    echo "  - $ORIG_DIR/checkpoints/    (training checkpoints)"
    echo ""
    echo "To view results:"
    echo "  ./view_results.sh"
    echo ""
    echo "To view tensorboard logs:"
    echo "  tensorboard --logdir $ORIG_DIR/logs/ --host 0.0.0.0 --port 6006"
    echo ""
    echo "SLURM output files:"
    echo "  - slurm_continue_energynet_${SLURM_JOB_ID}.out (standard output)"
    echo "  - slurm_continue_energynet_${SLURM_JOB_ID}.err (error output)"
else
    echo ""
    echo "=========================================="
    echo "EnergyNet continue training failed with errors!"
    echo "=========================================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Exit code: $EXIT_CODE"
    echo "Check slurm_continue_energynet_${SLURM_JOB_ID}.err for error details"
    echo ""
    echo "Working directory contents at failure:"
    ls -la "$WORK_DIR"
    echo ""
    echo "Common troubleshooting:"
    echo "  1. Check if model path exists: $MODEL_PATH"
    echo "  2. Check if model file is compatible with current code"
    echo "  3. Check Python package compatibility"
    echo "  4. Check available GPU memory"
    echo "  5. Try with CPU-only training (remove --gres=gpu:1)"
fi

# Cleanup temporary directory if we created it
if [[ "$WORK_DIR" == "/tmp/energynet_continue_training_${USER}_"* ]]; then
    echo "Cleaning up temporary directory: $WORK_DIR"
    rm -rf "$WORK_DIR"
fi

exit $EXIT_CODE
