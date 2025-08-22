#!/bin/bash

# Simple test script for continue training (without SLURM)
# This script helps test the continue training functionality locally

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo -e "MO-SAC Continue Training Test Script"
echo -e "===========================================${NC}"

# Check if model path is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Model path is required as first argument${NC}"
    echo ""
    echo "Usage: $0 <model_path> [timesteps] [experiment_name]"
    echo ""
    echo "Examples:"
    echo "  $0 ../best_model.zip"
    echo "  $0 ../best_model.zip 50000"
    echo "  $0 ../best_model.zip 50000 test_continue"
    echo ""
    echo "Environment variables:"
    echo "  TRAINED_PCS_MODEL      - Path to trained PCS model (optional)"
    echo "  USE_DISPATCH_ACTION    - Enable dispatch action (true/false)"
    echo "  DISPATCH_STRATEGY      - Dispatch strategy (PROPORTIONAL/UNIFORM/etc)"
    echo "  ENABLE_LR_ANNEALING    - Enable learning rate annealing (true/false)"
    echo "  RANDOM_SEED            - Random seed for reproducibility"
    exit 1
fi

MODEL_PATH="$1"
TIMESTEPS=${2:-50000}  # Default: 50k timesteps for testing
EXPERIMENT_NAME=${3:-"test_continue"}

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

echo -e "${GREEN}Found model file: $MODEL_PATH${NC}"
echo -e "${BLUE}Script directory: $SCRIPT_DIR${NC}"

# Check if Python script exists
if [ ! -f "$SCRIPT_DIR/continue_train_energynet.py" ]; then
    echo -e "${RED}Error: continue_train_energynet.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Check if we're in conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${GREEN}✓ Running in conda environment: $CONDA_DEFAULT_ENV${NC}"
    if [ "$CONDA_DEFAULT_ENV" != "IsoMOEnergyNet" ]; then
        echo -e "${YELLOW}⚠ Warning: Expected 'IsoMOEnergyNet' environment, but running in '$CONDA_DEFAULT_ENV'${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Warning: Not running in a conda environment${NC}"
    echo "It's recommended to activate the IsoMOEnergyNet environment:"
    echo "  conda activate IsoMOEnergyNet"
fi

# Try different Python commands
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        echo -e "${GREEN}Found Python: $PYTHON_CMD ($(which $cmd))${NC}"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: No Python interpreter found${NC}"
    exit 1
fi

echo "Python version: $($PYTHON_CMD --version)"

# Test imports
echo -e "${BLUE}Testing Python imports...${NC}"
if $PYTHON_CMD -c "import torch, numpy, gymnasium" 2>/dev/null; then
    echo -e "${GREEN}✓ Basic dependencies available${NC}"
else
    echo -e "${RED}✗ Basic dependencies missing${NC}"
    echo "Please install required packages:"
    echo "  pip install torch numpy gymnasium"
    exit 1
fi

# Test specific imports
if $PYTHON_CMD -c "import sys; sys.path.insert(0, '..'); import multi_objective_sac" 2>/dev/null; then
    echo -e "${GREEN}✓ multi_objective_sac import successful${NC}"
else
    echo -e "${RED}✗ multi_objective_sac import failed${NC}"
    echo "Make sure multi_objective_sac.py is in the parent directory"
    exit 1
fi

# Check GPU availability
CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo -e "${GREEN}✓ CUDA available for GPU training${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not available, will use CPU training${NC}"
fi

# Environment configuration
echo -e "${BLUE}Environment Configuration:${NC}"
echo "  Use Dispatch Action: ${USE_DISPATCH_ACTION:-false}"
echo "  Dispatch Strategy: ${DISPATCH_STRATEGY:-PROPORTIONAL}"
if [ -n "$TRAINED_PCS_MODEL" ]; then
    echo "  Trained PCS Model: $TRAINED_PCS_MODEL"
    if [ ! -f "$TRAINED_PCS_MODEL" ]; then
        echo -e "${YELLOW}  ⚠ Warning: PCS model file not found${NC}"
    else
        echo -e "${GREEN}  ✓ PCS model file found${NC}"
    fi
else
    echo "  Trained PCS Model: None (using default)"
fi

echo -e "${BLUE}Training Parameters:${NC}"
echo "  Model Path: $MODEL_PATH"
echo "  Additional Timesteps: $TIMESTEPS"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Random Seed: ${RANDOM_SEED:-Not set}"

# Create Python arguments
PYTHON_ARGS=(
    "--model-path" "$MODEL_PATH"
    "--total-timesteps" "$TIMESTEPS"
    "--save-dir" "."
    "--experiment-name" "test_continue_${EXPERIMENT_NAME}"
    "--eval-freq" "2000"  # More frequent for testing
    "--save-freq" "5000"  # More frequent for testing
    "--verbose"
)

# Add optional parameters
if [ -n "$RANDOM_SEED" ]; then
    PYTHON_ARGS+=("--seed" "$RANDOM_SEED")
fi

if [ "$USE_DISPATCH_ACTION" = "true" ]; then
    PYTHON_ARGS+=("--use-dispatch-action")
fi

PYTHON_ARGS+=("--dispatch-strategy" "${DISPATCH_STRATEGY:-PROPORTIONAL}")

if [ -n "$TRAINED_PCS_MODEL" ]; then
    PYTHON_ARGS+=("--trained-pcs-model" "$TRAINED_PCS_MODEL")
fi

if [ "$ENABLE_LR_ANNEALING" = "true" ]; then
    PYTHON_ARGS+=("--use-lr-annealing")
fi

echo -e "${BLUE}Starting continue training...${NC}"
echo "Command: $PYTHON_CMD continue_train_energynet.py ${PYTHON_ARGS[*]}"
echo ""

# Run the training
cd "$SCRIPT_DIR"
$PYTHON_CMD continue_train_energynet.py "${PYTHON_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo -e "Continue training completed successfully!"
    echo -e "==========================================${NC}"
    echo ""
    echo "Results should be in:"
    echo "  - Config files: test_continue_${EXPERIMENT_NAME}_*_config.json"
    echo "  - Results files: test_continue_${EXPERIMENT_NAME}_*_results.json"
    echo "  - Models: models/test_continue_${EXPERIMENT_NAME}_*"
    echo "  - Logs: logs/test_continue_${EXPERIMENT_NAME}_*"
    echo ""
    echo "To view files:"
    echo "  ls -la test_continue_${EXPERIMENT_NAME}_*"
    echo "  ls -la models/ logs/"
else
    echo ""
    echo -e "${RED}=========================================="
    echo -e "Continue training failed!"
    echo -e "==========================================${NC}"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  1. Model file format compatibility"
    echo "  2. Missing dependencies"
    echo "  3. Insufficient memory"
    echo "  4. Wrong conda environment"
fi

exit $EXIT_CODE
