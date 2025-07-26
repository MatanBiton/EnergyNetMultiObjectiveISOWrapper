#!/bin/bash

# Multi-Objective SAC Testing Script
# This script runs comprehensive tests on all available test environments
# 
# Usage: 
#   ./run_test_mo_sac.sh                    # Run comprehensive tests on all environments
#   ./run_test_mo_sac.sh CartPole           # Test specific environment (partial name match)
#   ./run_test_mo_sac.sh Pendulum           # Test specific environment (partial name match)

echo "=========================================="
echo "Multi-Objective SAC Testing Script"
echo "=========================================="

# Set the working directory to the script location
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
# echo "Checking dependencies..."
# python -c "import torch, numpy, gymnasium, matplotlib" 2>/dev/null
# if [ $? -ne 0 ]; then
#     echo "Warning: Some required packages might be missing. Installing..."
#     pip install -r requirements.txt
# fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p models
mkdir -p plots
mkdir -p logs

# Set environment variables for better performance (optional)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available, comment out to use CPU
export OMP_NUM_THREADS=4       # Number of CPU threads for PyTorch

# Environment selection
if [ $# -eq 0 ]; then
    echo "Running comprehensive tests on all environments..."
    echo "This will take a while (30-60 minutes depending on hardware)"
    echo ""
    
    # Run comprehensive tests with default parameters
    python test_mo_sac.py
    
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
    python test_mo_sac.py "$FULL_ENV_NAME"
fi

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Testing completed successfully!"
    echo "=========================================="
    echo "Results saved in:"
    echo "  - models/     (trained model files)"
    echo "  - plots/      (training plots and analysis)"
    echo "  - runs/       (tensorboard logs)"
    echo ""
    echo "To view tensorboard logs:"
    echo "  tensorboard --logdir runs/"
    echo ""
    echo "To view plots:"
    echo "  ls plots/*.png"
else
    echo ""
    echo "=========================================="
    echo "Testing failed with errors!"
    echo "=========================================="
    echo "Check the error messages above for troubleshooting."
    exit 1
fi
