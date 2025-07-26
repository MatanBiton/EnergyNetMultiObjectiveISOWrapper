#!/bin/bash

# Multi-Objective SAC Training Script for EnergyNet MoISO Environment
# This script trains the MO-SAC algorithm on the EnergyNet Multi-Objective ISO environment
#
# Usage: ./run_train_energynet.sh [OPTIONS]
# 
# Examples:
#   ./run_train_energynet.sh                    # Run with default parameters
#   ./run_train_energynet.sh --quick-test       # Quick test run (shorter training)
#   ./run_train_energynet.sh --cost-priority    # Prioritize cost reduction over stability
#   ./run_train_energynet.sh --stability-priority # Prioritize stability over cost

echo "=========================================="
echo "MO-SAC EnergyNet Training Script"
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
# python -c "import torch, numpy, gymnasium, matplotlib, tensorboard" 2>/dev/null
# if [ $? -ne 0 ]; then
#     echo "Warning: Some required packages might be missing. Installing..."
#     pip install -r requirements.txt
# fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p energynet_experiments
mkdir -p energynet_experiments/models
mkdir -p energynet_experiments/logs
mkdir -p energynet_experiments/plots

# Set environment variables for better performance (optional)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available, comment out to use CPU only
export OMP_NUM_THREADS=4       # Number of CPU threads for PyTorch

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
python train_energynet.py \
    `# Experiment Configuration` \
    --experiment-name "$EXP_NAME" \
    --save-dir "energynet_experiments" \
    \
    `# Training Parameters - Modify these to change training behavior` \
    --total-timesteps $TOTAL_TIMESTEPS \
    --learning-starts $LEARNING_STARTS \
    --eval-freq $EVAL_FREQ \
    --save-freq $SAVE_FREQ \
    \
    `# Multi-Objective Weights - [cost_weight, stability_weight] must sum to 1.0` \
    --weights $WEIGHTS \
    \
    `# Network Learning Rates - Decrease if training is unstable, increase if learning is slow` \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --alpha-lr 3e-4 \
    \
    `# SAC Algorithm Parameters` \
    --gamma 0.99 \        # Discount factor (0.9-0.999)
    --tau 0.005 \         # Target network update rate (0.001-0.01)
    \
    `# Experience Replay Configuration` \
    --buffer-size 1000000 \  # Replay buffer size (larger = more stable but more memory)
    --batch-size 256 \       # Training batch size (64-512, larger = more stable)
    \
    `# Environment Configuration` \
    --dispatch-strategy "PROPORTIONAL" \  # Dispatch strategy: PROPORTIONAL, EQUAL, etc.
    `# --use-dispatch-action \`           # Uncomment to enable dispatch actions
    \
    `# Logging Configuration` \
    --verbose \              # Enable detailed logging
    `# --no-tensorboard \`   # Uncomment to disable tensorboard logging

# Store the exit code
EXIT_CODE=$?

# Report results
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Results saved in: energynet_experiments/"
    echo ""
    echo "Files created:"
    echo "  - energynet_experiments/${EXP_NAME}_config.json    (configuration)"
    echo "  - energynet_experiments/${EXP_NAME}_results.json   (training results)"
    echo "  - energynet_experiments/models/${EXP_NAME}_final.pth (trained model)"
    echo "  - energynet_experiments/logs/                      (tensorboard logs)"
    echo ""
    echo "To view training progress:"
    echo "  tensorboard --logdir energynet_experiments/logs/"
    echo ""
    echo "To load and evaluate the trained model:"
    echo "  python -c \""
    echo "    from multi_objective_sac import MultiObjectiveSAC"
    echo "    agent = MultiObjectiveSAC(...)"
    echo "    agent.load('energynet_experiments/models/${EXP_NAME}_final.pth')"
    echo "  \""
    echo ""
    echo "Parameter Tuning Tips:"
    echo "  - If learning is slow: increase learning rates (--actor-lr, --critic-lr)"
    echo "  - If training is unstable: decrease learning rates or increase --tau"
    echo "  - For different objectives: adjust --weights [cost_weight stability_weight]"
    echo "  - For longer training: increase --total-timesteps"
    echo "  - For more evaluation: decrease --eval-freq"
    
else
    echo ""
    echo "=========================================="
    echo "Training failed with errors!"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Common troubleshooting:"
    echo "  1. Check that EnergyNet environment is properly installed"
    echo "  2. Verify all dependencies are installed: pip install -r requirements.txt"
    echo "  3. Check CUDA availability if using GPU: python -c 'import torch; print(torch.cuda.is_available())'"
    echo "  4. Try running with --quick-test flag for faster debugging"
    echo "  5. Check the error messages above for specific issues"
    
    exit $EXIT_CODE
fi

# Additional analysis suggestions
echo ""
echo "Analysis Suggestions:"
echo "========================"
echo "1. Compare different weight configurations:"
echo "   ./run_train_energynet.sh --cost-priority"
echo "   ./run_train_energynet.sh --stability-priority"
echo ""
echo "2. Monitor training in real-time:"
echo "   tensorboard --logdir energynet_experiments/logs/ --host 0.0.0.0"
echo ""
echo "3. Run multiple seeds for statistical significance:"
echo "   for i in {1..5}; do ./run_train_energynet.sh; done"
echo ""
echo "4. Hyperparameter tuning experiments:"
echo "   - Try different learning rates: 1e-4, 5e-4, 1e-3"
echo "   - Try different network sizes by modifying the script"
echo "   - Try different discount factors: 0.95, 0.99, 0.995"
