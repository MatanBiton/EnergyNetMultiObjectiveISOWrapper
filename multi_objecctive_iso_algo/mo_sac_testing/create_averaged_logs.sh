#!/bin/bash

# Helper script to manually average TensorBoard logs from multiseed runs
# This can be used if the automatic averaging in the multiseed script fails
#
# Usage:
#   ./create_averaged_logs.sh <experiment_name> [num_runs]
#
# Examples:
#   ./create_averaged_logs.sh my_experiment 5
#   ./create_averaged_logs.sh baseline_test 3

echo "=========================================="
echo "Manual TensorBoard Log Averaging Script"
echo "=========================================="

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [num_runs]"
    echo ""
    echo "Examples:"
    echo "  $0 my_experiment 5"
    echo "  $0 baseline_test 3"
    echo ""
    echo "This script will look for logs in the 'logs/' directory matching:"
    echo "  logs/train_energynet_<experiment_name>_1_*"
    echo "  logs/train_energynet_<experiment_name>_2_*"
    echo "  etc."
    exit 1
fi

EXPERIMENT_NAME=$1
NUM_RUNS=${2:-5}

# Get script directory
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

echo "Experiment Name: $EXPERIMENT_NAME"
echo "Number of runs: $NUM_RUNS"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Check if logs directory exists
LOGS_DIR="$SCRIPT_DIR/logs"
if [ ! -d "$LOGS_DIR" ]; then
    echo "Error: Logs directory not found: $LOGS_DIR"
    echo "Make sure you're running this from the correct directory"
    exit 1
fi

echo "Logs directory: $LOGS_DIR"

# Check if averaging script exists
AVERAGING_SCRIPT="$SCRIPT_DIR/average_tensorboard_logs.py"
if [ ! -f "$AVERAGING_SCRIPT" ]; then
    echo "Error: Averaging script not found: $AVERAGING_SCRIPT"
    exit 1
fi

echo "Averaging script: $AVERAGING_SCRIPT"

# Find Python command
PYTHON_CMD=""
for cmd in python3 python python3.8 python3.9 python3.10; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: No Python interpreter found"
    exit 1
fi

echo "Python command: $PYTHON_CMD"

# Check for available log directories
echo ""
echo "Checking for existing log directories..."
FOUND_RUNS=0
for i in $(seq 1 $NUM_RUNS); do
    RUN_NAME="${EXPERIMENT_NAME}_${i}"
    PATTERN="$LOGS_DIR/train_energynet_${RUN_NAME}_*"
    
    if ls $PATTERN &> /dev/null; then
        echo "  ✓ Found logs for run $i: $(basename $(ls $PATTERN | head -1))"
        FOUND_RUNS=$((FOUND_RUNS + 1))
    else
        echo "  ✗ No logs found for run $i (pattern: train_energynet_${RUN_NAME}_*)"
    fi
done

echo ""
echo "Found $FOUND_RUNS out of $NUM_RUNS runs"

if [ $FOUND_RUNS -lt 2 ]; then
    echo "Error: Need at least 2 runs to create averaged logs"
    echo ""
    echo "Available log directories:"
    ls -la "$LOGS_DIR"/train_energynet_* 2>/dev/null | head -10
    exit 1
fi

# Check if TensorBoard is available
echo ""
echo "Checking TensorBoard availability..."
if $PYTHON_CMD -c "import tensorboard" 2>/dev/null; then
    echo "✓ TensorBoard is available"
else
    echo "⚠ Warning: TensorBoard not found"
    echo "Install with: pip install tensorboard"
    echo "Attempting to continue anyway..."
fi

# Run the averaging script
echo ""
echo "=========================================="
echo "Running TensorBoard Log Averaging"
echo "=========================================="

# Create seeds array
SEEDS=(42 123 456 789 1337)
SEEDS_STR="${SEEDS[*]:0:$NUM_RUNS}"

echo "Command: $PYTHON_CMD $AVERAGING_SCRIPT $LOGS_DIR $EXPERIMENT_NAME --num-runs $NUM_RUNS --seeds $SEEDS_STR"
echo ""

if $PYTHON_CMD "$AVERAGING_SCRIPT" "$LOGS_DIR" "$EXPERIMENT_NAME" --num-runs "$NUM_RUNS" --seeds ${SEEDS[*]:0:$NUM_RUNS}; then
    echo ""
    echo "✓ Successfully created averaged TensorBoard logs!"
    echo ""
    echo "Results:"
    echo "  Averaged logs: $LOGS_DIR/train_energynet_${EXPERIMENT_NAME}_averaged"
    echo "  Summary report: $SCRIPT_DIR/averaged_summary_${EXPERIMENT_NAME}.txt"
    echo ""
    echo "To view results:"
    echo "  # View averaged results only:"
    echo "  tensorboard --logdir $LOGS_DIR/train_energynet_${EXPERIMENT_NAME}_averaged --host 0.0.0.0 --port 6006"
    echo ""
    echo "  # View all runs (including averaged):"
    echo "  tensorboard --logdir $LOGS_DIR/ --host 0.0.0.0 --port 6006"
    echo ""
    echo "  # View summary report:"
    echo "  cat $SCRIPT_DIR/averaged_summary_${EXPERIMENT_NAME}.txt"
else
    echo ""
    echo "✗ Failed to create averaged logs"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check that TensorBoard is installed: pip install tensorboard"
    echo "  2. Verify log directories exist and contain data"
    echo "  3. Check Python environment and dependencies"
    echo ""
    echo "Available logs:"
    ls -la "$LOGS_DIR"/train_energynet_${EXPERIMENT_NAME}_* 2>/dev/null
    exit 1
fi
