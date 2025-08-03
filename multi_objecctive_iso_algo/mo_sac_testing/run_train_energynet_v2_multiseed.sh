#!/bin/bash

#SBATCH --job-name=mo_sac_energynet_v2_multiseed
#SBATCH --output=slurm_energynet_multiseed_%j.out
#SBATCH --error=slurm_energynet_multiseed_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Multi-Seed Multi-Objective SAC EnergyNet Training Script for SLURM
# This script runs the original training script with 5 different seeds for robust testing
# All arguments are passed through to the original script, with seeds automatically handled
#
# Usage Examples:
#   # Basic training with 5 seeds (default parameters, default experiment name)
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh
#
#   # Custom experiment name with 5 seeds
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh experiment1
#
#   # Custom experiment name and timesteps with 5 seeds
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh baseline_test 200000
#
#   # With optimizations (use environment variables) - 5 seeds
#   ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh optimized_run 500000
#
#   # Full optimization suite with 5 seeds
#   ENABLE_ALL_OPTIMIZATIONS=true LR_ANNEALING_TYPE=cosine VALUE_CLIP_RANGE=150.0 \
#   sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh full_opt 1000000

echo "=========================================="
echo "Multi-Seed Multi-Objective SAC EnergyNet Training (SLURM)"
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
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
fi

echo "Script directory: $SCRIPT_DIR"

# Check if the original script exists
ORIGINAL_SCRIPT="$SCRIPT_DIR/run_train_energynet_v2.sh"
if [ ! -f "$ORIGINAL_SCRIPT" ]; then
    echo "Error: Original script not found at $ORIGINAL_SCRIPT"
    exit 1
fi

echo "Original script: $ORIGINAL_SCRIPT"

# Parse arguments (same as original script)
EXPERIMENT_NAME=${1:-"default"}  # First argument: experiment name (default: "default")
EPISODES=${2:-100000}  # Second argument: episodes (default: 100k episodes)
LEARNING_RATE=${3:-0.0003}  # Third argument: learning rate
BATCH_SIZE=${4:-256}  # Fourth argument: batch size

# Fixed seeds for reproducible multi-seed experiments
SEEDS=(42 123 456 789 1337)

echo "Multi-Seed Training Parameters:"
echo "  Base Experiment Name: $EXPERIMENT_NAME"
echo "  Episodes per run: $EPISODES"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Seeds: ${SEEDS[*]}"
echo ""

# Store all environment variables that should be passed through
ENV_VARS=(
    "ENABLE_LR_ANNEALING"
    "LR_ANNEALING_TYPE"
    "LR_ANNEALING_STEPS"
    "LR_MIN_FACTOR"
    "LR_DECAY_RATE"
    "ENABLE_REWARD_SCALING"
    "REWARD_SCALE_EPSILON"
    "ENABLE_ORTHOGONAL_INIT"
    "ORTHOGONAL_GAIN"
    "ACTOR_ORTHOGONAL_GAIN"
    "CRITIC_ORTHOGONAL_GAIN"
    "ENABLE_VALUE_CLIPPING"
    "VALUE_CLIP_RANGE"
    "ENABLE_ALL_OPTIMIZATIONS"
)

echo "Environment variables to pass through:"
for var in "${ENV_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        echo "  $var=${!var}"
    fi
done
echo ""

# Initialize tracking variables
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
FAILED_SEEDS=()
START_TIME=$(date +%s)

# Run the original script 5 times with different seeds
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    RUN_NUMBER=$((i + 1))
    SEED_EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RUN_NUMBER}"
    
    echo "=========================================="
    echo "Starting Run $RUN_NUMBER/5 (Seed: $SEED)"
    echo "Experiment Name: $SEED_EXPERIMENT_NAME"
    echo "=========================================="
    
    # Set the seed environment variable
    export PYTHONHASHSEED=$SEED
    export TORCH_MANUAL_SEED=$SEED
    export NUMPY_RANDOM_SEED=$SEED
    
    # Prepare environment variables for the subprocess
    ENV_STRING=""
    for var in "${ENV_VARS[@]}"; do
        if [ -n "${!var}" ]; then
            ENV_STRING="$ENV_STRING $var=${!var}"
        fi
    done
    
    # Add seed-related environment variables
    ENV_STRING="$ENV_STRING PYTHONHASHSEED=$SEED TORCH_MANUAL_SEED=$SEED NUMPY_RANDOM_SEED=$SEED"
    
    # Construct the command
    CMD="env $ENV_STRING bash $ORIGINAL_SCRIPT $SEED_EXPERIMENT_NAME $EPISODES $LEARNING_RATE $BATCH_SIZE"
    
    echo "Running command:"
    echo "$CMD"
    echo ""
    
    # Run the original script with the modified experiment name
    RUN_START_TIME=$(date +%s)
    
    # Execute the command
    eval "$CMD"
    EXIT_CODE=$?
    
    RUN_END_TIME=$(date +%s)
    RUN_DURATION=$((RUN_END_TIME - RUN_START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Run $RUN_NUMBER completed successfully (Duration: ${RUN_DURATION}s)"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    else
        echo "✗ Run $RUN_NUMBER failed with exit code $EXIT_CODE (Duration: ${RUN_DURATION}s)"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_SEEDS+=($SEED)
    fi
    
    echo ""
    
    # Brief pause between runs to avoid potential resource conflicts
    if [ $RUN_NUMBER -lt 5 ]; then
        echo "Pausing 10 seconds before next run..."
        sleep 10
    fi
done

# Calculate total duration
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo "=========================================="
echo "Multi-Seed Training Summary"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Base Experiment Name: $EXPERIMENT_NAME"
echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results:"
echo "  Successful runs: $SUCCESSFUL_RUNS/5"
echo "  Failed runs: $FAILED_RUNS/5"

if [ $FAILED_RUNS -gt 0 ]; then
    echo "  Failed seeds: ${FAILED_SEEDS[*]}"
fi

echo ""
echo "Generated experiments:"
for i in "${!SEEDS[@]}"; do
    RUN_NUMBER=$((i + 1))
    SEED_EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RUN_NUMBER}"
    SEED=${SEEDS[$i]}
    
    # Check if this run was successful by looking for expected output files
    if [ -f "$SCRIPT_DIR/train_energynet_${SEED_EXPERIMENT_NAME}_config.json" ] || \
       [ -f "$SCRIPT_DIR/train_energynet_${SEED_EXPERIMENT_NAME}_results.json" ]; then
        STATUS="✓"
    else
        STATUS="✗"
    fi
    
    echo "  $STATUS Run $RUN_NUMBER: train_energynet_${SEED_EXPERIMENT_NAME} (seed: $SEED)"
done

echo ""
echo "Results location: $SCRIPT_DIR"
echo ""
echo "To analyze results across all seeds:"
echo "  # View all config files:"
echo "  ls -la $SCRIPT_DIR/train_energynet_${EXPERIMENT_NAME}_*_config.json"
echo ""
echo "  # View all results files:"
echo "  ls -la $SCRIPT_DIR/train_energynet_${EXPERIMENT_NAME}_*_results.json"
echo ""
echo "  # View all models:"
echo "  ls -la $SCRIPT_DIR/models/train_energynet_${EXPERIMENT_NAME}_*"
echo ""
echo "  # View all tensorboard logs:"
echo "  ls -la $SCRIPT_DIR/logs/train_energynet_${EXPERIMENT_NAME}_*"
echo ""
echo "  # Start tensorboard for all runs:"
echo "  tensorboard --logdir $SCRIPT_DIR/logs/ --host 0.0.0.0 --port 6006"
echo ""

# Create a summary file with all run information
SUMMARY_FILE="$SCRIPT_DIR/multiseed_summary_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Multi-Seed Training Summary"
    echo "=========================="
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Base Experiment Name: $EXPERIMENT_NAME"
    echo "Training Episodes per run: $EPISODES"
    echo "Learning Rate: $LEARNING_RATE"
    echo "Batch Size: $BATCH_SIZE"
    echo "Seeds used: ${SEEDS[*]}"
    echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "Environment Variables:"
    for var in "${ENV_VARS[@]}"; do
        if [ -n "${!var}" ]; then
            echo "  $var=${!var}"
        fi
    done
    echo ""
    echo "Results:"
    echo "  Successful runs: $SUCCESSFUL_RUNS/5"
    echo "  Failed runs: $FAILED_RUNS/5"
    if [ $FAILED_RUNS -gt 0 ]; then
        echo "  Failed seeds: ${FAILED_SEEDS[*]}"
    fi
    echo ""
    echo "Generated experiments:"
    for i in "${!SEEDS[@]}"; do
        RUN_NUMBER=$((i + 1))
        SEED_EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RUN_NUMBER}"
        SEED=${SEEDS[$i]}
        
        if [ -f "$SCRIPT_DIR/train_energynet_${SEED_EXPERIMENT_NAME}_config.json" ] || \
           [ -f "$SCRIPT_DIR/train_energynet_${SEED_EXPERIMENT_NAME}_results.json" ]; then
            STATUS="SUCCESS"
        else
            STATUS="FAILED"
        fi
        
        echo "  Run $RUN_NUMBER: train_energynet_${SEED_EXPERIMENT_NAME} (seed: $SEED) - $STATUS"
    done
} > "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"

# Exit with appropriate code
if [ $FAILED_RUNS -eq 0 ]; then
    echo ""
    echo "🎉 All 5 runs completed successfully!"
    exit 0
elif [ $SUCCESSFUL_RUNS -gt 0 ]; then
    echo ""
    echo "⚠ Partial success: $SUCCESSFUL_RUNS out of 5 runs completed"
    exit 1
else
    echo ""
    echo "❌ All runs failed!"
    exit 1
fi
