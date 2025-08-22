#!/bin/bash
#SBATCH --job-name=sac_pcs_multiseed
#SBATCH --output=slurm_sac_pcs_multiseed_%j.out
#SBATCH --error=slurm_sac_pcs_multiseed_%j.err
#SBATCH --time=240:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Multi-seed SAC PCS Training Script
# This script runs the SAC PCS training multiple times with different seeds
# to get statistically significant results.

# Usage examples:
#   # Default training with 5 seeds
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs_multiseed.sh
#
#   # Custom experiment name with 5 seeds
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs_multiseed.sh my_experiment
#
#   # Custom parameters
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs_multiseed.sh baseline_test 200000 0.001 512
#
#   # With ISO model and optimizations
#   ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs_multiseed.sh optimized_run 500000 3e-4 256 /path/to/iso_model.pth
#
#   # Full optimization suite
#   ENABLE_ALL_OPTIMIZATIONS=true \
#   sbatch -c 4 --gres=gpu:1 ./run_train_sac_pcs_multiseed.sh full_opt 1000000

echo "=========================================="
echo "SAC PCS Multi-Seed Training (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Parse command line arguments
BASE_EXPERIMENT_NAME=${1:-"sac_pcs_multiseed"}
TOTAL_TIMESTEPS=${2:-500000}
LEARNING_RATE=${3:-3e-4}
BATCH_SIZE=${4:-256}
ISO_MODEL_PATH=${5:-""}

# Number of seeds to run
NUM_SEEDS=5
SEEDS=(42 123 456 789 1234)

echo "Multi-seed training configuration:"
echo "  Base Experiment Name: $BASE_EXPERIMENT_NAME"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  ISO Model Path: $ISO_MODEL_PATH"
echo "  Number of seeds: $NUM_SEEDS"
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

# Run the original script multiple times with different seeds
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    SEED_NUM=$((i + 1))
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_seed${SEED}"
    
    echo "=========================================="
    echo "Starting run $SEED_NUM/$NUM_SEEDS with seed $SEED"
    echo "Experiment name: $EXPERIMENT_NAME"
    echo "=========================================="
    
    # Set up environment variables for this run
    for var in "${ENV_VARS[@]}"; do
        if [ -n "${!var}" ]; then
            export $var="${!var}"
        fi
    done
    
    # Build command
    CMD="./run_train_sac_pcs.sh"
    CMD="$CMD $EXPERIMENT_NAME"
    CMD="$CMD $TOTAL_TIMESTEPS"
    CMD="$CMD $LEARNING_RATE"
    CMD="$CMD $BATCH_SIZE"
    if [ -n "$ISO_MODEL_PATH" ] && [ "$ISO_MODEL_PATH" != "" ]; then
        CMD="$CMD $ISO_MODEL_PATH"
    fi
    CMD="$CMD $SEED"
    
    echo "Running command: $CMD"
    echo ""
    
    # Run training
    RUN_START_TIME=$(date +%s)
    eval $CMD
    EXIT_CODE=$?
    RUN_END_TIME=$(date +%s)
    RUN_DURATION=$((RUN_END_TIME - RUN_START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Run $SEED_NUM/$NUM_SEEDS completed successfully (seed $SEED) in ${RUN_DURATION}s"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    else
        echo "✗ Run $SEED_NUM/$NUM_SEEDS failed (seed $SEED) with exit code $EXIT_CODE"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_SEEDS+=($SEED)
    fi
    
    echo ""
    echo "Progress: $SEED_NUM/$NUM_SEEDS completed"
    echo "Successful: $SUCCESSFUL_RUNS, Failed: $FAILED_RUNS"
    echo ""
    
    # Brief pause between runs
    sleep 10
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo "=========================================="
echo "Multi-Seed Training Summary"
echo "=========================================="
echo "Base experiment name: $BASE_EXPERIMENT_NAME"
echo "Total runs: $NUM_SEEDS"
echo "Successful runs: $SUCCESSFUL_RUNS"
echo "Failed runs: $FAILED_RUNS"

if [ $FAILED_RUNS -gt 0 ]; then
    echo "Failed seeds: ${FAILED_SEEDS[*]}"
fi

echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Generate summary statistics if we have successful runs
if [ $SUCCESSFUL_RUNS -gt 0 ]; then
    echo "Generating summary statistics..."
    
    # Create summary directory
    SUMMARY_DIR="pcs_experiments/multiseed_summary_${BASE_EXPERIMENT_NAME}"
    mkdir -p "$SUMMARY_DIR"
    
    # Collect results from all successful runs
    RESULTS_FILES=()
    for i in "${!SEEDS[@]}"; do
        SEED=${SEEDS[$i]}
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_seed${SEED}"
        RESULTS_FILE="pcs_experiments/${EXPERIMENT_NAME}_results.json"
        
        if [ -f "$RESULTS_FILE" ]; then
            RESULTS_FILES+=("$RESULTS_FILE")
        fi
    done
    
    echo "Found ${#RESULTS_FILES[@]} result files"
    
    # Create Python script to generate summary
    cat > "$SUMMARY_DIR/generate_summary.py" << 'EOF'
import json
import numpy as np
import sys
import os

def load_results(files):
    results = []
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    return results

def generate_summary(results):
    if not results:
        print("No results to summarize")
        return {}
    
    # Extract key metrics
    final_means = [r['final_evaluation']['mean'] for r in results]
    final_stds = [r['final_evaluation']['std'] for r in results]
    best_evals = [r['best_evaluation'] for r in results]
    total_episodes = [r['total_episodes'] for r in results]
    training_times = [r['training_time'] for r in results]
    
    summary = {
        'num_seeds': len(results),
        'final_evaluation': {
            'mean': float(np.mean(final_means)),
            'std': float(np.std(final_means)),
            'min': float(np.min(final_means)),
            'max': float(np.max(final_means)),
            'median': float(np.median(final_means)),
            'all_values': final_means
        },
        'best_evaluation': {
            'mean': float(np.mean(best_evals)),
            'std': float(np.std(best_evals)),
            'min': float(np.min(best_evals)),
            'max': float(np.max(best_evals)),
            'median': float(np.median(best_evals)),
            'all_values': best_evals
        },
        'total_episodes': {
            'mean': float(np.mean(total_episodes)),
            'std': float(np.std(total_episodes)),
            'min': int(np.min(total_episodes)),
            'max': int(np.max(total_episodes)),
            'all_values': total_episodes
        },
        'training_time': {
            'mean': float(np.mean(training_times)),
            'std': float(np.std(training_times)),
            'min': float(np.min(training_times)),
            'max': float(np.max(training_times)),
            'total': float(np.sum(training_times)),
            'all_values': training_times
        },
        'config': results[0]['config'] if results else {},
        'individual_results': [
            {
                'final_eval': r['final_evaluation']['mean'],
                'best_eval': r['best_evaluation'],
                'episodes': r['total_episodes'],
                'time': r['training_time']
            } for r in results
        ]
    }
    
    return summary

if __name__ == "__main__":
    files = sys.argv[1:]
    results = load_results(files)
    summary = generate_summary(results)
    
    # Save summary
    output_file = "multiseed_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nMulti-seed Training Summary:")
    print("=" * 50)
    print(f"Number of seeds: {summary['num_seeds']}")
    print(f"Final evaluation: {summary['final_evaluation']['mean']:.3f} ± {summary['final_evaluation']['std']:.3f}")
    print(f"Best evaluation: {summary['best_evaluation']['mean']:.3f} ± {summary['best_evaluation']['std']:.3f}")
    print(f"Total episodes (avg): {summary['total_episodes']['mean']:.0f} ± {summary['total_episodes']['std']:.0f}")
    print(f"Training time (total): {summary['training_time']['total']:.1f}s ({summary['training_time']['total']/3600:.1f}h)")
    print("")
    print("Individual results:")
    for i, result in enumerate(summary['individual_results']):
        print(f"  Seed {i+1}: Final={result['final_eval']:.3f}, Best={result['best_eval']:.3f}, Episodes={result['episodes']}, Time={result['time']:.1f}s")
    
    print(f"\nSummary saved to: {output_file}")
EOF
    
    # Run summary generation
    cd "$SUMMARY_DIR"
    if command -v python3 > /dev/null 2>&1; then
        python3 generate_summary.py "${RESULTS_FILES[@]}"
    elif command -v python > /dev/null 2>&1; then
        python generate_summary.py "${RESULTS_FILES[@]}"
    else
        echo "Python not found for summary generation"
    fi
    
    cd - > /dev/null
    
    echo "Summary files saved to: $SUMMARY_DIR"
fi

echo ""
echo "Model files available:"
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_seed${SEED}"
    MODEL_DIR="pcs_experiments/models"
    
    if [ -f "$MODEL_DIR/${EXPERIMENT_NAME}_final.pth" ]; then
        echo "  $MODEL_DIR/${EXPERIMENT_NAME}_final.pth"
    fi
    if [ -f "$MODEL_DIR/${EXPERIMENT_NAME}_best.pth" ]; then
        echo "  $MODEL_DIR/${EXPERIMENT_NAME}_best.pth"
    fi
done

echo ""
echo "TensorBoard logs:"
echo "  tensorboard --logdir pcs_experiments/logs/"

# Determine overall exit code
if [ $FAILED_RUNS -eq 0 ]; then
    OVERALL_EXIT_CODE=0
    echo "✓ All runs completed successfully!"
elif [ $SUCCESSFUL_RUNS -gt 0 ]; then
    OVERALL_EXIT_CODE=1
    echo "⚠ Some runs failed, but at least one succeeded"
else
    OVERALL_EXIT_CODE=2
    echo "✗ All runs failed"
fi

echo ""
echo "=========================================="
echo "Multi-Seed SLURM Job Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Base experiment: $BASE_EXPERIMENT_NAME"
echo "Successful runs: $SUCCESSFUL_RUNS/$NUM_SEEDS"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit code: $OVERALL_EXIT_CODE"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

exit $OVERALL_EXIT_CODE
