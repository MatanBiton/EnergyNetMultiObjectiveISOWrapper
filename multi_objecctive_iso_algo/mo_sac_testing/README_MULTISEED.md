# Multi-Seed EnergyNet Training Guide

This guide explains how to use the multi-seed wrapper script for robust Deep RL algorithm testing with multiple random seeds.

## Overview

The `run_train_energynet_v2_multiseed.sh` script is a wrapper around the original `run_train_energynet_v2.sh` that automatically runs the same experiment with 5 different seeds for more robust statistical analysis.

## Key Features

- **Automatic seed management**: Uses 5 fixed seeds (42, 123, 456, 789, 1337) for reproducible multi-seed experiments
- **Same interface**: Accepts all the same arguments as the original script
- **Organized output**: Each run gets a unique experiment name with `_1`, `_2`, `_3`, `_4`, `_5` suffixes
- **Comprehensive reporting**: Provides detailed summary of all runs and their success/failure status
- **Environment variable passthrough**: All optimization flags are properly passed to each run

## Usage Examples

### Basic Training with 5 Seeds
```bash
# Default parameters, 5 seeds
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh
```

### Custom Experiment Name
```bash
# Custom experiment name with 5 seeds
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh my_experiment
```

### Custom Parameters
```bash
# Custom experiment name, episodes, learning rate, and batch size
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh baseline_test 200000 0.001 512
```

### With Optimizations
```bash
# Learning rate annealing and reward scaling with 5 seeds
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh optimized_run 500000
```

### Full Optimization Suite
```bash
# All optimizations enabled with 5 seeds
ENABLE_ALL_OPTIMIZATIONS=true LR_ANNEALING_TYPE=cosine VALUE_CLIP_RANGE=150.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh full_opt 1000000
```

## Output Structure

When you run an experiment named `my_experiment`, the script will create 5 separate runs:

### Experiment Names
- `train_energynet_my_experiment_1` (seed: 42)
- `train_energynet_my_experiment_2` (seed: 123)
- `train_energynet_my_experiment_3` (seed: 456)
- `train_energynet_my_experiment_4` (seed: 789)
- `train_energynet_my_experiment_5` (seed: 1337)

### Generated Files
For each run, you'll get the standard files with the seed-specific experiment name:

**Config and Results:**
```
train_energynet_my_experiment_1_config.json
train_energynet_my_experiment_1_results.json
train_energynet_my_experiment_2_config.json
train_energynet_my_experiment_2_results.json
...
```

**Models:**
```
models/train_energynet_my_experiment_1_*
models/train_energynet_my_experiment_2_*
...
```

**Logs (TensorBoard):**
```
logs/train_energynet_my_experiment_1/
logs/train_energynet_my_experiment_2/
...
```

**Checkpoints:**
```
checkpoints/train_energynet_my_experiment_1_*
checkpoints/train_energynet_my_experiment_2_*
...
```

**Best Models:**
```
train_energynet_my_experiment_1_best_model.zip
train_energynet_my_experiment_2_best_model.zip
...
```

### Summary File
The script also creates a comprehensive summary file:
```
multiseed_summary_my_experiment_YYYYMMDD_HHMMSS.txt
```

This file contains:
- All run parameters and environment variables
- Success/failure status of each run
- Total duration and timing information
- Seeds used for each run

## Analyzing Multi-Seed Results

### View All Results
```bash
# List all config files
ls -la train_energynet_my_experiment_*_config.json

# List all results files
ls -la train_energynet_my_experiment_*_results.json

# List all models
ls -la models/train_energynet_my_experiment_*

# List all tensorboard logs
ls -la logs/train_energynet_my_experiment_*
```

### TensorBoard Analysis
```bash
# View all runs in TensorBoard
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# Or use the helper script
./view_results.sh --tensorboard
```

### Statistical Analysis
You can now perform robust statistical analysis across the 5 seeds:
- Calculate mean and standard deviation of final rewards
- Compare convergence rates across seeds
- Identify outlier runs
- Report confidence intervals

## Environment Variables

All optimization environment variables from the original script are supported:

### Learning Rate Annealing
- `ENABLE_LR_ANNEALING=true`
- `LR_ANNEALING_TYPE=cosine|linear|exponential`
- `LR_ANNEALING_STEPS=<number>`
- `LR_MIN_FACTOR=<float>`
- `LR_DECAY_RATE=<float>`

### Reward Scaling
- `ENABLE_REWARD_SCALING=true`
- `REWARD_SCALE_EPSILON=<float>`

### Orthogonal Initialization
- `ENABLE_ORTHOGONAL_INIT=true`
- `ORTHOGONAL_GAIN=<float>`
- `ACTOR_ORTHOGONAL_GAIN=<float>`
- `CRITIC_ORTHOGONAL_GAIN=<float>`

### Value Clipping
- `ENABLE_VALUE_CLIPPING=true`
- `VALUE_CLIP_RANGE=<float>`

### All Optimizations
- `ENABLE_ALL_OPTIMIZATIONS=true` (enables all optimizations with good defaults)

## Seeds Used

The script uses these fixed seeds for reproducibility:
1. **Seed 42** (Run 1)
2. **Seed 123** (Run 2)
3. **Seed 456** (Run 3)
4. **Seed 789** (Run 4)
5. **Seed 1337** (Run 5)

These seeds are:
- Set in `PYTHONHASHSEED`
- Set in `TORCH_MANUAL_SEED`
- Set in `NUMPY_RANDOM_SEED`

## Monitoring Progress

### SLURM Output
The script creates SLURM output files:
- `slurm_energynet_multiseed_<job_id>.out` - Standard output
- `slurm_energynet_multiseed_<job_id>.err` - Error output

### Real-time Monitoring
```bash
# Check job status
squeue -u $USER

# Monitor output in real-time
tail -f slurm_energynet_multiseed_<job_id>.out

# Check specific run progress
tail -f logs/train_energynet_my_experiment_1/progress.log
```

## Error Handling

The script is designed to be robust:

### Partial Failures
- If some runs fail, the script continues with remaining runs
- Failed runs are clearly reported in the summary
- Exit code indicates partial success (1) or complete failure (2)

### Recovery
- Each run is independent, so you can manually re-run failed seeds
- Use the same environment variables and just modify the experiment name

### Troubleshooting
1. Check the SLURM error file for detailed error messages
2. Verify all dependencies are available in your environment
3. Ensure sufficient disk space for all 5 runs
4. Check that the original script works with a single seed first

## Resource Considerations

### Memory and CPU
- Each run uses the same resources as the original script
- Runs are sequential, not parallel, so resource usage is the same per run
- Total time ≈ 5 × single run time + overhead

### Disk Space
- You'll need ~5× the disk space of a single run
- Plan accordingly for models, logs, and checkpoints

### Time Limits
- Set SLURM time limits appropriately (5× single run time + buffer)
- Consider using `--time=<hours>:00:00` in your sbatch command

## Best Practices

1. **Test single seed first**: Verify the original script works before using multi-seed
2. **Use descriptive experiment names**: Makes it easier to track different experiments
3. **Monitor disk usage**: Multi-seed experiments generate lots of data
4. **Set appropriate time limits**: Account for 5× the single run duration
5. **Keep backups**: Copy important results to permanent storage

## Comparison with Single Seed

| Aspect | Single Seed | Multi-Seed |
|--------|-------------|------------|
| Runs | 1 | 5 |
| Time | T | ~5T |
| Disk Space | D | ~5D |
| Statistical Power | Low | High |
| Reproducibility | Seed-dependent | Robust |
| Analysis Confidence | Limited | High |

## Example Complete Workflow

```bash
# 1. Run multi-seed experiment
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 --time=10:00:00 \
./run_train_energynet_v2_multiseed.sh robust_test 500000

# 2. Monitor progress
squeue -u $USER
tail -f slurm_energynet_multiseed_*.out

# 3. After completion, analyze results
ls -la train_energynet_robust_test_*_results.json
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# 4. Review summary
cat multiseed_summary_robust_test_*.txt
```

This multi-seed approach provides much more reliable and statistically significant results for your Deep RL experiments!
