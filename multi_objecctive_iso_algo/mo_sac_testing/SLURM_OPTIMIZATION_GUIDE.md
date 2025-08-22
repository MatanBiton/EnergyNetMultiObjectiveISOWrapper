# SLURM Training Script Usage Guide

## Overview

The `run_train_energynet_v2.sh` script is designed for training Multi-Objective SAC on SLURM clusters with full support for all optimization features. It handles environment setup, file copying, and result management automatically.

## Basic Usage

### Simple Training
```bash
# Default training (100k timesteps, NO optimizations - true baseline)
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh

# Custom timesteps with no optimizations
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Custom timesteps with learning rate and batch size (still no optimizations)
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 1000000 0.0005 512
```

### Resource Requirements
```bash
# Standard GPU job
sbatch -c 4 --gres=gpu:1 --mem=32G ./run_train_energynet_v2.sh

# High-memory job for large experiments
sbatch -c 8 --gres=gpu:1 --mem=64G ./run_train_energynet_v2.sh

# Multi-GPU job (if supported by cluster)
sbatch -c 8 --gres=gpu:2 --mem=64G ./run_train_energynet_v2.sh
```

## Optimization Parameters

All optimization features are controlled via **environment variables** that you set before running sbatch.

### 1. Learning Rate Annealing 📉

**Enable LR Annealing:**
```bash
ENABLE_LR_ANNEALING=true sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh
```

**Configure LR Annealing:**
```bash
# Cosine annealing (smooth decay)
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine LR_MIN_FACTOR=0.1 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Linear annealing (constant decay)
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=linear LR_MIN_FACTOR=0.2 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Exponential annealing
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=exponential LR_DECAY_RATE=0.95 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Custom annealing steps
ENABLE_LR_ANNEALING=true LR_ANNEALING_STEPS=100000 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

**Available Parameters:**
- `ENABLE_LR_ANNEALING` (true/false) - Enable learning rate annealing
- `LR_ANNEALING_TYPE` (cosine/linear/exponential) - Type of annealing
- `LR_MIN_FACTOR` (float) - Minimum LR as fraction of initial (default: 0.1)
- `LR_ANNEALING_STEPS` (int) - Number of annealing steps (default: auto-calculated)
- `LR_DECAY_RATE` (float) - Decay rate for exponential annealing (default: 0.95)

### 2. Reward Scaling 🎯

**Enable Reward Scaling:**
```bash
ENABLE_REWARD_SCALING=true sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh
```

**Configure Reward Scaling:**
```bash
# With custom epsilon
ENABLE_REWARD_SCALING=true REWARD_SCALE_EPSILON=1e-5 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

**Available Parameters:**
- `ENABLE_REWARD_SCALING` (true/false) - Enable reward normalization
- `REWARD_SCALE_EPSILON` (float) - Epsilon for numerical stability (default: 1e-4)

### 3. Orthogonal Initialization 🎲

**Enable Orthogonal Initialization:**
```bash
ENABLE_ORTHOGONAL_INIT=true sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh
```

**Configure Orthogonal Initialization:**
```bash
# Custom gains
ENABLE_ORTHOGONAL_INIT=true ACTOR_ORTHOGONAL_GAIN=0.05 CRITIC_ORTHOGONAL_GAIN=1.2 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

**Available Parameters:**
- `ENABLE_ORTHOGONAL_INIT` (true/false) - Enable orthogonal initialization (disabled by default)
- `ORTHOGONAL_GAIN` (float) - General orthogonal gain (default: 1.0)
- `ACTOR_ORTHOGONAL_GAIN` (float) - Actor-specific gain (default: 0.01)
- `CRITIC_ORTHOGONAL_GAIN` (float) - Critic-specific gain (default: 1.0)

### 4. Value Clipping 🔒

**Enable Value Clipping:**
```bash
ENABLE_VALUE_CLIPPING=true sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh
```

**Configure Value Clipping:**
```bash
# Custom clip range
ENABLE_VALUE_CLIPPING=true VALUE_CLIP_RANGE=150.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

**Available Parameters:**
- `ENABLE_VALUE_CLIPPING` (true/false) - Enable value clipping
- `VALUE_CLIP_RANGE` (float) - Clipping range (default: 200.0)

## Combined Optimization Examples

### Conservative Optimization (Recommended for Production)
```bash
# Safe optimizations that improve stability
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

### Moderate Optimization (Balanced Performance)
```bash
# Good balance of performance and stability
ENABLE_LR_ANNEALING=true \
ENABLE_REWARD_SCALING=true \
ENABLE_VALUE_CLIPPING=true \
LR_ANNEALING_TYPE=cosine \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

### Aggressive Optimization (Maximum Performance)
```bash
# All optimizations with aggressive settings
ENABLE_LR_ANNEALING=true \
LR_ANNEALING_TYPE=cosine \
LR_MIN_FACTOR=0.05 \
ENABLE_REWARD_SCALING=true \
REWARD_SCALE_EPSILON=1e-5 \
ENABLE_VALUE_CLIPPING=true \
VALUE_CLIP_RANGE=100.0 \
ACTOR_ORTHOGONAL_GAIN=0.1 \
CRITIC_ORTHOGONAL_GAIN=1.5 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

### Quick Enable All Optimizations
```bash
# Special environment variable to enable all optimizations with good defaults
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

## 🧪 **For Fair Comparison Testing** (All use same timesteps)

### **Systematic Testing Protocol**
```bash
# Baseline (no optimizations)
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Individual optimizations (same timesteps for fair comparison)
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

ENABLE_ORTHOGONAL_INIT=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

ENABLE_VALUE_CLIPPING=true VALUE_CLIP_RANGE=200.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Combined optimizations (same timesteps for fair comparison)
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

## Advanced Usage

### Long Training Jobs
```bash
# For very long training runs (multiple days)
ENABLE_LR_ANNEALING=true \
LR_ANNEALING_TYPE=cosine \
ENABLE_REWARD_SCALING=true \
ENABLE_VALUE_CLIPPING=true \
sbatch -c 8 --gres=gpu:1 --mem=64G --time=72:00:00 \
./run_train_energynet_v2.sh 5000000
```

### Hyperparameter Sweeps
```bash
# Script for running multiple experiments
#!/bin/bash

# Conservative sweep
for lr in 0.0001 0.0003 0.0005; do
    for clip in 100.0 150.0 200.0; do
        ENABLE_LR_ANNEALING=true \
        ENABLE_REWARD_SCALING=true \
        ENABLE_VALUE_CLIPPING=true \
        VALUE_CLIP_RANGE=$clip \
        sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 $lr
    done
done
```

### Different Annealing Strategies
```bash
# Cosine annealing for smooth convergence
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine LR_MIN_FACTOR=0.1 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 1000000

# Linear annealing for more control
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=linear LR_MIN_FACTOR=0.2 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 1000000

# Exponential annealing for fast initial decay
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=exponential LR_DECAY_RATE=0.98 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 1000000
```

## Environment Variable Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_LR_ANNEALING` | bool | false | Enable learning rate annealing |
| `LR_ANNEALING_TYPE` | string | cosine | Type: cosine, linear, exponential |
| `LR_ANNEALING_STEPS` | int | auto | Number of annealing steps |
| `LR_MIN_FACTOR` | float | 0.1 | Minimum LR as fraction of initial |
| `LR_DECAY_RATE` | float | 0.95 | Exponential decay rate |
| `ENABLE_REWARD_SCALING` | bool | false | Enable reward normalization |
| `REWARD_SCALE_EPSILON` | float | 1e-4 | Numerical stability epsilon |
| `ENABLE_ORTHOGONAL_INIT` | bool | false | Enable orthogonal initialization |
| `ORTHOGONAL_GAIN` | float | 1.0 | General orthogonal gain |
| `ACTOR_ORTHOGONAL_GAIN` | float | 0.01 | Actor network gain |
| `CRITIC_ORTHOGONAL_GAIN` | float | 1.0 | Critic network gain |
| `ENABLE_VALUE_CLIPPING` | bool | false | Enable value clipping |
| `VALUE_CLIP_RANGE` | float | 200.0 | Clipping range |
| `ENABLE_ALL_OPTIMIZATIONS` | bool | false | Enable all with defaults |

## Output and Results

### File Locations
After training completes, results are automatically copied back:
```
mo_sac_testing/
├── models/                    # Trained model files
├── logs/                      # TensorBoard logs
├── plots/                     # Training plots
├── checkpoints/               # Training checkpoints
├── energynet_training_*.json  # Config and results files
└── slurm_energynet_*.{out,err} # SLURM job logs
```

### Monitoring Training
```bash
# Check job status
squeue -u $USER

# View real-time logs
tail -f slurm_energynet_JOBID.out

# View TensorBoard (after job completes)
module load python
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

### Testing Results
```bash
# Test the trained model
sbatch -c 2 --gres=gpu:1 ./run_test_trained_model.sh

# View results summary
./view_results.sh
```

## Troubleshooting

### Common Issues

**1. Environment Variables Not Working:**
```bash
# Make sure to set variables before sbatch command
ENABLE_LR_ANNEALING=true sbatch ...  # ✓ Correct
sbatch ENABLE_LR_ANNEALING=true ...  # ✗ Wrong
```

**2. Memory Issues:**
```bash
# Increase memory allocation
sbatch -c 8 --gres=gpu:1 --mem=64G ./run_train_energynet_v2.sh
```

**3. GPU Not Available:**
```bash
# The script will automatically fall back to CPU training
# Check SLURM output for GPU status information
```

**4. Training Instability:**
```bash
# Try more conservative settings
ENABLE_VALUE_CLIPPING=true VALUE_CLIP_RANGE=200.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh
```

### Debugging
```bash
# Run with verbose output
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 10000  # Short run

# Check all environment variables are set correctly
echo "LR Annealing: $ENABLE_LR_ANNEALING"
echo "Reward Scaling: $ENABLE_REWARD_SCALING"
# etc.
```

## Performance Tips

1. **Start Small**: Test with 10k-50k timesteps first
2. **Use Conservative Settings**: Begin with reward scaling only
3. **Monitor Resources**: Check GPU/CPU usage in SLURM logs
4. **Save Frequently**: Use shorter save intervals for long jobs
5. **Compare Baselines**: Always test against unoptimized version

## Best Practices

1. **Reproducible Experiments**: Document exact environment variables used
2. **Resource Planning**: Estimate training time based on pilot runs
3. **Result Management**: Use meaningful experiment names
4. **Monitoring**: Set up TensorBoard viewing during training
5. **Backup**: Keep copies of successful configurations

The SLURM script now fully supports all optimization features! 🚀
