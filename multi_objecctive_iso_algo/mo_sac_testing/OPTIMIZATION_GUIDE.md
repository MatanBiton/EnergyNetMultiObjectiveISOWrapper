# Multi-Objective SAC Optimizations Guide

## Overview

This document describes the optimization features added to the Multi-Objective SAC algorithm to improve training performance and stability. All optimizations are configurable and can be enabled/disabled for testing and tuning purposes.

## New Optimization Features

### 1. Learning Rate Annealing 📉

**Purpose**: Gradually reduce learning rates during training for better convergence and fine-tuning.

**Parameters**:
- `use_lr_annealing` (bool): Enable/disable LR annealing
- `lr_annealing_type` (str): Type of annealing ('cosine', 'linear', 'exponential')
- `lr_annealing_steps` (int): Number of steps for annealing (default: total_timesteps/train_freq)
- `lr_min_factor` (float): Minimum LR as fraction of initial LR (default: 0.1)
- `lr_decay_rate` (float): Decay rate for exponential annealing (default: 0.95)

**Usage Examples**:
```python
# Cosine annealing (smooth decay)
agent = MultiObjectiveSAC(
    ...,
    use_lr_annealing=True,
    lr_annealing_type='cosine',
    lr_annealing_steps=100000,
    lr_min_factor=0.1
)

# Linear annealing (constant decay)
agent = MultiObjectiveSAC(
    ...,
    use_lr_annealing=True,
    lr_annealing_type='linear',
    lr_min_factor=0.2
)

# Exponential annealing (geometric decay)
agent = MultiObjectiveSAC(
    ...,
    use_lr_annealing=True,
    lr_annealing_type='exponential',
    lr_decay_rate=0.95
)
```

**Command Line**:
```bash
python train_energynet.py --use-lr-annealing --lr-annealing-type cosine --lr-min-factor 0.1
```

### 2. Reward Scaling/Normalization 🎯

**Purpose**: Normalize rewards to have zero mean and unit variance, improving training stability when objectives have different scales.

**Parameters**:
- `use_reward_scaling` (bool): Enable/disable reward scaling
- `reward_scale_epsilon` (float): Small value to prevent division by zero (default: 1e-4)

**Usage Examples**:
```python
# Enable reward scaling
agent = MultiObjectiveSAC(
    ...,
    use_reward_scaling=True,
    reward_scale_epsilon=1e-4
)
```

**Command Line**:
```bash
python train_energynet.py --use-reward-scaling --reward-scale-epsilon 1e-5
```

**When to Use**: Enable when your objectives have very different scales (e.g., cost in thousands vs. stability in 0-1 range).

### 3. Orthogonal Initialization 🎲

**Purpose**: Better weight initialization that can improve initial exploration and training stability.

**Parameters**:
- `use_orthogonal_init` (bool): Enable orthogonal initialization (default: True)
- `orthogonal_gain` (float): General gain for orthogonal initialization (default: 1.0)
- `actor_orthogonal_gain` (float): Specific gain for actor networks (default: 0.01)
- `critic_orthogonal_gain` (float): Specific gain for critic networks (default: 1.0)

**Usage Examples**:
```python
# Standard orthogonal initialization
agent = MultiObjectiveSAC(
    ...,
    use_orthogonal_init=True,
    actor_orthogonal_gain=0.01,
    critic_orthogonal_gain=1.0
)

# Disable orthogonal init (use Xavier instead)
agent = MultiObjectiveSAC(
    ...,
    use_orthogonal_init=False
)
```

**Command Line**:
```bash
# Enable with custom gains
python train_energynet.py --actor-orthogonal-gain 0.05 --critic-orthogonal-gain 1.2

# Disable orthogonal initialization
python train_energynet.py --disable-orthogonal-init
```

### 4. Value Clipping 🔒

**Purpose**: Clip Q-values to prevent numerical instabilities and training divergence.

**Parameters**:
- `use_value_clipping` (bool): Enable/disable value clipping
- `value_clip_range` (float): Range for clipping [-range, +range] (default: 200.0)

**Usage Examples**:
```python
# Enable value clipping
agent = MultiObjectiveSAC(
    ...,
    use_value_clipping=True,
    value_clip_range=100.0
)
```

**Command Line**:
```bash
python train_energynet.py --use-value-clipping --value-clip-range 150.0
```

**When to Use**: Enable if you experience training instability or diverging Q-values.

## Recommended Configurations

### 1. Conservative (Safe for Production) 🟢
```python
agent = MultiObjectiveSAC(
    ...,
    use_lr_annealing=False,          # No LR annealing
    use_reward_scaling=True,         # Normalize rewards
    use_orthogonal_init=True,        # Better initialization
    use_value_clipping=False         # No clipping
)
```

### 2. Moderate (Balanced Performance) 🟡
```python
agent = MultiObjectiveSAC(
    ...,
    use_lr_annealing=True,
    lr_annealing_type='cosine',
    lr_min_factor=0.1,
    use_reward_scaling=True,
    use_orthogonal_init=True,
    use_value_clipping=True,
    value_clip_range=200.0
)
```

### 3. Aggressive (Maximum Optimization) 🔴
```python
agent = MultiObjectiveSAC(
    ...,
    actor_lr=5e-4,                   # Higher learning rates
    use_lr_annealing=True,
    lr_annealing_type='cosine',
    lr_min_factor=0.05,              # More aggressive decay
    use_reward_scaling=True,
    reward_scale_epsilon=1e-5,       # More sensitive scaling
    use_orthogonal_init=True,
    actor_orthogonal_gain=0.1,       # Higher gains
    critic_orthogonal_gain=1.5,
    use_value_clipping=True,
    value_clip_range=100.0           # Tighter clipping
)
```

## Command Line Examples

### Basic Training with Optimizations
```bash
# Conservative approach
python train_energynet.py --use-reward-scaling --total-timesteps 500000

# Moderate optimization
python train_energynet.py \
    --use-lr-annealing --lr-annealing-type cosine \
    --use-reward-scaling \
    --use-value-clipping \
    --total-timesteps 1000000

# Aggressive optimization
python train_energynet.py \
    --use-lr-annealing --lr-annealing-type cosine --lr-min-factor 0.05 \
    --use-reward-scaling --reward-scale-epsilon 1e-5 \
    --use-value-clipping --value-clip-range 100.0 \
    --actor-lr 5e-4 --critic-lr 5e-4 \
    --total-timesteps 1000000
```

### SLURM Cluster Usage

For SLURM clusters, use the enhanced `run_train_energynet_v2.sh` script with environment variables:

```bash
# Conservative optimization on SLURM
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Moderate optimization on SLURM
ENABLE_LR_ANNEALING=true \
ENABLE_REWARD_SCALING=true \
ENABLE_VALUE_CLIPPING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 1000000

# All optimizations enabled
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 1000000

# Custom optimization settings
ENABLE_LR_ANNEALING=true \
LR_ANNEALING_TYPE=cosine \
LR_MIN_FACTOR=0.05 \
ENABLE_REWARD_SCALING=true \
ENABLE_VALUE_CLIPPING=true \
VALUE_CLIP_RANGE=150.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 1000000
```

**See `SLURM_OPTIMIZATION_GUIDE.md` for complete SLURM usage documentation.**

### Testing Different Configurations
```bash
# Test without optimizations (baseline)
python train_energynet.py --disable-orthogonal-init --total-timesteps 100000

# Test with all optimizations
python train_energynet.py \
    --use-lr-annealing --lr-annealing-type linear \
    --use-reward-scaling \
    --use-value-clipping \
    --total-timesteps 100000
```

## Monitoring and Logging

All optimization features log additional metrics to TensorBoard:

- **Learning Rates**: `LR/Actor`, `LR/Critic1`, `LR/Critic2`, `LR/Alpha`
- **Reward Scaling**: `RewardScaling/Mean_Obj_X`, `RewardScaling/Std_Obj_X`
- **Training Progress**: Enhanced logging with optimization status

## Performance Tips

1. **Start Conservative**: Begin with reward scaling and orthogonal initialization only
2. **Add Gradually**: Enable additional optimizations one at a time to assess impact
3. **Monitor Closely**: Watch TensorBoard logs for signs of instability
4. **Adjust Parameters**: Fine-tune gains and ranges based on your specific environment
5. **Compare Baselines**: Always compare against a baseline without optimizations

## Troubleshooting

### Common Issues and Solutions

**Training Instability**:
- Enable value clipping with a conservative range (200.0)
- Reduce learning rates
- Use more conservative orthogonal gains

**Slow Convergence**:
- Enable learning rate annealing with cosine decay
- Increase orthogonal gains slightly
- Enable reward scaling if objectives have different scales

**Poor Final Performance**:
- Reduce LR annealing minimum factor (don't decay too much)
- Adjust value clipping range
- Check reward scaling statistics in logs

## Compatibility

- ✅ Fully backward compatible with existing code
- ✅ All optimizations are optional (default: disabled except orthogonal init)
- ✅ Supports save/load with optimization states
- ✅ Works with existing training scripts

## Files Modified

1. `multi_objective_sac.py` - Core algorithm with optimizations
2. `train_energynet.py` - Training script with new CLI arguments
3. `run_train_energynet_v2.sh` - SLURM script with optimization support
4. `test_optimizations.py` - Comprehensive test suite
5. `optimization_examples.py` - Usage examples and demos
6. `SLURM_OPTIMIZATION_GUIDE.md` - Complete SLURM usage guide

The optimizations are now ready for production use! 🚀
