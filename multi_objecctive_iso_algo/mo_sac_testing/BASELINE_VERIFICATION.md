# Quick Test: Verify Baseline vs Optimizations

This script helps verify that the baseline truly has no optimizations enabled.

## Test Commands

### 1. True Baseline (No Optimizations)
```bash
# Should use Xavier init, no LR annealing, no reward scaling, no value clipping
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 10000
```

### 2. Individual Optimizations
```bash
# Only reward scaling
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 10000

# Only orthogonal initialization  
ENABLE_ORTHOGONAL_INIT=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 10000

# Only LR annealing
ENABLE_LR_ANNEALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 10000

# Only value clipping
ENABLE_VALUE_CLIPPING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 10000
```

### 3. All Optimizations
```bash
# All optimizations enabled
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 10000
```

## What to Look For

### Console Output
Each run should clearly show optimization status:
```
Optimization features enabled:
  LR Annealing: False
  Reward Scaling: False  
  Orthogonal Init: False
  Value Clipping: False
```

### TensorBoard Metrics
- **Baseline**: No LR scheduling plots, no reward scaling statistics
- **With optimizations**: Additional plots for enabled features

### Performance Differences
- **Baseline**: Standard SAC performance with Xavier initialization
- **Individual optimizations**: Isolated effect of each optimization
- **All optimizations**: Combined effect

## Expected Default Status (All False)
- `use_lr_annealing: False`
- `use_reward_scaling: False` 
- `use_orthogonal_init: False`
- `use_value_clipping: False`

This ensures a **true vanilla SAC baseline** for fair comparison!
