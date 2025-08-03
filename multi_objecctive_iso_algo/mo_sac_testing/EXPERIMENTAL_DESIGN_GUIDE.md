# 🧪 Experimental Design Guide for Multi-Objective SAC Optimizations

## 📊 **Controlled Experiments for Optimization Testing**

This guide provides a systematic approach to testing the optimization features fairly and scientifically.

## ⚖️ **Principle: Fair Comparison**

**CRITICAL**: Keep all parameters identical except for the optimization being tested.

### **Fixed Parameters for All Experiments**
```bash
TIMESTEPS=500000          # Same training duration
BATCH_SIZE=256           # Same batch size  
BUFFER_SIZE=1000000      # Same replay buffer
WEIGHTS="1.0 1.0 1.0"    # Equal objective weights
SEEDS=42                 # Same random seed (for initial comparison)
```

## 🔬 **Phase 1: Individual Optimization Testing**

Test each optimization **independently** to understand individual contributions:

```bash
# Baseline (no optimizations) - Control group
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Test 1: Reward scaling only
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Test 2: LR annealing only  
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Test 3: Value clipping only
ENABLE_VALUE_CLIPPING=true VALUE_CLIP_RANGE=200.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Test 4: Orthogonal initialization only (already default, but test without others)
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42
```

## 🔗 **Phase 2: Combination Testing**

Test combinations to find **synergistic effects**:

```bash
# Conservative combination
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Moderate combination
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Aggressive combination  
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true ENABLE_VALUE_CLIPPING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Full optimization suite
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42
```

## 🎲 **Phase 3: Statistical Validation**

Run **multiple seeds** for the best performing configurations:

```bash
# Example: If "moderate combination" performed best
BEST_CONFIG="ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true"

# Run with 5 different seeds for statistical significance
for SEED in 42 123 456 789 999; do
    eval $BEST_CONFIG \
    sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 $SEED
done

# Also run baseline with same seeds for comparison
for SEED in 42 123 456 789 999; do
    sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 $SEED
done
```

## 🔧 **Phase 4: Hyperparameter Sensitivity** (Optional)

After finding best optimization combination, test hyperparameter sensitivity:

```bash
# Test different LR annealing types (keep other params fixed)
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true

# Cosine annealing
LR_ANNEALING_TYPE=cosine eval $BEST_CONFIG \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Linear annealing  
LR_ANNEALING_TYPE=linear eval $BEST_CONFIG \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# Exponential annealing
LR_ANNEALING_TYPE=exponential eval $BEST_CONFIG \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42
```

```bash
# Test different value clipping ranges
ENABLE_VALUE_CLIPPING=true ENABLE_REWARD_SCALING=true

for CLIP_RANGE in 100.0 150.0 200.0 300.0; do
    VALUE_CLIP_RANGE=$CLIP_RANGE eval $BEST_CONFIG \
    sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42
done
```

## 📈 **Metrics to Track and Compare**

### **Primary Metrics**
1. **Final Scalarized Reward**: Performance at end of training
2. **Sample Efficiency**: Timesteps to reach 90% of final performance  
3. **Training Stability**: Standard deviation of episode rewards
4. **Convergence Speed**: Slope of learning curve

### **Secondary Metrics**
1. **Actor/Critic Loss Trends**: Training stability indicators
2. **Learning Rate Schedule**: Verify annealing is working
3. **Reward Scaling Statistics**: Mean/std of normalized rewards
4. **Computational Overhead**: Training time per timestep

## 📊 **Analysis Framework**

### **Results Organization**
```
results/
├── baseline/
│   ├── seed_42/
│   ├── seed_123/
│   └── ...
├── reward_scaling_only/
│   ├── seed_42/
│   └── ...
├── lr_annealing_only/
├── value_clipping_only/
├── moderate_combination/
├── aggressive_combination/
└── full_optimizations/
```

### **Comparison Script Template**
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_configurations(result_dirs, labels):
    """Compare multiple optimization configurations"""
    
    results = {}
    for dir_path, label in zip(result_dirs, labels):
        # Load results from each configuration
        rewards = load_episode_rewards(dir_path)
        results[label] = {
            'final_reward': np.mean(rewards[-100:]),  # Average of last 100 episodes
            'sample_efficiency': calculate_sample_efficiency(rewards),
            'stability': np.std(rewards[-1000:]),     # Std of last 1000 episodes  
            'learning_curve': rewards
        }
    
    # Create comparison plots
    plot_learning_curves(results)
    plot_final_performance(results)
    plot_sample_efficiency(results)
    
    return results

# Usage example
configurations = [
    'results/baseline/',
    'results/reward_scaling_only/', 
    'results/moderate_combination/',
    'results/full_optimizations/'
]

labels = ['Baseline', 'Reward Scaling', 'Moderate', 'Full Optimizations']
comparison_results = compare_configurations(configurations, labels)
```

## ⚠️ **Important Experimental Considerations**

### **What to Keep Constant**
- ✅ Total timesteps (500k recommended)
- ✅ Network architectures  
- ✅ Batch size and buffer size
- ✅ Environment configuration
- ✅ Objective weights
- ✅ Initial random seed (for initial comparison)

### **What to Vary Systematically**
- 🔄 Optimization features (one at a time, then combinations)
- 🔄 Random seeds (for statistical validation)
- 🔄 Hyperparameters (after finding best optimization set)

### **Common Pitfalls to Avoid**
- ❌ **Different timesteps** between configurations
- ❌ **Single seed comparison** (not statistically valid)
- ❌ **Testing all combinations** simultaneously (combinatorial explosion)
- ❌ **Ignoring computational overhead** (some optimizations may slow training)
- ❌ **Cherry-picking results** (report all experiments, not just best)

## 🎯 **Quick Start Experimental Protocol**

### **Minimal Viable Experiment** (4 runs)
```bash
# 1. Baseline
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# 2. Reward scaling (most promising single optimization)
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# 3. Moderate combination
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42

# 4. Full optimizations
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000 42
```

### **Comprehensive Experiment** (20+ runs)
- All individual optimizations (4 runs)
- Key combinations (4 runs)  
- Best configuration with multiple seeds (5 runs)
- Baseline with multiple seeds (5 runs)
- Hyperparameter sensitivity (5+ runs)

## 📋 **Experiment Tracking Template**

| Configuration | Timesteps | Seed | Final Reward | Sample Efficiency | Stability | Notes |
|---------------|-----------|------|--------------|-------------------|-----------|-------|
| Baseline | 500k | 42 | | | | |
| Reward Scaling | 500k | 42 | | | | |
| LR Annealing | 500k | 42 | | | | |
| Moderate Combo | 500k | 42 | | | | |
| Full Optimizations | 500k | 42 | | | | |

This systematic approach will give you **reliable, scientifically valid results** about which optimizations provide the most benefit for your Multi-Objective SAC algorithm! 🚀
