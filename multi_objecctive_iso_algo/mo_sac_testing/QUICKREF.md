# MO-SAC Optimization Quick Reference

## 🚀 SLURM Quick Commands

### Basic Training
```bash
# Default (100k timesteps, orthogonal init only)
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh

# Custom timesteps
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

### Recommended Configurations

#### 🟢 Conservative (Production Safe)
```bash
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

#### 🟡 Moderate (Balanced)
```bash
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true ENABLE_VALUE_CLIPPING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

#### 🔴 Aggressive (Max Performance)
```bash
ENABLE_ALL_OPTIMIZATIONS=true LR_MIN_FACTOR=0.05 VALUE_CLIP_RANGE=100.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

#### ⚡ Quick Enable All
```bash
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

### 🧪 Fair Comparison Testing (Same timesteps: 500k)

```bash
# Baseline
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Individual optimizations
ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

ENABLE_VALUE_CLIPPING=true VALUE_CLIP_RANGE=200.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

# Combined optimizations
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000

ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
```

## 🎛️ Environment Variables

| Feature | Enable | Configure |
|---------|--------|-----------|
| **LR Annealing** | `ENABLE_LR_ANNEALING=true` | `LR_ANNEALING_TYPE=cosine` `LR_MIN_FACTOR=0.1` |
| **Reward Scaling** | `ENABLE_REWARD_SCALING=true` | `REWARD_SCALE_EPSILON=1e-4` |
| **Orthogonal Init** | (enabled by default) | `ACTOR_ORTHOGONAL_GAIN=0.01` `CRITIC_ORTHOGONAL_GAIN=1.0` |
| **Value Clipping** | `ENABLE_VALUE_CLIPPING=true` | `VALUE_CLIP_RANGE=200.0` |
| **All Optimizations** | `ENABLE_ALL_OPTIMIZATIONS=true` | Uses good defaults |

## 🎯 Common Use Cases

### Research Experiment
```bash
# Multiple runs with different settings
for lr_factor in 0.05 0.1 0.2; do
    ENABLE_LR_ANNEALING=true LR_MIN_FACTOR=$lr_factor \
    ENABLE_REWARD_SCALING=true ENABLE_VALUE_CLIPPING=true \
    sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
done
```

### Long Training Run
```bash
# 5M timesteps with all optimizations
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 8 --gres=gpu:1 --mem=64G --time=72:00:00 \
./run_train_energynet_v2.sh 5000000
```

### Hyperparameter Tuning
```bash
# Different clipping ranges
for clip in 100.0 150.0 200.0; do
    ENABLE_VALUE_CLIPPING=true VALUE_CLIP_RANGE=$clip \
    sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 500000
done
```

## 🐍 Direct Python Usage

### Conservative
```bash
python train_energynet.py --use-reward-scaling --total-timesteps 500000
```

### Moderate
```bash
python train_energynet.py \
    --use-lr-annealing --lr-annealing-type cosine \
    --use-reward-scaling --use-value-clipping \
    --total-timesteps 1000000
```

### Aggressive
```bash
python train_energynet.py \
    --use-lr-annealing --lr-annealing-type cosine --lr-min-factor 0.05 \
    --use-reward-scaling --reward-scale-epsilon 1e-5 \
    --use-value-clipping --value-clip-range 100.0 \
    --actor-lr 5e-4 --total-timesteps 1000000
```

## 📊 Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### View Logs
```bash
tail -f slurm_energynet_*.out
```

### TensorBoard
```bash
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

## 🔧 Troubleshooting

### Environment Variables Not Working
```bash
# Correct: Set before sbatch
ENABLE_LR_ANNEALING=true sbatch ...

# Wrong: Set after sbatch  
sbatch ENABLE_LR_ANNEALING=true ...
```

### Training Unstable
```bash
# Add value clipping
ENABLE_VALUE_CLIPPING=true VALUE_CLIP_RANGE=200.0 \
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh
```

### Memory Issues
```bash
# Increase memory allocation
sbatch -c 8 --gres=gpu:1 --mem=64G ./run_train_energynet_v2.sh
```

## 📋 Testing Checklist

- [ ] Test baseline (no optimizations): `sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2.sh 10000`
- [ ] Test conservative: `ENABLE_REWARD_SCALING=true sbatch ...`
- [ ] Test moderate: `ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true ...`
- [ ] Test aggressive: `ENABLE_ALL_OPTIMIZATIONS=true ...`
- [ ] Monitor TensorBoard logs for optimization metrics
- [ ] Compare final performance across configurations

## 📚 Documentation Files

- `OPTIMIZATION_GUIDE.md` - Complete feature documentation
- `SLURM_OPTIMIZATION_GUIDE.md` - Detailed SLURM usage
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

---
**Quick Help**: All optimizations are backward compatible and optional! Start with `ENABLE_REWARD_SCALING=true` for safe improvements.
