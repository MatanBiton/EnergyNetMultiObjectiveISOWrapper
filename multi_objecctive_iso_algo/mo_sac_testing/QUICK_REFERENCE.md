# Quick Reference Guide for MO-SAC Scripts

## Running Scripts

### On Windows:
```cmd
# Test the algorithm on various environments
run_test_mo_sac.bat
run_test_mo_sac.bat CartPole
run_test_mo_sac.bat Pendulum

# Train on EnergyNet environment
run_train_energynet.bat
run_train_energynet.bat quick-test
run_train_energynet.bat cost-priority
run_train_energynet.bat stability-priority
```

### On Linux/Mac:
```bash
# Make scripts executable first
chmod +x *.sh

# Test the algorithm on various environments
./run_test_mo_sac.sh
./run_test_mo_sac.sh CartPole
./run_test_mo_sac.sh Pendulum

# Train on EnergyNet environment
./run_train_energynet.sh
./run_train_energynet.sh --quick-test
./run_train_energynet.sh --cost-priority
./run_train_energynet.sh --stability-priority
```

## Key Parameters to Modify

### In `run_train_energynet` scripts:

#### Multi-Objective Weights (most important to tune)
```bash
--weights 0.6 0.4     # Default: slightly prefer cost over stability
--weights 1.0 0.0     # Only optimize cost (ignore stability)
--weights 0.0 1.0     # Only optimize stability (ignore cost)
--weights 0.5 0.5     # Equal importance to both objectives
--weights 0.8 0.2     # Strongly prefer cost reduction
--weights 0.3 0.7     # Strongly prefer stability
```

#### Training Duration
```bash
--total-timesteps 50000     # Quick test (~10 minutes)
--total-timesteps 500000    # Medium training (~1-2 hours)
--total-timesteps 1000000   # Full training (~2-4 hours)
--total-timesteps 2000000   # Extended training (~4-8 hours)
```

#### Learning Rates (if having convergence issues)
```bash
# If learning is slow, increase:
--actor-lr 1e-3 --critic-lr 1e-3 --alpha-lr 1e-3

# If training is unstable, decrease:
--actor-lr 1e-4 --critic-lr 1e-4 --alpha-lr 1e-4
```

#### Network Size (edit in the Python files)
```python
# In multi_objective_sac.py constructor:
actor_hidden_dims=[128, 128]     # Smaller networks (faster)
actor_hidden_dims=[256, 256]     # Default
actor_hidden_dims=[512, 512]     # Larger networks (more capacity)
actor_hidden_dims=[256, 256, 128] # Deeper networks
```

#### Evaluation Frequency
```bash
--eval-freq 5000      # Evaluate every 5k steps (more frequent)
--eval-freq 20000     # Default
--eval-freq 50000     # Less frequent evaluation (faster training)
```

## Output Files

### After running test scripts:
- `models/` - Trained model checkpoints
- `plots/` - Training curves and analysis plots
- `runs/` - TensorBoard logs

### After running training scripts:
- `energynet_experiments/` - All training results
  - `models/` - Model checkpoints (.pth files)
  - `logs/` - TensorBoard logs
  - `*_config.json` - Training configuration
  - `*_results.json` - Final results and statistics

## Monitoring Training

### TensorBoard (real-time monitoring):
```bash
# For test environments
tensorboard --logdir runs/

# For EnergyNet training
tensorboard --logdir energynet_experiments/logs/
```

### View plots:
```bash
# Windows
dir plots\*.png
start plots\mo_sac_CartPole_training.png

# Linux/Mac
ls plots/*.png
open plots/mo_sac_CartPole_training.png  # Mac
xdg-open plots/mo_sac_CartPole_training.png  # Linux
```

## Troubleshooting

### Common Issues:

1. **"Python not found"**
   - Install Python 3.8+ and ensure it's in PATH
   - Try `python3` instead of `python`

2. **"Module not found"**
   - Install requirements: `pip install -r requirements.txt`
   - Check virtual environment activation

3. **"CUDA out of memory"**
   - Reduce batch size: `--batch-size 128`
   - Reduce buffer size: `--buffer-size 500000`
   - Use CPU: comment out `CUDA_VISIBLE_DEVICES` line

4. **Slow training**
   - Use GPU if available
   - Increase learning rates
   - Reduce evaluation frequency

5. **Unstable training**
   - Decrease learning rates
   - Increase tau: `--tau 0.01`
   - Check reward scaling

## Quick Experiments

### Compare different objectives priorities:
```bash
# Run multiple experiments
./run_train_energynet.sh --cost-priority
./run_train_energynet.sh --stability-priority

# Compare results in tensorboard
tensorboard --logdir energynet_experiments/logs/
```

### Test different environments:
```bash
# Quick tests on all environments
./run_test_mo_sac.sh CartPole
./run_test_mo_sac.sh Mountain
./run_test_mo_sac.sh Pendulum
./run_test_mo_sac.sh Lunar
```

### Hyperparameter sweep (manual):
Edit the script files and change:
- Learning rates: `1e-4, 3e-4, 1e-3`
- Discount factors: `0.95, 0.99, 0.995`
- Network sizes: `[128,128], [256,256], [512,512]`
- Batch sizes: `128, 256, 512`

## Performance Tips

1. **Use GPU if available** (scripts detect automatically)
2. **Start with quick tests** before long training runs
3. **Monitor TensorBoard** to catch issues early
4. **Save intermediate checkpoints** (scripts do this automatically)
5. **Run multiple seeds** for statistical significance
6. **Keep notes** of what parameter changes work best
