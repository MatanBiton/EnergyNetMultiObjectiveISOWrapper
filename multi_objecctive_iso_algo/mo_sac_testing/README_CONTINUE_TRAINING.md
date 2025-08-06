# Continue Training MO-SAC on EnergyNet

This directory contains scripts for continuing training of an existing Multi-Objective SAC agent on the EnergyNet MoISO environment.

## Files

- `continue_train_energynet.py` - Python script for loading and continuing training of MO-SAC agents
- `run_continue_train_energynet.sh` - SLURM batch script for cluster execution
- `test_continue_train.sh` - Local testing script (without SLURM)
- `README_CONTINUE_TRAINING.md` - This documentation

## Quick Start

### Local Testing (Recommended First)

```bash
# Activate the conda environment
conda activate IsoMOEnergyNet

# Test with a minimal example
./test_continue_train.sh ../best_model.zip 10000 test_run

# With trained PCS model
TRAINED_PCS_MODEL=/path/to/pcs_model.pth ./test_continue_train.sh ../best_model.zip 10000 with_pcs

# With optimizations
ENABLE_LR_ANNEALING=true RANDOM_SEED=42 ./test_continue_train.sh ../best_model.zip 20000 optimized
```

### SLURM Cluster Execution

```bash
# Basic continue training
sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/trained_model.pth

# With additional timesteps and custom name
sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/trained_model.pth 500000 experiment_name

# With trained PCS model
TRAINED_PCS_MODEL=/path/to/pcs_model.pth \
sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/trained_model.pth 500000 with_pcs

# With full configuration
USE_DISPATCH_ACTION=true DISPATCH_STRATEGY=UNIFORM TRAINED_PCS_MODEL=/path/to/pcs.pth \
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine RANDOM_SEED=123 \
sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh /path/to/trained_model.pth 1000000 full_config
```

## Parameters

### Required Parameters

- **Model Path**: Path to the trained MO-SAC model file (`.pth` format)

### Optional Parameters

1. **Timesteps** (default: 500,000): Number of additional training timesteps
2. **Experiment Name** (default: "mo_sac_continued"): Name for the continued training experiment
3. **Random Seed**: For reproducibility

### Environment Variables

#### Environment Configuration
- `TRAINED_PCS_MODEL`: Path to trained PCS model (SAC PCS agent)
- `USE_DISPATCH_ACTION`: Enable dispatch action (`true`/`false`)
- `DISPATCH_STRATEGY`: Dispatch strategy (`PROPORTIONAL`, `UNIFORM`, etc.)

#### Optimization Overrides
- `ENABLE_LR_ANNEALING`: Enable learning rate annealing (`true`/`false`)
- `LR_ANNEALING_TYPE`: Type of annealing (`cosine`, `linear`, `exponential`)
- `LR_ANNEALING_STEPS`: Number of steps for LR annealing
- `LR_MIN_FACTOR`: Minimum LR as fraction of initial LR (default: 0.1)
- `LR_DECAY_RATE`: Decay rate for exponential annealing (default: 0.95)
- `ENABLE_REWARD_SCALING`: Enable reward scaling (`true`/`false`)
- `REWARD_SCALE_EPSILON`: Epsilon for reward scaling (default: 1e-4)
- `ENABLE_VALUE_CLIPPING`: Enable value clipping (`true`/`false`)
- `VALUE_CLIP_RANGE`: Value clipping range (default: 200.0)

#### Other
- `RANDOM_SEED`: Random seed for reproducibility

## Usage Examples

### 1. Basic Continue Training

Continue training an existing model for additional 500k timesteps:

```bash
# Local test
./test_continue_train.sh ../best_model.zip 50000

# SLURM cluster
sbatch ./run_continue_train_energynet.sh ../best_model.zip 500000
```

### 2. With Trained PCS Model

Use a trained SAC PCS agent in the environment:

```bash
# Local test
TRAINED_PCS_MODEL=../training/pcs_experiments/sac_pcs_best_model.pth \
./test_continue_train.sh ../best_model.zip 50000 with_pcs

# SLURM cluster
TRAINED_PCS_MODEL=/path/to/sac_pcs_model.pth \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 500000 with_pcs
```

### 3. With Optimization Overrides

Override optimization settings from the original training:

```bash
# Enable new optimizations
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine ENABLE_REWARD_SCALING=true \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 1000000 optimized

# Change environment configuration
USE_DISPATCH_ACTION=true DISPATCH_STRATEGY=UNIFORM \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 500000 dispatch_uniform
```

### 4. Reproducible Training

Set random seed for reproducible results:

```bash
RANDOM_SEED=42 sbatch ./run_continue_train_energynet.sh ../best_model.zip 500000 reproducible
```

### 5. Complete Configuration

Full example with all options:

```bash
USE_DISPATCH_ACTION=true \
DISPATCH_STRATEGY=PROPORTIONAL \
TRAINED_PCS_MODEL=/path/to/pcs_model.pth \
ENABLE_LR_ANNEALING=true \
LR_ANNEALING_TYPE=cosine \
LR_MIN_FACTOR=0.05 \
ENABLE_REWARD_SCALING=true \
ENABLE_VALUE_CLIPPING=true \
VALUE_CLIP_RANGE=150.0 \
RANDOM_SEED=123 \
sbatch -c 4 --gres=gpu:1 ./run_continue_train_energynet.sh \
    /path/to/trained_model.pth 1000000 complete_config
```

## Model Compatibility

### Supported Model Formats

- **MO-SAC models**: `.pth` files saved by the `MultiObjectiveSAC.save()` method
- **Legacy models**: Most models from previous training runs should be compatible

### Required Model Components

The loaded model must contain:
- Actor, critic, and target network states
- Optimizer states
- Training configuration (weights, hyperparameters)
- Training progress (step count, episode count)

### PCS Model Compatibility

- **SAC PCS models**: Trained using `sac_pcs_training.py`
- **Model format**: `.pth` files compatible with `SACPCSAgent.load()`
- **Location**: Should be accessible from the compute node

## Output Files

After training, the following files will be created:

### Configuration and Results
- `continue_train_energynet_<name>_<timestamp>_config.json`: Training configuration
- `continue_train_energynet_<name>_<timestamp>_results.json`: Training results and metrics

### Models
- `models/continue_train_energynet_<name>_<timestamp>_final.pth`: Final trained model
- `models/continue_train_energynet_<name>_<timestamp>_<step>.pth`: Intermediate checkpoints

### Logs and Analysis
- `logs/continue_train_energynet_<name>_<timestamp>/`: TensorBoard logs
- `plots/`: Training plots and analysis (if generated)

## Performance Tracking

The script automatically tracks:

### Before Continuing Training
- Current model performance (10 episodes)
- Baseline metrics for comparison

### During Training
- Episode rewards
- Training losses
- Learning rate schedules (if annealing enabled)
- Optimization metrics

### After Training
- Final performance (50 episodes)
- Improvement metrics
- Comparison with baseline

### Metrics Tracked
- **Cost Reward**: Economic efficiency
- **Stability Reward**: System stability
- **Scalarized Reward**: Weighted combination
- **Training Statistics**: Episode count, convergence

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Error: Model file not found
   ```
   - Check if the model path is correct
   - Ensure the file is accessible from the compute node

2. **Compatibility Issues**
   ```
   KeyError: 'weights'
   ```
   - Model might be from an older version
   - Try loading with different parameters

3. **Memory Issues**
   ```
   CUDA out of memory
   ```
   - Reduce batch size
   - Use smaller models
   - Switch to CPU training

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'multi_objective_sac'
   ```
   - Ensure you're in the correct conda environment
   - Check if all dependencies are installed

### Performance Issues

1. **Slow Training**
   - Check if GPU is being used
   - Reduce evaluation frequency
   - Use smaller networks

2. **Poor Convergence**
   - Try different learning rates
   - Enable optimization features
   - Check if the original model was well-trained

### Environment Issues

1. **Conda Environment**
   ```bash
   conda activate IsoMOEnergyNet
   ```

2. **Missing Dependencies**
   ```bash
   pip install torch numpy gymnasium matplotlib
   ```

3. **CUDA Issues**
   - Check CUDA installation
   - Verify PyTorch CUDA compatibility

## Best Practices

### Before Starting

1. **Test Locally First**: Use `test_continue_train.sh` with small timesteps
2. **Verify Model**: Ensure the base model loads and runs correctly
3. **Check Environment**: Confirm all dependencies are available

### Training Configuration

1. **Gradual Increases**: Start with smaller timestep increases
2. **Monitor Progress**: Use frequent evaluation during testing
3. **Save Frequently**: Use reasonable save frequencies to avoid losing progress

### Resource Management

1. **GPU Memory**: Monitor GPU usage, especially with large models
2. **Storage**: Ensure sufficient disk space for logs and checkpoints
3. **Time Limits**: Consider SLURM time limits for long training runs

### Experiment Organization

1. **Clear Naming**: Use descriptive experiment names
2. **Version Control**: Keep track of code changes
3. **Documentation**: Document parameter choices and results

## Integration with Existing Workflow

### With Original Training

```bash
# Original training
sbatch ./run_train_energynet_v2.sh baseline 1000000

# Continue training from best model
sbatch ./run_continue_train_energynet.sh baseline_best_model.pth 500000 baseline_continued
```

### With PCS Training

```bash
# Train PCS agent first
sbatch ../training/run_train_sac_pcs.sh

# Use PCS model in continue training
TRAINED_PCS_MODEL=../training/pcs_experiments/sac_pcs_best_model.pth \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 500000 with_trained_pcs
```

### Results Analysis

```bash
# View results
./view_results.sh

# Compare with original training
python compare_training_results.py original_results.json continued_results.json
```

## Advanced Usage

### Custom Optimization Schedules

Modify learning rate schedules for fine-tuning:

```bash
# Aggressive annealing for convergence
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=exponential LR_DECAY_RATE=0.9 LR_MIN_FACTOR=0.01 \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 200000 aggressive_tuning

# Gentle annealing for stability
ENABLE_LR_ANNEALING=true LR_ANNEALING_TYPE=cosine LR_MIN_FACTOR=0.3 \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 500000 gentle_tuning
```

### Multi-Stage Training

Chain multiple continue training runs:

```bash
# Stage 1: Basic continue training
sbatch ./run_continue_train_energynet.sh original_model.pth 500000 stage1

# Stage 2: With optimizations (after stage 1 completes)
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
sbatch ./run_continue_train_energynet.sh stage1_final_model.pth 300000 stage2

# Stage 3: Fine-tuning (after stage 2 completes)
ENABLE_VALUE_CLIPPING=true \
sbatch ./run_continue_train_energynet.sh stage2_final_model.pth 200000 stage3_final
```

### Hyperparameter Exploration

Use continue training to explore different configurations:

```bash
# Different dispatch strategies
for strategy in PROPORTIONAL UNIFORM RANDOM; do
    DISPATCH_STRATEGY=$strategy \
    sbatch ./run_continue_train_energynet.sh ../best_model.zip 200000 dispatch_${strategy,,}
done

# Different optimization combinations
ENABLE_LR_ANNEALING=true \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 200000 opt_lr_only

ENABLE_REWARD_SCALING=true \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 200000 opt_reward_only

ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true ENABLE_VALUE_CLIPPING=true \
sbatch ./run_continue_train_energynet.sh ../best_model.zip 200000 opt_all
```
