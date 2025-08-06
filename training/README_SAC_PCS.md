# SAC PCS Agent for EnergyNet

This directory contains a complete implementation of a SAC-based PCS (Power Control Strategy) agent that can be trained on the EnergyNet PCSUnitEnv environment.

## Overview

The SAC PCS Agent (`SACPCSAgent`) is a deep reinforcement learning agent that:
- Inherits from `GenericPCSAgent` for compatibility with the EnergyNet ISO environment
- Implements Soft Actor-Critic (SAC) algorithm with all optimizations from the multi-objective SAC
- Can be trained to control battery storage systems in energy markets
- Supports training with or without a pre-trained ISO model

## Features

### Core SAC Implementation
- **Actor-Critic Architecture**: Continuous action spaces with twin critic networks
- **Entropy Regularization**: Automatic or manual entropy coefficient tuning
- **Target Networks**: Soft updates for stable training
- **Experience Replay**: Efficient sampling from replay buffer

### Optimization Features
- **Learning Rate Annealing**: Cosine, linear, or exponential decay schedules
- **Reward Scaling**: Running normalization of rewards for stable training
- **Orthogonal Initialization**: Better network initialization for faster convergence
- **Value Clipping**: Prevents unstable value function estimates
- **Device Support**: Automatic GPU/CPU detection and usage

### Training Features
- **Multi-seed Training**: Statistical significance through multiple runs
- **Comprehensive Logging**: TensorBoard integration and JSON result files
- **Model Checkpointing**: Save/load functionality with complete state
- **Evaluation**: Regular policy evaluation during training

## Files

```
EnergyNetMoISO/pcs_models/
├── sac_pcs_agent.py              # Main SAC PCS Agent implementation
├── generic_pcs_agent.py          # Base class for PCS agents
└── __init__.py                   # Package initialization

training/
├── sac_pcs_training.py           # Training script
├── run_train_sac_pcs.sh          # SLURM training script
└── run_train_sac_pcs_multiseed.sh # Multi-seed SLURM script

testing/
└── test_sac_pcs.py              # Test suite for SAC PCS agent
```

## Usage

### Basic Training

```bash
# Train SAC PCS agent without ISO model
python training/sac_pcs_training.py --experiment-name basic_pcs --total-timesteps 500000

# Train with pre-trained ISO model
python training/sac_pcs_training.py \
    --experiment-name pcs_with_iso \
    --trained-iso-model-path /path/to/iso_model.pth \
    --total-timesteps 1000000
```

### SLURM Training

```bash
# Basic SLURM job
sbatch -c 4 --gres=gpu:1 training/run_train_sac_pcs.sh

# With custom parameters
sbatch -c 4 --gres=gpu:1 training/run_train_sac_pcs.sh my_experiment 500000 3e-4 256

# With ISO model
sbatch -c 4 --gres=gpu:1 training/run_train_sac_pcs.sh \
    pcs_with_iso 500000 3e-4 256 /path/to/iso_model.pth

# With optimizations
ENABLE_LR_ANNEALING=true ENABLE_REWARD_SCALING=true \
sbatch -c 4 --gres=gpu:1 training/run_train_sac_pcs.sh optimized_pcs 1000000

# Full optimization suite
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 training/run_train_sac_pcs.sh full_opt 1000000
```

### Multi-Seed Training

```bash
# Run with 5 different seeds for statistical significance
sbatch -c 4 --gres=gpu:1 training/run_train_sac_pcs_multiseed.sh baseline_study

# Multi-seed with optimizations
ENABLE_ALL_OPTIMIZATIONS=true \
sbatch -c 4 --gres=gpu:1 training/run_train_sac_pcs_multiseed.sh optimized_study
```

## Environment Variables for Optimizations

### Learning Rate Annealing
- `ENABLE_LR_ANNEALING=true`: Enable learning rate annealing
- `LR_ANNEALING_TYPE=cosine|linear|exponential`: Annealing schedule type
- `LR_ANNEALING_STEPS=<number>`: Number of steps for annealing (auto if not set)
- `LR_MIN_FACTOR=<float>`: Minimum LR as fraction of initial (default: 0.1)
- `LR_DECAY_RATE=<float>`: Decay rate for exponential annealing (default: 0.95)

### Reward Scaling
- `ENABLE_REWARD_SCALING=true`: Enable reward normalization
- `REWARD_SCALE_EPSILON=<float>`: Epsilon for numerical stability (default: 1e-4)

### Orthogonal Initialization
- `ENABLE_ORTHOGONAL_INIT=true`: Enable orthogonal weight initialization
- `ORTHOGONAL_GAIN=<float>`: General orthogonal gain (default: 1.0)
- `ACTOR_ORTHOGONAL_GAIN=<float>`: Actor-specific gain (default: 0.01)
- `CRITIC_ORTHOGONAL_GAIN=<float>`: Critic-specific gain (default: 1.0)

### Value Clipping
- `ENABLE_VALUE_CLIPPING=true`: Enable value function clipping
- `VALUE_CLIP_RANGE=<float>`: Clipping range (default: 200.0)

### All Optimizations
- `ENABLE_ALL_OPTIMIZATIONS=true`: Enable all optimizations with good defaults

## Command Line Arguments

### Training Arguments
- `--experiment-name`: Name for the experiment (default: sac_pcs_training)
- `--total-timesteps`: Total training timesteps (default: 1000000)
- `--learning-starts`: Timesteps before training starts (default: 10000)
- `--eval-freq`: Evaluation frequency (default: 20000)
- `--eval-episodes`: Episodes per evaluation (default: 10)
- `--save-freq`: Model save frequency (default: 100000)

### Agent Arguments
- `--actor-lr`: Actor learning rate (default: 3e-4)
- `--critic-lr`: Critic learning rate (default: 3e-4)
- `--alpha-lr`: Alpha learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--tau`: Target network update rate (default: 0.005)
- `--batch-size`: Training batch size (default: 256)
- `--buffer-capacity`: Replay buffer capacity (default: 1000000)

### Environment Arguments
- `--trained-iso-model-path`: Path to pre-trained ISO model (optional)

## Code Example

```python
from EnergyNetMoISO.pcs_models.sac_pcs_agent import SACPCSAgent
from energy_net.env.pcs_unit_v0 import PCSUnitEnv

# Create environment
env = PCSUnitEnv()

# Create agent
agent = SACPCSAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_bounds=(env.action_space.low[0], env.action_space.high[0]),
    use_lr_annealing=True,
    use_reward_scaling=True,
    verbose=True
)

# Training loop
state, _ = env.reset()
for step in range(100000):
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    
    agent.store_transition(state, action, reward, next_state, done)
    
    if step > 1000:  # Start training after some exploration
        agent.update()
    
    if done:
        state, _ = env.reset()
    else:
        state = next_state

# Save trained agent
agent.save("trained_pcs_agent.pth")

# Use agent for inference
action, _ = agent.predict(state, deterministic=True)
```

## Results and Monitoring

### TensorBoard
```bash
tensorboard --logdir pcs_experiments/logs/
```

### Result Files
Training generates:
- `{experiment}_results.json`: Complete training statistics
- `{experiment}_final.pth`: Final model checkpoint
- `{experiment}_best.pth`: Best performing model during training

### Multi-Seed Summary
Multi-seed runs generate additional summary statistics:
- `multiseed_summary_{experiment}/multiseed_summary.json`: Aggregated statistics across seeds
- Individual result files for each seed

## Testing

Run the test suite to verify the implementation:

```bash
python testing/test_sac_pcs.py
```

Tests include:
- Basic functionality (action selection, storage)
- Training mechanics (updates, optimization)
- Save/load functionality
- Integration with PCSUnitEnv
- Compatibility with ISO models

## Integration with ISO Training

The trained SAC PCS agent can be used during ISO training:

```python
# Load trained PCS agent
pcs_agent = SACPCSAgent(state_dim=4, action_dim=1, action_bounds=(-100, 100))
pcs_agent.load("trained_pcs_agent.pth")

# Use in ISO environment
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
iso_env = MultiObjectiveISOEnv(trained_pcs_model=pcs_agent)
```

## Performance Tips

1. **Use GPU**: Training is significantly faster with CUDA
2. **Tune Learning Rates**: Start with 3e-4, adjust based on convergence
3. **Enable Optimizations**: Use environment variables for better performance
4. **Monitor Training**: Watch TensorBoard for loss curves and policy performance
5. **Multi-Seed**: Run multiple seeds for robust evaluation
6. **Buffer Size**: Larger buffers generally improve performance but use more memory

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure `energy_net` package is properly installed
2. **CUDA Issues**: Check GPU availability with `torch.cuda.is_available()`
3. **Memory Issues**: Reduce batch size or buffer capacity
4. **Slow Training**: Enable GPU, reduce network sizes, or increase learning rates
5. **Unstable Training**: Enable value clipping, reduce learning rates

### Debug Tips

1. Use `--verbose` flag for detailed output
2. Check TensorBoard for training curves
3. Verify environment setup with test script
4. Start with smaller timestep counts for debugging

## References

- [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)
- [EnergyNet Documentation](https://github.com/mihirp1998/EnergyNet)
- [SAC Implementation Details](https://spinningup.openai.com/en/latest/algorithms/sac.html)
