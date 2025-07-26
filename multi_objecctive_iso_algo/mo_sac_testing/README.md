# Multi-Objective Soft Actor-Critic (MO-SAC) Testing Module

This module provides comprehensive testing capabilities for the Multi-Objective Soft Actor-Critic algorithm on various continuous control environments, including the EnergyNet MoISO environment.

## Features

- **Complete MO-SAC Implementation**: State-of-the-art multi-objective reinforcement learning algorithm
- **Multiple Test Environments**: Various continuous control environments with multi-objective rewards
- **EnergyNet Integration**: Specific training utilities for the EnergyNet MoISO environment
- **Comprehensive Analysis**: Weight sensitivity analysis, Pareto front visualization, and performance metrics
- **Flexible Configuration**: Tunable hyperparameters for networks, optimizers, and training

## Test Environments

### 1. MultiObjectiveContinuousCartPole-v0
- **Objectives**: Balance pole (angle) + Keep cart centered (position)
- **State**: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
- **Action**: [force] (continuous)
- **Good for**: Testing basic multi-objective trade-offs

### 2. MultiObjectiveMountainCarContinuous-v0
- **Objectives**: Reach goal quickly (time) + Minimize energy consumption
- **State**: [position, velocity]
- **Action**: [force] (continuous)
- **Good for**: Testing exploration vs exploitation trade-offs

### 3. MultiObjectivePendulum-v0
- **Objectives**: Keep pendulum upright + Minimize control effort
- **State**: [cos(theta), sin(theta), angular_velocity]
- **Action**: [torque] (continuous)
- **Good for**: Testing stability vs efficiency trade-offs

### 4. MultiObjectiveLunarLander-v0
- **Objectives**: Safe landing + Fuel efficiency
- **State**: [x, y, vx, vy, angle, angular_velocity]
- **Action**: [main_engine, left_engine, right_engine] (continuous)
- **Good for**: Testing complex multi-objective scenarios

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure EnergyNet is properly installed according to its documentation.

## Usage

### Quick Test on a Single Environment

```bash
# Test MO-SAC on CartPole
python test_mo_sac.py MultiObjectiveContinuousCartPole-v0

# Test on Pendulum
python test_mo_sac.py MultiObjectivePendulum-v0
```

### Comprehensive Testing

```bash
# Run tests on all environments with weight sensitivity analysis
python test_mo_sac.py
```

### Train on EnergyNet Environment

```bash
# Basic training with default parameters
python train_energynet.py

# Advanced training with custom parameters
python train_energynet.py --experiment-name my_experiment \
                         --total-timesteps 2000000 \
                         --weights 0.7 0.3 \
                         --actor-lr 1e-4 \
                         --critic-lr 1e-4 \
                         --verbose
```

### Training Parameters

#### Environment Parameters
- `--use-dispatch-action`: Use dispatch action in environment
- `--dispatch-strategy`: Dispatch strategy (default: PROPORTIONAL)

#### Training Parameters
- `--total-timesteps`: Total training timesteps (default: 1000000)
- `--learning-starts`: Timesteps before learning starts (default: 10000)
- `--eval-freq`: Evaluation frequency (default: 20000)
- `--save-freq`: Model save frequency (default: 100000)

#### Agent Parameters
- `--weights`: Multi-objective weights [cost_weight, stability_weight] (default: [0.6, 0.4])
- `--actor-lr`: Actor learning rate (default: 3e-4)
- `--critic-lr`: Critic learning rate (default: 3e-4)
- `--alpha-lr`: Alpha learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--tau`: Target network update rate (default: 0.005)
- `--buffer-size`: Replay buffer size (default: 1000000)
- `--batch-size`: Batch size (default: 256)

## Algorithm Details

### Multi-Objective SAC

The Multi-Objective SAC algorithm extends the standard SAC algorithm to handle multiple objectives:

1. **Scalarization**: Uses weighted sum to combine multiple objectives
2. **Dual Critics**: Two critic networks for each objective to reduce overestimation bias
3. **Entropy Regularization**: Automatic entropy tuning for exploration
4. **Soft Updates**: Target network updates for stability

### Key Features

- **Continuous Action Spaces**: Designed for continuous control problems
- **Multi-Objective Rewards**: Handles environments with multiple reward signals
- **Configurable Networks**: Customizable actor and critic architectures
- **Automatic Hyperparameter Tuning**: Automatic entropy coefficient tuning
- **Comprehensive Logging**: TensorBoard integration and detailed metrics

## Output Files

### Training Results
- `models/`: Saved model checkpoints
- `plots/`: Training plots and visualizations
- `logs/`: TensorBoard logs
- `*_config.json`: Experiment configuration
- `*_results.json`: Training results and statistics

### Analysis Tools

1. **Training Curves**: Individual objective rewards, scalarized rewards, episode lengths
2. **Pareto Fronts**: Visualization of trade-offs between objectives (for 2-objective problems)
3. **Weight Sensitivity**: Performance comparison across different weight configurations
4. **Correlation Analysis**: Objective correlation matrices

## Code Structure

```
mo_sac_testing/
├── __init__.py
├── test_environments.py    # Multi-objective test environments
├── test_mo_sac.py         # Comprehensive testing script
├── train_energynet.py     # EnergyNet-specific training
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Example: Weight Sensitivity Analysis

```python
# Test different weight configurations
weight_configs = [
    np.array([1.0, 0.0]),  # Only first objective
    np.array([0.0, 1.0]),  # Only second objective
    np.array([0.5, 0.5]),  # Equal weights
    np.array([0.7, 0.3]),  # Prefer first objective
    np.array([0.3, 0.7])   # Prefer second objective
]

results = test_weight_sensitivity(
    env_name='MultiObjectivePendulum-v0',
    weight_configs=weight_configs,
    total_timesteps=50000
)
```

## Performance Tips

1. **Learning Starts**: Use enough learning_starts (e.g., 10k) for good initial buffer diversity
2. **Buffer Size**: Larger buffers (1M) generally improve performance
3. **Network Size**: Start with [256, 256] hidden layers, increase if needed
4. **Learning Rates**: 3e-4 is generally a good starting point
5. **Weights**: Start with equal weights, then tune based on preference

## Monitoring Training

1. **TensorBoard**: Use `tensorboard --logdir runs/` to monitor training
2. **Console Output**: Enable verbose mode for detailed progress
3. **Evaluation**: Regular evaluation episodes show generalization
4. **Plots**: Automatic plot generation for visual analysis

## Troubleshooting

### Common Issues

1. **Slow Learning**: 
   - Increase learning rates
   - Check environment reward scales
   - Ensure proper exploration

2. **Unstable Training**:
   - Decrease learning rates
   - Increase tau (slower target updates)
   - Check reward normalization

3. **Poor Performance**:
   - Adjust weight configuration
   - Increase network size
   - Longer training time

4. **Memory Issues**:
   - Reduce buffer size
   - Reduce batch size
   - Use smaller networks

### Environment-Specific Tips

- **CartPole**: Usually converges quickly (50k-100k steps)
- **MountainCar**: Requires more exploration (150k+ steps)
- **Pendulum**: Sensitive to weight configuration
- **LunarLander**: Benefits from larger networks
- **EnergyNet**: May require domain-specific tuning

## Contributing

When adding new test environments:

1. Inherit from `gym.Env`
2. Implement multi-objective `reward_space`
3. Return numpy arrays for multi-objective rewards
4. Add to `MULTI_OBJECTIVE_ENVS` registry
5. Test with the existing MO-SAC implementation

## References

- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
- [Multi-Objective Reinforcement Learning](https://link.springer.com/article/10.1007/s10994-013-5373-6)
- [EnergyNet Environment](https://github.com/YuzhongMa/EnergyNet)
