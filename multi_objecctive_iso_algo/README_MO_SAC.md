# Multi-Objective Soft Actor-Critic (MO-SAC) Algorithm

A comprehensive implementation of Multi-Objective Soft Actor-Critic for continuous control environments with multiple reward objectives.

## Features

- **Multi-Objective Learning**: Handles environments with multiple reward signals using weighted scalarization
- **Continuous Action Spaces**: Designed for continuous control problems
- **Automatic Entropy Tuning**: Adaptive entropy coefficient for optimal exploration-exploitation balance
- **Dual Critic Architecture**: Reduces overestimation bias with twin Q-networks
- **Comprehensive Logging**: TensorBoard integration and detailed training metrics
- **Flexible Configuration**: Tunable network architectures, optimizers, and hyperparameters
- **Save/Load Functionality**: Model persistence for deployment and continued training

## Algorithm Overview

Multi-Objective SAC extends the standard Soft Actor-Critic algorithm to handle multiple objectives:

1. **Scalarization**: Uses weighted linear combination to convert multi-objective rewards into scalar values
2. **Policy Learning**: Actor network learns stochastic policy with maximum entropy regularization
3. **Value Learning**: Twin critic networks estimate Q-values for each objective
4. **Entropy Regularization**: Automatic tuning of entropy coefficient for exploration

## Installation

Required dependencies:
```bash
pip install torch numpy gymnasium matplotlib tensorboard
```

## Quick Start

```python
import numpy as np
from multi_objective_sac import MultiObjectiveSAC, train_mo_sac

# Create your multi-objective environment
# env = YourMultiObjectiveEnv()

# Initialize agent
agent = MultiObjectiveSAC(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    reward_dim=env.reward_dim,  # Number of objectives
    action_bounds=(env.action_space.low[0], env.action_space.high[0]),
    weights=np.array([0.6, 0.4]),  # Objective weights
    verbose=True
)

# Train agent
train_mo_sac(
    env=env,
    agent=agent,
    total_timesteps=100000,
    verbose=True
)
```

## API Reference

### MultiObjectiveSAC

Main algorithm class for multi-objective reinforcement learning.

#### Constructor Parameters

**Environment Configuration:**
- `state_dim` (int): Dimension of observation space
- `action_dim` (int): Dimension of action space  
- `reward_dim` (int): Number of objectives/reward components
- `action_bounds` (tuple): (min, max) action bounds
- `weights` (np.ndarray): Weights for objective scalarization

**Network Architecture:**
- `actor_hidden_dims` (list): Hidden layer sizes for actor [256, 256]
- `critic_hidden_dims` (list): Hidden layer sizes for critics [256, 256]

**Optimizer Configuration:**
- `actor_lr` (float): Actor learning rate (3e-4)
- `critic_lr` (float): Critic learning rate (3e-4)
- `alpha_lr` (float): Entropy coefficient learning rate (3e-4)

**Algorithm Parameters:**
- `gamma` (float): Discount factor (0.99)
- `tau` (float): Target network update rate (0.005)
- `alpha` (float): Initial entropy coefficient (0.2)
- `auto_tune_alpha` (bool): Enable automatic entropy tuning (True)
- `target_entropy` (float): Target entropy for auto-tuning (-action_dim)

**Experience Replay:**
- `buffer_capacity` (int): Replay buffer size (1000000)
- `batch_size` (int): Training batch size (256)

**Logging:**
- `device` (str): Computing device ('auto', 'cpu', 'cuda')
- `verbose` (bool): Enable verbose logging (False)
- `tensorboard_log` (str): TensorBoard log directory (None)

#### Methods

**Action Selection:**
```python
action = agent.select_action(state, deterministic=False)
```

**Experience Storage:**
```python
agent.store_transition(state, action, reward, next_state, done)
```

**Network Updates:**
```python
losses = agent.update()  # Returns dict of loss values
```

**Model Persistence:**
```python
agent.save("model.pth")
agent.load("model.pth")
```

### Training Functions

#### train_mo_sac()

Main training loop for MO-SAC agent.

```python
stats = train_mo_sac(
    env,                    # Environment instance
    agent,                  # MO-SAC agent
    total_timesteps=1000000,# Total training steps
    learning_starts=10000,  # Steps before learning
    train_freq=1,          # Training frequency
    eval_freq=10000,       # Evaluation frequency
    eval_episodes=10,      # Episodes per evaluation
    save_freq=50000,       # Model save frequency
    save_path="model",     # Save path prefix
    verbose=False          # Verbose output
)
```

**Returns:** Dictionary with training statistics
- `episode_rewards`: List of episode rewards for each objective
- `episode_lengths`: List of episode lengths

#### evaluate_mo_sac()

Evaluate trained agent performance.

```python
rewards = evaluate_mo_sac(
    env,              # Environment instance
    agent,            # Trained agent
    num_episodes=10,  # Number of evaluation episodes
    verbose=False     # Verbose output
)
```

**Returns:** NumPy array of shape (num_episodes, reward_dim)

## Multi-Objective Considerations

### Weight Selection

Objective weights determine the trade-off between different goals:

```python
# Equal importance
weights = np.array([0.5, 0.5])

# Prefer first objective  
weights = np.array([0.8, 0.2])

# Three objectives
weights = np.array([0.5, 0.3, 0.2])
```

### Reward Scaling

Ensure objectives are on similar scales for effective learning:

```python
# In your environment step function
reward1_scaled = reward1 / reward1_scale
reward2_scaled = reward2 / reward2_scale
return np.array([reward1_scaled, reward2_scaled])
```

### Objective Correlation

Monitor objective correlations to understand trade-offs:
- Positive correlation: Objectives align (easier optimization)
- Negative correlation: Objectives conflict (harder optimization)
- Zero correlation: Independent objectives

## Network Architecture

### Actor Network

Outputs stochastic policy with reparameterization trick:

```
Input: State
↓
Hidden Layers (ReLU activation)
↓
Mean Head → Action Mean
Log Std Head → Action Log Std (clamped)
↓
Normal Distribution → Tanh → Final Action
```

### Critic Networks

Twin networks estimate Q-values for each objective:

```
Input: State + Action
↓
Hidden Layers (ReLU activation)  
↓
Output: Q-values for each objective
```

## Hyperparameter Tuning

### Learning Rates
- Start with 3e-4 for all networks
- Increase if learning is slow
- Decrease if training is unstable

### Network Size
- Default: [256, 256] hidden layers
- Increase for complex environments
- Decrease for faster training

### Buffer Size
- Larger buffers improve sample efficiency
- Balance with memory constraints
- Typical range: 100k - 1M

### Batch Size
- Larger batches more stable but slower
- Typical range: 64 - 512
- Ensure batch_size < buffer_size

### Update Frequencies
- `train_freq=1`: Update every step (standard)
- `tau=0.005`: Slow target updates (stable)
- Adjust based on environment dynamics

## Troubleshooting

### Poor Performance
1. Check reward scaling across objectives
2. Verify weight configuration
3. Increase network capacity
4. Extend training time
5. Adjust learning rates

### Training Instability
1. Decrease learning rates
2. Increase tau (slower updates)
3. Check for reward clipping
4. Verify action space bounds

### Slow Learning
1. Increase learning rates
2. Decrease learning_starts
3. Check exploration behavior
4. Verify environment rewards

### Memory Issues
1. Reduce buffer_capacity
2. Reduce batch_size
3. Use smaller networks
4. Enable gradient checkpointing

## Advanced Usage

### Custom Scalarization

Override the scalarization method for advanced multi-objective techniques:

```python
class CustomMOSAC(MultiObjectiveSAC):
    def scalarize_rewards(self, rewards):
        # Implement Chebyshev scalarization
        return torch.max(rewards * self.weights, dim=-1, keepdim=True)[0]
```

### Dynamic Weights

Implement time-varying objective weights:

```python
def update_weights(agent, timestep):
    # Gradually shift from exploration to exploitation
    alpha = min(timestep / 100000, 1.0)
    agent.weights = np.array([alpha, 1.0 - alpha])
```

### Multi-Objective Evaluation

Analyze Pareto fronts and dominated solutions:

```python
def analyze_pareto_front(eval_rewards):
    # Identify non-dominated solutions
    pareto_front = []
    for i, reward_i in enumerate(eval_rewards):
        dominated = False
        for j, reward_j in enumerate(eval_rewards):
            if i != j and np.all(reward_j >= reward_i) and np.any(reward_j > reward_i):
                dominated = True
                break
        if not dominated:
            pareto_front.append(reward_i)
    return np.array(pareto_front)
```

## Examples

See the `mo_sac_testing/` directory for comprehensive examples:
- `examples.py`: Basic usage examples
- `test_mo_sac.py`: Comprehensive testing on various environments
- `train_energynet.py`: EnergyNet-specific training

## References

1. Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
2. Roijers, D. M., & Whiteson, S. "Multi-objective decision making." Synthesis Lectures on Artificial Intelligence and Machine Learning, 2017.
3. Vamplew, P., et al. "Empirical evaluation methods for multiobjective reinforcement learning algorithms." Machine learning, 2011.
