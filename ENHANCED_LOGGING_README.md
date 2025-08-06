# Enhanced EnergyNet Info Dict Logging for MOSAC

## Overview

This enhancement adds comprehensive logging for all EnergyNet environment info dict entries when training the Multi-Objective Soft Actor-Critic (MOSAC) algorithm. The logging is designed to be compatible with TensorBoard and handles all data types present in the EnergyNet environment's info dictionary.

## Features Added

### 1. Step-Level Logging (`EnergyNet_Step/`)
- Logs every single step of training
- Captures all 20 info dict entries per step
- Properly handles both scalar values and list values
- For list values (battery_level, battery_actions, pcs_actions), logs the last element (current step value)

### 2. Episode-Level Aggregated Logging (`EnergyNet_Episode/`)
- Provides statistical summaries for each episode
- Computes and logs the following statistics for each info dict entry:
  - `mean`: Average value across the episode
  - `std`: Standard deviation across the episode  
  - `sum`: Total sum across the episode
  - `min`: Minimum value in the episode
  - `max`: Maximum value in the episode
  - `final`: Final value of the episode

### 3. Evaluation Logging (`EnergyNet_Eval/`)
- Logs the same aggregated statistics during evaluation episodes
- Helps monitor performance during evaluation phases
- Uses global episode numbering for consistent tracking

## Info Dict Entries Logged

Based on the EnergyNet environment, the following 20 entries are logged:

### Scalar Values (float)
- `predicted_demand`: Predicted energy demand
- `realized_demand`: Actual realized energy demand
- `pcs_demand`: PCS (Power Conversion System) demand
- `net_demand`: Net energy demand
- `dispatch`: Energy dispatch amount
- `shortfall`: Energy shortfall amount
- `dispatch_cost`: Cost of energy dispatch
- `reserve_cost`: Cost of energy reserves
- `pcs_costs`: PCS operational costs
- `production`: Energy production amount
- `consumption`: Energy consumption amount
- `buy_price`: Energy buy price
- `sell_price`: Energy sell price
- `iso_buy_price`: ISO buy price
- `iso_sell_price`: ISO sell price
- `net_exchange`: Net energy exchange
- `pcs_cost`: Individual PCS cost

### List Values (last element logged)
- `battery_level`: Current battery level (logs final value per step)
- `battery_actions`: Battery actions taken (logs final action per step)
- `pcs_actions`: PCS actions taken (logs final action per step)

## Implementation Details

### Data Type Handling
- **Scalar values**: Logged directly as float values
- **List values**: Extracts the last element (`list[-1]`) for step-level logging
- **Type safety**: All values are converted to float for TensorBoard compatibility
- **Validation**: Checks for key existence and proper data types before logging

### Memory Efficiency
- Episode-level aggregation uses lists that are reset after each episode
- No unnecessary data accumulation across episodes
- Efficient numpy operations for statistical calculations

### Logging Frequency
- **Step-level**: Every environment step during training
- **Episode-level**: At the end of each training episode
- **Evaluation**: During evaluation episodes at specified intervals

## Usage

The enhanced logging is automatically enabled when:
1. Training with MOSAC on an EnergyNet environment
2. TensorBoard logging is enabled (`tensorboard_log` parameter is set)
3. The environment returns an info dict with the expected structure

### Example Usage

```python
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from multi_objective_sac import MultiObjectiveSAC, train_mo_sac

# Create environment
env = MultiObjectiveISOEnv(use_dispatch_action=True)

# Create agent with tensorboard logging
agent = MultiObjectiveSAC(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    reward_dim=2,
    tensorboard_log="./logs/energynet_training",
    verbose=True
)

# Train with enhanced logging
train_mo_sac(
    env=env,
    agent=agent,
    total_timesteps=1000000,
    verbose=True
)
```

## TensorBoard Visualization

After training, use TensorBoard to visualize the logged data:

```bash
tensorboard --logdir ./logs/energynet_training
```

### Viewing Logged Data

1. **Step-level data**: Navigate to `EnergyNet_Step/` to see individual step values
2. **Episode statistics**: Check `EnergyNet_Episode/` for episode-level aggregations
3. **Evaluation data**: Look at `EnergyNet_Eval/` for evaluation episode statistics

## Benefits

1. **Comprehensive Monitoring**: Track all aspects of the EnergyNet environment
2. **Performance Analysis**: Understand energy dispatch strategies and costs
3. **Debugging**: Identify issues with specific environment variables
4. **Research Insights**: Analyze relationships between different energy metrics
5. **Reproducibility**: Complete logging for experiment reproduction

## Environment Compatibility

This enhanced logging is specifically designed for:
- **EnergyNet environments** (MultiObjectiveISOEnv)
- **MOSAC algorithm** with TensorBoard logging enabled
- **IsoMOEnergyNet conda environment** to avoid import errors

## Technical Notes

- All logged values are converted to float for TensorBoard compatibility
- Episode aggregation data is automatically reset between episodes
- Memory usage is optimized for long training runs
- Compatible with both CPU and GPU training
- Evaluation logging uses global episode numbering for consistency

## Testing

Run the test script to verify functionality:

```bash
conda activate IsoMOEnergyNet
python test_enhanced_logging.py
```

The test verifies:
- Environment creation and info dict structure
- Proper logging setup and data capture
- TensorBoard log file generation
- Compatibility with short training runs
