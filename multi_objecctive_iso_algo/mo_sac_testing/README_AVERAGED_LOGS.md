# Multi-Seed TensorBoard Log Averaging

This document describes the automatic TensorBoard log averaging functionality that has been added to the multi-seed training script.

## Overview

When you run the multi-seed training script (`run_train_energynet_v2_multiseed.sh`), it now automatically creates averaged TensorBoard logs after all 5 runs complete. This provides:

1. **Averaged metrics** across all successful runs
2. **Standard deviation** information for each metric
3. **Summary report** with key performance statistics
4. **Easy visualization** of training stability and performance

## How It Works

### Automatic Averaging
After all 5 seed runs complete, the script automatically:

1. Checks if at least 2 runs were successful
2. Extracts scalar data from each individual TensorBoard log
3. Aligns metrics by timestep across all runs
4. Computes mean and standard deviation for each metric at each timestep
5. Creates a new TensorBoard log with averaged results
6. Generates a text summary report

### Metrics Averaged

All scalar metrics logged during training are averaged, including:

- **Loss metrics**: `Loss/Critic`, `Loss/Actor`, `Loss/Alpha`
- **Episode metrics**: `Episode/Reward_Objective_X`, `Episode/Scalarized_Reward`, `Episode/Length`
- **Learning rates**: `LR/Actor`, `LR/Critic1`, `LR/Critic2`, `LR/Alpha`
- **Alpha values**: `Alpha`
- **Reward scaling stats**: `RewardScaling/Mean_Obj_X`, `RewardScaling/Std_Obj_X`

### Output Structure

After successful averaging, you'll find:

```
logs/
├── train_energynet_experiment_1_TIMESTAMP/    # Individual run 1
├── train_energynet_experiment_2_TIMESTAMP/    # Individual run 2
├── train_energynet_experiment_3_TIMESTAMP/    # Individual run 3
├── train_energynet_experiment_4_TIMESTAMP/    # Individual run 4
├── train_energynet_experiment_5_TIMESTAMP/    # Individual run 5
└── train_energynet_experiment_averaged/       # Averaged results
    ├── events.out.tfevents.TIMESTAMP          # TensorBoard event file
    └── ...

averaged_summary_experiment.txt                # Text summary report
```

## Usage

### Automatic (Recommended)
Simply run the multi-seed script as usual:

```bash
sbatch -c 4 --gres=gpu:1 ./run_train_energynet_v2_multiseed.sh my_experiment
```

The averaging happens automatically after all runs complete.

### Manual Averaging
If automatic averaging fails or you want to average logs manually:

```bash
# Use the helper script
./create_averaged_logs.sh my_experiment 5

# Or use the Python script directly
python3 average_tensorboard_logs.py logs/ my_experiment --num-runs 5 --seeds 42 123 456 789 1337
```

## Viewing Results

### TensorBoard Visualization

View averaged results only:
```bash
tensorboard --logdir logs/train_energynet_my_experiment_averaged --host 0.0.0.0 --port 6006
```

View all runs (individual + averaged):
```bash
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

### Text Summary
View the summary report:
```bash
cat averaged_summary_my_experiment.txt
```

## Understanding the Averaged Data

### TensorBoard Metrics
- **`Averaged/` prefix**: Mean values across all successful runs
- **`Std/` prefix**: Standard deviation across all successful runs

### Interpreting Results
- **Low standard deviation**: Consistent performance across seeds
- **High standard deviation**: High variance, may need more runs or hyperparameter tuning
- **Smooth averaged curves**: Better understanding of true learning dynamics

## Troubleshooting

### "TensorBoard not available"
Install TensorBoard:
```bash
pip install tensorboard
```

### "No data extracted from any runs"
- Check that individual runs completed successfully
- Verify TensorBoard logs exist in the `logs/` directory
- Ensure event files are not corrupted

### "Need at least 2 successful runs"
- Check SLURM output for individual run failures
- Look at `slurm_energynet_multiseed_*.err` files for error details
- Ensure sufficient resources (GPU, memory, time limits)

### Manual Recovery
If automatic averaging fails, you can manually average any subset of successful runs:

```bash
# Average only runs 1, 2, and 4 (if run 3 and 5 failed)
python3 average_tensorboard_logs.py logs/ my_experiment --num-runs 3 --seeds 42 123 789
```

## Dependencies

The averaging functionality requires:
- Python 3.8+
- `tensorboard` package
- `numpy` package
- `torch` package (for SummaryWriter)

All dependencies are listed in `requirements.txt`.

## Technical Details

### Data Alignment
- Metrics are aligned by exact timestep across runs
- If a timestep is missing in some runs, it's only included if at least 50% of runs have data
- This prevents sparse data from skewing averages

### File Handling
- Automatically finds the most recent log directory for each run
- Handles multiple event files by using the most recent one
- Creates clean output structure for easy navigation

### Performance
- Memory efficient: processes one metric at a time
- Handles large log files gracefully
- Minimal computational overhead

## Examples

### Successful Output
```
✓ Successfully created averaged TensorBoard logs

📊 Averaged Results Available:
  Averaged logs: logs/train_energynet_my_experiment_averaged
  Summary report: averaged_summary_my_experiment.txt

To view averaged results in TensorBoard:
  tensorboard --logdir logs/train_energynet_my_experiment_averaged --host 0.0.0.0 --port 6006
```

### Sample Summary Report
```
Multi-Seed Training Averaged Results
====================================

Experiment: my_experiment
Number of runs: 5
Seeds used: [42, 123, 456, 789, 1337]
Total timesteps per run: 1000000

Key Performance Metrics:
========================

Objective 0 Final Reward: -245.6789 ± 12.3456
Objective 1 Final Reward: 0.8234 ± 0.0567
Scalarized Reward: -146.5432 ± 8.9012
Episode Length: 498.23 ± 15.67
```

This averaged logging system provides a comprehensive way to analyze the stability and performance of your multi-objective SAC training across multiple random seeds.
