#!/usr/bin/env python3
"""
Script to average TensorBoard logs from multiple runs and create a combined log.
This script is used by the multi-seed training script to create averaged results.
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import glob

# Try importing tensorboard
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False


def extract_scalar_data(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract scalar data from a TensorBoard log directory.
    
    Returns:
        Dictionary mapping metric names to lists of (step, value) tuples
    """
    if not TENSORBOARD_AVAILABLE:
        return {}
    
    # Find the event file
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return {}
    
    # Use the most recent event file
    event_file = max(event_files, key=os.path.getmtime)
    print(f"  Reading: {os.path.basename(event_file)}")
    
    # Load the data
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    scalar_data = {}
    scalar_tags = ea.Tags()['scalars']
    
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        scalar_data[tag] = [(event.step, event.value) for event in events]
    
    return scalar_data


def align_and_average_metrics(all_runs_data: List[Dict[str, List[Tuple[int, float]]]]) -> Dict[str, List[Tuple[int, float, float]]]:
    """
    Align metrics by timestep and compute averages and standard deviations.
    
    Returns:
        Dictionary mapping metric names to lists of (step, mean_value, std_value) tuples
    """
    # Get all unique metric names across all runs
    all_metrics = set()
    for run_data in all_runs_data:
        all_metrics.update(run_data.keys())
    
    averaged_data = {}
    
    for metric in all_metrics:
        print(f"  Processing metric: {metric}")
        
        # Collect all timesteps for this metric across all runs
        all_steps = set()
        for run_data in all_runs_data:
            if metric in run_data:
                steps = [step for step, _ in run_data[metric]]
                all_steps.update(steps)
        
        # Sort timesteps
        sorted_steps = sorted(all_steps)
        
        # For each timestep, collect values from all runs that have data for that step
        averaged_metric_data = []
        
        for step in sorted_steps:
            values_at_step = []
            
            for run_data in all_runs_data:
                if metric in run_data:
                    # Find the value at this exact step
                    for data_step, value in run_data[metric]:
                        if data_step == step:
                            values_at_step.append(value)
                            break
            
            # Only include timesteps where at least half the runs have data
            if len(values_at_step) >= len(all_runs_data) // 2:
                mean_value = np.mean(values_at_step)
                std_value = np.std(values_at_step) if len(values_at_step) > 1 else 0.0
                averaged_metric_data.append((step, mean_value, std_value))
        
        averaged_data[metric] = averaged_metric_data
        print(f"    {len(averaged_metric_data)} data points averaged")
    
    return averaged_data


def write_averaged_logs(averaged_data: Dict[str, List[Tuple[int, float, float]]], output_dir: str):
    """
    Write averaged data to a new TensorBoard log directory.
    """
    if not TENSORBOARD_AVAILABLE:
        print("Warning: Cannot write TensorBoard logs - TensorBoard not available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
    
    for metric, data_points in averaged_data.items():
        for step, mean_value, std_value in data_points:
            # Write the mean value
            writer.add_scalar(f'Averaged/{metric}', mean_value, step)
            # Write the standard deviation as a separate metric
            writer.add_scalar(f'Std/{metric}', std_value, step)
    
    writer.close()
    print(f"Averaged logs written to: {output_dir}")


def create_summary_report(averaged_data: Dict[str, List[Tuple[int, float, float]]], 
                         output_dir: str, 
                         experiment_name: str, 
                         run_info: Dict):
    """
    Create a text summary report of the averaged results.
    """
    summary_file = os.path.join(output_dir, f"averaged_summary_{experiment_name}.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"Multi-Seed Training Averaged Results\n")
        f.write(f"====================================\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Number of runs: {run_info.get('num_runs', 'Unknown')}\n")
        f.write(f"Seeds used: {run_info.get('seeds', 'Unknown')}\n")
        f.write(f"Total timesteps per run: {run_info.get('total_timesteps', 'Unknown')}\n")
        f.write(f"\n")
        
        f.write(f"Metrics Summary:\n")
        f.write(f"================\n\n")
        
        for metric, data_points in averaged_data.items():
            if data_points:
                final_step, final_mean, final_std = data_points[-1]
                f.write(f"{metric}:\n")
                f.write(f"  Final value: {final_mean:.4f} ± {final_std:.4f}\n")
                f.write(f"  Final timestep: {final_step}\n")
                f.write(f"  Data points: {len(data_points)}\n\n")
        
        # Special summary for key metrics
        f.write(f"Key Performance Metrics:\n")
        f.write(f"========================\n\n")
        
        # Episode metrics
        for i in range(10):  # Check for up to 10 objectives
            objective_metric = f"Episode/Reward_Objective_{i}"
            if objective_metric in averaged_data and averaged_data[objective_metric]:
                final_step, final_mean, final_std = averaged_data[objective_metric][-1]
                f.write(f"Objective {i} Final Reward: {final_mean:.4f} ± {final_std:.4f}\n")
        
        if "Episode/Scalarized_Reward" in averaged_data and averaged_data["Episode/Scalarized_Reward"]:
            final_step, final_mean, final_std = averaged_data["Episode/Scalarized_Reward"][-1]
            f.write(f"Scalarized Reward: {final_mean:.4f} ± {final_std:.4f}\n")
        
        if "Episode/Length" in averaged_data and averaged_data["Episode/Length"]:
            final_step, final_mean, final_std = averaged_data["Episode/Length"][-1]
            f.write(f"Episode Length: {final_mean:.2f} ± {final_std:.2f}\n")
    
    print(f"Summary report written to: {summary_file}")


def average_multiseed_logs(logs_base_dir: str, 
                          experiment_name: str, 
                          num_runs: int = 5,
                          seeds: List[int] = None) -> str:
    """
    Average TensorBoard logs from multiple seed runs.
    
    Args:
        logs_base_dir: Base directory containing logs
        experiment_name: Base experiment name
        num_runs: Number of runs to average
        seeds: List of seeds used (for reporting)
    
    Returns:
        Path to the averaged logs directory
    """
    if not TENSORBOARD_AVAILABLE:
        print("Error: TensorBoard not available. Cannot average logs.")
        return ""
    
    print(f"Averaging TensorBoard logs for experiment: {experiment_name}")
    print(f"Looking for {num_runs} runs in: {logs_base_dir}")
    
    # Find all run log directories
    run_log_dirs = []
    for i in range(1, num_runs + 1):
        run_name = f"{experiment_name}_{i}"
        # Look for log directories matching the pattern
        pattern = os.path.join(logs_base_dir, f"train_energynet_{run_name}_*")
        matching_dirs = glob.glob(pattern)
        
        if matching_dirs:
            # Use the most recent one if multiple matches
            log_dir = max(matching_dirs, key=os.path.getmtime)
            run_log_dirs.append(log_dir)
            print(f"  Found run {i}: {os.path.basename(log_dir)}")
        else:
            print(f"  Warning: No logs found for run {i} (pattern: train_energynet_{run_name}_*)")
    
    if len(run_log_dirs) < 2:
        print(f"Error: Found only {len(run_log_dirs)} runs. Need at least 2 runs to average.")
        return ""
    
    print(f"\nProcessing {len(run_log_dirs)} runs...")
    
    # Extract data from all runs
    all_runs_data = []
    for i, log_dir in enumerate(run_log_dirs):
        print(f"Extracting data from run {i+1}...")
        run_data = extract_scalar_data(log_dir)
        if run_data:
            all_runs_data.append(run_data)
            print(f"  Found {len(run_data)} metrics")
        else:
            print(f"  Warning: No data extracted from {log_dir}")
    
    if not all_runs_data:
        print("Error: No data extracted from any runs.")
        return ""
    
    print(f"\nAveraging data across {len(all_runs_data)} runs...")
    averaged_data = align_and_average_metrics(all_runs_data)
    
    # Create output directory
    output_dir = os.path.join(logs_base_dir, f"train_energynet_{experiment_name}_averaged")
    
    print(f"\nWriting averaged logs...")
    write_averaged_logs(averaged_data, output_dir)
    
    # Create summary report
    run_info = {
        'num_runs': len(all_runs_data),
        'seeds': seeds if seeds else list(range(1, len(all_runs_data) + 1)),
        'total_timesteps': 'Variable'  # Could extract this from the logs if needed
    }
    create_summary_report(averaged_data, logs_base_dir, experiment_name, run_info)
    
    print(f"\n✓ Averaging complete!")
    print(f"Averaged logs: {output_dir}")
    print(f"View with: tensorboard --logdir {logs_base_dir}")
    
    return output_dir


def main():
    """Command line interface for the averaging script."""
    parser = argparse.ArgumentParser(description="Average TensorBoard logs from multi-seed runs")
    parser.add_argument('logs_dir', type=str, help='Directory containing the log files')
    parser.add_argument('experiment_name', type=str, help='Base experiment name')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs to average')
    parser.add_argument('--seeds', type=int, nargs='+', help='Seeds used in the runs')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.logs_dir):
        print(f"Error: Logs directory not found: {args.logs_dir}")
        sys.exit(1)
    
    # Average the logs
    output_dir = average_multiseed_logs(
        logs_base_dir=args.logs_dir,
        experiment_name=args.experiment_name,
        num_runs=args.num_runs,
        seeds=args.seeds
    )
    
    if output_dir:
        print(f"\nSuccess! Averaged logs created at: {output_dir}")
    else:
        print(f"\nFailed to create averaged logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
