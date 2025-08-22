#!/usr/bin/env python3
"""
Quick start script for testing Multi-Objective SAC optimizations.
Run this script to test different optimization configurations.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Command completed successfully!")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ Command failed!")
            if result.stderr:
                print("Error:", result.stderr[-500:])  # Last 500 chars
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
    
    print("\nPress Enter to continue to next test...")
    input()

def main():
    """Run optimization tests."""
    
    print("🎯 Multi-Objective SAC Optimization Quick Tests")
    print("=" * 60)
    print("This script will run several short training tests to demonstrate")
    print("the different optimization configurations.")
    print()
    print("Each test runs for only 5000 timesteps to show functionality.")
    print("For real training, use much higher timestep counts (100k-1M+).")
    print()
    
    # Base command
    base_cmd = "conda run --live-stream --name ENMOISOW python train_energynet.py"
    common_args = "--total-timesteps 5000 --learning-starts 1000 --eval-freq 2500 --verbose"
    
    tests = [
        {
            "name": "Baseline (No Optimizations)",
            "args": "--disable-orthogonal-init --experiment-name baseline_test",
            "description": "Standard SAC without optimizations"
        },
        {
            "name": "Conservative Optimization",
            "args": "--use-reward-scaling --experiment-name conservative_test",
            "description": "Safe optimizations: reward scaling + orthogonal init"
        },
        {
            "name": "Moderate Optimization", 
            "args": "--use-lr-annealing --lr-annealing-type cosine --use-reward-scaling --use-value-clipping --experiment-name moderate_test",
            "description": "Balanced approach with multiple optimizations"
        },
        {
            "name": "Aggressive Optimization",
            "args": "--use-lr-annealing --lr-annealing-type cosine --lr-min-factor 0.05 --use-reward-scaling --use-value-clipping --value-clip-range 100.0 --actor-lr 5e-4 --experiment-name aggressive_test",
            "description": "Maximum optimizations for fastest convergence"
        }
    ]
    
    print("Available tests:")
    for i, test in enumerate(tests, 1):
        print(f"{i}. {test['name']} - {test['description']}")
    
    print("\nOptions:")
    print("- Enter test number (1-4) to run specific test")
    print("- Enter 'all' to run all tests sequentially") 
    print("- Enter 'quit' to exit")
    
    while True:
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'quit':
            print("👋 Goodbye!")
            break
        elif choice == 'all':
            for test in tests:
                cmd = f"{base_cmd} {common_args} {test['args']}"
                run_command(cmd, test['name'])
            break
        elif choice in ['1', '2', '3', '4']:
            test = tests[int(choice) - 1]
            cmd = f"{base_cmd} {common_args} {test['args']}"
            run_command(cmd, test['name'])
        else:
            print("❌ Invalid choice. Please enter 1-4, 'all', or 'quit'")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("train_energynet.py"):
        print("❌ Error: train_energynet.py not found in current directory")
        print("Please run this script from the mo_sac_testing directory")
        sys.exit(1)
    
    main()
