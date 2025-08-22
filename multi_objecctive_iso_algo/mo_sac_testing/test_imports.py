#!/usr/bin/env python3
"""
Simple script to test imports for MO-SAC in SLURM environment.
This helps debug import issues before running the full training.
"""

import sys
import os

print("Testing imports for MO-SAC...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Add paths for imports - same logic as in train_energynet.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

print(f"Current dir: {current_dir}")
print(f"Parent dir: {parent_dir}")
print(f"Root dir: {root_dir}")

# Add current directory (for when files are copied to temp dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add parent directory (for multi_objective_sac.py)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add root directory (for EnergyNetMoISO package)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print(f"Python path (first 5): {sys.path[:5]}")
print("")

# Test basic packages
print("1. Testing basic packages...")
try:
    import numpy as np
    print(f"   ✓ numpy {np.__version__}")
except ImportError as e:
    print(f"   ✗ numpy: {e}")

try:
    import torch
    print(f"   ✓ torch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"     - CUDA available: {torch.cuda.device_count()} devices")
    else:
        print(f"     - CUDA not available")
except ImportError as e:
    print(f"   ✗ torch: {e}")

try:
    import gymnasium as gym
    print(f"   ✓ gymnasium {gym.__version__}")
except ImportError:
    try:
        import gym
        print(f"   ✓ gym {gym.__version__}")
    except ImportError as e:
        print(f"   ✗ gym/gymnasium: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"   ✓ matplotlib")
except ImportError as e:
    print(f"   ✗ matplotlib: {e}")

try:
    from torch.utils.tensorboard import SummaryWriter
    print(f"   ✓ tensorboard")
except ImportError as e:
    print(f"   ✗ tensorboard: {e}")

print("")

# Test MO-SAC imports
print("2. Testing MO-SAC algorithm import...")
try:
    from multi_objective_sac import MultiObjectiveSAC, train_mo_sac, evaluate_mo_sac
    print("   ✓ multi_objective_sac module imported successfully")
except ImportError as e:
    print(f"   ✗ multi_objective_sac: {e}")

print("")

# Test environment imports
print("3. Testing test environments import...")
try:
    from mo_sac_testing.test_environments import make_env, get_available_envs
    print("   ✓ mo_sac_testing.test_environments imported (module style)")
    method = "module"
except ImportError:
    try:
        from test_environments import make_env, get_available_envs
        print("   ✓ test_environments imported (direct style)")
        method = "direct"
    except ImportError as e:
        print(f"   ✗ test_environments: {e}")
        method = "failed"

if method != "failed":
    print("   Testing environment creation...")
    try:
        available_envs = get_available_envs()
        print(f"   ✓ Available environments: {available_envs}")
        
        # Test creating one environment
        env_name = available_envs[0]
        env = make_env(env_name)
        print(f"   ✓ Created environment: {env_name}")
        print(f"     - Observation space: {env.observation_space}")
        print(f"     - Action space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"   ✗ Environment creation failed: {e}")

print("")

# Test EnergyNet import (if available)
print("4. Testing EnergyNet import...")
try:
    from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
    print("   ✓ EnergyNetMoISO.MoISOEnv imported successfully")
    try:
        # Test creating environment (might fail if not properly configured)
        # env = MultiObjectiveISOEnv()
        print("   ✓ EnergyNet environment class available")
    except Exception as e:
        print(f"   ! EnergyNet environment creation might need configuration: {e}")
except ImportError as e:
    print(f"   ✗ EnergyNetMoISO: {e}")

print("")

# File system check
print("5. File system check...")
print("   Files in current directory:")
for f in sorted(os.listdir(".")):
    if f.endswith(('.py', '.txt', '.md')):
        print(f"     {f}")

print("")
print("Import test completed!")
print("If all imports show ✓, the environment should work for MO-SAC training.")
