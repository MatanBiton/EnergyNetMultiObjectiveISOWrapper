#!/usr/bin/env python3
"""
Test script to verify all imports for SAC PCS training work correctly.
"""

import sys
import os

print("Testing imports for SAC PCS training...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Add paths for imports - same as in sac_pcs_training.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

print(f"Current dir: {current_dir}")
print(f"Parent dir: {parent_dir}")
print(f"Root dir: {root_dir}")

# Add parent directory to Python path - this is where EnergyNetMoISO is located
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also add the multi_objective_iso_algo directory
algo_dir = os.path.join(parent_dir, 'multi_objecctive_iso_algo')
if algo_dir not in sys.path:
    sys.path.insert(0, algo_dir)

print(f"Python path (first 5): {sys.path[:5]}")
print("")

# Test imports
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA devices: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    from energy_net.env.pcs_unit_v0 import PCSUnitEnv
    print("✓ PCSUnitEnv imported successfully")
except ImportError as e:
    print(f"✗ PCSUnitEnv: {e}")

try:
    from EnergyNetMoISO.pcs_models.sac_pcs_agent import SACPCSAgent
    print("✓ SACPCSAgent imported successfully")
except ImportError as e:
    print(f"✗ SACPCSAgent: {e}")

try:
    from multi_objective_sac import MultiObjectiveSAC
    print("✓ MultiObjectiveSAC imported successfully")
except ImportError as e:
    print(f"✗ MultiObjectiveSAC: {e}")

print("")
print("Testing environment creation...")
try:
    env = PCSUnitEnv()
    print(f"✓ PCSUnitEnv created successfully")
    print(f"  State dimension: {env.observation_space.shape[0]}")
    print(f"  Action dimension: {env.action_space.shape[0]}")
    print(f"  Action bounds: ({env.action_space.low[0]}, {env.action_space.high[0]})")
    env.close()
except Exception as e:
    print(f"✗ PCSUnitEnv creation failed: {e}")

print("")
print("Testing agent creation...")
try:
    agent = SACPCSAgent(
        state_dim=4,
        action_dim=1,
        action_bounds=(-1.0, 1.0),
        verbose=False
    )
    print("✓ SACPCSAgent created successfully")
except Exception as e:
    print(f"✗ SACPCSAgent creation failed: {e}")

print("")
print("All import tests completed!")
