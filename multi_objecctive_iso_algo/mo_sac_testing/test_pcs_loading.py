#!/usr/bin/env python3
"""
Quick test script to verify PCS model loading functionality.
"""

import sys
import os
import torch

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def test_pcs_loading():
    """Test PCS model loading functionality"""
    print("Testing PCS model loading...")
    
    try:
        from EnergyNetMoISO.pcs_models.sac_pcs_agent import SACPCSAgent
        print("✓ Successfully imported SACPCSAgent")
    except Exception as e:
        print(f"✗ Failed to import SACPCSAgent: {e}")
        return False
    
    try:
        from energy_net.env.pcs_unit_v0 import PCSUnitEnv
        print("✓ Successfully imported PCSUnitEnv")
        
        # Create dummy environment to get dimensions
        temp_env = PCSUnitEnv()
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]
        action_bounds = (float(temp_env.action_space.low[0]), float(temp_env.action_space.high[0]))
        temp_env.close()
        
        print(f"✓ PCS environment dimensions: state={state_dim}, action={action_dim}, bounds={action_bounds}")
        
    except Exception as e:
        print(f"✗ Failed to import or create PCSUnitEnv: {e}")
        return False
    
    try:
        # Create SAC PCS agent
        pcs_agent = SACPCSAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds
        )
        print("✓ Successfully created SACPCSAgent")
        
        # Test that it has predict method
        if hasattr(pcs_agent, 'predict'):
            print("✓ SACPCSAgent has predict method")
        else:
            print("✗ SACPCSAgent missing predict method")
            return False
            
    except Exception as e:
        print(f"✗ Failed to create SACPCSAgent: {e}")
        return False
    
    # Test loading a model file (if it exists)
    pcs_model_path = "/home/matan.biton/EnergyNetMultiObjectiveISOWrapper/training/pcs_experiments/models/with_opt_iso_opt_no_disp_final.pth"
    
    if os.path.exists(pcs_model_path):
        print(f"Testing model loading from: {pcs_model_path}")
        try:
            pcs_agent.load(pcs_model_path)
            print("✓ Successfully loaded PCS model")
        except Exception as e:
            print(f"✗ Failed to load PCS model: {e}")
            return False
    else:
        print(f"⚠ PCS model file not found: {pcs_model_path}")
    
    print("\n✓ All PCS loading tests passed!")
    return True

def test_environment_creation():
    """Test environment creation with PCS model"""
    print("\nTesting environment creation with PCS model...")
    
    try:
        from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
        print("✓ Successfully imported MultiObjectiveISOEnv")
    except Exception as e:
        print(f"✗ Failed to import MultiObjectiveISOEnv: {e}")
        return False
    
    # Import the updated create_energynet_env function
    try:
        # Add parent directory to import multi_objective_sac
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from continue_train_energynet import create_energynet_env
        print("✓ Successfully imported create_energynet_env")
    except Exception as e:
        print(f"✗ Failed to import create_energynet_env: {e}")
        return False
    
    # Test environment creation without PCS model
    try:
        env = create_energynet_env(use_dispatch_action=False)
        print("✓ Successfully created environment without PCS model")
        env.close()
    except Exception as e:
        print(f"✗ Failed to create environment without PCS: {e}")
        return False
    
    # Test environment creation with PCS model path
    pcs_model_path = "/home/matan.biton/EnergyNetMultiObjectiveISOWrapper/training/pcs_experiments/models/with_opt_iso_opt_no_disp_final.pth"
    
    if os.path.exists(pcs_model_path):
        try:
            env = create_energynet_env(
                use_dispatch_action=False,
                trained_pcs_model=pcs_model_path
            )
            print("✓ Successfully created environment with PCS model")
            env.close()
        except Exception as e:
            print(f"✗ Failed to create environment with PCS: {e}")
            return False
    else:
        print(f"⚠ Skipping PCS model test - file not found: {pcs_model_path}")
    
    print("✓ All environment creation tests passed!")
    return True

if __name__ == "__main__":
    print("PCS Model Loading Test")
    print("=" * 50)
    
    success = True
    success &= test_pcs_loading()
    success &= test_environment_creation()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
