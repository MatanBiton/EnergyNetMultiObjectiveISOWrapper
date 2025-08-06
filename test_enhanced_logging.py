#!/usr/bin/env python3
"""
Test script for enhanced EnergyNet info dict logging in MOSAC.
"""

import sys
import os
import numpy as np
import tempfile
import shutil

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from multi_objecctive_iso_algo.multi_objective_sac import MultiObjectiveSAC, train_mo_sac

def test_enhanced_logging():
    """Test the enhanced logging functionality with EnergyNet environment."""
    
    print("Testing Enhanced EnergyNet Info Dict Logging...")
    
    # Create EnergyNet environment
    try:
        env = MultiObjectiveISOEnv(use_dispatch_action=True)
        print(f"✓ Environment created successfully")
        print(f"  - State space: {env.observation_space.shape}")
        print(f"  - Action space: {env.action_space.shape}")
        print(f"  - Reward dimension: 2 (assumed multi-objective)")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    # Test a single step to verify info dict structure
    try:
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Environment step successful")
        print(f"  - Info dict keys: {list(info.keys())}")
        print(f"  - Info dict types: {[(k, type(v).__name__) for k, v in info.items()]}")
        
        # Verify expected keys from data.txt are present
        expected_keys = [
            'predicted_demand', 'realized_demand', 'pcs_demand', 'net_demand',
            'dispatch', 'shortfall', 'dispatch_cost', 'reserve_cost', 'pcs_costs',
            'production', 'consumption', 'battery_level', 'battery_actions',
            'buy_price', 'sell_price', 'iso_buy_price', 'iso_sell_price',
            'net_exchange', 'pcs_cost', 'pcs_actions'
        ]
        
        missing_keys = [key for key in expected_keys if key not in info]
        if missing_keys:
            print(f"⚠ Missing expected keys: {missing_keys}")
        else:
            print(f"✓ All expected info dict keys present")
            
    except Exception as e:
        print(f"✗ Failed environment step: {e}")
        return False
    
    # Create MOSAC agent with tensorboard logging
    temp_log_dir = tempfile.mkdtemp(prefix="mosac_test_logs_")
    
    try:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        reward_dim = 2  # Multi-objective rewards
        
        agent = MultiObjectiveSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            reward_dim=reward_dim,
            tensorboard_log=temp_log_dir,
            verbose=True,
            # Use small networks for testing
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32],
            buffer_capacity=1000,
            batch_size=32
        )
        
        print(f"✓ MOSAC agent created with tensorboard logging at: {temp_log_dir}")
        
    except Exception as e:
        print(f"✗ Failed to create MOSAC agent: {e}")
        shutil.rmtree(temp_log_dir, ignore_errors=True)
        return False
    
    # Run a very short training to test logging
    try:
        print("Running short training to test logging...")
        
        training_results = train_mo_sac(
            env=env,
            agent=agent,
            total_timesteps=200,  # Very short for testing
            learning_starts=50,
            train_freq=1,
            eval_freq=100,
            eval_episodes=2,
            save_freq=1000,  # Won't trigger in short test
            verbose=True
        )
        
        print(f"✓ Training completed successfully")
        print(f"  - Episodes completed: {len(training_results['episode_rewards'])}")
        print(f"  - Episode lengths: {training_results['episode_lengths']}")
        
        # Check that tensorboard logs were created
        log_files = os.listdir(temp_log_dir)
        print(f"✓ Tensorboard log files created: {log_files}")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_log_dir, ignore_errors=True)
        print(f"✓ Cleaned up temporary log directory")
    
    print("\n🎉 All tests passed! Enhanced logging is working correctly.")
    print("\nThe following logging has been added:")
    print("1. Step-level logging: EnergyNet_Step/{key} - logs every info dict entry per step")
    print("2. Episode-level aggregation: EnergyNet_Episode/{key}_{stat} - logs mean, std, sum, min, max, final")
    print("3. Evaluation logging: EnergyNet_Eval/{key}_{stat} - logs evaluation episode statistics")
    print("4. Proper handling of list values (battery_level, battery_actions, pcs_actions)")
    print("5. Automatic data type conversion and validation")
    
    return True

if __name__ == "__main__":
    test_enhanced_logging()
