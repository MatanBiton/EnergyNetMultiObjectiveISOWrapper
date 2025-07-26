#!/usr/bin/env python3
"""
Quick test script for Multi-Objective SAC - runs faster tests for SLURM validation.
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path to import MO-SAC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_objective_sac import MultiObjectiveSAC, train_mo_sac, evaluate_mo_sac

# Import test environments - handle both module and direct import cases
try:
    from mo_sac_testing.test_environments import make_env, get_available_envs
except ImportError:
    # Fallback for when files are copied to working directory (SLURM case)
    from test_environments import make_env, get_available_envs


def quick_test_environment(env_name: str, 
                          total_timesteps: int = 10000,
                          save_plots: bool = True) -> bool:
    """Quick test of MO-SAC on a single environment with reduced timesteps."""
    
    print(f"\n{'='*60}")
    print(f"Quick Testing MO-SAC on {env_name}")
    print(f"Timesteps: {total_timesteps} (reduced for quick testing)")
    print(f"{'='*60}")
    
    try:
        # Create environment
        env = make_env(env_name)
        print(f"Environment created successfully:")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Reward space: {env.reward_space}")
        
        # Set up default weights (equal weighting)
        num_objectives = env.reward_space.shape[0]
        weights = np.ones(num_objectives) / num_objectives
        print(f"  Using equal weights: {weights}")
        
        # Create agent with reduced network sizes for faster training
        agent = MultiObjectiveSAC(
            env.observation_space,
            env.action_space,
            num_objectives=num_objectives,
            weights=weights,
            device='cuda' if sys.platform != 'darwin' else 'cpu',
            # Smaller networks for faster testing
            hidden_dim=128,  # Reduced from default
            batch_size=128,  # Reduced batch size
            memory_size=10000,  # Reduced memory
        )
        
        print(f"Agent created with device: {agent.device}")
        
        # Quick training
        start_time = time.time()
        results = train_mo_sac(
            agent=agent,
            env=env,
            total_timesteps=total_timesteps,
            eval_freq=max(1000, total_timesteps // 10),  # Evaluate less frequently
            save_freq=max(2000, total_timesteps // 5),   # Save less frequently
            verbose=True,
            log_dir=f"./runs/quick_{env_name}",
            save_path=f"./models/quick_{env_name}.pth"
        )
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final episode reward: {results['episode_rewards'][-1]:.4f}")
        
        # Quick evaluation
        print("\nRunning quick evaluation...")
        start_time = time.time()
        eval_results = evaluate_mo_sac(
            agent=agent,
            env=env,
            num_episodes=3,  # Very quick evaluation
            render=False
        )
        eval_time = time.time() - start_time
        
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        print(f"Mean evaluation reward: {eval_results['mean_reward']:.4f}")
        print(f"Std evaluation reward: {eval_results['std_reward']:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"\nERROR: Testing failed for {env_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for quick testing."""
    
    print("="*80)
    print("Multi-Objective SAC Quick Test")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Test imports first
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not available")
        return False
    
    try:
        import gymnasium as gym
        print(f"Gymnasium version: {gym.__version__}")
    except ImportError:
        try:
            import gym
            print(f"Gym version: {gym.__version__}")
        except ImportError:
            print("Neither gymnasium nor gym available")
            return False
    
    # Get available environments
    try:
        available_envs = get_available_envs()
        print(f"\nAvailable environments: {available_envs}")
    except Exception as e:
        print(f"Error getting available environments: {e}")
        return False
    
    # Test specific environment or default
    if len(sys.argv) > 1:
        env_name = sys.argv[1]
        if env_name not in available_envs:
            print(f"Error: Environment '{env_name}' not available")
            print(f"Available: {available_envs}")
            return False
        test_envs = [env_name]
    else:
        # Default to CartPole for quick test
        test_envs = ["MultiObjectiveContinuousCartPole-v0"]
    
    # Run tests
    success_count = 0
    total_start_time = time.time()
    
    for env_name in test_envs:
        print(f"\n{'-'*60}")
        print(f"Testing environment: {env_name}")
        print(f"{'-'*60}")
        
        if quick_test_environment(env_name, total_timesteps=10000):
            success_count += 1
            print(f"✓ {env_name} test PASSED")
        else:
            print(f"✗ {env_name} test FAILED")
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("QUICK TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {len(test_envs)}")
    print(f"Tests passed: {success_count}")
    print(f"Tests failed: {len(test_envs) - success_count}")
    print(f"Total time: {total_time:.2f} seconds")
    
    if success_count == len(test_envs):
        print("🎉 All tests PASSED! MO-SAC is working correctly.")
        print("\nTo run full tests (slower but more comprehensive):")
        print("  python test_mo_sac.py")
        return True
    else:
        print("❌ Some tests FAILED. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
