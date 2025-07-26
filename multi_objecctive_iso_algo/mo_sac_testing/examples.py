"""
Example usage of Multi-Objective SAC algorithm.

This script demonstrates basic usage of the MO-SAC implementation
on both test environments and the EnergyNet environment.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_objective_sac import MultiObjectiveSAC, train_mo_sac, evaluate_mo_sac
from mo_sac_testing.test_environments import make_env


def example_simple_training():
    """Simple example of training MO-SAC on a test environment."""
    
    print("="*60)
    print("Example 1: Simple MO-SAC Training")
    print("="*60)
    
    # Create environment
    env = make_env('MultiObjectivePendulum-v0')
    
    # Get environment properties
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (env.action_space.low[0], env.action_space.high[0])
    
    print(f"Environment: MultiObjectivePendulum-v0")
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Reward dim: {reward_dim}")
    
    # Create agent with equal weights
    weights = np.array([0.5, 0.5])
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=weights,
        verbose=True
    )
    
    # Train for a short time
    print("Training for 20,000 timesteps...")
    training_stats = train_mo_sac(
        env=env,
        agent=agent,
        total_timesteps=20000,
        learning_starts=1000,
        eval_freq=5000,
        eval_episodes=5,
        verbose=True
    )
    
    # Evaluate final performance
    print("\nFinal evaluation...")
    final_rewards = evaluate_mo_sac(env, agent, num_episodes=10)
    mean_rewards = np.mean(final_rewards, axis=0)
    
    print(f"Final performance: {mean_rewards}")
    print(f"Scalarized reward: {np.sum(mean_rewards * weights):.3f}")


def example_weight_comparison():
    """Example of comparing different weight configurations."""
    
    print("\n" + "="*60)
    print("Example 2: Weight Configuration Comparison")
    print("="*60)
    
    # Create environment
    env = make_env('MultiObjectiveContinuousCartPole-v0')
    
    # Get environment properties
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (env.action_space.low[0], env.action_space.high[0])
    
    print(f"Environment: MultiObjectiveContinuousCartPole-v0")
    print(f"Objectives: Balance pole + Keep cart centered")
    
    # Test different weight configurations
    weight_configs = [
        ("Balance Priority", np.array([0.8, 0.2])),
        ("Equal Weights", np.array([0.5, 0.5])),
        ("Position Priority", np.array([0.2, 0.8]))
    ]
    
    results = []
    
    for name, weights in weight_configs:
        print(f"\nTesting {name}: weights = {weights}")
        
        # Create agent
        agent = MultiObjectiveSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            reward_dim=reward_dim,
            action_bounds=action_bounds,
            weights=weights,
            verbose=False
        )
        
        # Quick training
        train_mo_sac(
            env=env,
            agent=agent,
            total_timesteps=15000,
            learning_starts=1000,
            eval_freq=10000,
            eval_episodes=3,
            verbose=False
        )
        
        # Evaluate
        eval_rewards = evaluate_mo_sac(env, agent, num_episodes=10, verbose=False)
        mean_rewards = np.mean(eval_rewards, axis=0)
        scalarized = np.sum(mean_rewards * weights)
        
        print(f"  Balance reward: {mean_rewards[0]:.3f}")
        print(f"  Position reward: {mean_rewards[1]:.3f}")
        print(f"  Scalarized: {scalarized:.3f}")
        
        results.append((name, weights, mean_rewards, scalarized))
    
    # Summary
    print(f"\n{'Config':<20} {'Balance':<10} {'Position':<10} {'Scalarized':<12}")
    print("-" * 60)
    for name, weights, rewards, scalarized in results:
        print(f"{name:<20} {rewards[0]:<10.3f} {rewards[1]:<10.3f} {scalarized:<12.3f}")


def example_custom_configuration():
    """Example of using custom network and training configurations."""
    
    print("\n" + "="*60)
    print("Example 3: Custom Configuration")
    print("="*60)
    
    # Create environment
    env = make_env('MultiObjectiveMountainCarContinuous-v0')
    
    # Get environment properties
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (env.action_space.low[0], env.action_space.high[0])
    
    print(f"Environment: MultiObjectiveMountainCarContinuous-v0")
    print(f"Objectives: Reach goal quickly + Minimize energy")
    
    # Create agent with custom configuration
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=np.array([0.7, 0.3]),  # Prefer speed over efficiency
        
        # Custom network architecture
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        
        # Custom learning rates
        actor_lr=1e-4,
        critic_lr=1e-4,
        alpha_lr=1e-4,
        
        # Custom hyperparameters
        gamma=0.995,  # Slightly higher discount
        tau=0.01,     # Faster target updates
        batch_size=128,  # Smaller batch size
        
        verbose=True
    )
    
    print("Using custom configuration:")
    print("  Networks: [128, 128, 64]")
    print("  Learning rates: 1e-4")
    print("  Gamma: 0.995, Tau: 0.01, Batch size: 128")
    
    # Train with custom schedule
    print("\nTraining with custom schedule...")
    train_mo_sac(
        env=env,
        agent=agent,
        total_timesteps=30000,
        learning_starts=2000,   # Later start
        train_freq=2,           # Less frequent updates
        eval_freq=7500,         # More frequent evaluation
        eval_episodes=8,        # More evaluation episodes
        verbose=True
    )
    
    # Final evaluation
    final_rewards = evaluate_mo_sac(env, agent, num_episodes=15)
    mean_rewards = np.mean(final_rewards, axis=0)
    
    print(f"\nFinal performance:")
    print(f"  Time reward: {mean_rewards[0]:.3f}")
    print(f"  Energy reward: {mean_rewards[1]:.3f}")
    print(f"  Scalarized: {np.sum(mean_rewards * agent.weights):.3f}")


def example_save_and_load():
    """Example of saving and loading trained models."""
    
    print("\n" + "="*60)
    print("Example 4: Save and Load Models")
    print("="*60)
    
    # Create environment
    env = make_env('MultiObjectivePendulum-v0')
    
    # Get environment properties
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (env.action_space.low[0], env.action_space.high[0])
    
    # Create and train agent
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=np.array([0.6, 0.4]),
        verbose=False
    )
    
    print("Training agent...")
    train_mo_sac(
        env=env,
        agent=agent,
        total_timesteps=15000,
        learning_starts=1000,
        verbose=False
    )
    
    # Evaluate before saving
    rewards_before = evaluate_mo_sac(env, agent, num_episodes=5, verbose=False)
    mean_before = np.mean(rewards_before, axis=0)
    
    # Save model
    os.makedirs('example_models', exist_ok=True)
    model_path = 'example_models/demo_model.pth'
    agent.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Create new agent and load model
    new_agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=np.array([0.6, 0.4]),
        verbose=False
    )
    
    new_agent.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Evaluate after loading
    rewards_after = evaluate_mo_sac(env, new_agent, num_episodes=5, verbose=False)
    mean_after = np.mean(rewards_after, axis=0)
    
    print(f"\nPerformance comparison:")
    print(f"  Before save: {mean_before}")
    print(f"  After load:  {mean_after}")
    print(f"  Difference:  {np.abs(mean_before - mean_after)}")


if __name__ == "__main__":
    print("Multi-Objective SAC Examples")
    print("=" * 80)
    
    try:
        # Run examples
        example_simple_training()
        example_weight_comparison()
        example_custom_configuration()
        example_save_and_load()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
        print("\nNext steps:")
        print("1. Run comprehensive tests: python test_mo_sac.py")
        print("2. Train on EnergyNet: python train_energynet.py")
        print("3. Customize parameters for your specific use case")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError during examples: {e}")
        print("Make sure all dependencies are installed and paths are correct.")
