"""
Test script for Multi-Objective SAC on various continuous control environments.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# Add parent directory to path to import MO-SAC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_objective_sac import MultiObjectiveSAC, train_mo_sac, evaluate_mo_sac
from mo_sac_testing.test_environments import make_env, get_available_envs


def test_single_environment(env_name: str, 
                          total_timesteps: int = 100000,
                          weights: np.ndarray = None,
                          save_plots: bool = True,
                          verbose: bool = True) -> Dict:
    """Test MO-SAC on a single environment."""
    
    print(f"\n{'='*60}")
    print(f"Testing MO-SAC on {env_name}")
    print(f"{'='*60}")
    
    # Create environment
    env = make_env(env_name)
    
    # Get environment info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (env.action_space.low[0], env.action_space.high[0])
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Reward dimension: {reward_dim}")
    print(f"Action bounds: {action_bounds}")
    
    # Create agent
    if weights is None:
        weights = np.ones(reward_dim) / reward_dim  # Equal weights
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=weights,
        verbose=verbose,
        tensorboard_log=f"runs/mo_sac_{env_name}_{int(time.time())}"
    )
    
    print(f"Using weights: {weights}")
    
    # Train agent
    start_time = time.time()
    training_stats = train_mo_sac(
        env=env,
        agent=agent,
        total_timesteps=total_timesteps,
        learning_starts=1000,
        train_freq=1,
        eval_freq=10000,
        eval_episodes=5,
        save_freq=20000,
        save_path=f"models/mo_sac_{env_name}",
        verbose=verbose
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_rewards = evaluate_mo_sac(env, agent, num_episodes=20, verbose=False)
    mean_rewards = np.mean(final_rewards, axis=0)
    std_rewards = np.std(final_rewards, axis=0)
    
    print(f"Final performance:")
    for i, (mean_r, std_r) in enumerate(zip(mean_rewards, std_rewards)):
        print(f"  Objective {i+1}: {mean_r:.3f} ± {std_r:.3f}")
    
    scalarized_rewards = np.sum(final_rewards * weights, axis=1)
    print(f"  Scalarized: {np.mean(scalarized_rewards):.3f} ± {np.std(scalarized_rewards):.3f}")
    
    # Plot results
    if save_plots and training_stats['episode_rewards']:
        plot_training_results(training_stats, env_name, weights, save=True)
    
    # Save final model
    final_model_path = f"models/mo_sac_{env_name}_final.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    agent.save(final_model_path)
    
    return {
        'env_name': env_name,
        'training_time': training_time,
        'final_rewards': final_rewards,
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'weights': weights,
        'model_path': final_model_path
    }


def plot_training_results(training_stats: Dict, 
                         env_name: str, 
                         weights: np.ndarray,
                         save: bool = True):
    """Plot training results."""
    
    episode_rewards = np.array(training_stats['episode_rewards'])
    episode_lengths = training_stats['episode_lengths']
    
    num_objectives = episode_rewards.shape[1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'MO-SAC Training Results: {env_name}', fontsize=16)
    
    # Plot individual objective rewards
    ax = axes[0, 0]
    for i in range(num_objectives):
        # Moving average for smoothing
        window_size = min(50, len(episode_rewards) // 10)
        if window_size > 1:
            smoothed_rewards = np.convolve(episode_rewards[:, i], 
                                         np.ones(window_size)/window_size, 
                                         mode='valid')
            episodes = range(window_size-1, len(episode_rewards))
            ax.plot(episodes, smoothed_rewards, label=f'Objective {i+1}')
        else:
            ax.plot(episode_rewards[:, i], label=f'Objective {i+1}')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Individual Objective Rewards')
    ax.legend()
    ax.grid(True)
    
    # Plot scalarized rewards
    ax = axes[0, 1]
    scalarized_rewards = np.sum(episode_rewards * weights, axis=1)
    window_size = min(50, len(scalarized_rewards) // 10)
    if window_size > 1:
        smoothed_scalarized = np.convolve(scalarized_rewards, 
                                        np.ones(window_size)/window_size, 
                                        mode='valid')
        episodes = range(window_size-1, len(scalarized_rewards))
        ax.plot(episodes, smoothed_scalarized, 'purple', linewidth=2)
    else:
        ax.plot(scalarized_rewards, 'purple', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Scalarized Reward')
    ax.set_title(f'Scalarized Reward (weights: {weights})')
    ax.grid(True)
    
    # Plot episode lengths
    ax = axes[1, 0]
    window_size = min(50, len(episode_lengths) // 10)
    if window_size > 1:
        smoothed_lengths = np.convolve(episode_lengths, 
                                     np.ones(window_size)/window_size, 
                                     mode='valid')
        episodes = range(window_size-1, len(episode_lengths))
        ax.plot(episodes, smoothed_lengths, 'green')
    else:
        ax.plot(episode_lengths, 'green')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Lengths')
    ax.grid(True)
    
    # Plot Pareto front (for 2-objective problems)
    if num_objectives == 2:
        ax = axes[1, 1]
        # Take final episodes for Pareto front
        final_episodes = max(1, len(episode_rewards) // 10)
        final_rewards = episode_rewards[-final_episodes:]
        
        ax.scatter(final_rewards[:, 0], final_rewards[:, 1], alpha=0.6, s=20)
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title('Pareto Front (Final Episodes)')
        ax.grid(True)
    else:
        # For >2 objectives, show correlation matrix
        ax = axes[1, 1]
        corr_matrix = np.corrcoef(episode_rewards.T)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('Objective Correlation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
        
        # Add text annotations
        for i in range(num_objectives):
            for j in range(num_objectives):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")
    
    plt.tight_layout()
    
    if save:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/mo_sac_{env_name}_training.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved: plots/mo_sac_{env_name}_training.png")
    
    plt.show()


def test_weight_sensitivity(env_name: str, 
                          weight_configs: List[np.ndarray],
                          total_timesteps: int = 50000) -> Dict:
    """Test MO-SAC with different weight configurations."""
    
    print(f"\n{'='*60}")
    print(f"Weight Sensitivity Analysis: {env_name}")
    print(f"{'='*60}")
    
    results = []
    
    for i, weights in enumerate(weight_configs):
        print(f"\nTesting weight configuration {i+1}/{len(weight_configs)}: {weights}")
        
        result = test_single_environment(
            env_name=env_name,
            total_timesteps=total_timesteps,
            weights=weights,
            save_plots=False,
            verbose=False
        )
        
        results.append(result)
        
        print(f"Results: {result['mean_rewards']}")
    
    # Plot comparison
    plot_weight_comparison(results, env_name)
    
    return results


def plot_weight_comparison(results: List[Dict], env_name: str):
    """Plot comparison of different weight configurations."""
    
    num_configs = len(results)
    num_objectives = len(results[0]['mean_rewards'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Weight Sensitivity Analysis: {env_name}', fontsize=16)
    
    # Plot mean rewards for each objective
    ax = axes[0]
    x = np.arange(num_configs)
    width = 0.35
    
    for obj in range(num_objectives):
        means = [r['mean_rewards'][obj] for r in results]
        stds = [r['std_rewards'][obj] for r in results]
        ax.bar(x + obj * width, means, width, yerr=stds, 
               label=f'Objective {obj+1}', alpha=0.8)
    
    ax.set_xlabel('Weight Configuration')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Rewards by Weight Configuration')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([f"W{i+1}" for i in range(num_configs)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot Pareto front comparison (for 2-objective problems)
    if num_objectives == 2:
        ax = axes[1]
        colors = plt.cm.tab10(np.linspace(0, 1, num_configs))
        
        for i, result in enumerate(results):
            final_rewards = result['final_rewards']
            weights = result['weights']
            ax.scatter(final_rewards[:, 0], final_rewards[:, 1], 
                      c=[colors[i]], alpha=0.6, s=30, 
                      label=f'W{i+1}: {weights}')
        
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title('Pareto Fronts Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/mo_sac_{env_name}_weight_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved: plots/mo_sac_{env_name}_weight_comparison.png")
    plt.show()


def run_comprehensive_tests():
    """Run comprehensive tests on all available environments."""
    
    print("Starting comprehensive MO-SAC tests...")
    print(f"Available environments: {get_available_envs()}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Test configurations
    test_configs = [
        {
            'env_name': 'MultiObjectiveContinuousCartPole-v0',
            'timesteps': 100000,
            'weight_configs': [
                np.array([1.0, 0.0]),  # Only balance
                np.array([0.0, 1.0]),  # Only position
                np.array([0.5, 0.5]),  # Equal weights
                np.array([0.7, 0.3]),  # Prefer balance
                np.array([0.3, 0.7])   # Prefer position
            ]
        },
        {
            'env_name': 'MultiObjectiveMountainCarContinuous-v0',
            'timesteps': 150000,
            'weight_configs': [
                np.array([1.0, 0.0]),  # Only time
                np.array([0.0, 1.0]),  # Only energy
                np.array([0.5, 0.5]),  # Equal weights
                np.array([0.8, 0.2]),  # Prefer speed
                np.array([0.2, 0.8])   # Prefer efficiency
            ]
        },
        {
            'env_name': 'MultiObjectivePendulum-v0',
            'timesteps': 80000,
            'weight_configs': [
                np.array([1.0, 0.0]),  # Only stability
                np.array([0.0, 1.0]),  # Only control effort
                np.array([0.5, 0.5]),  # Equal weights
                np.array([0.9, 0.1]),  # Prefer stability
                np.array([0.1, 0.9])   # Prefer efficiency
            ]
        }
    ]
    
    all_results = {}
    
    for config in test_configs:
        env_name = config['env_name']
        
        print(f"\n{'#'*80}")
        print(f"TESTING ENVIRONMENT: {env_name}")
        print(f"{'#'*80}")
        
        # Single test with equal weights
        single_result = test_single_environment(
            env_name=env_name,
            total_timesteps=config['timesteps'],
            weights=np.array([0.5, 0.5]),
            save_plots=True,
            verbose=True
        )
        
        # Weight sensitivity analysis
        weight_results = test_weight_sensitivity(
            env_name=env_name,
            weight_configs=config['weight_configs'],
            total_timesteps=config['timesteps'] // 2  # Shorter for sensitivity analysis
        )
        
        all_results[env_name] = {
            'single_test': single_result,
            'weight_sensitivity': weight_results
        }
    
    # Save summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    for env_name, results in all_results.items():
        print(f"\n{env_name}:")
        single = results['single_test']
        print(f"  Training time: {single['training_time']:.2f}s")
        print(f"  Final performance: {single['mean_rewards']}")
        print(f"  Best weight config from sensitivity analysis:")
        
        # Find best weight config based on scalarized performance
        best_idx = 0
        best_score = -np.inf
        for i, result in enumerate(results['weight_sensitivity']):
            score = np.sum(result['mean_rewards'] * result['weights'])
            if score > best_score:
                best_score = score
                best_idx = i
        
        best_result = results['weight_sensitivity'][best_idx]
        print(f"    Weights: {best_result['weights']}")
        print(f"    Performance: {best_result['mean_rewards']}")
    
    print(f"\nAll models saved in: models/")
    print(f"All plots saved in: plots/")
    print(f"Tensorboard logs in: runs/")


if __name__ == "__main__":
    # Quick test on a single environment
    if len(sys.argv) > 1:
        env_name = sys.argv[1]
        if env_name in get_available_envs():
            test_single_environment(env_name, total_timesteps=50000)
        else:
            print(f"Unknown environment: {env_name}")
            print(f"Available environments: {get_available_envs()}")
    else:
        # Run comprehensive tests
        run_comprehensive_tests()
