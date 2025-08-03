"""
Training script for Multi-Objective SAC on EnergyNet MoISO environment.
"""

import sys
import os
import numpy as np
import argparse
from typing import Dict, Any
import time
import json

# Add paths for imports - handle both original and SLURM temporary directory cases
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# Add root directory (for EnergyNetMoISO package)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from multi_objective_sac import MultiObjectiveSAC, train_mo_sac, evaluate_mo_sac
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from EnergyNetMoISO.pcs_models.constant_pcs_agent import ConstantPCSAgent


def create_energynet_env(**kwargs) -> MultiObjectiveISOEnv:
    """Create EnergyNet MoISO environment with default configuration."""
    default_config = {
        'use_dispatch_action': False,
        'dispatch_strategy': "PROPORTIONAL",
        'trained_pcs_model': ConstantPCSAgent(1),
        # Add other environment parameters as needed
    }
    default_config.update(kwargs)
    
    return MultiObjectiveISOEnv(**default_config)


def train_mo_sac_on_energynet(
    # Environment parameters
    env_config: Dict[str, Any] = None,
    
    # Training parameters
    total_timesteps: int = 1000000,
    learning_starts: int = 10000,
    train_freq: int = 1,
    eval_freq: int = 20000,
    eval_episodes: int = 10,
    save_freq: int = 100000,
    
    # Agent parameters
    weights: np.ndarray = None,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    auto_tune_alpha: bool = True,
    actor_hidden_dims: list = [256, 256],
    critic_hidden_dims: list = [256, 256],
    buffer_capacity: int = 1000000,
    batch_size: int = 256,
    
    # Optimization parameters (NEW)
    use_lr_annealing: bool = False,
    lr_annealing_type: str = 'cosine',
    lr_annealing_steps: int = None,
    lr_min_factor: float = 0.1,
    lr_decay_rate: float = 0.95,
    use_reward_scaling: bool = False,
    reward_scale_epsilon: float = 1e-4,
    use_orthogonal_init: bool = True,
    orthogonal_gain: float = 1.0,
    actor_orthogonal_gain: float = 0.01,
    critic_orthogonal_gain: float = 1.0,
    use_value_clipping: bool = False,
    value_clip_range: float = 200.0,
    
    # Logging parameters
    experiment_name: str = "mo_sac_energynet",
    save_dir: str = "energynet_experiments",
    verbose: bool = True,
    tensorboard_log: bool = True
) -> Dict[str, Any]:
    """
    Train Multi-Objective SAC on EnergyNet MoISO environment.
    
    Returns:
        Dictionary containing training results and statistics.
    """
    
    print(f"{'='*80}")
    print(f"Training MO-SAC on EnergyNet MoISO Environment")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}")
    
    # Create directories - use base directory structure
    base_dir = os.path.dirname(save_dir) if save_dir.endswith(('models', 'logs', 'plots')) else save_dir
    models_dir = f"{base_dir}/models" 
    logs_dir = f"{base_dir}/logs"
    plots_dir = f"{base_dir}/plots"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True) 
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create environment
    if env_config is None:
        env_config = {}
    
    env = create_energynet_env(**env_config)
    
    # Get environment info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    print(f"Environment Configuration:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Reward dimension: {reward_dim}")
    print(f"  Action bounds: {action_bounds}")
    print(f"  Environment config: {env_config}")
    
    # Set default weights if not provided
    if weights is None:
        weights = np.array([0.6, 0.4])  # Slightly prefer cost over stability
        print(f"  Using default weights: {weights}")
    else:
        print(f"  Using provided weights: {weights}")
    
    # Create agent with timestamp for consistent naming
    timestamp = int(time.time())
    tensorboard_path = f"{logs_dir}/{experiment_name}_{timestamp}" if tensorboard_log else None
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=weights,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        auto_tune_alpha=auto_tune_alpha,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        # Optimization parameters
        use_lr_annealing=use_lr_annealing,
        lr_annealing_type=lr_annealing_type,
        lr_annealing_steps=lr_annealing_steps,
        lr_min_factor=lr_min_factor,
        lr_decay_rate=lr_decay_rate,
        use_reward_scaling=use_reward_scaling,
        reward_scale_epsilon=reward_scale_epsilon,
        use_orthogonal_init=use_orthogonal_init,
        orthogonal_gain=orthogonal_gain,
        actor_orthogonal_gain=actor_orthogonal_gain,
        critic_orthogonal_gain=critic_orthogonal_gain,
        use_value_clipping=use_value_clipping,
        value_clip_range=value_clip_range,
        verbose=verbose,
        tensorboard_log=tensorboard_path
    )
    
    print(f"\nAgent Configuration:")
    print(f"  Actor LR: {actor_lr}, Critic LR: {critic_lr}, Alpha LR: {alpha_lr}")
    print(f"  Gamma: {gamma}, Tau: {tau}, Initial Alpha: {alpha}")
    print(f"  Auto-tune Alpha: {auto_tune_alpha}")
    print(f"  Actor hidden dims: {actor_hidden_dims}")
    print(f"  Critic hidden dims: {critic_hidden_dims}")
    print(f"  Buffer capacity: {buffer_capacity}, Batch size: {batch_size}")
    
    print(f"\nOptimization Features:")
    print(f"  LR Annealing: {use_lr_annealing}")
    if use_lr_annealing:
        print(f"    Type: {lr_annealing_type}")
        print(f"    Steps: {lr_annealing_steps}")
        print(f"    Min factor: {lr_min_factor}")
        if lr_annealing_type == 'exponential':
            print(f"    Decay rate: {lr_decay_rate}")
    print(f"  Reward Scaling: {use_reward_scaling}")
    if use_reward_scaling:
        print(f"    Epsilon: {reward_scale_epsilon}")
    print(f"  Orthogonal Init: {use_orthogonal_init}")
    if use_orthogonal_init:
        print(f"    Actor gain: {actor_orthogonal_gain}")
        print(f"    Critic gain: {critic_orthogonal_gain}")
    print(f"  Value Clipping: {use_value_clipping}")
    if use_value_clipping:
        print(f"    Clip range: {value_clip_range}")
    
    if tensorboard_path:
        print(f"  Tensorboard log: {tensorboard_path}")
    
    # Save configuration
    config = {
        'env_config': env_config,
        'training_params': {
            'total_timesteps': total_timesteps,
            'learning_starts': learning_starts,
            'train_freq': train_freq,
            'eval_freq': eval_freq,
            'eval_episodes': eval_episodes,
            'save_freq': save_freq
        },
        'agent_params': {
            'weights': weights.tolist(),
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'alpha_lr': alpha_lr,
            'gamma': gamma,
            'tau': tau,
            'alpha': alpha,
            'auto_tune_alpha': auto_tune_alpha,
            'actor_hidden_dims': actor_hidden_dims,
            'critic_hidden_dims': critic_hidden_dims,
            'buffer_capacity': buffer_capacity,
            'batch_size': batch_size,
            # Optimization parameters
            'use_lr_annealing': use_lr_annealing,
            'lr_annealing_type': lr_annealing_type,
            'lr_annealing_steps': lr_annealing_steps,
            'lr_min_factor': lr_min_factor,
            'lr_decay_rate': lr_decay_rate,
            'use_reward_scaling': use_reward_scaling,
            'reward_scale_epsilon': reward_scale_epsilon,
            'use_orthogonal_init': use_orthogonal_init,
            'orthogonal_gain': orthogonal_gain,
            'actor_orthogonal_gain': actor_orthogonal_gain,
            'critic_orthogonal_gain': critic_orthogonal_gain,
            'use_value_clipping': use_value_clipping,
            'value_clip_range': value_clip_range,
        }
    }
    
    with open(f"{base_dir}/{experiment_name}_{timestamp}_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train agent
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    training_stats = train_mo_sac(
        env=env,
        agent=agent,
        total_timesteps=total_timesteps,
        learning_starts=learning_starts,
        train_freq=train_freq,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        save_freq=save_freq,
        save_path=f"{models_dir}/{experiment_name}_{timestamp}",
        verbose=verbose
    )
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    final_rewards = evaluate_mo_sac(env, agent, num_episodes=50, verbose=False)
    mean_rewards = np.mean(final_rewards, axis=0)
    std_rewards = np.std(final_rewards, axis=0)
    
    print(f"\nFinal Performance (50 episodes):")
    print(f"  Cost reward: {mean_rewards[0]:.3f} ± {std_rewards[0]:.3f}")
    print(f"  Stability reward: {mean_rewards[1]:.3f} ± {std_rewards[1]:.3f}")
    
    scalarized_rewards = np.sum(final_rewards * weights, axis=1)
    print(f"  Scalarized reward: {np.mean(scalarized_rewards):.3f} ± {np.std(scalarized_rewards):.3f}")
    
    # Save final model and results
    final_model_path = f"{models_dir}/{experiment_name}_{timestamp}_final.pth"
    agent.save(final_model_path)
    
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'final_evaluation': {
            'mean_rewards': mean_rewards.tolist(),
            'std_rewards': std_rewards.tolist(),
            'scalarized_mean': float(np.mean(scalarized_rewards)),
            'scalarized_std': float(np.std(scalarized_rewards)),
            'all_rewards': final_rewards.tolist()
        },
        'training_stats': {
            'num_episodes': len(training_stats['episode_rewards']),
            'final_episode_reward': training_stats['episode_rewards'][-1].tolist() if training_stats['episode_rewards'] else None
        },
        'model_path': final_model_path
    }
    
    # Save results
    with open(f"{base_dir}/{experiment_name}_{timestamp}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  Config: {base_dir}/{experiment_name}_{timestamp}_config.json")
    print(f"  Results: {base_dir}/{experiment_name}_{timestamp}_results.json")
    print(f"  Final model: {final_model_path}")
    if tensorboard_path:
        print(f"  Tensorboard logs: {tensorboard_path}")
    
    return results


def main():
    """Main training script with command line arguments."""
    parser = argparse.ArgumentParser(description="Train MO-SAC on EnergyNet MoISO")
    
    # Experiment parameters
    parser.add_argument('--experiment-name', type=str, default='mo_sac_energynet',
                       help='Name of the experiment')
    parser.add_argument('--save-dir', type=str, default='energynet_experiments',
                       help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--learning-starts', type=int, default=10000,
                       help='Timesteps before learning starts')
    parser.add_argument('--eval-freq', type=int, default=20000,
                       help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=100000,
                       help='Model save frequency')
    
    # Agent parameters
    parser.add_argument('--weights', type=float, nargs=2, default=[0.6, 0.4],
                       help='Multi-objective weights [cost_weight, stability_weight]')
    parser.add_argument('--actor-lr', type=float, default=3e-4,
                       help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=3e-4,
                       help='Critic learning rate')
    parser.add_argument('--alpha-lr', type=float, default=3e-4,
                       help='Alpha learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Target network update rate')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                       help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    
    # Optimization parameters (NEW)
    parser.add_argument('--use-lr-annealing', action='store_true',
                       help='Enable learning rate annealing')
    parser.add_argument('--lr-annealing-type', type=str, default='cosine',
                       choices=['cosine', 'linear', 'exponential'],
                       help='Type of learning rate annealing')
    parser.add_argument('--lr-annealing-steps', type=int, default=None,
                       help='Number of steps for LR annealing (default: total_timesteps/train_freq)')
    parser.add_argument('--lr-min-factor', type=float, default=0.1,
                       help='Minimum LR as fraction of initial LR')
    parser.add_argument('--lr-decay-rate', type=float, default=0.95,
                       help='Decay rate for exponential LR annealing')
    parser.add_argument('--use-reward-scaling', action='store_true',
                       help='Enable reward scaling/normalization')
    parser.add_argument('--reward-scale-epsilon', type=float, default=1e-4,
                       help='Epsilon for reward scaling')
    parser.add_argument('--disable-orthogonal-init', action='store_true',
                       help='Disable orthogonal initialization (use Xavier instead)')
    parser.add_argument('--orthogonal-gain', type=float, default=1.0,
                       help='Gain for orthogonal initialization')
    parser.add_argument('--actor-orthogonal-gain', type=float, default=0.01,
                       help='Gain for actor orthogonal initialization')
    parser.add_argument('--critic-orthogonal-gain', type=float, default=1.0,
                       help='Gain for critic orthogonal initialization')
    parser.add_argument('--use-value-clipping', action='store_true',
                       help='Enable value clipping for stability')
    parser.add_argument('--value-clip-range', type=float, default=200.0,
                       help='Value clipping range')
    
    # Environment parameters
    parser.add_argument('--use-dispatch-action', action='store_true',
                       help='Use dispatch action in environment')
    parser.add_argument('--dispatch-strategy', type=str, default='PROPORTIONAL',
                       help='Dispatch strategy')
    
    # Logging parameters
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose logging')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable tensorboard logging')
    
    args = parser.parse_args()
    
    # Convert weights to numpy array and normalize
    weights = np.array(args.weights)
    weights = weights / np.sum(weights)
    
    # Environment configuration
    env_config = {
        'use_dispatch_action': args.use_dispatch_action,
        'dispatch_strategy': args.dispatch_strategy
    }
    
    # Train the agent
    results = train_mo_sac_on_energynet(
        env_config=env_config,
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        weights=weights,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        tau=args.tau,
        buffer_capacity=args.buffer_size,
        batch_size=args.batch_size,
        # Optimization parameters
        use_lr_annealing=args.use_lr_annealing,
        lr_annealing_type=args.lr_annealing_type,
        lr_annealing_steps=args.lr_annealing_steps,
        lr_min_factor=args.lr_min_factor,
        lr_decay_rate=args.lr_decay_rate,
        use_reward_scaling=args.use_reward_scaling,
        reward_scale_epsilon=args.reward_scale_epsilon,
        use_orthogonal_init=not args.disable_orthogonal_init,
        orthogonal_gain=args.orthogonal_gain,
        actor_orthogonal_gain=args.actor_orthogonal_gain,
        critic_orthogonal_gain=args.critic_orthogonal_gain,
        use_value_clipping=args.use_value_clipping,
        value_clip_range=args.value_clip_range,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir,
        verbose=args.verbose,
        tensorboard_log=not args.no_tensorboard
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Final scalarized performance: {results['final_evaluation']['scalarized_mean']:.3f}")


if __name__ == "__main__":
    main()
