"""
Continue training script for Multi-Objective SAC on EnergyNet MoISO environment.
This script loads an existing trained MO-SAC agent and continues training.
"""

import sys
import os
import numpy as np
import argparse
from typing import Dict, Any
import time
import json
import torch

# Add paths for imports - handle both original and SLURM temporary directory cases
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# Add root directory (for EnergyNetMoISO package)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Add parent directory (for multi_objective_sac.py)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from multi_objective_sac import MultiObjectiveSAC, train_mo_sac, evaluate_mo_sac
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from EnergyNetMoISO.pcs_models.constant_pcs_agent import ConstantPCSAgent
from EnergyNetMoISO.pcs_models.sac_pcs_agent import SACPCSAgent


def create_energynet_env(**kwargs) -> MultiObjectiveISOEnv:
    """Create EnergyNet MoISO environment with default configuration."""
    default_config = {
        'use_dispatch_action': True,
        'dispatch_strategy': "PROPORTIONAL",
        'trained_pcs_model': None,
        # Add other environment parameters as needed
    }
    default_config.update(kwargs)
    
    # Handle trained_pcs_model if it's a path string
    trained_pcs_model_path = default_config.get('trained_pcs_model')
    if trained_pcs_model_path and isinstance(trained_pcs_model_path, str):
        if os.path.exists(trained_pcs_model_path):
            print(f"Loading PCS model from: {trained_pcs_model_path}")
            try:
                # Create SAC PCS agent and load the trained model
                # We need to create a dummy environment to get the action/observation spaces
                from energy_net.env.pcs_unit_v0 import PCSUnitEnv
                temp_env = PCSUnitEnv()
                state_dim = temp_env.observation_space.shape[0]
                action_dim = temp_env.action_space.shape[0]
                action_bounds = (float(temp_env.action_space.low[0]), float(temp_env.action_space.high[0]))
                temp_env.close()
                
                # Create and load the SAC PCS agent
                pcs_agent = SACPCSAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    action_bounds=action_bounds
                )
                pcs_agent.load(trained_pcs_model_path)
                default_config['trained_pcs_model'] = pcs_agent
                print(f"✓ Successfully loaded PCS model")
            except Exception as e:
                print(f"Warning: Failed to load PCS model from {trained_pcs_model_path}: {e}")
                print("Using default PCS behavior instead")
                default_config['trained_pcs_model'] = None
        else:
            print(f"Warning: PCS model file not found: {trained_pcs_model_path}")
            print("Using default PCS behavior instead")
            default_config['trained_pcs_model'] = None
    
    return MultiObjectiveISOEnv(**default_config)


def continue_train_mo_sac_on_energynet(
    # Model loading parameters
    model_path: str,
    
    # Environment parameters
    env_config: Dict[str, Any] = None,
    
    # Training parameters
    total_timesteps: int = 1000000,
    learning_starts: int = 10000,
    train_freq: int = 1,
    eval_freq: int = 20000,
    eval_episodes: int = 10,
    save_freq: int = 100000,
    
    # Optimization parameters that can be modified for continued training
    use_lr_annealing: bool = None,  # None means keep existing setting
    lr_annealing_type: str = None,
    lr_annealing_steps: int = None,
    lr_min_factor: float = None,
    lr_decay_rate: float = None,
    use_reward_scaling: bool = None,
    reward_scale_epsilon: float = None,
    use_value_clipping: bool = None,
    value_clip_range: float = None,
    
    # Reproducibility parameters
    seed: int = None,
    
    # Logging parameters
    experiment_name: str = "mo_sac_energynet_continued",
    save_dir: str = "energynet_experiments",
    verbose: bool = True,
    tensorboard_log: bool = True
) -> Dict[str, Any]:
    """
    Continue training Multi-Objective SAC on EnergyNet MoISO environment.
    
    Args:
        model_path: Path to the trained model file (.pth)
        
    Returns:
        Dictionary containing training results and statistics.
    """
    
    print(f"{'='*80}")
    print(f"Continuing MO-SAC Training on EnergyNet MoISO Environment")
    print(f"Loading from: {model_path}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
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
    
    # Load the existing model to get its configuration
    print(f"\\nLoading existing model configuration...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract existing configuration
    existing_weights = checkpoint.get('weights', np.array([0.6, 0.4]))
    existing_config = checkpoint.get('config', {})
    
    print(f"  Loaded weights: {existing_weights}")
    print(f"  Training step: {checkpoint.get('training_step', 0)}")
    print(f"  Episode count: {checkpoint.get('episode_count', 0)}")
    
    # Create agent with same configuration as loaded model
    agent_config = existing_config.get('agent_params', {})
    
    # Override optimization parameters if specified
    if use_lr_annealing is not None:
        agent_config['use_lr_annealing'] = use_lr_annealing
    if lr_annealing_type is not None:
        agent_config['lr_annealing_type'] = lr_annealing_type
    if lr_annealing_steps is not None:
        agent_config['lr_annealing_steps'] = lr_annealing_steps
    if lr_min_factor is not None:
        agent_config['lr_min_factor'] = lr_min_factor
    if lr_decay_rate is not None:
        agent_config['lr_decay_rate'] = lr_decay_rate
    if use_reward_scaling is not None:
        agent_config['use_reward_scaling'] = use_reward_scaling
    if reward_scale_epsilon is not None:
        agent_config['reward_scale_epsilon'] = reward_scale_epsilon
    if use_value_clipping is not None:
        agent_config['use_value_clipping'] = use_value_clipping
    if value_clip_range is not None:
        agent_config['value_clip_range'] = value_clip_range
    
    # Create timestamp for consistent naming
    timestamp = int(time.time())
    tensorboard_path = f"{logs_dir}/{experiment_name}_{timestamp}" if tensorboard_log else None
    
    # Create agent with loaded configuration
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=existing_weights,
        actor_lr=agent_config.get('actor_lr', 3e-4),
        critic_lr=agent_config.get('critic_lr', 3e-4),
        alpha_lr=agent_config.get('alpha_lr', 3e-4),
        gamma=agent_config.get('gamma', 0.99),
        tau=agent_config.get('tau', 0.005),
        alpha=agent_config.get('alpha', 0.2),
        auto_tune_alpha=agent_config.get('auto_tune_alpha', True),
        actor_hidden_dims=agent_config.get('actor_hidden_dims', [256, 256]),
        critic_hidden_dims=agent_config.get('critic_hidden_dims', [256, 256]),
        buffer_capacity=agent_config.get('buffer_capacity', 1000000),
        batch_size=agent_config.get('batch_size', 1024),
        # Optimization parameters (use updated values if provided)
        use_lr_annealing=agent_config.get('use_lr_annealing', False),
        lr_annealing_type=agent_config.get('lr_annealing_type', 'cosine'),
        lr_annealing_steps=agent_config.get('lr_annealing_steps', None),
        lr_min_factor=agent_config.get('lr_min_factor', 0.1),
        lr_decay_rate=agent_config.get('lr_decay_rate', 0.95),
        use_reward_scaling=agent_config.get('use_reward_scaling', False),
        reward_scale_epsilon=agent_config.get('reward_scale_epsilon', 1e-4),
        use_orthogonal_init=agent_config.get('use_orthogonal_init', False),
        orthogonal_gain=agent_config.get('orthogonal_gain', 1.0),
        actor_orthogonal_gain=agent_config.get('actor_orthogonal_gain', 0.01),
        critic_orthogonal_gain=agent_config.get('critic_orthogonal_gain', 1.0),
        use_value_clipping=agent_config.get('use_value_clipping', False),
        value_clip_range=agent_config.get('value_clip_range', 200.0),
        verbose=verbose,
        tensorboard_log=tensorboard_path
    )
    
    # Load the trained model
    print(f"\\nLoading trained model state...")
    agent.load(model_path)
    
    print(f"\\nAgent Configuration (after loading):")
    print(f"  Actor LR: {agent_config.get('actor_lr', 3e-4)}")
    print(f"  Critic LR: {agent_config.get('critic_lr', 3e-4)}")
    print(f"  Alpha LR: {agent_config.get('alpha_lr', 3e-4)}")
    print(f"  Gamma: {agent_config.get('gamma', 0.99)}")
    print(f"  Tau: {agent_config.get('tau', 0.005)}")
    print(f"  Buffer capacity: {agent_config.get('buffer_capacity', 1000000)}")
    print(f"  Batch size: {agent_config.get('batch_size', 256)}")
    
    print(f"\\nOptimization Features:")
    print(f"  LR Annealing: {agent_config.get('use_lr_annealing', False)}")
    if agent_config.get('use_lr_annealing', False):
        print(f"    Type: {agent_config.get('lr_annealing_type', 'cosine')}")
        print(f"    Steps: {agent_config.get('lr_annealing_steps', None)}")
        print(f"    Min factor: {agent_config.get('lr_min_factor', 0.1)}")
    print(f"  Reward Scaling: {agent_config.get('use_reward_scaling', False)}")
    print(f"  Value Clipping: {agent_config.get('use_value_clipping', False)}")
    
    if tensorboard_path:
        print(f"  Tensorboard log: {tensorboard_path}")
    
    # Evaluate current performance before continuing training
    print(f"\\nEvaluating current model performance...")
    eval_rewards = evaluate_mo_sac(env, agent, num_episodes=10, verbose=False)
    eval_mean = np.mean(eval_rewards, axis=0)
    eval_std = np.std(eval_rewards, axis=0)
    eval_scalarized = np.mean(np.sum(eval_rewards * existing_weights, axis=1))
    
    print(f"  Current performance (10 episodes):")
    print(f"    Cost reward: {eval_mean[0]:.3f} ± {eval_std[0]:.3f}")
    print(f"    Stability reward: {eval_mean[1]:.3f} ± {eval_std[1]:.3f}")
    print(f"    Scalarized: {eval_scalarized:.3f}")
    
    # Save configuration for continued training
    config = {
        'continued_from': model_path,
        'env_config': env_config,
        'training_params': {
            'total_timesteps': total_timesteps,
            'learning_starts': learning_starts,
            'train_freq': train_freq,
            'eval_freq': eval_freq,
            'eval_episodes': eval_episodes,
            'save_freq': save_freq
        },
        'agent_params': agent_config,
        'reproducibility': {
            'seed': seed
        },
        'initial_performance': {
            'mean_rewards': eval_mean.tolist(),
            'std_rewards': eval_std.tolist(),
            'scalarized_mean': float(eval_scalarized)
        }
    }
    
    with open(f"{base_dir}/{experiment_name}_{timestamp}_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Continue training
    print(f"\\n{'='*60}")
    print("Continuing Training...")
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
    
    print(f"\\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"Additional training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
    # Final evaluation
    print("\\nPerforming final evaluation...")
    final_rewards = evaluate_mo_sac(env, agent, num_episodes=50, verbose=False)
    final_mean = np.mean(final_rewards, axis=0)
    final_std = np.std(final_rewards, axis=0)
    final_scalarized = np.mean(np.sum(final_rewards * existing_weights, axis=1))
    
    print(f"\\nFinal Performance (50 episodes):")
    print(f"  Cost reward: {final_mean[0]:.3f} ± {final_std[0]:.3f}")
    print(f"  Stability reward: {final_mean[1]:.3f} ± {final_std[1]:.3f}")
    print(f"  Scalarized reward: {final_scalarized:.3f} ± {np.std(np.sum(final_rewards * existing_weights, axis=1)):.3f}")
    
    print(f"\\nImprovement:")
    print(f"  Cost reward: {final_mean[0] - eval_mean[0]:+.3f}")
    print(f"  Stability reward: {final_mean[1] - eval_mean[1]:+.3f}")
    print(f"  Scalarized: {final_scalarized - eval_scalarized:+.3f}")
    
    # Save final model and results
    final_model_path = f"{models_dir}/{experiment_name}_{timestamp}_final.pth"
    agent.save(final_model_path)
    
    results = {
        'experiment_name': experiment_name,
        'continued_from': model_path,
        'config': config,
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'initial_evaluation': {
            'mean_rewards': eval_mean.tolist(),
            'std_rewards': eval_std.tolist(),
            'scalarized_mean': float(eval_scalarized)
        },
        'final_evaluation': {
            'mean_rewards': final_mean.tolist(),
            'std_rewards': final_std.tolist(),
            'scalarized_mean': float(final_scalarized),
            'scalarized_std': float(np.std(np.sum(final_rewards * existing_weights, axis=1))),
            'all_rewards': final_rewards.tolist()
        },
        'improvement': {
            'cost_reward': float(final_mean[0] - eval_mean[0]),
            'stability_reward': float(final_mean[1] - eval_mean[1]),
            'scalarized': float(final_scalarized - eval_scalarized)
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
    
    print(f"\\nResults saved:")
    print(f"  Config: {base_dir}/{experiment_name}_{timestamp}_config.json")
    print(f"  Results: {base_dir}/{experiment_name}_{timestamp}_results.json")
    print(f"  Final model: {final_model_path}")
    if tensorboard_path:
        print(f"  Tensorboard logs: {tensorboard_path}")
    
    return results


def main():
    """Main continue training script with command line arguments."""
    parser = argparse.ArgumentParser(description="Continue training MO-SAC on EnergyNet MoISO")
    
    # Model loading parameters
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model file (.pth)')
    
    # Experiment parameters
    parser.add_argument('--experiment-name', type=str, default='mo_sac_energynet_continued',
                       help='Name of the continued training experiment')
    parser.add_argument('--save-dir', type=str, default='energynet_experiments',
                       help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Additional training timesteps')
    parser.add_argument('--learning-starts', type=int, default=10000,
                       help='Timesteps before learning starts')
    parser.add_argument('--eval-freq', type=int, default=20000,
                       help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=100000,
                       help='Model save frequency')
    
    # Optimization parameters that can be modified for continued training
    parser.add_argument('--use-lr-annealing', action='store_true',
                       help='Enable learning rate annealing (overrides loaded setting)')
    parser.add_argument('--lr-annealing-type', type=str, 
                       choices=['cosine', 'linear', 'exponential'],
                       help='Type of learning rate annealing')
    parser.add_argument('--lr-annealing-steps', type=int,
                       help='Number of steps for LR annealing')
    parser.add_argument('--lr-min-factor', type=float,
                       help='Minimum LR as fraction of initial LR')
    parser.add_argument('--lr-decay-rate', type=float,
                       help='Decay rate for exponential LR annealing')
    parser.add_argument('--use-reward-scaling', action='store_true',
                       help='Enable reward scaling (overrides loaded setting)')
    parser.add_argument('--reward-scale-epsilon', type=float,
                       help='Epsilon for reward scaling')
    parser.add_argument('--use-value-clipping', action='store_true',
                       help='Enable value clipping (overrides loaded setting)')
    parser.add_argument('--value-clip-range', type=float,
                       help='Value clipping range')
    
    # Environment parameters
    parser.add_argument('--use-dispatch-action', action='store_true',
                       help='Use dispatch action in environment')
    parser.add_argument('--dispatch-strategy', type=str, default='PROPORTIONAL',
                       help='Dispatch strategy')
    parser.add_argument('--trained-pcs-model', type=str, default=None,
                       help='Path to trained PCS model (SAC PCS agent)')
    
    # Logging parameters
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose logging')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable tensorboard logging')
    
    # Reproducibility parameters
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        import random
        import torch
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Environment configuration
    env_config = {
        'use_dispatch_action': args.use_dispatch_action,
        'dispatch_strategy': args.dispatch_strategy,
        'trained_pcs_model': args.trained_pcs_model
    }
    
    # Only include optimization parameters if they were explicitly set
    opt_params = {}
    if args.use_lr_annealing:
        opt_params['use_lr_annealing'] = True
    if args.lr_annealing_type:
        opt_params['lr_annealing_type'] = args.lr_annealing_type
    if args.lr_annealing_steps:
        opt_params['lr_annealing_steps'] = args.lr_annealing_steps
    if args.lr_min_factor:
        opt_params['lr_min_factor'] = args.lr_min_factor
    if args.lr_decay_rate:
        opt_params['lr_decay_rate'] = args.lr_decay_rate
    if args.use_reward_scaling:
        opt_params['use_reward_scaling'] = True
    if args.reward_scale_epsilon:
        opt_params['reward_scale_epsilon'] = args.reward_scale_epsilon
    if args.use_value_clipping:
        opt_params['use_value_clipping'] = True
    if args.value_clip_range:
        opt_params['value_clip_range'] = args.value_clip_range
    
    # Continue training
    results = continue_train_mo_sac_on_energynet(
        model_path=args.model_path,
        env_config=env_config,
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir,
        verbose=args.verbose,
        tensorboard_log=not args.no_tensorboard,
        seed=args.seed,
        **opt_params
    )
    
    print(f"\\nContinued training completed successfully!")
    print(f"Final scalarized performance: {results['final_evaluation']['scalarized_mean']:.3f}")
    print(f"Improvement: {results['improvement']['scalarized']:+.3f}")


if __name__ == "__main__":
    main()
