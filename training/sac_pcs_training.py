"""
Training s# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# Add parent directory to Python path - this is where EnergyNetMoISO is located
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also add the multi_objective_iso_algo directory
algo_dir = os.path.join(parent_dir, 'multi_objecctive_iso_algo')
if algo_dir not in sys.path:
    sys.path.insert(0, algo_dir)

# For SLURM jobs, also add current directory (in case files were copied)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)-based PCS Agent on PCSUnitEnv.
"""

import sys
import os
import numpy as np
import argparse
from typing import Dict, Any, Optional
import time
import json
import random
import torch

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# Add root directory to Python path - this is where EnergyNetMoISO is located
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also add the multi_objective_iso_algo directory
algo_dir = os.path.join(parent_dir, 'multi_objecctive_iso_algo')
if algo_dir not in sys.path:
    sys.path.insert(0, algo_dir)

try:
    from energy_net.env.pcs_unit_v0 import PCSUnitEnv
except ImportError:
    print("Warning: Could not import PCSUnitEnv. Make sure energy_net is properly installed.")
    PCSUnitEnv = None

from EnergyNetMoISO.pcs_models.sac_pcs_agent import SACPCSAgent
from multi_objective_sac import MultiObjectiveSAC


def create_pcs_env(trained_iso_model_path: Optional[str] = None, **kwargs):
    """Create PCS Unit environment with optional trained ISO model."""
    if PCSUnitEnv is None:
        raise ImportError("PCSUnitEnv not available. Make sure energy_net is properly installed.")
    
    trained_iso_model_instance = None
    if trained_iso_model_path:
        print(f"Loading trained ISO model from: {trained_iso_model_path}")
        # Load the ISO model
        # We need to determine the dimensions from the saved model or use defaults
        try:
            # Try to load checkpoint to get dimensions
            checkpoint = torch.load(trained_iso_model_path, map_location='cpu')
            state_dim = checkpoint.get('state_dim', 3)
            action_dim = checkpoint.get('action_dim', 3) 
            reward_dim = checkpoint.get('reward_dim', 2)
            action_bounds = checkpoint.get('action_bounds', (-1.0, 1.0))
            
            trained_iso_model_instance = MultiObjectiveSAC(
                state_dim=state_dim,
                action_dim=action_dim,
                reward_dim=reward_dim,
                action_bounds=action_bounds
            )
            trained_iso_model_instance.load(trained_iso_model_path)
            print(f"✓ Successfully loaded ISO model with dimensions: state={state_dim}, action={action_dim}")
        except Exception as e:
            print(f"Warning: Could not load ISO model: {e}")
            print("Proceeding without trained ISO model")
            trained_iso_model_instance = None
    
    return PCSUnitEnv(trained_iso_model_instance=trained_iso_model_instance, **kwargs)


def train_sac_pcs(
    # Environment parameters
    trained_iso_model_path: Optional[str] = None,
    
    # Training parameters
    total_timesteps: int = 1000000,
    learning_starts: int = 10000,
    train_freq: int = 1,
    eval_freq: int = 20000,
    eval_episodes: int = 10,
    save_freq: int = 100000,
    
    # Agent parameters
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
    
    # Optimization parameters
    use_lr_annealing: bool = False,
    lr_annealing_type: str = 'cosine',
    lr_annealing_steps: int = None,
    lr_min_factor: float = 0.1,
    lr_decay_rate: float = 0.95,
    use_reward_scaling: bool = False,
    reward_scale_epsilon: float = 1e-4,
    use_orthogonal_init: bool = False,
    orthogonal_gain: float = 1.0,
    actor_orthogonal_gain: float = 0.01,
    critic_orthogonal_gain: float = 1.0,
    use_value_clipping: bool = False,
    value_clip_range: float = 200.0,
    
    # Training configuration
    seed: int = None,
    experiment_name: str = "sac_pcs_training",
    save_dir: str = "pcs_experiments",
    verbose: bool = True,
    tensorboard_log: bool = True
) -> Dict[str, Any]:
    """
    Train SAC on PCS Unit environment.
    
    Returns:
        Dictionary containing training results and statistics.
    """
    
    print(f"{'='*80}")
    print(f"Training SAC on PCS Unit Environment")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}")
    
    # Set random seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"Set random seed: {seed}")
    
    # Create directories
    base_dir = os.path.dirname(save_dir) if save_dir.endswith(('models', 'logs', 'plots')) else save_dir
    models_dir = f"{base_dir}/models" 
    logs_dir = f"{base_dir}/logs"
    plots_dir = f"{base_dir}/plots"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True) 
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create environment
    env = create_pcs_env(trained_iso_model_path=trained_iso_model_path)
    
    # Get environment info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    print(f"Environment Configuration:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Action bounds: {action_bounds}")
    print(f"  ISO model path: {trained_iso_model_path}")
    
    # Set up annealing steps if not provided
    if use_lr_annealing and lr_annealing_steps is None:
        lr_annealing_steps = total_timesteps // 4  # Anneal over 1/4 of training
    
    # Create agent
    tensorboard_dir = f"{logs_dir}/{experiment_name}" if tensorboard_log else None
    
    agent = SACPCSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=action_bounds,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        auto_tune_alpha=auto_tune_alpha,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
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
        tensorboard_log=tensorboard_dir
    )
    
    print(f"\nAgent Configuration:")
    print(f"  Actor LR: {actor_lr}")
    print(f"  Critic LR: {critic_lr}")
    print(f"  Alpha LR: {alpha_lr}")
    print(f"  Gamma: {gamma}")
    print(f"  Tau: {tau}")
    print(f"  Buffer capacity: {buffer_capacity}")
    print(f"  Batch size: {batch_size}")
    
    # Training loop
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    start_time = time.time()
    
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    # Tracking variables
    episode_rewards = []
    episode_lengths = []
    training_metrics = []
    evaluation_results = []
    
    best_eval_reward = float('-inf')
    
    for timestep in range(total_timesteps):
        # Select action
        if timestep < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # Update agent
        if timestep >= learning_starts and timestep % train_freq == 0:
            metrics = agent.update()
            if metrics and timestep % 1000 == 0:  # Log every 1000 steps
                training_metrics.append({
                    'timestep': timestep,
                    **metrics
                })
        
        # Handle episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1
            agent.episode_count = episode_count
            
            if verbose and episode_count % 100 == 0:
                recent_rewards = episode_rewards[-100:]
                print(f"Episode {episode_count:5d} | "
                      f"Timestep {timestep:7d} | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"Avg(100): {np.mean(recent_rewards):8.2f}")
            
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        
        # Evaluation
        if timestep > 0 and timestep % eval_freq == 0:
            print(f"\nEvaluating at timestep {timestep:,}...")
            eval_rewards = evaluate_sac_pcs(env, agent, eval_episodes, verbose=False)
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            
            evaluation_results.append({
                'timestep': timestep,
                'mean_reward': eval_mean,
                'std_reward': eval_std,
                'rewards': eval_rewards.tolist()
            })
            
            print(f"Evaluation: {eval_mean:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                best_model_path = f"{models_dir}/{experiment_name}_best.pth"
                agent.save(best_model_path)
                if verbose:
                    print(f"New best model saved: {eval_mean:.2f}")
        
        # Save checkpoint
        if timestep > 0 and timestep % save_freq == 0:
            checkpoint_path = f"{models_dir}/{experiment_name}_checkpoint_{timestep}.pth"
            agent.save(checkpoint_path)
            if verbose:
                print(f"Checkpoint saved: {checkpoint_path}")
    
    # Final evaluation
    print(f"\nFinal evaluation...")
    final_eval_rewards = evaluate_sac_pcs(env, agent, eval_episodes * 2, verbose=True)
    final_eval_mean = np.mean(final_eval_rewards)
    final_eval_std = np.std(final_eval_rewards)
    
    # Save final model
    final_model_path = f"{models_dir}/{experiment_name}_final.pth"
    agent.save(final_model_path)
    
    # Training time
    training_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    print(f"Total episodes: {episode_count}")
    print(f"Final evaluation: {final_eval_mean:.2f} ± {final_eval_std:.2f}")
    print(f"Best evaluation: {best_eval_reward:.2f}")
    print(f"Models saved to: {models_dir}")
    print(f"{'='*80}")
    
    # Prepare results
    results = {
        'experiment_name': experiment_name,
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'total_episodes': episode_count,
        'final_evaluation': {
            'mean': final_eval_mean,
            'std': final_eval_std,
            'rewards': final_eval_rewards.tolist()
        },
        'best_evaluation': best_eval_reward,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'evaluation_results': evaluation_results,
        'training_metrics': training_metrics[-100:],  # Keep last 100
        'model_paths': {
            'final': final_model_path,
            'best': f"{models_dir}/{experiment_name}_best.pth"
        },
        'config': {
            'trained_iso_model_path': trained_iso_model_path,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'action_bounds': action_bounds,
            'agent_params': {
                'actor_lr': actor_lr,
                'critic_lr': critic_lr,
                'alpha_lr': alpha_lr,
                'gamma': gamma,
                'tau': tau,
                'alpha': alpha,
                'auto_tune_alpha': auto_tune_alpha,
                'buffer_capacity': buffer_capacity,
                'batch_size': batch_size,
                'use_lr_annealing': use_lr_annealing,
                'lr_annealing_type': lr_annealing_type,
                'lr_annealing_steps': lr_annealing_steps,
                'use_reward_scaling': use_reward_scaling,
                'use_orthogonal_init': use_orthogonal_init,
                'use_value_clipping': use_value_clipping,
            }
        }
    }
    
    # Save results
    results_path = f"{base_dir}/{experiment_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results


def evaluate_sac_pcs(env, agent, num_episodes: int = 10, verbose: bool = False) -> np.ndarray:
    """Evaluate SAC PCS agent."""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            # Prevent infinite episodes
            if step_count > 1000:
                break
        
        episode_rewards.append(episode_reward)
        
        if verbose:
            print(f"  Episode {episode+1:2d}: Reward = {episode_reward:8.2f}, Steps = {step_count:3d}")
    
    return np.array(episode_rewards)


def main():
    parser = argparse.ArgumentParser(description="Train SAC on PCS Unit Environment")
    
    # Environment arguments
    parser.add_argument('--trained-iso-model-path', type=str, default=None,
                       help='Path to trained ISO model (optional)')
    
    # Training arguments
    parser.add_argument('--experiment-name', type=str, default='sac_pcs_training',
                       help='Experiment name for logging')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--learning-starts', type=int, default=10000,
                       help='Timesteps before training starts')
    parser.add_argument('--eval-freq', type=int, default=20000,
                       help='Evaluation frequency')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--save-freq', type=int, default=100000,
                       help='Model save frequency')
    parser.add_argument('--save-dir', type=str, default='pcs_experiments',
                       help='Directory to save results')
    
    # Agent arguments
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
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Entropy coefficient (if not auto-tuning)')
    parser.add_argument('--no-auto-tune-alpha', action='store_true',
                       help='Disable automatic alpha tuning')
    parser.add_argument('--buffer-capacity', type=int, default=1000000,
                       help='Replay buffer capacity')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    
    # Optimization arguments
    parser.add_argument('--use-lr-annealing', action='store_true',
                       help='Enable learning rate annealing')
    parser.add_argument('--lr-annealing-type', type=str, default='cosine',
                       choices=['cosine', 'linear', 'exponential'],
                       help='Learning rate annealing type')
    parser.add_argument('--lr-annealing-steps', type=int, default=None,
                       help='Steps for learning rate annealing')
    parser.add_argument('--lr-min-factor', type=float, default=0.1,
                       help='Minimum learning rate factor')
    parser.add_argument('--lr-decay-rate', type=float, default=0.95,
                       help='Learning rate decay rate (exponential)')
    parser.add_argument('--use-reward-scaling', action='store_true',
                       help='Enable reward scaling/normalization')
    parser.add_argument('--reward-scale-epsilon', type=float, default=1e-4,
                       help='Epsilon for reward scaling')
    parser.add_argument('--use-orthogonal-init', action='store_true',
                       help='Enable orthogonal weight initialization')
    parser.add_argument('--orthogonal-gain', type=float, default=1.0,
                       help='Orthogonal initialization gain')
    parser.add_argument('--actor-orthogonal-gain', type=float, default=0.01,
                       help='Actor orthogonal initialization gain')
    parser.add_argument('--critic-orthogonal-gain', type=float, default=1.0,
                       help='Critic orthogonal initialization gain')
    parser.add_argument('--use-value-clipping', action='store_true',
                       help='Enable value function clipping')
    parser.add_argument('--value-clip-range', type=float, default=200.0,
                       help='Value clipping range')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable tensorboard logging')
    
    args = parser.parse_args()
    
    # Train the agent
    results = train_sac_pcs(
        trained_iso_model_path=args.trained_iso_model_path,
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_tune_alpha=not args.no_auto_tune_alpha,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        use_lr_annealing=args.use_lr_annealing,
        lr_annealing_type=args.lr_annealing_type,
        lr_annealing_steps=args.lr_annealing_steps,
        lr_min_factor=args.lr_min_factor,
        lr_decay_rate=args.lr_decay_rate,
        use_reward_scaling=args.use_reward_scaling,
        reward_scale_epsilon=args.reward_scale_epsilon,
        use_orthogonal_init=args.use_orthogonal_init,
        orthogonal_gain=args.orthogonal_gain,
        actor_orthogonal_gain=args.actor_orthogonal_gain,
        critic_orthogonal_gain=args.critic_orthogonal_gain,
        use_value_clipping=args.use_value_clipping,
        value_clip_range=args.value_clip_range,
        seed=args.seed,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir,
        verbose=args.verbose,
        tensorboard_log=not args.no_tensorboard
    )


if __name__ == "__main__":
    main()
