"""
Example script showing how to use the different optimization features in Multi-Objective SAC.
This script demonstrates various optimization configurations for performance tuning.
"""

import sys
import os
import numpy as np

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from multi_objective_sac import MultiObjectiveSAC, train_mo_sac
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from EnergyNetMoISO.pcs_models.constant_pcs_agent import ConstantPCSAgent


def create_energynet_env():
    """Create EnergyNet MoISO environment."""
    return MultiObjectiveISOEnv(
        use_dispatch_action=False,
        dispatch_strategy="PROPORTIONAL",
        trained_pcs_model=ConstantPCSAgent(1)
    )


def baseline_configuration():
    """Baseline configuration without optimizations."""
    print("🔵 BASELINE CONFIGURATION (No Optimizations)")
    print("-" * 50)
    
    env = create_energynet_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=np.array([0.6, 0.4]),
        # Standard parameters
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        # No optimizations
        use_lr_annealing=False,
        use_reward_scaling=False,
        use_orthogonal_init=False,  # Use Xavier initialization
        use_value_clipping=False,
        verbose=True
    )
    
    return env, agent


def conservative_optimization():
    """Conservative optimization configuration - safe improvements."""
    print("🟡 CONSERVATIVE OPTIMIZATION")
    print("-" * 50)
    
    env = create_energynet_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=np.array([0.6, 0.4]),
        # Standard parameters
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        # Conservative optimizations
        use_lr_annealing=False,  # No LR annealing yet
        use_reward_scaling=True,  # Enable reward scaling for stability
        reward_scale_epsilon=1e-4,
        use_orthogonal_init=True,  # Better initialization
        actor_orthogonal_gain=0.01,
        critic_orthogonal_gain=1.0,
        use_value_clipping=False,  # No clipping yet
        verbose=True
    )
    
    return env, agent


def moderate_optimization():
    """Moderate optimization configuration - balanced approach."""
    print("🟠 MODERATE OPTIMIZATION")
    print("-" * 50)
    
    env = create_energynet_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=np.array([0.6, 0.4]),
        # Standard parameters
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        # Moderate optimizations
        use_lr_annealing=True,
        lr_annealing_type='cosine',  # Smooth decay
        lr_min_factor=0.1,  # Decay to 10% of initial LR
        use_reward_scaling=True,
        reward_scale_epsilon=1e-4,
        use_orthogonal_init=True,
        actor_orthogonal_gain=0.01,
        critic_orthogonal_gain=1.0,
        use_value_clipping=True,
        value_clip_range=200.0,  # Conservative clipping
        verbose=True
    )
    
    return env, agent


def aggressive_optimization():
    """Aggressive optimization configuration - maximum performance."""
    print("🔴 AGGRESSIVE OPTIMIZATION")
    print("-" * 50)
    
    env = create_energynet_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=np.array([0.6, 0.4]),
        # Higher learning rates for faster learning
        actor_lr=5e-4,
        critic_lr=5e-4,
        alpha_lr=5e-4,
        # Aggressive optimizations
        use_lr_annealing=True,
        lr_annealing_type='cosine',
        lr_min_factor=0.05,  # Decay to 5% of initial LR
        use_reward_scaling=True,
        reward_scale_epsilon=1e-5,  # More sensitive scaling
        use_orthogonal_init=True,
        actor_orthogonal_gain=0.1,  # Higher actor gain
        critic_orthogonal_gain=1.5,  # Higher critic gain
        use_value_clipping=True,
        value_clip_range=100.0,  # Tighter clipping
        verbose=True
    )
    
    return env, agent


def custom_tuned_optimization():
    """Custom tuned configuration for specific use cases."""
    print("🟣 CUSTOM TUNED OPTIMIZATION")
    print("-" * 50)
    
    env = create_energynet_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        weights=np.array([0.7, 0.3]),  # Prioritize cost over stability
        # Tuned learning rates
        actor_lr=4e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        # Custom optimizations
        use_lr_annealing=True,
        lr_annealing_type='linear',  # Linear decay for more control
        lr_min_factor=0.2,  # Don't decay too much
        use_reward_scaling=True,
        reward_scale_epsilon=1e-4,
        use_orthogonal_init=True,
        actor_orthogonal_gain=0.05,  # Balanced gains
        critic_orthogonal_gain=1.2,
        use_value_clipping=True,
        value_clip_range=150.0,  # Moderate clipping
        # Larger networks for more capacity
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512],
        verbose=True
    )
    
    return env, agent


def run_short_demo(env, agent, config_name):
    """Run a short demo with the given configuration."""
    print(f"\n▶️ Running short demo for {config_name}...")
    
    try:
        results = train_mo_sac(
            env=env,
            agent=agent,
            total_timesteps=2000,  # Short demo
            learning_starts=500,
            train_freq=1,
            eval_freq=1000,
            eval_episodes=3,
            save_freq=10000,  # Don't save during demo
            verbose=False
        )
        
        if results['episode_rewards']:
            final_reward = results['episode_rewards'][-1]
            scalarized = np.sum(final_reward * agent.weights)
            print(f"✅ Demo completed! Final reward: {final_reward}, Scalarized: {scalarized:.3f}")
        else:
            print("⚠️ Demo completed but no episodes finished")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Demonstrate different optimization configurations."""
    print("="*70)
    print("Multi-Objective SAC Optimization Configurations Demo")
    print("="*70)
    
    configurations = [
        ("Baseline", baseline_configuration),
        ("Conservative", conservative_optimization),
        ("Moderate", moderate_optimization),
        ("Aggressive", aggressive_optimization),
        ("Custom Tuned", custom_tuned_optimization),
    ]
    
    for config_name, config_func in configurations:
        print(f"\n{'='*70}")
        
        env, agent = config_func()
        
        print(f"\nConfiguration Summary for {config_name}:")
        print(f"  Weights: {agent.weights}")
        print(f"  LR Annealing: {agent.use_lr_annealing}")
        if agent.use_lr_annealing:
            print(f"    Type: {agent.lr_annealing_type}")
            print(f"    Min factor: {agent.lr_min_factor}")
        print(f"  Reward Scaling: {agent.use_reward_scaling}")
        print(f"  Orthogonal Init: {agent.use_orthogonal_init}")
        if agent.use_orthogonal_init:
            print(f"    Actor gain: {agent.actor_orthogonal_gain}")
            print(f"    Critic gain: {agent.critic_orthogonal_gain}")
        print(f"  Value Clipping: {agent.use_value_clipping}")
        if agent.use_value_clipping:
            print(f"    Clip range: {agent.value_clip_range}")
        
        # Run short demo
        run_short_demo(env, agent, config_name)
    
    print(f"\n{'='*70}")
    print("🎯 RECOMMENDATIONS:")
    print("="*70)
    print("1. Start with CONSERVATIVE for stable training")
    print("2. Use MODERATE for general good performance")
    print("3. Try AGGRESSIVE if you need faster convergence")
    print("4. Use CUSTOM TUNED as a starting point for fine-tuning")
    print("5. Always test on a small scale before full training")
    print("")
    print("🔧 TUNING TIPS:")
    print("- Enable reward scaling if rewards have very different scales")
    print("- Use orthogonal init for better initial exploration")
    print("- Enable LR annealing for fine-tuning in later training")
    print("- Use value clipping if training becomes unstable")
    print("- Adjust weights based on your objective priorities")
    print("="*70)


if __name__ == "__main__":
    main()
