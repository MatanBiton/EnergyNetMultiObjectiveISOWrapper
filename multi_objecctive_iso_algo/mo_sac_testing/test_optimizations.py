"""
Test script for Multi-Objective SAC optimizations.
Tests that the new optimization features work correctly and don't break existing functionality.
"""

import sys
import os
import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Any

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from multi_objective_sac import MultiObjectiveSAC, train_mo_sac, evaluate_mo_sac
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from EnergyNetMoISO.pcs_models.constant_pcs_agent import ConstantPCSAgent


def create_test_env():
    """Create a simple test environment."""
    return MultiObjectiveISOEnv(
        use_dispatch_action=False,
        dispatch_strategy="PROPORTIONAL",
        trained_pcs_model=ConstantPCSAgent(1)
    )


def test_basic_functionality():
    """Test basic SAC functionality without optimizations."""
    print("Testing basic SAC functionality...")
    
    env = create_test_env()
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
        verbose=False
    )
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state)
    assert action.shape == (action_dim,), f"Action shape mismatch: {action.shape} vs {(action_dim,)}"
    print("✓ Action selection works")
    
    # Test storing transitions
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, terminated or truncated)
    print("✓ Transition storage works")
    
    print("✓ Basic functionality test passed!\n")


def test_orthogonal_initialization():
    """Test orthogonal initialization."""
    print("Testing orthogonal initialization...")
    
    env = create_test_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    # Test with orthogonal initialization
    agent_ortho = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_orthogonal_init=True,
        actor_orthogonal_gain=0.01,
        critic_orthogonal_gain=1.0,
        verbose=False
    )
    
    # Test without orthogonal initialization
    agent_xavier = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_orthogonal_init=False,
        verbose=False
    )
    
    # Check that weights are different (they should be initialized differently)
    ortho_weights = agent_ortho.actor.mean_head.weight.data.clone()
    xavier_weights = agent_xavier.actor.mean_head.weight.data.clone()
    
    weight_diff = torch.norm(ortho_weights - xavier_weights).item()
    assert weight_diff > 0.1, f"Initialization methods should produce different weights, diff: {weight_diff}"
    
    print("✓ Orthogonal initialization test passed!\n")


def test_reward_scaling():
    """Test reward scaling functionality."""
    print("Testing reward scaling...")
    
    env = create_test_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_reward_scaling=True,
        reward_scale_epsilon=1e-4,
        verbose=False
    )
    
    # Generate some dummy rewards with different scales
    rewards = [
        np.array([100.0, -50.0]),
        np.array([150.0, -75.0]),
        np.array([200.0, -100.0]),
        np.array([80.0, -40.0]),
        np.array([120.0, -60.0])
    ]
    
    # Store transitions to build up reward statistics
    state, _ = env.reset()
    for reward in rewards:
        action = agent.select_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, terminated or truncated)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
    
    # Check that reward statistics have been updated
    assert agent.reward_rms.count > 0, "Reward statistics should be updated"
    assert np.all(agent.reward_rms.var > 0), "Reward variance should be positive"
    
    print("✓ Reward scaling test passed!\n")


def test_value_clipping():
    """Test value clipping functionality."""
    print("Testing value clipping...")
    
    env = create_test_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    agent = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_value_clipping=True,
        value_clip_range=10.0,  # Small range for testing
        verbose=False
    )
    
    # Check that the value clipping parameters are set
    assert agent.use_value_clipping == True
    assert agent.value_clip_range == 10.0
    
    print("✓ Value clipping test passed!\n")


def test_lr_annealing():
    """Test learning rate annealing."""
    print("Testing learning rate annealing...")
    
    env = create_test_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    # Test cosine annealing
    agent_cosine = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_lr_annealing=True,
        lr_annealing_type='cosine',
        lr_annealing_steps=100,
        lr_min_factor=0.1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        verbose=False
    )
    
    # Test linear annealing
    agent_linear = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_lr_annealing=True,
        lr_annealing_type='linear',
        lr_annealing_steps=100,
        lr_min_factor=0.1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        verbose=False
    )
    
    # Test exponential annealing
    agent_exp = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_lr_annealing=True,
        lr_annealing_type='exponential',
        lr_decay_rate=0.95,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        verbose=False
    )
    
    # Check that schedulers are created
    assert agent_cosine.actor_scheduler is not None
    assert agent_linear.actor_scheduler is not None
    assert agent_exp.actor_scheduler is not None
    
    # Test scheduler step
    initial_lr = agent_cosine.actor_optimizer.param_groups[0]['lr']
    agent_cosine._update_schedulers()
    after_lr = agent_cosine.actor_optimizer.param_groups[0]['lr']
    
    # For cosine annealing, LR should change
    assert initial_lr != after_lr, f"LR should change after scheduler step: {initial_lr} -> {after_lr}"
    
    print("✓ Learning rate annealing test passed!\n")


def test_short_training():
    """Test short training run with all optimizations enabled."""
    print("Testing short training run with optimizations...")
    
    env = create_test_env()
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
        use_lr_annealing=True,
        lr_annealing_type='cosine',
        lr_annealing_steps=50,
        use_reward_scaling=True,
        use_orthogonal_init=True,
        use_value_clipping=True,
        value_clip_range=100.0,
        verbose=False
    )
    
    # Run short training
    try:
        results = train_mo_sac(
            env=env,
            agent=agent,
            total_timesteps=1000,  # Short training
            learning_starts=100,
            train_freq=1,
            eval_freq=500,
            eval_episodes=2,
            save_freq=10000,  # Don't save during test
            verbose=False
        )
        
        assert len(results['episode_rewards']) > 0, "Should have completed at least one episode"
        assert len(results['episode_lengths']) > 0, "Should have episode length data"
        
        print("✓ Short training with optimizations completed successfully!")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise
    
    print("✓ Short training test passed!\n")


def test_save_load_with_optimizations():
    """Test saving and loading with optimization features."""
    print("Testing save/load with optimizations...")
    
    env = create_test_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
    
    # Create agent with optimizations
    agent1 = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_lr_annealing=True,
        lr_annealing_type='cosine',
        lr_annealing_steps=100,
        use_reward_scaling=True,
        use_orthogonal_init=True,
        use_value_clipping=True,
        verbose=False
    )
    
    # Store some transitions to build reward statistics
    state, _ = env.reset()
    for _ in range(10):
        action = agent1.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent1.store_transition(state, action, reward, next_state, terminated or truncated)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
    
    # Update a few times to set training step
    for _ in range(5):
        if len(agent1.replay_buffer) >= agent1.batch_size:
            agent1.update()
    
    # Save model
    temp_path = "temp_test_model.pth"
    agent1.save(temp_path)
    
    # Create new agent and load
    agent2 = MultiObjectiveSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_bounds=action_bounds,
        use_lr_annealing=True,
        lr_annealing_type='cosine',
        lr_annealing_steps=100,
        use_reward_scaling=True,
        use_orthogonal_init=True,
        use_value_clipping=True,
        verbose=False
    )
    
    agent2.load(temp_path)
    
    # Check that key parameters match
    assert agent2.training_step == agent1.training_step
    assert np.allclose(agent2.weights, agent1.weights)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print("✓ Save/load with optimizations test passed!\n")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Multi-Objective SAC Optimizations")
    print("="*60)
    
    try:
        test_basic_functionality()
        test_orthogonal_initialization()
        test_reward_scaling()
        test_value_clipping()
        test_lr_annealing()
        test_short_training()
        test_save_load_with_optimizations()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("The optimization features are working correctly.")
        print("="*60)
        
    except Exception as e:
        print("="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    main()
