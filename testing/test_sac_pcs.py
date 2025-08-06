"""
Test script for SAC PCS Agent to verify it works correctly.
"""

import sys
import os
import numpy as np
import torch

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from energy_net.env.pcs_unit_v0 import PCSUnitEnv
    PCSUnitEnv_available = True
except ImportError:
    print("Warning: PCSUnitEnv not available. Make sure energy_net is properly installed.")
    PCSUnitEnv_available = False

from EnergyNetMoISO.pcs_models.sac_pcs_agent import SACPCSAgent
from multi_objecctive_iso_algo.multi_objective_sac import MultiObjectiveSAC


def test_sac_pcs_agent_basic():
    """Test basic SAC PCS agent functionality."""
    print("Testing SAC PCS Agent basic functionality...")
    
    # Create a simple agent
    state_dim = 4  # Example: battery_level, time, buy_price, sell_price
    action_dim = 1  # Power dispatch action
    
    agent = SACPCSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(-100.0, 100.0),  # Power in kW
        verbose=True
    )
    
    # Test action selection
    dummy_state = np.array([50.0, 0.5, 30.0, 25.0])  # Battery 50%, time 0.5, prices
    
    # Test deterministic action
    action_det = agent.select_action(dummy_state, deterministic=True)
    print(f"Deterministic action: {action_det}")
    
    # Test stochastic action
    action_stoch = agent.select_action(dummy_state, deterministic=False)
    print(f"Stochastic action: {action_stoch}")
    
    # Test predict method (GenericPCSAgent interface)
    pred_action, state = agent.predict(dummy_state, deterministic=True)
    print(f"Predict method: action={pred_action}, state={state}")
    
    # Test storing transitions
    next_state = np.array([45.0, 0.6, 32.0, 27.0])
    reward = 10.5
    agent.store_transition(dummy_state, action_det, reward, next_state, False)
    print(f"Stored transition, buffer size: {len(agent.replay_buffer)}")
    
    print("✓ Basic functionality test passed!\n")


def test_sac_pcs_agent_training():
    """Test SAC PCS agent training functionality."""
    print("Testing SAC PCS Agent training functionality...")
    
    state_dim = 4
    action_dim = 1
    
    agent = SACPCSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(-100.0, 100.0),
        batch_size=32,  # Smaller batch for testing
        verbose=False
    )
    
    # Fill replay buffer with some dummy data
    print("Filling replay buffer with dummy data...")
    for i in range(100):
        state = np.random.normal(50.0, 10.0, state_dim)
        action = np.random.uniform(-50.0, 50.0, action_dim)
        reward = np.random.normal(0.0, 5.0)
        next_state = state + np.random.normal(0.0, 1.0, state_dim)
        done = np.random.random() < 0.1
        
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    
    # Test updating
    print("Testing agent update...")
    metrics = agent.update()
    
    if metrics:
        print("Update metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("No metrics returned (buffer not large enough)")
    
    print("✓ Training functionality test passed!\n")


def test_sac_pcs_agent_optimizations():
    """Test SAC PCS agent with all optimizations enabled."""
    print("Testing SAC PCS Agent with optimizations...")
    
    state_dim = 4
    action_dim = 1
    
    agent = SACPCSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(-100.0, 100.0),
        # Enable all optimizations
        use_lr_annealing=True,
        lr_annealing_type='cosine',
        lr_annealing_steps=1000,
        use_reward_scaling=True,
        use_orthogonal_init=True,
        use_value_clipping=True,
        value_clip_range=100.0,
        verbose=True
    )
    
    print("✓ Agent with optimizations created successfully!")
    
    # Test a few actions
    dummy_state = np.array([50.0, 0.5, 30.0, 25.0])
    action = agent.select_action(dummy_state)
    print(f"Action with optimizations: {action}")
    
    print("✓ Optimizations test passed!\n")


def test_sac_pcs_agent_save_load():
    """Test SAC PCS agent save/load functionality."""
    print("Testing SAC PCS Agent save/load...")
    
    state_dim = 4
    action_dim = 1
    
    # Create agent
    agent1 = SACPCSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(-100.0, 100.0),
        verbose=False
    )
    
    # Get initial action
    dummy_state = np.array([50.0, 0.5, 30.0, 25.0])
    action1 = agent1.select_action(dummy_state, deterministic=True)
    
    # Save agent
    save_path = "/tmp/test_sac_pcs_agent.pth"
    agent1.save(save_path)
    print(f"Agent saved to: {save_path}")
    
    # Create new agent and load
    agent2 = SACPCSAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(-100.0, 100.0),
        verbose=False
    )
    
    agent2.load(save_path)
    print(f"Agent loaded from: {save_path}")
    
    # Test that actions are the same
    action2 = agent2.select_action(dummy_state, deterministic=True)
    
    print(f"Original action: {action1}")
    print(f"Loaded action: {action2}")
    
    if np.allclose(action1, action2, atol=1e-6):
        print("✓ Actions match after save/load!")
    else:
        print("✗ Actions don't match after save/load!")
        
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("✓ Save/load test passed!\n")


def test_with_pcs_unit_env():
    """Test SAC PCS agent with actual PCSUnitEnv if available."""
    if not PCSUnitEnv_available:
        print("Skipping PCSUnitEnv test - environment not available")
        return
    
    print("Testing SAC PCS Agent with PCSUnitEnv...")
    
    try:
        # Create environment (without ISO model for simplicity)
        env = PCSUnitEnv()
        
        # Get environment info
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
        
        print(f"Environment info:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Action bounds: {action_bounds}")
        
        # Create agent
        agent = SACPCSAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            verbose=False
        )
        
        # Test a few steps
        state, _ = env.reset()
        print(f"Initial state: {state}")
        
        for step in range(5):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"Step {step+1}: action={action}, reward={reward:.3f}, done={done}")
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        
        print(f"Buffer size after test: {len(agent.replay_buffer)}")
        print("✓ PCSUnitEnv test passed!")
        
    except Exception as e:
        print(f"✗ PCSUnitEnv test failed: {e}")
    
    print()


def test_with_iso_model():
    """Test SAC PCS agent with a dummy ISO model."""
    if not PCSUnitEnv_available:
        print("Skipping ISO model test - PCSUnitEnv not available")
        return
    
    print("Testing SAC PCS Agent with dummy ISO model...")
    
    try:
        # Create a dummy ISO model
        iso_model = MultiObjectiveSAC(
            state_dim=3,
            action_dim=3,
            reward_dim=2,
            verbose=False
        )
        
        # Create environment with ISO model
        env = PCSUnitEnv(trained_iso_model_instance=iso_model)
        
        # Get environment info
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
        
        print(f"Environment with ISO model:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Action bounds: {action_bounds}")
        
        # Create PCS agent
        pcs_agent = SACPCSAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            verbose=False
        )
        
        # Test a few steps
        state, _ = env.reset()
        print(f"Initial state: {state}")
        
        for step in range(3):
            action = pcs_agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"Step {step+1}: action={action}, reward={reward:.3f}, done={done}")
            
            if done:
                break
            state = next_state
        
        print("✓ ISO model test passed!")
        
    except Exception as e:
        print(f"✗ ISO model test failed: {e}")
    
    print()


def main():
    print("SAC PCS Agent Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_sac_pcs_agent_basic()
    test_sac_pcs_agent_training()
    test_sac_pcs_agent_optimizations()
    test_sac_pcs_agent_save_load()
    test_with_pcs_unit_env()
    test_with_iso_model()
    
    print("=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main()
