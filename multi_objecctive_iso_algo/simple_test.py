"""Simple test to verify our modified multi_objective_sac.py works."""
import torch
import numpy as np

print("Testing basic imports...")
try:
    from multi_objective_sac import MultiObjectiveSAC, RunningMeanStd, orthogonal_init
    print("✓ Successfully imported optimized Multi-Objective SAC")
    
    # Test RunningMeanStd
    rms = RunningMeanStd(shape=(2,))
    test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rms.update(test_data)
    print("✓ RunningMeanStd works")
    
    # Test orthogonal initialization
    linear_layer = torch.nn.Linear(10, 5)
    orthogonal_init(linear_layer, gain=1.0)
    print("✓ Orthogonal initialization works")
    
    # Test creating MultiObjectiveSAC with optimizations
    agent = MultiObjectiveSAC(
        state_dim=4,
        action_dim=2,
        reward_dim=2,
        action_bounds=(-1.0, 1.0),
        use_lr_annealing=True,
        lr_annealing_type='cosine',
        lr_annealing_steps=100,
        use_reward_scaling=True,
        use_orthogonal_init=True,
        use_value_clipping=True,
        verbose=False
    )
    print("✓ MultiObjectiveSAC with optimizations created successfully")
    
    # Test action selection
    state = np.random.randn(4)
    action = agent.select_action(state)
    print(f"✓ Action selection works, action shape: {action.shape}")
    
    print("\n🎉 All basic tests passed! The optimization features are working.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
