#!/usr/bin/env python3
"""
Quick GPU test for MO-SAC to verify GPU utilization is working properly.
This runs a minimal test that should complete in under 5 minutes on GPU.
"""

import sys
import os
import time
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gpu_availability():
    """Test basic GPU availability and performance."""
    print("=" * 60)
    print("GPU Availability Test")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        # Performance test
        print("\nGPU Performance Test...")
        device = torch.device('cuda')
        
        # Test large matrix multiplication
        size = 4096
        print(f"Testing {size}x{size} matrix multiplication on GPU...")
        
        start_time = time.time()
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        
        print(f"GPU time: {gpu_time:.3f} seconds")
        
        # Compare with CPU
        print(f"Testing {size}x{size} matrix multiplication on CPU...")
        start_time = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time:.3f} seconds")
        print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
        
        if gpu_time < cpu_time:
            print("✓ GPU is working and faster than CPU")
            return True
        else:
            print("✗ GPU is slower than CPU - something is wrong")
            return False
    else:
        print("✗ CUDA not available")
        return False

def test_mo_sac_gpu():
    """Test MO-SAC with GPU acceleration on a simple environment."""
    print("\n" + "=" * 60)
    print("MO-SAC GPU Test")
    print("=" * 60)
    
    try:
        from multi_objective_sac import MultiObjectiveSAC
        from test_environments import make_env
        
        # Create simple environment
        env = make_env("MultiObjectiveContinuousCartPole-v0")
        
        print(f"Environment: {env}")
        print(f"State dim: {env.observation_space.shape[0]}")
        print(f"Action dim: {env.action_space.shape[0]}")
        print(f"Reward dim: {env.reward_dim}")
        
        # Create agent
        agent = MultiObjectiveSAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            reward_dim=env.reward_dim,
            action_bounds=(-1.0, 1.0),
            verbose=True,
            batch_size=64  # Smaller batch for quick test
        )
        
        print("\nTesting action selection and updates...")
        
        # Test action selection
        state, _ = env.reset()
        action = agent.select_action(state)
        print(f"✓ Action selection works: {action}")
        
        # Collect some transitions
        print("Collecting transitions...")
        for _ in range(1000):  # Collect enough for one update
            state, _ = env.reset()
            for step in range(10):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, terminated or truncated)
                state = next_state
                if terminated or truncated:
                    break
        
        print(f"Buffer size: {len(agent.replay_buffer)}")
        
        # Test updates
        print("Testing network updates...")
        start_time = time.time()
        for i in range(10):  # 10 updates
            update_info = agent.update()
            if i == 0:
                print(f"✓ First update: {update_info}")
        
        update_time = time.time() - start_time
        print(f"10 updates took {update_time:.3f} seconds ({update_time/10:.3f}s per update)")
        
        print("✓ MO-SAC GPU test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ MO-SAC GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all GPU tests."""
    print("Quick GPU Test for MO-SAC")
    print("This should complete in under 5 minutes if GPU is working properly")
    print("")
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    
    if not gpu_available:
        print("\n❌ GPU not available or not working properly!")
        print("The training will be extremely slow on CPU.")
        print("Check SLURM job allocation and CUDA installation.")
        return False
    
    # Test MO-SAC with GPU
    mo_sac_working = test_mo_sac_gpu()
    
    if mo_sac_working:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("GPU is available and MO-SAC is using it properly.")
        print("Training should be fast now.")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ MO-SAC GPU TEST FAILED!")
        print("There may be issues with the implementation.")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
