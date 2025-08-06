#!/usr/bin/env python3
"""
Test script to verify that both standard MOSAC logging (loss curves, alpha) 
and enhanced EnergyNet logging are working correctly.
"""

import sys
import os
import numpy as np
import tempfile
import shutil
from datetime import datetime

# Add the current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from multi_objecctive_iso_algo.multi_objective_sac import MultiObjectiveSAC, train_mo_sac

def test_complete_logging():
    """Test both standard and enhanced logging functionality."""
    
    print("🧪 Testing Complete MOSAC Logging (Standard + Enhanced EnergyNet)")
    print("=" * 70)
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/complete_logging_test_{timestamp}"
    
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    
    # Create EnergyNet environment
    print("🏭 Creating EnergyNet environment...")
    env = MultiObjectiveISOEnv(use_dispatch_action=True)
    print(f"  ✓ Environment created")
    
    # Create MOSAC agent with enhanced logging
    print("🤖 Creating MOSAC agent...")
    agent = MultiObjectiveSAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        reward_dim=env.reward_dim,
        
        # Logging configuration
        tensorboard_log=log_dir,
        verbose=True,
        
        # Small networks for testing
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        buffer_capacity=5000,
        batch_size=32,
        
        # Multi-objective weights
        weights=np.array([0.6, 0.4])
    )
    
    print(f"  ✓ Agent created with TensorBoard logging enabled")
    
    # Run training with frequent logging
    print("🏋️ Running short training to test all logging...")
    training_results = train_mo_sac(
        env=env,
        agent=agent,
        total_timesteps=500,   # Short test
        learning_starts=100,   # Start learning early
        train_freq=1,          # Update every step
        eval_freq=250,         # Mid-training evaluation
        eval_episodes=3,       # Quick evaluation
        save_freq=1000,        # Won't trigger in short test
        verbose=True
    )
    
    print(f"✓ Training completed")
    
    # Check what was logged
    print("\n📋 Checking logged data...")
    
    try:
        # Try to read the tensorboard event file to verify logging
        import glob
        event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        
        if event_files:
            print(f"  ✓ TensorBoard event file created: {os.path.basename(event_files[0])}")
            
            # Try to read the events to check what's logged
            try:
                from tensorflow.python.summary.summary_iterator import summary_iterator
                
                logged_scalars = set()
                step_count = 0
                
                for event in summary_iterator(event_files[0]):
                    for value in event.summary.value:
                        logged_scalars.add(value.tag)
                        step_count += 1
                        if step_count > 100:  # Don't read everything, just sample
                            break
                    if step_count > 100:
                        break
                
                print(f"  ✓ Found {len(logged_scalars)} different logged metrics")
                
                # Check for standard MOSAC metrics
                standard_metrics = ['Loss/Critic', 'Loss/Actor', 'Loss/Alpha', 'Training/Alpha']
                found_standard = [m for m in standard_metrics if m in logged_scalars]
                print(f"  ✓ Standard MOSAC metrics found: {found_standard}")
                
                # Check for episode metrics
                episode_metrics = [m for m in logged_scalars if m.startswith('Episode/')]
                print(f"  ✓ Episode metrics found: {len(episode_metrics)} (e.g., {episode_metrics[:3] if episode_metrics else 'None'})")
                
                # Check for enhanced EnergyNet metrics
                step_metrics = [m for m in logged_scalars if m.startswith('EnergyNet_Step/')]
                episode_en_metrics = [m for m in logged_scalars if m.startswith('EnergyNet_Episode/')]
                print(f"  ✓ EnergyNet step metrics found: {len(step_metrics)}")
                print(f"  ✓ EnergyNet episode metrics found: {len(episode_en_metrics)}")
                
                if len(found_standard) >= 3:
                    print("  🎉 Standard MOSAC logging is working!")
                else:
                    print("  ⚠️ Some standard MOSAC metrics are missing")
                
                if len(step_metrics) > 10 and len(episode_en_metrics) > 50:
                    print("  🎉 Enhanced EnergyNet logging is working!")
                else:
                    print("  ⚠️ Enhanced EnergyNet logging may be incomplete")
                
            except ImportError:
                print("  ⚠️ Cannot read TensorBoard events (tensorflow not available), but file exists")
            except Exception as e:
                print(f"  ⚠️ Error reading TensorBoard events: {e}")
        
        else:
            print("  ❌ No TensorBoard event files found")
            
    except Exception as e:
        print(f"  ❌ Error checking logs: {e}")
    
    # Give instructions for viewing
    print(f"\n📊 To view all logged metrics:")
    print(f"   tensorboard --logdir {log_dir}")
    print()
    print("Look for the following metric categories:")
    print("  • Loss/* - Training loss curves (Critic, Actor, Alpha)")
    print("  • Training/Alpha - Alpha parameter evolution")
    print("  • Episode/* - Episode-level rewards and metrics")
    print("  • EnergyNet_Step/* - Step-level EnergyNet info dict entries")
    print("  • EnergyNet_Episode/* - Episode-level EnergyNet statistics")
    print("  • EnergyNet_Eval/* - Evaluation EnergyNet metrics")
    
    # Clean up
    try:
        shutil.rmtree(log_dir)
        print(f"\n🧹 Cleaned up test logs")
    except:
        print(f"\n⚠️ Could not clean up {log_dir}")
    
    return True

if __name__ == "__main__":
    # Ensure proper environment
    print("🔧 Environment check...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch not available")
        exit(1)
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("  ✓ TensorBoard available")
    except ImportError:
        print("  ❌ TensorBoard not available")
        exit(1)
    
    print()
    test_complete_logging()
