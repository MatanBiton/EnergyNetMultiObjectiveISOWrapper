#!/usr/bin/env python3
"""
Example script demonstrating enhanced EnergyNet info dict logging with MOSAC.

This script shows how to set up and run MOSAC training with comprehensive
EnergyNet environment logging enabled.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add the current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv
from multi_objecctive_iso_algo.multi_objective_sac import MultiObjectiveSAC, train_mo_sac

def main():
    """Main training function with enhanced EnergyNet logging."""
    
    print("🚀 Starting MOSAC Training with Enhanced EnergyNet Logging")
    print("=" * 60)
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/energynet_mosac_{timestamp}"
    
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    
    # Create EnergyNet environment
    print("🏭 Creating EnergyNet environment...")
    env = MultiObjectiveISOEnv(use_dispatch_action=True)
    
    print(f"  ✓ Environment created")
    print(f"  - State dimension: {env.observation_space.shape[0]}")
    print(f"  - Action dimension: {env.action_space.shape[0]}")
    print(f"  - Reward dimension: {env.reward_dim}")
    
    # Create MOSAC agent with enhanced logging
    print("🤖 Creating MOSAC agent...")
    agent = MultiObjectiveSAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        reward_dim=env.reward_dim,
        
        # Enhanced logging configuration
        tensorboard_log=log_dir,
        verbose=True,
        
        # Training parameters (adjust as needed)
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=1000000,
        batch_size=256,
        
        # Network architecture
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        
        # Optional optimizations
        use_lr_annealing=False,
        use_reward_scaling=False,
        use_orthogonal_init=False,
        use_value_clipping=False,
        
        # Multi-objective weights (equal weighting)
        weights=np.array([0.5, 0.5])  # [cost_weight, stability_weight]
    )
    
    print(f"  ✓ Agent created with enhanced logging enabled")
    
    # Training configuration
    training_config = {
        'total_timesteps': 100000,     # Adjust for your needs
        'learning_starts': 10000,
        'train_freq': 1,
        'eval_freq': 10000,
        'eval_episodes': 10,
        'save_freq': 25000,
        'save_path': f"./models/energynet_mosac_{timestamp}",
        'verbose': True
    }
    
    print("🏋️ Training configuration:")
    for key, value in training_config.items():
        print(f"  - {key}: {value}")
    
    print("\n📈 Enhanced logging will capture:")
    print("  Step-level (EnergyNet_Step/):")
    print("    • All 20 info dict entries per step")
    print("    • Scalar values: demand, costs, prices, etc.")
    print("    • List values: battery_level, actions (last element)")
    print()
    print("  Episode-level (EnergyNet_Episode/):")
    print("    • Statistical aggregations: mean, std, sum, min, max, final")
    print("    • Complete episode summaries")
    print()
    print("  Evaluation (EnergyNet_Eval/):")
    print("    • Evaluation episode statistics")
    print("    • Performance monitoring during eval")
    
    # Create models directory
    os.makedirs("./models", exist_ok=True)
    
    print(f"\n🔄 Starting training...")
    print("=" * 60)
    
    # Run training with enhanced logging
    try:
        training_results = train_mo_sac(
            env=env,
            agent=agent,
            **training_config
        )
        
        print("\n✅ Training completed successfully!")
        print(f"  - Total episodes: {len(training_results['episode_rewards'])}")
        print(f"  - Average episode length: {np.mean(training_results['episode_lengths']):.1f}")
        
        # Save final model
        final_model_path = f"./models/energynet_mosac_{timestamp}_final.pth"
        agent.save(final_model_path)
        print(f"  - Final model saved: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        interrupt_model_path = f"./models/energynet_mosac_{timestamp}_interrupted.pth"
        agent.save(interrupt_model_path)
        print(f"  - Model saved: {interrupt_model_path}")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        return False
    
    print(f"\n📊 View training logs with:")
    print(f"   tensorboard --logdir {log_dir}")
    print("\n🎉 Enhanced logging captured all EnergyNet environment data!")
    
    return True

if __name__ == "__main__":
    # Ensure we're in the IsoMOEnergyNet environment
    import subprocess
    result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
    if 'IsoMOEnergyNet' not in result.stdout:
        print("⚠️  Warning: Make sure you're in the IsoMOEnergyNet conda environment")
        print("   Run: conda activate IsoMOEnergyNet")
        print()
    
    main()
