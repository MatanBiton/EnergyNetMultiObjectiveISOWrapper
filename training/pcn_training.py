#!/usr/bin/env python3
import argparse
import numpy as np
import gymnasium as gym
from multi_objecctive_iso_algo.pcn import PCNAgent
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Predicted Control Network (PCN) agent"
    )
    # Environment options (mirror MOSAC script)
    parser.add_argument(
        "--iso_dispatch", action="store_true",
        help="Enable ISO dispatch action"
    )
    # PCN hyperparameters
    parser.add_argument(
        "--scaling-factor", type=float, nargs='+', default=[1,1],
        help="Scaling factor for each objective (list of floats)"
    )
    parser.add_argument(
        "--ref-point", type=float, nargs='+', default=[0,0],
        help="Reference point for hypervolume (list of floats)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate for the PCN optimizer"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="Discount factor for return-to-go accumulation"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for model updates"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64,
        help="Hidden layer size for the PCN network embeddings"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1,
        help="Gaussian noise scale for action sampling"
    )
    parser.add_argument(
        "--log-dir", type=str, default="runs/PCN",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: 'cpu' or 'cuda'"
    )
    # Training control
    parser.add_argument(
        "--total-timesteps", type=int, default=100000,
        help="Total number of environment steps to train for"
    )
    parser.add_argument(
        "--max-buffer-size", type=int, default=100,
        help="Maximum number of episodes to buffer"
    )
    parser.add_argument(
        "--num-er-episodes", type=int, default=20,
        help="Number of episodes for initial experience replay filling"
    )
    parser.add_argument(
        "--num-step-episodes", type=int, default=10,
        help="Number of new episodes per training loop"
    )
    parser.add_argument(
        "--num-model-updates", type=int, default=50,
        help="Model update steps per training loop"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Number of episodes to evaluate after training"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--load-path", type=str, default=None,
        help="Path to load a saved PCN checkpoint"
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="Path to save the trained PCN checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create training and (optional) separate eval environments
    env = MultiObjectiveISOEnv(use_dispatch_action=args.iso_dispatch)
    eval_env = env  # using same env for evaluation

    # Prepare PCN parameters
    scaling = np.array(args.scaling_factor, dtype=np.float32)
    ref_point = np.array(args.ref_point, dtype=np.float32)

    # Instantiate PCN agent
    agent = PCNAgent(
        env=env,
        scaling_factor=scaling,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        noise=args.noise,
        log_dir=args.log_dir,
        seed=args.seed,
        device=args.device
    )

    # Optionally load existing checkpoint
    if args.load_path:
        print(f"Loading checkpoint from {args.load_path}...")
        agent.load(args.load_path)

    # Training
    print(f"Starting PCN training for {args.total_timesteps} timesteps...")
    agent.train(
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        max_buffer_size=args.max_buffer_size,
        num_er_episodes=args.num_er_episodes,
        num_step_episodes=args.num_step_episodes,
        num_model_updates=args.num_model_updates
    )
    print("Training completed.")

    # Optionally save checkpoint
    if args.save_path:
        print(f"Saving checkpoint to {args.save_path}...")
        agent.save(args.save_path)

    # Evaluation
    print(f"Evaluating over {args.eval_episodes} episodes...")
    eval_returns = agent.evaluate(
        num_episodes=args.eval_episodes,
        eval_env=eval_env,
        render=args.render
    )
    print("Average evaluation returns:", eval_returns)


if __name__ == "__main__":
    main()
