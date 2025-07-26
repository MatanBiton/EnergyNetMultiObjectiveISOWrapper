#!/usr/bin/env python3
import argparse
import gymnasium as gym
from multi_objecctive_iso_algo.multi_objective_sac import MOSAC
from EnergyNetMoISO.MoISOEnv import MultiObjectiveISOEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a Multi-Objective SAC (MOSAC) agent")
    # environment
    parser.add_argument("--iso_dispatch", action="store_true",
                        help="Enable iso dispatch action")
    # TODO: add PCS policy options
    # MOSAC hyperparameters
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256],
                        help="Sizes of hidden layers in actor and critic networks")
    parser.add_argument("--actor-lr", type=float, default=3e-4,
                        help="Learning rate for the actor network")
    parser.add_argument("--critic-lr", type=float, default=3e-4,
                        help="Learning rate for the critic networks")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Entropy coefficient for SAC")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient for target networks")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for updates")
    parser.add_argument("--buffer-capacity", type=int, default=100_000,
                        help="Replay buffer capacity")
    parser.add_argument("--max-steps-per-episode", type=int, default=500,
                        help="Maximum environment steps per episode")
    parser.add_argument("--writer-filename", type=str, default="mosac_runs",
                        help="Directory name for TensorBoard logs")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable raw action logging")
    # training control
    parser.add_argument("--num-episodes", type=int, default=1000,
                        help="Number of episodes to train")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of episodes to run evaluation after training")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create environment
    env = MultiObjectiveISOEnv(use_dispatch_action=args.iso_dispatch)

    # Instantiate agent
    agent = MOSAC(
        env=env,
        objectives=2,
        hidden_sizes=args.hidden_sizes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha=args.alpha,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        max_steps_per_episode=args.max_steps_per_episode,
        writer_filename=args.writer_filename,
        verbose=args.verbose
    )

    # Train
    print(f"Starting training for {args.num_episodes} episodes...")
    agent.train(args.num_episodes)
    print("Training completed.")

    # Evaluate
    print(f"Evaluating over {args.eval_episodes} episodes...")
    eval_rewards = agent.evaluate(episodes=args.eval_episodes)
    print("Average evaluation rewards per objective:", eval_rewards)

if __name__ == "__main__":
    main()
