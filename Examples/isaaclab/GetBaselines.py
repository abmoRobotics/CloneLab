#!/usr/bin/env python3

"""
Baseline Evaluation Script for CloneLab

Evaluate pretrained baseline policies from RLRoverLab on multiple environments
to get consistent metrics with CloneLab IQL evaluations.

Uses existing RLRoverLab infrastructure for environment loading and agent creation.
"""

import argparse
import random
import sys
import os
import gymnasium as gym
from isaaclab.app import AppLauncher

# Available environments (from rover_envs __init__.py)
AVAILABLE_ENVS = [
    "AAURoverEnv-v0",
    "AAURoverEnvSimple-v0",
    "AAURoverEnvDict-v0",
    "AAURoverEnvCamera-v0",
    "AAURoverEnvCosmos-v0",
    "AAURoverEnvRGBDRaw-v0",
    "AAURoverEnvRGBDRawTemp-v0"
]

# Available baseline agents (from RLRoverLab)
BASELINE_AGENTS = [
    "PPO",
    "SAC",
    "TD3",
    "TRPO",
    "RPO"
]


def evaluate_baseline_agent(agent_name: str, env_name: str, num_steps: int = 100000, num_envs: int = 20):
    """Evaluate a pretrained baseline agent on an environment"""

    print(f"\n{'='*60}")
    print(f"Evaluating {agent_name} baseline on {env_name}")
    print(f"{'='*60}")

    # Parse environment configuration
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed
    import rover_envs
    import rover_envs.envs.navigation.robots
    from rover_envs.learning.agents import create_agent
    from rover_envs.utils.config import parse_skrl_cfg
    from rover_envs.utils.logging_utils import log_setup
    from CloneRL.utils.wrappers import SuccessTrackerWrapper

    # Set up environment
    env_cfg = parse_env_cfg(env_name, device="cuda:0", num_envs=num_envs)

    # Get agent configuration
    experiment_cfg_file = gym.spec(env_name).kwargs.get("skrl_cfgs")[
        agent_name.upper()]
    experiment_cfg = parse_skrl_cfg(experiment_cfg_file)

    # Create environment
    env = gym.make(env_name, cfg=env_cfg, render_mode=None)
    env = SuccessTrackerWrapper(env, success_buffer_size=1_000_000)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")


    # Create agent
    agent = create_agent(agent_name, env, experiment_cfg)

    # Load pretrained weights
    try:
        agent_policy_path = gym.spec(env_name).kwargs.get("best_model_path")
        if agent_policy_path and os.path.exists(agent_policy_path):
            agent.load(agent_policy_path)
            print(f"Loaded pretrained policy from: {agent_policy_path}")
        else:
            print(
                f"Warning: No pretrained policy found for {agent_name} on {env_name}")
            print(f"Expected path: {agent_policy_path}")
    except Exception as e:
        print(f"Error loading pretrained policy: {e}")
        return None

    # Set up trainer
    trainer_cfg = experiment_cfg["trainer"]
    trainer_cfg["timesteps"] = num_steps

    trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)

    print(f"Starting evaluation for {num_steps} steps...")

    # Run evaluation
    try:
        trainer.eval()

        # Get metrics (these should be logged by the agent during evaluation)
        print(f"Evaluation completed for {agent_name} on {env_name}")

        # Extract success rate from the environment's episode stats
        success_rate = 0.0
        num_episodes = 0
        if hasattr(env, "episode_stats") and "success" in env.episode_stats:
            success_deque = env.episode_stats["success"]
            num_episodes = len(success_deque)
            if num_episodes > 0:
                success_rate = sum(success_deque) / num_episodes
            print(f"Success rate: {success_rate:.2%}")
            print(f"Total episodes: {num_episodes}")

        return {
            'agent_name': agent_name,
            'env_name': env_name,
            'num_steps': num_steps,
            'status': 'completed',
            'success_rate': success_rate,
            'num_episodes': num_episodes
        }

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {
            'agent_name': agent_name,
            'env_name': env_name,
            'error': str(e),
            'status': 'failed'
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        "Evaluate pretrained baseline policies from RLRoverLab")
    parser.add_argument("--video", action="store_true",
                        default=False, help="Record videos during evaluation.")
    parser.add_argument("--video_length", type=int, default=200,
                        help="Length of recorded video (in steps).")
    parser.add_argument("--video_interval", type=int,
                        default=2000, help="Interval between video recordings.")
    parser.add_argument("--num_envs", type=int, default=120,
                        help="Number of environments to simulate.")
    parser.add_argument("--envs", nargs="+", choices=AVAILABLE_ENVS, default=["AAURoverEnv-v0"],
                        help="Environments to evaluate on")
    parser.add_argument("--agents", nargs="+", choices=BASELINE_AGENTS, default=["PPO"],
                        help="Baseline agents to evaluate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed used for the environment")
    parser.add_argument("--num_steps", type=int,
                        default=500, help="Number of evaluation steps")

    # Add AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch the simulation app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import after app launch (required for Isaac Lab)

    print("Starting baseline evaluation...")
    print(f"Environments: {args_cli.envs}")
    print(f"Agents: {args_cli.agents}")
    print(f"Evaluation steps: {args_cli.num_steps}")

    results = []

    # Evaluate each agent on each environment
    for env_name in args_cli.envs:
        for agent_name in args_cli.agents:
            result = evaluate_baseline_agent(
                agent_name=agent_name,
                env_name=env_name,
                num_steps=args_cli.num_steps,
                num_envs=args_cli.num_envs
            )
            if result:
                results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print("BASELINE EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Environment':<25} {'Agent':<10} {'Steps':<10} {'Episodes':<10} {'Status':<10} {'SR (%)':<10}")
    print(f"{'-'*80}")

    for result in results:
        status = result.get('status', 'unknown')
        success_rate = result.get('success_rate', 0) * 100
        num_episodes = result.get('num_episodes', 0)
        print(
            f"{result['env_name']:<25} {result['agent_name']:<10} {result.get('num_steps', 'N/A'):<10} {num_episodes:<10} {status:<10} {success_rate:<10.2f}")

    # Clozse simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
