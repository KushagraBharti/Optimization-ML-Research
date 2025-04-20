# src/train/train_rl.py

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.envs.drone_cover_env import DroneCoverEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="data/raw/train.jsonl")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cov-reward", type=float, default=10.0)
    parser.add_argument("--invalid-penalty", type=float, default=-100.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/rl")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    # create checkpoint dir
    ckpt_path = Path(args.checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # instantiate env
    env = DroneCoverEnv(
        jsonl_path=args.jsonl,
        cov_reward=args.cov_reward,
        invalid_penalty=args.invalid_penalty,
    )

    # wrap callback to save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_drone",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # initialize PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        verbose=1,
        tensorboard_log="logs/ppo_drone/",
        device=args.device,
    )

    # train
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    # final save
    model.save(str(ckpt_path / "ppo_drone_final"))

if __name__ == "__main__":
    main()
