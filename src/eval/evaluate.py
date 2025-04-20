# src/eval/evaluate.py

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.solvers.greedy import cover_min_tours, tour_length
from src.solvers.dp     import cover_min_length
from src.envs.drone_cover_env import DroneCoverEnv

def eval_classical(instances, h):
    stats = {"greedy": [], "dp": []}
    for rec in instances:
        segs = [tuple(s) for s in rec["segments"]]
        L = float(rec["L"])
        g = cover_min_tours(segs, L, h)
        stats["greedy"].append((len(g), sum(tour_length(p,q,h) for p,q in g)))
        d = cover_min_length(segs, L, h)
        stats["dp"].append((len(d), sum(tour_length(p,q,h) for p,q in d)))
    return stats

def eval_rl(instances, model_path, h, cov_reward, invalid_penalty):
    base = DroneCoverEnv(jsonl_path=None, h=h,
                         cov_reward=cov_reward, invalid_penalty=invalid_penalty)
    base.instances = instances
    vec = DummyVecEnv([lambda: base])

    model = PPO.load(model_path)
    model.set_env(vec)

    stats = []
    for rec in instances:
        base.segments   = [tuple(s) for s in rec["segments"]]
        base.L0         = float(rec["L"])
        obs             = vec.reset()
        done = False; steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec.step(action)
            done = dones[0]
            steps += 1
        total_dist = base.L0 - base.remaining_L
        success    = all(base.seg_covered)
        stats.append((steps, total_dist, success))
    return stats

def print_summary(cls, rl):
    print("\nClassical:")
    for name, runs in cls.items():
        tours = [t for t,_ in runs]; dists = [d for _,d in runs]
        print(f" {name:>6} | avg tours={mean(tours):.2f}, avg dist={mean(dists):.2f}")
    print("\nRL policy:")
    steps = [s for s,_,_ in rl]; dists = [d for _,d,_ in rl]; succ=[ok for _,_,ok in rl]
    print(f" avg steps   = {mean(steps):.2f}")
    print(f" avg dist    = {mean(dists):.2f}")
    print(f" success rate= {100*mean(succ):.1f}%\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-jsonl",    type=str, default="data/raw/train.jsonl")
    p.add_argument("--n-test",        type=int,   default=200)
    p.add_argument("--rl-model",      type=str,   default="checkpoints/rl/ppo_drone_final.zip")
    p.add_argument("--h",             type=float, default=1.0)
    p.add_argument("--cov-reward",    type=float, default=10.0)
    p.add_argument("--invalid-penalty",type=float, default=-100.0)
    args = p.parse_args()

    lines = Path(args.test_jsonl).read_text().splitlines()
    insts = [json.loads(l) for l in lines[-args.n_test:]]
    cls_stats = eval_classical(insts, args.h)
    rl_stats  = eval_rl(insts, args.rl_model, args.h, args.cov_reward, args.invalid_penalty)
    print_summary(cls_stats, rl_stats)

if __name__ == "__main__":
    main()
