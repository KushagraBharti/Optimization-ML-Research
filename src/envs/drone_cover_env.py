# src/envs/drone_cover_env.py

import json
import random
import math
import gym
import numpy as np
from typing import List, Tuple, Optional

from src.solvers.greedy import tour_length

MAX_SEGMENTS  = 30
MAX_ENDPOINTS = 2 * MAX_SEGMENTS  # 60

class DroneCoverEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        jsonl_path: Optional[str] = "data/raw/train.jsonl",
        h: float = 1.0,
        cov_reward: float = 10.0,
        invalid_penalty: float = -100.0,
    ):
        super().__init__()
        self.h = h
        self.cov_reward = cov_reward
        self.invalid_penalty = invalid_penalty

        # load instances if provided
        if jsonl_path:
            with open(jsonl_path, "r") as fin:
                self.instances = [json.loads(l) for l in fin]
        else:
            self.instances = []

        # placeholders
        self.segments: List[Tuple[float, float]] = []
        self.L0 = 0.0
        self.remaining_L = 0.0
        self.endpoints: List[float] = []
        self.seg_covered: List[bool] = []

        # spaces
        self.action_space = gym.spaces.MultiDiscrete([MAX_ENDPOINTS, MAX_ENDPOINTS])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(2 * MAX_ENDPOINTS + 1,),
            dtype=np.float32,
        )

    def reset(self):
        assert self.instances, "No instances loaded!"
        rec = random.choice(self.instances)
        self.segments = [tuple(s) for s in rec["segments"]]
        self.L0 = float(rec["L"])
        self.remaining_L = self.L0

        # build endpoint list
        self.endpoints = []
        for a, b in self.segments:
            self.endpoints.extend([a, b])
        self.E = len(self.endpoints)
        assert self.E <= MAX_ENDPOINTS

        self.seg_covered = [False] * len(self.segments)
        return self._get_obs()

    def step(self, action):
        i, j = int(action[0]), int(action[1])
        if not (0 <= i <= j < self.E):
            return self._get_obs(), self.invalid_penalty, True, {}

        p, q = self.endpoints[i], self.endpoints[j]
        length = tour_length(p, q, self.h)
        if length > self.remaining_L:
            return self._get_obs(), self.invalid_penalty, True, {}

        newly = 0
        for idx, (a, b) in enumerate(self.segments):
            if not self.seg_covered[idx] and a >= p and b <= q:
                self.seg_covered[idx] = True
                newly += 1

        self.remaining_L -= length
        reward = self.cov_reward * newly - length
        done = all(self.seg_covered)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        ep = np.zeros(MAX_ENDPOINTS, dtype=np.float32)
        m  = np.zeros(MAX_ENDPOINTS, dtype=np.float32)
        for idx, v in enumerate(self.endpoints):
            ep[idx] = v
        for idx_seg, (a, b) in enumerate(self.segments):
            if not self.seg_covered[idx_seg]:
                for idx, v in enumerate(self.endpoints):
                    if v == a or v == b:
                        m[idx] = 1.0
        rem = np.array([self.remaining_L], dtype=np.float32)
        return np.concatenate([ep, m, rem])

    def render(self, mode="human"):
        uncovered = [seg for i, seg in enumerate(self.segments) if not self.seg_covered[i]]
        print(f"Remaining_L={self.remaining_L:.2f}, Uncovered={uncovered}")

    def close(self):
        pass
