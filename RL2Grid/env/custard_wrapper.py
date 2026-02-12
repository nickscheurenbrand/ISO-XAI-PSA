"""CUSTARD (Constrained Upper Confidence Trees for Any RL) wrapper.

Transforms a continuous redispatching Grid2Op/Gymnasium env into an
Interpretable Basis MDP (IBMDP) where the agent can either (a) split on a
feature to refine bounds or (b) emit a continuous redispatch action.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


class CustardWrapper(gym.Wrapper):
    """Wraps a redispatch environment to expose CUSTARD-style split actions.

    Observation: only the bound vector S_I (2 * N) is exposed to the agent to
    encourage a pure decision-tree policy. The true environment state S_M is
    kept internally and used for bound updates and env stepping.

    Action space: a Dict with a `mode` selector.
      mode == 0: split action, uses `feature` (Discrete) and `threshold` (Box)
      mode == 1: env action, uses `env_action` (Box, same as base env)
    """

    def __init__(self, env: gym.Env, threshold_low: float = -np.inf, threshold_high: float = np.inf):
        super().__init__(env)

        assert isinstance(env.action_space, gym.spaces.Box), "CustardWrapper expects continuous redispatch actions."

        self.n_features = env.observation_space.shape[0]
        self._hidden_obs = None  # last true observation S_M

        # Bounds S_I: concatenate lower and upper bounds (l_1..l_N, u_1..u_N)
        self._lower = np.full(self.n_features, -np.inf, dtype=np.float32)
        self._upper = np.full(self.n_features, np.inf, dtype=np.float32)

        # Observation space the agent sees (only bounds)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.n_features,),
            dtype=np.float32,
        )

        # Action space: union of split and env actions
        self.action_space = gym.spaces.Dict(
            {
                "mode": gym.spaces.Discrete(2),  # 0 = split (A_I), 1 = env action (A_M)
                "feature": gym.spaces.Discrete(self.n_features),
                "threshold": gym.spaces.Box(low=threshold_low, high=threshold_high, shape=(1,), dtype=np.float32),
                "env_action": env.action_space,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    def _reset_bounds(self):
        self._lower.fill(-np.inf)
        self._upper.fill(np.inf)

    def _obs_bounds(self) -> np.ndarray:
        # Agent only sees bounds; concat lowers then uppers
        return np.concatenate([self._lower, self._upper]).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._hidden_obs = np.array(obs, copy=True)
        self._reset_bounds()
        return self._obs_bounds(), info

    def step(self, action):
        mode = int(action.get("mode", 1))

        if mode == 0:
            # Split action: update bounds based on hidden observation value
            feature = int(action["feature"])
            threshold = float(np.array(action["threshold"]).reshape(-1)[0])
            val = float(self._hidden_obs[feature])
            if val <= threshold:
                self._upper[feature] = min(self._upper[feature], threshold)
            else:
                self._lower[feature] = max(self._lower[feature], threshold)

            obs = self._obs_bounds()
            reward = 0.0
            terminated = False
            truncated = False
            info = {"split": True, "feature": feature, "threshold": threshold, "value": val}
            return obs, reward, terminated, truncated, info

        # mode == 1: environment action
        env_act = np.array(action["env_action"], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = self.env.step(env_act)
        self._hidden_obs = np.array(next_obs, copy=True)

        # New decision path starts after a real env step
        self._reset_bounds()
        obs = self._obs_bounds()
        info = dict(info)
        info["split"] = False
        return obs, reward, terminated, truncated, info


__all__ = ["CustardWrapper"]