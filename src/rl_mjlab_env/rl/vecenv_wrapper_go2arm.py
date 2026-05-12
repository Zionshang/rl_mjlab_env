"""RSL-RL wrapper for the Go2Arm local runner."""

from __future__ import annotations

import numpy as np
import torch

from local_rsl_rl.env import VecEnv

from rl_mjlab_env.envs import Go2ArmManagerBasedRlEnv


class Go2ArmVecEnvWrapper(VecEnv):
    env: Go2ArmManagerBasedRlEnv

    def __init__(
        self,
        env: Go2ArmManagerBasedRlEnv,
        clip_actions: float | None = None,
    ) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.num_history, self.num_prop, self.num_priv = self._compute_obs_layout()
        self.env.reset()

    @property
    def unwrapped(self) -> Go2ArmManagerBasedRlEnv:
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict = self.unwrapped.observation_manager.compute()
        return obs_dict["policy"], {"observations": obs_dict}

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.reset()
        return obs_dict["policy"], {"observations": obs_dict, **extras}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, rew_arm, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return obs_dict["policy"], rew, rew_arm, dones, extras

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_obs_list_length(self) -> tuple[list[str], list[int]]:
        terms = self.unwrapped.observation_manager.get_active_iterable_terms(0)
        return [term[0] for term in terms], [len(term[1]) for term in terms]

    def _compute_obs_layout(self) -> tuple[int, int, int]:
        terms = self.cfg.observations["policy"].terms
        group_history = self.cfg.observations["policy"].history_length
        num_history = 0
        num_prop = 0
        num_priv = 0
        keys, lengths = self.get_obs_list_length()
        for key, length in zip(keys, lengths, strict=False):
            term_name = key.split("-", maxsplit=1)[1]
            if term_name.startswith("priv_"):
                num_priv += length
                continue
            term_history = group_history if group_history is not None else terms[term_name].history_length
            term_history = max(int(term_history), 1)
            num_history = max(num_history, term_history)
            num_prop += length // term_history
        return num_history, num_prop, num_priv

    def prepare_obs(self) -> None:
        self.total = np.zeros((self.num_history, self.num_prop))
        self.obs_new = torch.zeros(
            self.env.num_envs,
            self.num_prop * self.num_history,
            device=self.env.device,
        )
        keys, lengths = self.get_obs_list_length()
        prop_items = [(key, length) for key, length in zip(keys, lengths, strict=False) if "-priv_" not in key]
        ranges = {}
        offset = 0
        for key, length in prop_items:
            term_width = int(length / self.num_history)
            ranges[key] = np.array(list(range(offset + length - term_width, offset + length)))
            offset += length
        key_list = list(ranges.keys())
        for hist_idx in range(self.num_history):
            indices = []
            for key in key_list:
                indices.append(ranges[key] - hist_idx * ranges[key].shape[0])
            self.total[hist_idx, :] = np.concatenate(indices)

    def change_obs_order(self, obs: torch.Tensor) -> torch.Tensor:
        for hist_idx in range(self.num_history):
            obs_1 = obs[:, self.total[hist_idx, :]]
            self.obs_new = torch.cat([self.obs_new, obs_1], dim=-1)
        obs = torch.cat(
            [self.obs_new[:, self.num_prop * self.num_history :], obs[:, self.num_prop * self.num_history :]],
            dim=-1,
        )
        self.obs_new = torch.zeros(
            self.env.num_envs,
            self.num_prop * self.num_history,
            device=self.env.device,
        )
        return obs
