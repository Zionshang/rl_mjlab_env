"""RSL-RL wrapper for the vendored AMP runners."""

from __future__ import annotations

import torch

from mjlab.rl import RslRlVecEnvWrapper

from rl_mjlab_env.envs import AmpManagerBasedRlEnv


class AmpVecEnvWrapper(RslRlVecEnvWrapper):
    env: AmpManagerBasedRlEnv

    def __init__(
        self,
        env: AmpManagerBasedRlEnv,
        clip_actions: float | None = None,
        clip_obs: float | None = None,
    ) -> None:
        self.env = env
        self.clip_obs = clip_obs
        super().__init__(env, clip_actions=clip_actions)
        self.step_dt = self.unwrapped.step_dt
        self.max_episode_length_s = self.unwrapped.max_episode_length_s

    @property
    def unwrapped(self) -> AmpManagerBasedRlEnv:
        return self.env.unwrapped

    def step(self, actions: torch.Tensor, amp_out: torch.Tensor | None = None):
        self.env.update_amp_out(amp_out)
        return super().step(actions)
