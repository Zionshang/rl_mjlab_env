"""RSL-RL wrapper for the standalone VAE+AMP backend."""

from __future__ import annotations

import torch
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from rl_loco_mj.envs.vae_amp_rl_env import VaeAmpRlEnv


class VaeAmpVecEnvWrapper(RslRlVecEnvWrapper):
  """Adapt a MJLab environment to the original VAE+AMP runner API."""

  env: VaeAmpRlEnv

  def __init__(
    self,
    env: VaeAmpRlEnv,
    clip_actions: float | None = None,
    clip_obs: float | None = None,
  ) -> None:
    self.env = env
    self.clip_obs = clip_obs
    super().__init__(env, clip_actions=clip_actions)
    self.step_dt = self.unwrapped.step_dt
    self.max_episode_length_s = self.unwrapped.max_episode_length_s

  @property
  def unwrapped(self) -> VaeAmpRlEnv:
    return self.env.unwrapped

  def step(self, actions: torch.Tensor, amp_out: torch.Tensor | None = None):
    self.env.update_amp_out(amp_out)
    return super().step(actions)

  def get_observations(self) -> TensorDict:
    """Return cached observations without advancing physical-step delay buffers."""
    return TensorDict(self.unwrapped.obs_buf, batch_size=[self.num_envs])
