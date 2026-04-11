"""RSL-RL wrapper for the vendored pure AMP runner."""

from __future__ import annotations

import torch

from mjlab.utils.spaces import Space
from rsl_rl.env import VecEnv

from rl_mjlab_env.envs import Go2AmpManagerBasedRlEnv


class AmpVecEnvWrapper(VecEnv):
    def __init__(
        self,
        env: Go2AmpManagerBasedRlEnv,
        clip_actions: float | None = None,
        clip_obs: float | None = None,
    ) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self.clip_obs = clip_obs

        self.num_envs = self.unwrapped.num_envs
        self.step_dt = self.unwrapped.step_dt
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.max_episode_length_s = self.unwrapped.max_episode_length_s
        self.num_actions = self.unwrapped.num_actions

        self._modify_action_space()
        self.env.reset()

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> Space:
        return self.env.observation_space

    @property
    def action_space(self) -> Space:
        return self.env.action_space

    @property
    def unwrapped(self) -> Go2AmpManagerBasedRlEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_observations(self):
        if self.unwrapped.obs_buf:
            return self.unwrapped.obs_buf
        return self.unwrapped.observation_manager.compute()

    def reset(self):
        obs_dict, _ = self.env.reset()
        return obs_dict

    def step(self, actions: torch.Tensor, amp_out: torch.Tensor | None = None):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        (
            obs_dict,
            rew,
            terminated,
            truncated,
            extras,
            reset_env_ids,
            terminal_amp_states,
            episode_reward,
        ) = self.env.step(actions, amp_out)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return obs_dict, rew, dones, extras, reset_env_ids, terminal_amp_states, episode_reward

    def close(self) -> None:
        self.env.close()

    def _modify_action_space(self) -> None:
        if self.clip_actions is None:
            return
        from mjlab.utils.spaces import Box, batch_space

        self.unwrapped.single_action_space = Box(
            shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
        )
        self.unwrapped.action_space = batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )
