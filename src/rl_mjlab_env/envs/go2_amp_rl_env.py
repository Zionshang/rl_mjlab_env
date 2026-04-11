"""MJLab manager-based environment with rl_sim_env AMP step semantics."""

from __future__ import annotations

import torch

from mjlab.envs import types
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class Go2AmpManagerBasedRlEnv(ManagerBasedRlEnv):
    """Pure AMP environment adapter for Unitree Go2 flat locomotion."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clip_obs = 100.0
        self.only_positive_reward = True
        self.episode_reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.event_push_vel_buf = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float32
        )
        self.actions_history = torch.zeros(
            self.num_envs,
            3,
            self.action_manager.total_action_dim,
            device=self.device,
            dtype=torch.float32,
        )

    @property
    def num_actions(self) -> int:
        return self.action_manager.total_action_dim

    def update_amp_out(self, amp_out: torch.Tensor | None = None) -> None:
        if amp_out is not None:
            self.amp_out = amp_out

    def step(
        self, action: torch.Tensor, amp_out: torch.Tensor | None = None
    ) -> types.VecEnvStepReturn:
        self.update_amp_out(amp_out)
        action = action.to(self.device)
        self.actions_history = torch.roll(self.actions_history, shifts=1, dims=1)
        self.actions_history[:, 0, :] = action
        self.action_manager.process_action(action)

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if self.only_positive_reward:
            self.reward_buf.clamp_(min=0.0)
        self.episode_reward_buf += self.reward_buf
        self.metrics_manager.compute()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0 and "amp_obs" in self.observation_manager.active_terms:
            terminal_amp_states = self.observation_manager.compute_group("amp_obs")[
                reset_env_ids
            ]
        else:
            terminal_amp_states = None

        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()

        self.sim.forward()
        self.command_manager.compute(dt=self.step_dt)

        self.event_push_vel_buf.zero_()
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        for obs in self.obs_buf.values():
            if isinstance(obs, torch.Tensor):
                obs.clamp_(-self.clip_obs, self.clip_obs)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
            reset_env_ids,
            terminal_amp_states,
            self.episode_reward_buf,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.episode_reward_buf[env_ids] = 0.0
        self.actions_history[env_ids] = 0.0
