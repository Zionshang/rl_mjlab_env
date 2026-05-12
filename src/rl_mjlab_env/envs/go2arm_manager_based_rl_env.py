"""Go2Arm environment adapter with split leg and arm rewards."""

from __future__ import annotations

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardManager


class Go2ArmRewardManager(RewardManager):
    """Reward manager that keeps end-effector terms in a separate buffer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arm_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def compute(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        self._reward_buf[:] = 0.0
        self.arm_reward_buf[:] = 0.0
        scale = dt if self._scale_by_dt else 1.0
        for term_idx, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=False)
        ):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            if name.startswith("end_effector"):
                self.arm_reward_buf += value
            else:
                self._reward_buf += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / scale
        return self._reward_buf, self.arm_reward_buf


class Go2ArmManagerBasedRlEnv(ManagerBasedRlEnv):
    """Manager-based env that returns `(leg_reward, arm_reward)` for local_rsl_rl."""

    @property
    def num_actions(self) -> int:
        return self.action_manager.total_action_dim

    def load_managers(self) -> None:
        super().load_managers()
        self.reward_manager = Go2ArmRewardManager(
            self.cfg.rewards,
            self,
            scale_by_dt=self.cfg.scale_rewards_by_dt,
        )
        self.setup_manager_visualizers()

    def step(self, action: torch.Tensor):
        if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
            pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
            raise RuntimeError(
                f"Environments {pending_ids.cpu().tolist()} must be reset before step()."
            )

        self.action_manager.process_action(action.to(self.device))

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)
            self.metrics_manager.compute_substep()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf, self.arm_reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.metrics_manager.compute()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if self.cfg.auto_reset and len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()

        self.sim.forward()
        self.command_manager.compute(dt=self.step_dt)

        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if self.cfg.auto_reset and len(reset_env_ids) > 0:
            self.recorder_manager.record_post_reset(reset_env_ids)
        elif len(reset_env_ids) > 0:
            self._manual_reset_pending[reset_env_ids] = True

        self.recorder_manager.record_post_step()

        return (
            self.obs_buf,
            self.reward_buf,
            self.arm_reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )
