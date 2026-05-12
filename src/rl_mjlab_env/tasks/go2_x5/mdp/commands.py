"""Command generators for the Go2 + X5 task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import CommandTerm
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_unique,
    wrap_to_pi,
)

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class CurriculumVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command with Go2Arm-style linear range curriculum."""

    curriculum_coeff: int = 1000
    ranges_init: UniformVelocityCommandCfg.Ranges | None = None
    ranges_final: UniformVelocityCommandCfg.Ranges | None = None

    def build(self, env: ManagerBasedRlEnv) -> "CurriculumVelocityCommand":
        return CurriculumVelocityCommand(self, env)


class CurriculumVelocityCommand(CommandTerm):
    cfg: CurriculumVelocityCommandCfg

    def __init__(self, cfg: CurriculumVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
            / max_command_step
        )

    def _curriculum_blend(self) -> torch.Tensor:
        count = self._env.common_step_counter / (24.0 * max(self.cfg.curriculum_coeff, 1))
        return torch.tensor(count, device=self.device).clamp(0.0, 1.0)

    def _sample_range(
        self,
        env_ids: torch.Tensor,
        init_range: tuple[float, float],
        final_range: tuple[float, float],
    ) -> torch.Tensor:
        ratio = self._curriculum_blend()
        r = torch.empty(len(env_ids), device=self.device)
        return r.uniform_(*init_range) * (1.0 - ratio) + r.uniform_(*final_range) * ratio

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.ranges_init is not None and self.cfg.ranges_final is not None:
            ranges_init = self.cfg.ranges_init
            ranges_final = self.cfg.ranges_final
            self.vel_command_b[env_ids, 0] = self._sample_range(
                env_ids, ranges_init.lin_vel_x, ranges_final.lin_vel_x
            )
            self.vel_command_b[env_ids, 1] = self._sample_range(
                env_ids, ranges_init.lin_vel_y, ranges_final.lin_vel_y
            )
            self.vel_command_b[env_ids, 2] = self._sample_range(
                env_ids, ranges_init.ang_vel_z, ranges_final.ang_vel_z
            )
        else:
            r = torch.empty(len(env_ids), device=self.device)
            self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
            self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
            self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        r = torch.empty(len(env_ids), device=self.device)
        if self.cfg.heading_command:
            assert self.cfg.ranges.heading is not None
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self) -> None:
        if self.cfg.heading_command:
            heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error[env_ids],
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        cmds = self.command.detach().cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.detach().cpu().numpy()
        base_mat_ws = matrix_from_quat(self.robot.data.root_link_quat_w).detach().cpu().numpy()
        lin_vel_bs = self.robot.data.root_link_lin_vel_b.detach().cpu().numpy()
        ang_vel_bs = self.robot.data.root_link_ang_vel_b.detach().cpu().numpy()

        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset

        for batch in env_indices:
            base_pos_w = base_pos_ws[batch]
            if np.linalg.norm(base_pos_w) < 1.0e-6:
                continue
            base_mat_w = base_mat_ws[batch]

            def local_to_world(vec: np.ndarray) -> np.ndarray:
                return base_pos_w + base_mat_w @ vec

            cmd = cmds[batch]
            lin_vel_b = lin_vel_bs[batch]
            ang_vel_b = ang_vel_bs[batch]
            origin = np.array([0.0, 0.0, z_offset])

            cmd_from = local_to_world(origin * scale)
            cmd_to = local_to_world((origin + np.array([cmd[0], cmd[1], 0.0])) * scale)
            visualizer.add_arrow(
                cmd_from,
                cmd_to,
                color=(0.15, 0.2, 1.0, 0.85),
                width=0.018,
                label=f"base_cmd_lin_{batch}",
            )

            cmd_yaw_to = local_to_world((origin + np.array([0.0, 0.0, cmd[2]])) * scale)
            visualizer.add_arrow(
                cmd_from,
                cmd_yaw_to,
                color=(0.2, 0.8, 0.2, 0.8),
                width=0.014,
                label=f"base_cmd_yaw_{batch}",
            )

            vel_to = local_to_world((origin + np.array([lin_vel_b[0], lin_vel_b[1], 0.0])) * scale)
            visualizer.add_arrow(
                cmd_from,
                vel_to,
                color=(0.0, 0.85, 1.0, 0.75),
                width=0.012,
                label=f"base_actual_lin_{batch}",
            )

            yaw_to = local_to_world((origin + np.array([0.0, 0.0, ang_vel_b[2]])) * scale)
            visualizer.add_arrow(
                cmd_from,
                yaw_to,
                color=(0.0, 1.0, 0.45, 0.65),
                width=0.01,
                label=f"base_actual_yaw_{batch}",
            )


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """End-effector pose command in the robot root frame."""

    entity_name: str
    body_name: str
    curriculum_coeff: int = 1000
    make_quat_unique: bool = True
    is_go2arm: bool = True
    is_go2arm_play: bool = False

    @dataclass
    class Ranges:
        pos_x: tuple[float, float]
        pos_y: tuple[float, float]
        pos_z: tuple[float, float]
        roll: tuple[float, float]
        pitch: tuple[float, float]
        yaw: tuple[float, float]

    ranges: Ranges
    ranges_init: Ranges | None = None
    ranges_final: Ranges | None = None

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (1.0, 0.45, 0.0, 0.9)
        current_color: tuple[float, float, float, float] = (0.0, 0.8, 1.0, 0.7)
        target_radius: float = 0.035
        current_radius: float = 0.02
        frame_scale: float = 0.16

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> "UniformPoseCommand":
        return UniformPoseCommand(self, env)


class UniformPoseCommand(CommandTerm):
    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        body_ids, _ = self.robot.find_bodies((cfg.body_name,), preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one end-effector body, got {body_ids}")
        self.body_id = body_ids[0]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.pose_command_w_z = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _curriculum_blend(self) -> torch.Tensor:
        count = self._env.common_step_counter / (24.0 * max(self.cfg.curriculum_coeff, 1))
        return torch.tensor(count, device=self.device).clamp(0.0, 1.0)

    def _sample_range(
        self,
        env_ids: torch.Tensor,
        init_range: tuple[float, float],
        final_range: tuple[float, float],
    ) -> torch.Tensor:
        ratio = self._curriculum_blend()
        r = torch.empty(len(env_ids), device=self.device)
        return r.uniform_(*init_range) * (1.0 - ratio) + r.uniform_(*final_range) * ratio

    def _update_metrics(self) -> None:
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        if self.cfg.is_go2arm or self.cfg.is_go2arm_play:
            self.pose_command_w[:, 2] = self.pose_command_w_z[:, 0]

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_link_pos_w[:, self.body_id, :],
            self.robot.data.body_link_quat_w[:, self.body_id, :],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        euler_angles = torch.zeros((len(env_ids), 3), device=self.device)
        ranges = self.cfg.ranges

        if self.cfg.is_go2arm and self.cfg.ranges_init is not None and self.cfg.ranges_final is not None:
            ranges_init = self.cfg.ranges_init
            ranges_final = self.cfg.ranges_final
            self.pose_command_b[env_ids, 0] = self._sample_range(
                env_ids, ranges_init.pos_x, ranges_final.pos_x
            )
            self.pose_command_b[env_ids, 1] = self._sample_range(
                env_ids, ranges_init.pos_y, ranges_final.pos_y
            )
            self.pose_command_w_z[env_ids, 0] = self._sample_range(
                env_ids, ranges_init.pos_z, ranges_final.pos_z
            )
        else:
            r = torch.empty(len(env_ids), device=self.device)
            self.pose_command_b[env_ids, 0] = r.uniform_(*ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*ranges.pos_y)
            self.pose_command_w_z[env_ids, 0] = r.uniform_(*ranges.pos_z)

        self.pose_command_b[env_ids, 2] = (
            self.pose_command_w_z[env_ids, 0] - self.robot.data.root_link_pos_w[env_ids, 2]
        )
        self._reject_unreachable_goals(env_ids)

        delta_x = self.pose_command_b[env_ids, 0]
        delta_y = self.pose_command_b[env_ids, 1]
        delta_z = self.pose_command_b[env_ids, 2]
        r = torch.empty(len(env_ids), device=self.device)
        euler_angles[:, 0] = r.uniform_(*ranges.roll)
        euler_angles[:, 1] = -torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) + r.uniform_(
            *ranges.pitch
        )
        euler_angles[:, 2] = torch.atan2(delta_y, delta_x) + r.uniform_(*ranges.yaw)

        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _reject_unreachable_goals(self, env_ids: torch.Tensor) -> None:
        if not (self.cfg.is_go2arm or self.cfg.is_go2arm_play):
            return
        r = torch.empty(1, device=self.device)
        ranges = self.cfg.ranges
        root_z = self.robot.data.root_link_pos_w[:, 2]
        for env_id in env_ids.tolist():
            for _ in range(32):
                goal = self.pose_command_b[env_id, :3]
                length = torch.norm(goal)
                too_close_front = goal[0] < 0.2 and torch.abs(goal[1]) < 0.2
                if 0.3 <= length <= 0.7 and not too_close_front:
                    break
                self.pose_command_b[env_id, 0] = r.uniform_(*ranges.pos_x)
                self.pose_command_b[env_id, 1] = r.uniform_(*ranges.pos_y)
                self.pose_command_w_z[env_id, 0] = r.uniform_(*ranges.pos_z)
                self.pose_command_b[env_id, 2] = self.pose_command_w_z[env_id, 0] - root_z[env_id]

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        target_pos_ws = self.pose_command_w[:, :3].detach().cpu().numpy()
        target_mat_ws = matrix_from_quat(self.pose_command_w[:, 3:]).detach().cpu().numpy()
        current_pos_ws = self.robot.data.body_link_pos_w[:, self.body_id, :].detach().cpu().numpy()
        current_mat_ws = (
            matrix_from_quat(self.robot.data.body_link_quat_w[:, self.body_id, :])
            .detach()
            .cpu()
            .numpy()
        )

        for batch in env_indices:
            target_pos = target_pos_ws[batch]
            current_pos = current_pos_ws[batch]
            if np.linalg.norm(target_pos) < 1.0e-6:
                continue

            visualizer.add_sphere(
                center=target_pos,
                radius=self.cfg.viz.target_radius,
                color=self.cfg.viz.target_color,
                label=f"ee_cmd_target_{batch}",
            )
            visualizer.add_sphere(
                center=current_pos,
                radius=self.cfg.viz.current_radius,
                color=self.cfg.viz.current_color,
                label=f"ee_current_{batch}",
            )
            visualizer.add_frame(
                position=target_pos,
                rotation_matrix=target_mat_ws[batch],
                scale=self.cfg.viz.frame_scale,
                label=f"ee_cmd_frame_{batch}",
                axis_radius=0.008,
                alpha=0.9,
            )
            visualizer.add_frame(
                position=current_pos,
                rotation_matrix=current_mat_ws[batch],
                scale=self.cfg.viz.frame_scale * 0.75,
                label=f"ee_current_frame_{batch}",
                axis_radius=0.006,
                alpha=0.7,
            )
            visualizer.add_arrow(
                current_pos,
                target_pos,
                color=(1.0, 0.55, 0.0, 0.65),
                width=0.01,
                label=f"ee_error_{batch}",
            )
