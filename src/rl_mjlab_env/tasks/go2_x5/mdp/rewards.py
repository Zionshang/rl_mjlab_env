"""Reward terms for the Go2 + X5 task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    quat_mul,
    quat_apply_inverse,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def position_command_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_link_pos_w,
        asset.data.root_link_quat_w,
        des_pos_b,
    )
    des_pos_w[:, 2] = des_pos_b[:, 2] + asset.data.root_link_pos_w[:, 2]
    curr_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], :3]
    return torch.exp(-torch.sum(torch.abs(curr_pos_w - des_pos_w) / std, dim=1))


def orientation_command_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_link_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def action_rate_l2_arm(env: ManagerBasedRlEnv, arm_start: int = 12) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action[:, arm_start:] - env.action_manager.prev_action[:, arm_start:]),
        dim=1,
    )


def arm_action_smoothness_penalty(env: ManagerBasedRlEnv, arm_start: int = 12) -> torch.Tensor:
    return torch.linalg.norm(
        env.action_manager.action[:, arm_start:] - env.action_manager.prev_action[:, arm_start:],
        dim=1,
    )


def action_rate_l2_leg(env: ManagerBasedRlEnv, leg_dim: int = 12) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action[:, :leg_dim] - env.action_manager.prev_action[:, :leg_dim]),
        dim=1,
    )


def leg_action_smoothness_penalty(env: ManagerBasedRlEnv, leg_dim: int = 12) -> torch.Tensor:
    return torch.linalg.norm(
        env.action_manager.action[:, :leg_dim] - env.action_manager.prev_action[:, :leg_dim],
        dim=1,
    )


def track_lin_vel_xy_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std)


def track_ang_vel_z_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_torques_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1)


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    first_contact = sensor.compute_first_contact(env.step_dt)
    last_air_time = sensor.data.last_air_time
    assert last_air_time is not None
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    reward *= torch.norm(command[:, :2], dim=1) > 0.1
    return reward


def feet_height_body(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.site_names is not None:
        foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
        foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
        num_feet = len(asset_cfg.site_ids)
    else:
        foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
        foot_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
        num_feet = len(asset_cfg.body_ids)
    rel_pos_w = foot_pos_w - asset.data.root_link_pos_w[:, None, :]
    rel_vel_w = foot_vel_w - asset.data.root_link_lin_vel_w[:, None, :]
    root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, num_feet, -1)
    foot_pos_b = quat_apply_inverse(root_quat, rel_pos_w)
    foot_vel_b = quat_apply_inverse(root_quat, rel_vel_w)
    foot_z_error = torch.square(foot_pos_b[:, :, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_b[:, :, :2], dim=2))
    reward = torch.sum(foot_z_error * foot_velocity_tanh, dim=1)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    reward *= torch.linalg.norm(command, dim=1) > 0.1
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.site_names is not None:
        foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
        foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
    else:
        foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
        foot_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    foot_z_error = torch.square(foot_pos_w[:, :, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_w[:, :, :2], dim=2))
    reward = torch.sum(foot_z_error * foot_velocity_tanh, dim=1)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    reward *= torch.linalg.norm(command, dim=1) > 0.1
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def standing_feet_contact_force(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    force_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None
    contact_force = torch.norm(sensor.data.force, dim=-1)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    is_small_command = torch.norm(command[:, :2], dim=1) < command_threshold
    force = torch.clamp(torch.min(contact_force[:, 0], contact_force[:, 1]), 0.0, force_threshold)
    return torch.where(is_small_command, 2.0 * force, force)


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )


def base_height_l2(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    curr_height = torch.clamp(asset.data.root_link_pos_w[:, 2], max=0.4)
    return torch.square(curr_height - target_height)


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
