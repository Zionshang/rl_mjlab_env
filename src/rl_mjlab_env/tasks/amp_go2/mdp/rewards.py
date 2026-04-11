"""Reward terms for Go2 flat AMP."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
    return torch.exp(-error / std**2)


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
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_power(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    actuator_force = asset.data.actuator_force[:, asset_cfg.actuator_ids]
    return torch.sum(torch.abs(joint_vel * actuator_force), dim=1)


def joint_power_distribution(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    actuator_force = asset.data.actuator_force[:, asset_cfg.actuator_ids]
    return torch.var(torch.abs(joint_vel * actuator_force), dim=1)


def action_smoothness_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    actions = env.actions_history
    diff = torch.square(actions[:, 0, :] - 2.0 * actions[:, 1, :] + actions[:, 2, :])
    return torch.sum(diff, dim=1)


def undesired_contacts(
    env: ManagerBasedRlEnv, sensor_name: str, threshold: float
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None
    return torch.sum(torch.norm(sensor.data.force, dim=-1) > threshold, dim=1).float()


def amp_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    if not hasattr(env, "amp_out"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    reward = torch.clamp(1.0 - 0.25 * torch.square(env.amp_out - 1.0), min=0.0)
    return reward.squeeze(-1)
