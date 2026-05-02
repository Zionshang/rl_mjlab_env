"""Observation terms for Go2 AMP."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_lin_xy_vel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, :2]


def base_ang_yaw_vel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b[:, 2:3]


def joint_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def generated_commands_scale(
    env: ManagerBasedRlEnv, command_name: str, scale: tuple[float, ...]
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    return command * torch.tensor(scale, dtype=command.dtype, device=command.device)


def foot_positions(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_link_pos_w[:, None, :]
    root_quat_w = asset.data.root_link_quat_w[:, None, :].expand_as(
        torch.cat([foot_pos_w, foot_pos_w[..., :1]], dim=-1)
    )
    rel_pos_w = foot_pos_w - root_pos_w
    return quat_apply_inverse(root_quat_w, rel_pos_w).flatten(start_dim=1)


def push_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.event_push_vel_buf
