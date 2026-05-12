"""Event terms for the Go2 + X5 task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_joints_by_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset selected joints by scaling their default position.

    This mirrors the IsaacLab Go2Arm reset: non-zero nominal leg angles stay near
    their default posture, instead of receiving a large absolute radian offset.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert default_joint_pos is not None
    assert default_joint_vel is not None
    assert soft_joint_pos_limits is not None

    joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
