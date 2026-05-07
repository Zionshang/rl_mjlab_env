"""Observation terms for Go2 locomotion."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from rl_mjlab_env.tasks.amp.mdp.observations import (
    base_ang_yaw_vel,
    base_lin_xy_vel,
    foot_positions,
    generated_commands_scale,
    joint_pos,
    joint_vel,
    push_vel,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def random_com_obs(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
    return env.sim.model.body_ipos[:, body_ids, :3].reshape(env.num_envs, -1)


def random_mass_obs(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
    default_mass = env.sim.get_default_field("body_mass")[body_ids]
    current_mass = env.sim.model.body_mass[:, body_ids]
    return (current_mass - default_mass.unsqueeze(0)).reshape(env.num_envs, -1)


def random_material_obs(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
    return env.sim.model.geom_friction[:, geom_ids, :3].reshape(env.num_envs, -1)


def randomize_actuator_gains_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    kp_default: float | None = None,
    kd_default: float | None = None,
) -> torch.Tensor:
    del kp_default, kd_default  # Read defaults from the compiled model.

    asset: Entity = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.actuator_ids, slice):
        actuators = asset.actuators[asset_cfg.actuator_ids]
    else:
        actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]

    kp_terms = []
    kd_terms = []
    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    default_biasprm = env.sim.get_default_field("actuator_biasprm")
    for actuator in actuators:
        ctrl_ids = actuator.global_ctrl_ids
        current_kp = env.sim.model.actuator_gainprm[:, ctrl_ids, 0]
        default_kp = default_gainprm[ctrl_ids, 0].unsqueeze(0)
        current_kd = -env.sim.model.actuator_biasprm[:, ctrl_ids, 2]
        default_kd = -default_biasprm[ctrl_ids, 2].unsqueeze(0)
        kp_terms.append(current_kp - default_kp)
        kd_terms.append(current_kd - default_kd)
    return torch.cat([torch.cat(kp_terms, dim=1), torch.cat(kd_terms, dim=1)], dim=1)


def randomize_actuator_lag_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    if not asset.actuators:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

    delay_buffer = asset.actuators[0]._delay_buffer
    if delay_buffer is None:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return delay_buffer.current_lags.to(dtype=torch.float32).unsqueeze(1)
