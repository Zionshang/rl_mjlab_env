"""Observation terms for Go2 AMPVAE tasks."""

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
  scale_tensor = torch.tensor(scale, dtype=command.dtype, device=command.device)
  return command * scale_tensor


def foot_positions(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.site_names is not None:
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  else:
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand_as(
    torch.cat([foot_pos_w, foot_pos_w[..., :1]], dim=-1)
  )
  return quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w).flatten(start_dim=1)


def push_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.event_push_vel_buf


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
