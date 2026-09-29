"""Reward terms shared by the Go2 VAE+AMP environment."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
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
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
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


class UndesiredContacts:
  """Count colliding links while excluding foot geoms attached to calf bodies.

  The Go2 MJCF attaches each foot geom to its calf body. A body-based contact
  match therefore misclassifies every normal foot strike as a calf collision.
  The sensor instead exposes the individual thigh/calf collision geoms; this
  term groups the two calf geoms back into one logical link and takes the
  maximum force over all physics substeps in the policy step, matching the
  Isaac Lab reward semantics.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    groups: dict[str, list[int]] = {}
    for index, geom_name in enumerate(sensor.primary_names):
      if "_calf" in geom_name:
        link_name = geom_name.split("_calf", maxsplit=1)[0] + "_calf"
      else:
        link_name = geom_name.removesuffix("_collision")
      groups.setdefault(link_name, []).append(index)
    self._geom_groups = tuple(tuple(indices) for indices in groups.values())

  def __call__(
    self, env: ManagerBasedRlEnv, sensor_name: str, threshold: float
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force_history is not None:
      force_magnitude = torch.norm(sensor.data.force_history, dim=-1)
      geom_in_contact = force_magnitude.amax(dim=2) > threshold
    else:
      assert sensor.data.force is not None
      geom_in_contact = torch.norm(sensor.data.force, dim=-1) > threshold

    link_contacts = [
      geom_in_contact[:, indices].any(dim=1) for indices in self._geom_groups
    ]
    return torch.stack(link_contacts, dim=1).sum(dim=1).float()


def amp_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  if env.amp_out is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  reward = torch.clamp(1.0 - 0.25 * torch.square(env.amp_out - 1.0), min=0.0)
  return reward.squeeze(-1)
