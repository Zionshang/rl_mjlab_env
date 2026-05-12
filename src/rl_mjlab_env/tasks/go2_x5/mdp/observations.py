"""Observation terms for Go2 + X5."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    return command


def feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    return sensor.compute_first_contact(env.step_dt).float()


def get_mass_base(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
    current_mass = env.sim.model.body_mass[:, body_ids]
    return current_mass.reshape(env.num_envs, -1)


def get_mass_ee(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return get_mass_base(env, asset_cfg)


def get_joints_torques(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, asset_cfg.actuator_ids]
