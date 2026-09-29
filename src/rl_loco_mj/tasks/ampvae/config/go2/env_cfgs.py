"""Unitree Go2 rough-terrain AMPVAE configuration."""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from rl_loco_mj.asset_zoo.robots.go2.go2_constants import (
  GO2_BASE_LINK,
  GO2_CALF_GEOM_NAMES,
  GO2_FOOT_GEOM_NAMES,
  GO2_FOOT_SITE_NAMES,
  GO2_JOINT_ORDER,
  GO2_THIGH_GEOM_NAMES,
  get_go2_robot_cfg,
)
from rl_loco_mj.envs import VaeAmpRlEnvCfg
from rl_loco_mj.tasks.ampvae import mdp

COMMAND_NAME = "base_command"
POLICY_DECIMATION = 4
ROBOT_JOINT_CFG = SceneEntityCfg(
  "robot", joint_names=tuple(GO2_JOINT_ORDER), preserve_order=True
)
ROBOT_ACTUATOR_CFG = SceneEntityCfg(
  "robot", actuator_names=list(GO2_JOINT_ORDER), preserve_order=True
)
ROBOT_FOOT_BODY_CFG = SceneEntityCfg(
  "robot", site_names=tuple(GO2_FOOT_SITE_NAMES), preserve_order=True
)
ROBOT_FOOT_GEOM_CFG = SceneEntityCfg(
  "robot", geom_names=tuple(GO2_FOOT_GEOM_NAMES), preserve_order=True
)
ROBOT_BASE_BODY_CFG = SceneEntityCfg(
  "robot", body_names=GO2_BASE_LINK, preserve_order=True
)


def _actor_terms(play: bool) -> dict[str, ObservationTermCfg]:
  return {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      scale=0.25,
      noise=None if play else Unoise(n_min=-0.3, n_max=0.3),
      delay_min_lag=0,
      delay_max_lag=2,
      delay_update_period=1_000_000,
      delay_per_env_phase=False,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=None if play else Unoise(n_min=-0.05, n_max=0.05),
      delay_min_lag=0,
      delay_max_lag=2,
      delay_update_period=1_000_000,
      delay_per_env_phase=False,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": ROBOT_JOINT_CFG},
      noise=None if play else Unoise(n_min=-0.03, n_max=0.03),
      delay_min_lag=0,
      delay_max_lag=1,
      delay_update_period=1_000_000,
      delay_per_env_phase=False,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": ROBOT_JOINT_CFG},
      scale=0.05,
      noise=None if play else Unoise(n_min=-1.5, n_max=1.5),
      delay_min_lag=0,
      delay_max_lag=2,
      delay_update_period=1_000_000,
      delay_per_env_phase=False,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action, scale=0.25),
    "velocity_commands": ObservationTermCfg(
      func=mdp.generated_commands_scale,
      params={"command_name": COMMAND_NAME, "scale": (2.0, 2.0, 0.25)},
    ),
  }


def _critic_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel, scale=2.0),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, scale=0.25),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": ROBOT_JOINT_CFG},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": ROBOT_JOINT_CFG},
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action, scale=0.25),
    "velocity_commands": ObservationTermCfg(
      func=mdp.generated_commands_scale,
      params={"command_name": COMMAND_NAME, "scale": (2.0, 2.0, 0.25)},
    ),
  }


def _observations(play: bool = False) -> dict[str, ObservationGroupCfg]:
  actor_terms = _actor_terms(play)
  critic_terms = _critic_terms()
  return {
    "actor_obs": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic_obs": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "privileged_obs": ObservationGroupCfg(
      terms={
        "push_vel": ObservationTermCfg(func=mdp.push_vel),
        "random_mass": ObservationTermCfg(
          func=mdp.random_mass_obs,
          params={"asset_cfg": ROBOT_BASE_BODY_CFG},
          scale=0.1,
        ),
        "random_com": ObservationTermCfg(
          func=mdp.random_com_obs,
          params={"asset_cfg": ROBOT_BASE_BODY_CFG},
          scale=5.0,
        ),
        "random_material": ObservationTermCfg(
          func=mdp.random_material_obs,
          params={"asset_cfg": ROBOT_FOOT_GEOM_CFG},
        ),
        "random_actuator_gains": ObservationTermCfg(
          func=mdp.randomize_actuator_gains_obs,
          params={
            "asset_cfg": ROBOT_ACTUATOR_CFG,
            "kp_default": 70.0,
            "kd_default": 2.0,
          },
          scale=0.1,
        ),
        "random_actuator_lag": ObservationTermCfg(
          func=mdp.randomize_actuator_lag_obs,
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "gt_heightmap_obs": ObservationGroupCfg(
      terms={
        "height_map_yaw": ObservationTermCfg(
          func=envs_mdp.height_scan,
          params={"sensor_name": "terrain_scan", "offset": 0.5, "miss_value": 2.0},
          scale=5.0,
          clip=(-2.0, 2.0),
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "amp_obs": ObservationGroupCfg(
      terms={
        "base_lin_xy_vel": ObservationTermCfg(func=mdp.base_lin_xy_vel),
        "base_ang_yaw_vel": ObservationTermCfg(func=mdp.base_ang_yaw_vel),
        "joint_pos": ObservationTermCfg(
          func=mdp.joint_pos, params={"asset_cfg": ROBOT_JOINT_CFG}
        ),
        "joint_vel": ObservationTermCfg(
          func=mdp.joint_vel, params={"asset_cfg": ROBOT_JOINT_CFG}
        ),
        "foot_positions": ObservationTermCfg(
          func=mdp.foot_positions,
          params={"asset_cfg": ROBOT_FOOT_BODY_CFG},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "vae_ground_truth": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel, scale=2.0),
        "random_com": ObservationTermCfg(
          func=mdp.random_com_obs,
          params={"asset_cfg": ROBOT_BASE_BODY_CFG},
          scale=5.0,
        ),
        "random_mass": ObservationTermCfg(
          func=mdp.random_mass_obs,
          params={"asset_cfg": ROBOT_BASE_BODY_CFG},
          scale=0.1,
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def _events(play: bool = False) -> dict[str, EventTermCfg]:
  events = {
    "randomize_base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": ROBOT_BASE_BODY_CFG,
        "operation": "add",
        "ranges": (-1.0, 3.0),
      },
    ),
    "randomize_base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": ROBOT_BASE_BODY_CFG,
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.03, 0.03), 2: (-0.03, 0.05)},
      },
    ),
    "randomize_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": ROBOT_FOOT_GEOM_CFG,
        "operation": "abs",
        "ranges": (0.25, 1.2),
        "axes": [0, 1, 2],
        "shared_random": True,
      },
    ),
    "reset_base": EventTermCfg(
      mode="reset",
      func=envs_mdp.reset_root_state_from_flat_patches,
      params={
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (0.02, 0.05),
          "yaw": (-1.57, 1.57),
        },
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (0.0, 0.0),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      mode="reset",
      func=envs_mdp.reset_joints_by_offset,
      params={
        "position_range": (-0.5, 0.5),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": ROBOT_JOINT_CFG,
      },
    ),
    "reset_actuator_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": ROBOT_ACTUATOR_CFG,
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    "push_robot": EventTermCfg(
      mode="interval",
      interval_range_s=(10.0, 15.0),
      func=envs_mdp.push_by_setting_velocity,
      params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    ),
  }
  if play:
    for key in (
      "randomize_base_mass",
      "randomize_base_com",
      "randomize_friction",
      "reset_actuator_gains",
      "reset_robot_joints",
      "push_robot",
    ):
      events.pop(key, None)
  return events


def _rewards(play: bool = False) -> dict[str, RewardTermCfg]:
  rewards = {
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp,
      weight=1.5,
      params={"command_name": COMMAND_NAME, "std": math.sqrt(0.25)},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=0.5,
      params={"command_name": COMMAND_NAME, "std": math.sqrt(0.25)},
    ),
    "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
    "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
    "dof_vel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-1.0e-4,
      params={"asset_cfg": ROBOT_JOINT_CFG},
    ),
    "dof_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      # MuJoCo exposes instantaneous generalized acceleration (qacc), while
      # Isaac Lab finite-differences joint velocity. The qacc squared sum is
      # about 2-4x larger in matched rollouts, so retain the more physical
      # MuJoCo signal with a backend-calibrated coefficient.
      weight=-1.0e-7,
      params={"asset_cfg": ROBOT_JOINT_CFG},
    ),
    "dof_torques_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-1.0e-4,
      params={"asset_cfg": ROBOT_ACTUATOR_CFG},
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    "action_smoothness_l2": RewardTermCfg(func=mdp.action_smoothness_l2, weight=-0.01),
    "joint_power": RewardTermCfg(
      func=mdp.joint_power,
      weight=-2.0e-5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=tuple(GO2_JOINT_ORDER),
          actuator_names=list(GO2_JOINT_ORDER),
          preserve_order=True,
        )
      },
    ),
    "joint_power_distribution": RewardTermCfg(
      func=mdp.joint_power_distribution,
      weight=-1.0e-5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=tuple(GO2_JOINT_ORDER),
          actuator_names=list(GO2_JOINT_ORDER),
          preserve_order=True,
        )
      },
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.UndesiredContacts,
      weight=-0.1,
      params={"sensor_name": "undesired_contact", "threshold": 0.1},
    ),
    "amp_reward": RewardTermCfg(func=mdp.amp_reward, weight=0.5),
  }
  if play:
    rewards.pop("amp_reward", None)
  return rewards


def unitree_go2_ampvae_env_cfg(play: bool = False) -> VaeAmpRlEnvCfg:
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name=GO2_BASE_LINK, entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
  )
  undesired_contact = ContactSensorCfg(
    name="undesired_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=GO2_THIGH_GEOM_NAMES + GO2_CALF_GEOM_NAMES,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=POLICY_DECIMATION,
  )

  cfg = VaeAmpRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG, curriculum=not play),
        max_init_terrain_level=1,
      ),
      entities={"robot": get_go2_robot_cfg()},
      sensors=(terrain_scan, undesired_contact),
    ),
    observations=_observations(play=play),
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=list(GO2_JOINT_ORDER),
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
      )
    },
    commands={
      COMMAND_NAME: UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=not play,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(0.8, 1.0) if play else (-1.0, 1.0),
          lin_vel_y=(0.0, 0.0) if play else (-0.5, 0.5),
          ang_vel_z=(0.0, 0.0) if play else (-1.5, 1.5),
          heading=(0.0, 0.0) if play else (-1.57, 1.57),
        ),
      )
    },
    events=_events(play=play),
    rewards=_rewards(play=play),
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "bad_orientation": TerminationTermCfg(
        func=envs_mdp.bad_orientation,
        params={"limit_angle": 1.0},
      ),
    },
    curriculum=(
      {
        "terrain_levels": CurriculumTermCfg(
          func=terrain_levels_vel,
          params={"command_name": COMMAND_NAME},
        )
      }
      if not play
      else {}
    ),
    metrics={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=GO2_BASE_LINK,
      distance=3.0,
      elevation=-15.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      contact_sensor_maxmatch=500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=500,
        impratio=10.0,
        cone="elliptic",
        disableflags=("multiccd", "nativeccd"),
      ),
    ),
    decimation=POLICY_DECIMATION,
    episode_length_s=20.0,
    is_finite_horizon=False,
    scale_rewards_by_dt=True,
    seed=42,
  )

  if (
    play
    and cfg.scene.terrain is not None
    and cfg.scene.terrain.terrain_generator is not None
  ):
    cfg.scene.num_envs = 50
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.num_cols = 5

  return cfg
