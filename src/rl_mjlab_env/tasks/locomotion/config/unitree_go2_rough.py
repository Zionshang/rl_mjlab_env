"""Unitree Go2 rough-terrain locomotion configuration."""

from __future__ import annotations

from dataclasses import replace
import math

from mjlab.envs import ManagerBasedRlEnvCfg
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

from rl_mjlab_env.asset_zoo.robots.unitree_go2.go2_constants import (
    GO2_BASE_LINK,
    GO2_CALF_NAMES,
    GO2_FOOT_GEOM_NAMES,
    GO2_FOOT_SITE_NAMES,
    GO2_JOINT_ORDER,
    GO2_THIGH_NAMES,
    get_go2_robot_cfg,
)
from rl_mjlab_env.rl import LocomotionRunnerCfg
from rl_mjlab_env.tasks.amp.config.unitree_go2_flat import (
    go2_amp_observation_schema,
    go2_motion_files,
)
from rl_mjlab_env.tasks.locomotion import mdp

COMMAND_NAME = "base_command"
ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(GO2_JOINT_ORDER), preserve_order=True)
ROBOT_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=list(GO2_JOINT_ORDER), preserve_order=True)
ROBOT_FOOT_BODY_CFG = SceneEntityCfg("robot", site_names=tuple(GO2_FOOT_SITE_NAMES), preserve_order=True)
ROBOT_FOOT_GEOM_CFG = SceneEntityCfg("robot", geom_names=tuple(GO2_FOOT_GEOM_NAMES), preserve_order=True)
ROBOT_BASE_BODY_CFG = SceneEntityCfg("robot", body_names=GO2_BASE_LINK, preserve_order=True)


def make_go2_locomotion_runner_cfg() -> LocomotionRunnerCfg:
    return LocomotionRunnerCfg(
        clip_actions=100.0,
        experiment_name="unitree_go2_locomotion",
        wandb_tags=("go2", "locomotion", "amp", "vae", "mjlab"),
        policy_type={
            "actor_critic_type": "ActorCriticEncoder",
            "vae_type": "VAEBlind",
        },
        training_type="rl",
        module_cfg_dict={
            "amp": {
                "hidden_dims": [1024, 512],
            },
            "vae": {
                "encoder_in_dim": 225,
                "encoder_hidden_dims": [128],
                "encoder_out_dim": 64,
                "encoder_head_dim_dict": {
                    "obs_vel": 3,
                    "obs_com": 3,
                    "obs_mass": 1,
                    "obs_latent": 16,
                },
                "decoder_in_dim": 23,
                "decoder_hidden_dims": [64, 128],
                "decoder_out_dim": 48,
                "activation": "elu",
            },
            "actor_critic": {
                "actor": {
                    "num_actor_obs": 68,
                    "num_actions": 12,
                    "actor_hidden_dims": [512, 256, 128],
                    "actor_obs_normalization": False,
                },
                "privileged_encoder": {
                    "num_privileged_obs": 43,
                    "privileged_encoder_hidden_dims": [64, 32],
                    "num_privileged_encoder_out": 16,
                },
                "heightmap_encoder": {
                    "num_heightmap_obs": 187,
                    "heightmap_encoder_hidden_dims": [64, 32],
                    "num_heightmap_encoder_out": 32,
                },
                "critic": {
                    "num_critic_obs": 96,
                    "critic_hidden_dims": [512, 256, 128],
                    "critic_obs_normalization": False,
                },
                "init_noise_std": 1.0,
                "noise_std_type": "scalar",
                "activation": "elu",
                "min_normalized_std": [0.01, 0.01, 0.01] * 4,
            },
        },
        train_cfg_dict={
            "use_amp": True,
            "use_vae": True,
            "ppo_algorithm": {
                "value_loss_coef": 0.5,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.01,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 2.0e-5,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
                "normalize_advantage_per_mini_batch": False,
            },
            "amp": {
                "amp_replay_buffer_size": 1000000,
                "amp_disc_grad_penalty": 5.0,
                "motion_files": go2_motion_files(),
                "num_preload_transitions": 2000000,
                "observation_schema": go2_amp_observation_schema(),
            },
            "vae": {
                "use_exclusive_optimizer": False,
                "use_exclusive_lr": False,
                "learning_rate": 1.0e-3,
                "beta_adaptive": True,
                "beta_max_step": 5000,
                "beta": 1.0e-4,
                "beta_max": 0.01,
                "free_bits": 0.5,
                "use_adaboot": True,
                "adaboot_max_step": 500,
            },
        },
    )


def _actor_terms(play: bool) -> dict[str, ObservationTermCfg]:
    return {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            scale=0.25,
            noise=None if play else Unoise(n_min=-0.3, n_max=0.3),
            delay_min_lag=2,
            delay_max_lag=2,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=None if play else Unoise(n_min=-0.05, n_max=0.05),
            delay_min_lag=2,
            delay_max_lag=2,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": ROBOT_JOINT_CFG},
            noise=None if play else Unoise(n_min=-0.03, n_max=0.03),
            delay_min_lag=1,
            delay_max_lag=1,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": ROBOT_JOINT_CFG},
            scale=0.05,
            noise=None if play else Unoise(n_min=-1.5, n_max=1.5),
            delay_min_lag=2,
            delay_max_lag=2,
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
        "estimator_obs": ObservationGroupCfg(
            terms=_actor_terms(play),
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=5,
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
                    params={"asset_cfg": ROBOT_ACTUATOR_CFG, "kp_default": 70.0, "kd_default": 2.0},
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
                "joint_pos": ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": ROBOT_JOINT_CFG}),
                "joint_vel": ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": ROBOT_JOINT_CFG}),
                "foot_positions": ObservationTermCfg(
                    func=mdp.foot_positions,
                    params={"asset_cfg": ROBOT_FOOT_BODY_CFG},
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "next_obs": ObservationGroupCfg(
            terms=_critic_terms(),
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "gt_vel_obs": ObservationGroupCfg(
            terms={"base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel, scale=2.0)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "gt_mass_obs": ObservationGroupCfg(
            terms={
                "random_mass": ObservationTermCfg(
                    func=mdp.random_mass_obs,
                    params={"asset_cfg": ROBOT_BASE_BODY_CFG},
                    scale=0.1,
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "gt_com_obs": ObservationGroupCfg(
            terms={
                "random_com": ObservationTermCfg(
                    func=mdp.random_com_obs,
                    params={"asset_cfg": ROBOT_BASE_BODY_CFG},
                    scale=5.0,
                )
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
            weight=-2.5e-7,
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
            func=mdp.undesired_contacts,
            weight=-0.1,
            params={"sensor_name": "undesired_contact", "threshold": 0.1},
        ),
        "amp_reward": RewardTermCfg(func=mdp.amp_reward, weight=0.5),
    }
    if play:
        rewards.pop("amp_reward", None)
    return rewards


def unitree_go2_locomotion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name=GO2_BASE_LINK, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=play,
    )
    undesired_contact = ContactSensorCfg(
        name="undesired_contact",
        primary=ContactMatch(
            mode="body",
            pattern=GO2_THIGH_NAMES + GO2_CALF_NAMES,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=5,
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
                heading_command=True,
                heading_control_stiffness=0.5,
                debug_vis=play,
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-1.5, 1.5),
                    heading=(-1.57, 1.57),
                ),
            )
        },
        events=_events(play=play),
        rewards=_rewards(play=play),
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            "bad_orientation": TerminationTermCfg(
                func=envs_mdp.bad_orientation,
                params={"limit_angle": 1.4},
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
        decimation=4,
        episode_length_s=20.0,
        is_finite_horizon=False,
        scale_rewards_by_dt=True,
        seed=42,
    )

    if play and cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5

    return cfg
