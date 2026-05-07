"""Unitree Go2 flat pure AMP configuration."""

from __future__ import annotations

import glob
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from rl_mjlab_env.asset_zoo.robots.unitree_go2.go2_constants import (
    GO2_BASE_LINK,
    GO2_FOOT_GEOM_NAMES,
    GO2_FOOT_NAMES,
    GO2_FOOT_SITE_NAMES,
    GO2_JOINT_ORDER,
    GO2_THIGH_NAMES,
    get_go2_robot_cfg,
)
from rl_mjlab_env.rl.config import (
    AmpAlgorithmCfg,
    AmpModuleCfg,
    AmpPolicyCfg,
    AmpRunnerCfg,
)
from rl_mjlab_env.tasks.amp import mdp

COMMAND_NAME = "base_command"
_DEFAULT_MOTION_DIR = Path(__file__).resolve().parents[5] / "dataset" / "unitree_go2" / "trot" / "npz"
ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=tuple(GO2_JOINT_ORDER), preserve_order=True)
ROBOT_FOOT_BODY_CFG = SceneEntityCfg("robot", site_names=tuple(GO2_FOOT_SITE_NAMES), preserve_order=True)


def go2_amp_observation_schema() -> dict:
    root_pos_size = 3
    root_rot_size = 4
    root_linear_vel_size = 3
    root_angular_vel_size = 3
    frame_pos_size = 12
    frame_vel_size = 12
    joint_pos_size = 12
    joint_vel_size = 12

    root_pos_start = 0
    root_pos_end = root_pos_start + root_pos_size
    root_rot_start = root_pos_end
    root_rot_end = root_rot_start + root_rot_size
    root_linear_vel_start = root_rot_end
    root_linear_vel_end = root_linear_vel_start + root_linear_vel_size
    root_angular_vel_start = root_linear_vel_end
    root_angular_vel_end = root_angular_vel_start + root_angular_vel_size
    frame_pos_start = root_angular_vel_end
    frame_pos_end = frame_pos_start + frame_pos_size
    frame_vel_start = frame_pos_end
    frame_vel_end = frame_vel_start + frame_vel_size
    joint_pos_start = frame_vel_end
    joint_pos_end = joint_pos_start + joint_pos_size
    joint_vel_start = joint_pos_end
    joint_vel_end = joint_vel_start + joint_vel_size

    root_keys = [
        "root_position_world",
        "root_quaternion_wxyz",
        "root_linear_velocity_base",
        "root_angular_velocity_base",
    ]
    frame_pos_keys = [frame + "_position_base" for frame in GO2_FOOT_NAMES]
    frame_vel_keys = [frame + "_velocity_base" for frame in GO2_FOOT_NAMES]
    joint_pos_keys = [joint + "_q" for joint in GO2_JOINT_ORDER]
    joint_vel_keys = [joint + "_dq" for joint in GO2_JOINT_ORDER]

    return {
        "ROOT_POS_START_IDX": root_pos_start,
        "ROOT_POS_END_IDX": root_pos_end,
        "ROOT_ROT_START_IDX": root_rot_start,
        "ROOT_ROT_END_IDX": root_rot_end,
        "ROOT_LINEAR_VEL_START_IDX": root_linear_vel_start,
        "ROOT_LINEAR_VEL_END_IDX": root_linear_vel_end,
        "ROOT_ANGULAR_VEL_START_IDX": root_angular_vel_start,
        "ROOT_ANGULAR_VEL_END_IDX": root_angular_vel_end,
        "FRAME_POS_START_IDX": frame_pos_start,
        "FRAME_POS_END_IDX": frame_pos_end,
        "FRAME_VEL_START_IDX": frame_vel_start,
        "FRAME_VEL_END_IDX": frame_vel_end,
        "JOINT_POS_START_IDX": joint_pos_start,
        "JOINT_POS_END_IDX": joint_pos_end,
        "JOINT_VEL_START_IDX": joint_vel_start,
        "JOINT_VEL_END_IDX": joint_vel_end,
        "TOTAL_SIZE": joint_vel_end,
        "all_keys": root_keys + frame_pos_keys + frame_vel_keys + joint_pos_keys + joint_vel_keys,
        "combined_indices": list(range(root_linear_vel_start, root_linear_vel_end - 1))
        + list(range(root_angular_vel_start + 2, root_angular_vel_end))
        + list(range(joint_pos_start, joint_pos_end))
        + list(range(joint_vel_start, joint_vel_end))
        + list(range(frame_pos_start, frame_pos_end)),
    }


def go2_motion_files() -> tuple[str, ...]:
    return tuple(sorted(glob.glob(str(_DEFAULT_MOTION_DIR / "*"))))


def make_go2_amp_runner_cfg() -> AmpRunnerCfg:
    return AmpRunnerCfg(
        clip_actions=100.0,
        experiment_name="unitree_go2_flat_amp",
        wandb_tags=("go2", "amp", "flat", "mjlab"),
        policy=AmpPolicyCfg(
            num_actor_obs=45,
            num_critic_obs=48,
            num_actions=12,
            min_normalized_std=(0.01, 0.01, 0.01) * 4,
        ),
        algorithm=AmpAlgorithmCfg(),
        amp=AmpModuleCfg(
            motion_files=go2_motion_files(),
            observation_schema=go2_amp_observation_schema(),
        ),
    )


def _observations(play: bool = False) -> dict[str, ObservationGroupCfg]:
    command_scale = (2.0, 2.0, 0.25)
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            scale=0.25,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            scale=1.0,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": ROBOT_JOINT_CFG},
            scale=1.0,
            noise=Unoise(n_min=-0.03, n_max=0.03),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": ROBOT_JOINT_CFG},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands_scale,
            params={"command_name": COMMAND_NAME, "scale": command_scale},
        ),
    }
    critic_terms = {
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel, scale=2.0),
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, scale=0.25),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity, scale=1.0),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": ROBOT_JOINT_CFG},
            scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": ROBOT_JOINT_CFG},
            scale=0.05,
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands_scale,
            params={"command_name": COMMAND_NAME, "scale": command_scale},
        ),
    }
    amp_terms = {
        "base_lin_xy_vel": ObservationTermCfg(func=mdp.base_lin_xy_vel),
        "base_ang_yaw_vel": ObservationTermCfg(func=mdp.base_ang_yaw_vel),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": ROBOT_JOINT_CFG}),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": ROBOT_JOINT_CFG}),
        "foot_positions": ObservationTermCfg(
            func=mdp.foot_positions,
            params={"asset_cfg": ROBOT_FOOT_BODY_CFG},
        ),
    }
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
        "amp_obs": ObservationGroupCfg(
            terms=amp_terms,
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
                "asset_cfg": SceneEntityCfg("robot", body_names=GO2_BASE_LINK),
                "operation": "add",
                "ranges": (-1.0, 3.0),
            },
        ),
        "randomize_base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=GO2_BASE_LINK),
                "operation": "add",
                "ranges": {0: (-0.05, 0.05), 1: (-0.03, 0.03), 2: (-0.03, 0.05)},
            },
        ),
        "randomize_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=tuple(GO2_FOOT_GEOM_NAMES)),
                "operation": "abs",
                "ranges": (0.25, 1.2),
                "shared_random": True,
            },
        ),
        "reset_base": EventTermCfg(
            mode="reset",
            func=envs_mdp.reset_root_state_uniform,
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-1.57, 1.57)},
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
                "asset_cfg": SceneEntityCfg("robot", actuator_names=list(GO2_JOINT_ORDER)),
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
        events.pop("push_robot", None)
    return events


def _rewards() -> dict[str, RewardTermCfg]:
    joint_cfg = SceneEntityCfg(
        "robot",
        joint_names=tuple(GO2_JOINT_ORDER),
        actuator_names=list(GO2_JOINT_ORDER),
        preserve_order=True,
    )
    return {
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
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(GO2_JOINT_ORDER), preserve_order=True)},
        ),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        "action_smoothness_l2": RewardTermCfg(func=mdp.action_smoothness_l2, weight=-0.01),
        "joint_power": RewardTermCfg(
            func=mdp.joint_power,
            weight=-2.0e-5,
            params={"asset_cfg": joint_cfg},
        ),
        "joint_power_distribution": RewardTermCfg(
            func=mdp.joint_power_distribution,
            weight=-1.0e-5,
            params={"asset_cfg": joint_cfg},
        ),
        "undesired_contacts": RewardTermCfg(
            func=mdp.undesired_contacts,
            weight=-0.1,
            params={"sensor_name": "undesired_contact", "threshold": 0.1},
        ),
        "amp_reward": RewardTermCfg(func=mdp.amp_reward, weight=0.5),
    }


def unitree_go2_flat_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=GO2_FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    undesired_contact_cfg = ContactSensorCfg(
        name="undesired_contact",
        primary=ContactMatch(mode="subtree", pattern=GO2_THIGH_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_go2_robot_cfg()},
            sensors=(feet_ground_cfg, undesired_contact_cfg),
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
        rewards=_rewards(),
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            "bad_orientation": TerminationTermCfg(
                func=envs_mdp.bad_orientation,
                params={"limit_angle": 1.4},
            ),
        },
        curriculum={},
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
            nconmax=None,
            njmax=300,
            contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=20.0,
        is_finite_horizon=False,
        scale_rewards_by_dt=True,
        seed=42,
    )
