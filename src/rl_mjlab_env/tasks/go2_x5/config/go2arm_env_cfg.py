"""Go2 + X5 Go2Arm task configuration."""

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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from rl_mjlab_env.asset_zoo.robots.go2_x5.go2_x5_constants import (
    GO2_X5_ARM_CONTROL_JOINT_ORDER,
    GO2_X5_BASE_LINK,
    GO2_X5_CALF_GEOM_NAMES,
    GO2_X5_CALF_NAMES,
    GO2_X5_CONTROL_JOINT_ORDER,
    GO2_X5_FOOT_GEOM_NAMES,
    GO2_X5_FOOT_SITE_NAMES,
    GO2_X5_GRIPPER_TIP_GEOM_NAMES,
    GO2_X5_LEG_JOINT_ORDER,
    GO2_X5_THIGH_NAMES,
    get_go2_x5_robot_cfg,
)
from rl_mjlab_env.rl import Go2ArmRunnerCfg
from rl_mjlab_env.tasks.go2_x5 import mdp

BASE_COMMAND = "base_velocity"
EE_COMMAND = "ee_pose"
EE_BODY = "link6"

LEG_JOINT_ORDER = GO2_X5_LEG_JOINT_ORDER
ARM_CONTROL_JOINT_ORDER = GO2_X5_ARM_CONTROL_JOINT_ORDER
CONTROL_JOINT_ORDER = GO2_X5_CONTROL_JOINT_ORDER

ROBOT_CONTROL_JOINT_CFG = SceneEntityCfg("robot", joint_names=CONTROL_JOINT_ORDER, preserve_order=True)
ROBOT_CONTROL_ACTUATOR_CFG = SceneEntityCfg(
    "robot",
    joint_names=CONTROL_JOINT_ORDER,
    actuator_names=list(CONTROL_JOINT_ORDER),
    preserve_order=True,
)
ROBOT_LEG_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=LEG_JOINT_ORDER,
    actuator_names=list(LEG_JOINT_ORDER),
    preserve_order=True,
)
ROBOT_ARM_ACTUATOR_CFG = SceneEntityCfg(
    "robot",
    joint_names=ARM_CONTROL_JOINT_ORDER,
    actuator_names=list(ARM_CONTROL_JOINT_ORDER),
    preserve_order=True,
)
ROBOT_PD_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_ids=[0, 1, 2, 3, 4, 5])
ROBOT_BASE_BODY_CFG = SceneEntityCfg("robot", body_names=GO2_X5_BASE_LINK)
ROBOT_EE_BODY_CFG = SceneEntityCfg("robot", body_names=EE_BODY)
ROBOT_FOOT_BODY_CFG = SceneEntityCfg("robot", site_names=GO2_X5_FOOT_SITE_NAMES, preserve_order=True)


def make_go2arm_runner_cfg(rough: bool = False) -> Go2ArmRunnerCfg:
    cfg = Go2ArmRunnerCfg()
    if rough:
        cfg.experiment_name = "unitree_go2_x5_rough"
        cfg.max_iterations = 10000
        cfg.save_interval = 500
        cfg.algorithm["priv_reg_coef_schedual"] = [0, 0.1, 1500, 4000]
        cfg.algorithm["mixing_schedule"] = [1.0, 0, 3000]
    return cfg


def _observations(play: bool) -> dict[str, ObservationGroupCfg]:
    joint_noise = None if play else Unoise(n_min=-0.01, n_max=0.01)
    joint_vel_noise = None if play else Unoise(n_min=-0.5, n_max=0.5)
    gravity_noise = None if play else Unoise(n_min=-0.1, n_max=0.1)
    h = 10
    terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            history_length=h,
            noise=None,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": ROBOT_CONTROL_JOINT_CFG},
            history_length=h,
            noise=joint_noise,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": ROBOT_CONTROL_JOINT_CFG},
            history_length=h,
            noise=joint_vel_noise,
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action, history_length=h),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": BASE_COMMAND},
            history_length=h,
        ),
        "Go2_pose_command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": EE_COMMAND},
            history_length=h,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            history_length=h,
            noise=gravity_noise,
        ),
        "priv_mass_base": ObservationTermCfg(
            func=mdp.get_mass_base,
            params={"asset_cfg": ROBOT_BASE_BODY_CFG},
        ),
        "priv_mass_ee": ObservationTermCfg(
            func=mdp.get_mass_ee,
            params={"asset_cfg": ROBOT_EE_BODY_CFG},
        ),
        "priv_joint_torques": ObservationTermCfg(
            func=mdp.get_joints_torques,
            params={"asset_cfg": ROBOT_CONTROL_ACTUATOR_CFG},
        ),
        "priv_base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "priv_feet_contact": ObservationTermCfg(
            func=mdp.feet_contact,
            params={"sensor_name": "feet_contact"},
        ),
    }
    return {
        "policy": ObservationGroupCfg(
            terms=terms,
            concatenate_terms=True,
            enable_corruption=not play,
        )
    }


def _commands(rough: bool, play: bool) -> dict:
    if rough:
        vel_init = mdp.CurriculumVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        )
        vel_final = mdp.CurriculumVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        )
        pose_init = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.5),
            pos_y=(-0.05, 0.05),
            pos_z=(0.35, 0.4),
            roll=(0.0, 0.0),
            pitch=(-math.pi / 9, math.pi / 9),
            yaw=(-math.pi / 9, math.pi / 9),
        )
        pose_final = pose_init
        vel_coeff = 1000
        pose_coeff = 1000
    else:
        vel_init = mdp.CurriculumVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        )
        vel_final = mdp.CurriculumVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
        )
        pose_init = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.40),
            pos_y=(-0.05, 0.05),
            pos_z=(0.45, 0.5),
            roll=(0.0, 0.0),
            pitch=(-math.pi / 9, math.pi / 9),
            yaw=(-math.pi / 9, math.pi / 9),
        )
        pose_final = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.65),
            pos_y=(-0.35, 0.35),
            pos_z=(0.15, 0.6),
            roll=(0.0, 0.0),
            pitch=(-math.pi / 9, math.pi / 9),
            yaw=(-math.pi / 9, math.pi / 9),
        )
        vel_coeff = 4000
        pose_coeff = 5000

    pose_ranges = pose_final if play else pose_init
    vel_ranges = vel_final if play else vel_init
    return {
        EE_COMMAND: mdp.UniformPoseCommandCfg(
            entity_name="robot",
            body_name=EE_BODY,
            resampling_time_range=(4.0, 4.0) if play else (6.0, 8.0),
            debug_vis=True,
            is_go2arm=not play,
            is_go2arm_play=play,
            curriculum_coeff=pose_coeff,
            ranges=pose_ranges,
            ranges_init=pose_init,
            ranges_final=pose_final,
        ),
        BASE_COMMAND: mdp.CurriculumVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(5.0, 5.0) if play else (10.0, 10.0),
            rel_standing_envs=0.1,
            heading_command=False,
            debug_vis=True,
            curriculum_coeff=vel_coeff,
            ranges=vel_ranges,
            ranges_init=None if play else vel_init,
            ranges_final=vel_final,
            viz=mdp.CurriculumVelocityCommandCfg.VizCfg(z_offset=0.75, scale=1.0),
        ),
    }


def _events(rough: bool, play: bool) -> dict[str, EventTermCfg]:
    events = {
        "randomize_base_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": ROBOT_BASE_BODY_CFG,
                "operation": "add",
                "ranges": (-3.0, 3.0),
            },
        ),
        "randomize_ee_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": ROBOT_EE_BODY_CFG,
                "operation": "add",
                "ranges": (-0.1, 0.5),
            },
        ),
        "randomize_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=tuple(GO2_X5_FOOT_GEOM_NAMES)),
                "operation": "abs",
                "ranges": (0.5, 2.0),
                "shared_random": True,
            },
        ),
        "reset_base": EventTermCfg(
            mode="reset",
            func=envs_mdp.reset_root_state_uniform,
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            mode="reset",
            func=mdp.reset_joints_by_scale,
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": ROBOT_CONTROL_JOINT_CFG,
            },
        ),
        "reset_actuator_gains": EventTermCfg(
            mode="reset",
            func=dr.pd_gains,
            params={
                "asset_cfg": ROBOT_PD_ACTUATOR_CFG,
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
                "operation": "scale",
            },
        ),
        "push_robot": EventTermCfg(
            mode="interval",
            interval_range_s=(10.0, 15.0),
            func=envs_mdp.push_by_setting_velocity,
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
    }
    if play or not rough:
        events.pop("push_robot", None)
    if play:
        for key in (
            "randomize_base_mass",
            "randomize_ee_mass",
            "randomize_friction",
            "reset_actuator_gains",
        ):
            events.pop(key, None)
    return events


def _rewards(rough: bool) -> dict[str, RewardTermCfg]:
    rewards = {
        "end_effector_position_tracking": RewardTermCfg(
            func=mdp.position_command_error_exp,
            weight=2.5,
            params={"asset_cfg": ROBOT_EE_BODY_CFG, "command_name": EE_COMMAND, "std": 0.2},
        ),
        "end_effector_orientation_tracking": RewardTermCfg(
            func=mdp.orientation_command_error,
            weight=-1.5,
            params={"asset_cfg": ROBOT_EE_BODY_CFG, "command_name": EE_COMMAND},
        ),
        "end_effector_action_rate": RewardTermCfg(func=mdp.action_rate_l2_arm, weight=-0.005),
        "end_effector_action_smoothness": RewardTermCfg(func=mdp.arm_action_smoothness_penalty, weight=-0.02),
        "tracking_lin_vel_x_l1": RewardTermCfg(
            func=mdp.track_lin_vel_xy_exp,
            weight=1.5,
            params={"command_name": BASE_COMMAND, "std": 0.2},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=mdp.track_ang_vel_z_exp,
            weight=1.5,
            params={"command_name": BASE_COMMAND, "std": math.sqrt(0.2)},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.5),
        "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.02),
        "dof_torques_l2": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-2.0e-5,
            params={"asset_cfg": ROBOT_LEG_JOINT_CFG},
        ),
        "dof_acc_l2": RewardTermCfg(
            func=envs_mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": ROBOT_LEG_JOINT_CFG},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2_leg, weight=-0.01),
        "feet_air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.5,
            params={"sensor_name": "feet_contact", "command_name": BASE_COMMAND, "threshold": 0.5},
        ),
        "F_feet_air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.5,
            params={"sensor_name": "front_feet_contact", "command_name": BASE_COMMAND, "threshold": 0.5},
        ),
        "R_feet_air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=2.0,
            params={"sensor_name": "rear_feet_contact", "command_name": BASE_COMMAND, "threshold": 0.5},
        ),
        "feet_height": RewardTermCfg(
            func=mdp.feet_height,
            weight=0.0,
            params={
                "asset_cfg": ROBOT_FOOT_BODY_CFG,
                "tanh_mult": 2.0,
                "target_height": 0.08,
                "command_name": BASE_COMMAND,
            },
        ),
        "feet_height_body": RewardTermCfg(
            func=mdp.feet_height_body,
            weight=0.0,
            params={
                "asset_cfg": ROBOT_FOOT_BODY_CFG,
                "tanh_mult": 2.0,
                "target_height": -0.2,
                "command_name": BASE_COMMAND,
            },
        ),
        "foot_contact": RewardTermCfg(
            func=mdp.standing_feet_contact_force,
            weight=0.003,
            params={
                "sensor_name": "rear_feet_contact",
                "command_name": BASE_COMMAND,
                "force_threshold": 7.5,
                "command_threshold": 0.1,
            },
        ),
        "hip_deviation": RewardTermCfg(
            func=mdp.joint_deviation_l1,
            weight=-0.4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
        ),
        "joint_deviation": RewardTermCfg(
            func=mdp.joint_deviation_l1,
            weight=-0.04,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
        ),
        "action_smoothness": RewardTermCfg(func=mdp.leg_action_smoothness_penalty, weight=-0.02),
        "height_reward": RewardTermCfg(
            func=mdp.base_height_l2,
            weight=-2.0,
            params={"target_height": 0.3},
        ),
        "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
    }
    if not rough:
        rewards["end_effector_position_tracking"].weight = 3.0
        rewards["end_effector_orientation_tracking"].weight = -2.0
        rewards["end_effector_action_rate"].weight = -0.01
        rewards["end_effector_action_smoothness"].weight = -0.04
        # rewards["end_effector_position_tracking"].weight = 0.0
        # rewards["end_effector_orientation_tracking"].weight = 0.0
        # rewards["end_effector_action_rate"].weight = 0.0
        # rewards["end_effector_action_smoothness"].weight = 0.0
        rewards["tracking_lin_vel_x_l1"].weight = 3.5
        rewards["track_ang_vel_z_exp"].weight = 2.0
        rewards["ang_vel_xy_l2"].weight = -0.05
        rewards["feet_air_time"].weight = 0.0
        rewards["F_feet_air_time"].weight = 1.0
        rewards["R_feet_air_time"].weight = 1.0
        rewards["feet_height"].weight = -0.0
        rewards["feet_height_body"].weight = -3.0
        rewards["hip_deviation"].weight = -0.2
        rewards["joint_deviation"].weight = -0.01
    return rewards


def _contact_sensors() -> tuple[ContactSensorCfg, ...]:
    return (
        ContactSensorCfg(
            name="feet_contact",
            primary=ContactMatch(mode="geom", pattern=GO2_X5_FOOT_GEOM_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="front_feet_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=("FR_foot_collision", "FL_foot_collision"),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="rear_feet_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=("RR_foot_collision", "RL_foot_collision"),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="base_contact",
            primary=ContactMatch(mode="body", pattern=GO2_X5_BASE_LINK, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="thigh_contact",
            primary=ContactMatch(mode="body", pattern=GO2_X5_THIGH_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="calf_contact",
            primary=ContactMatch(mode="geom", pattern=GO2_X5_CALF_GEOM_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="arm_contact",
            primary=ContactMatch(
                mode="body",
                pattern=("arm_base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="gripper_contact",
            primary=ContactMatch(mode="geom", pattern=GO2_X5_GRIPPER_TIP_GEOM_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
    )


def go2arm_env_cfg(rough: bool = False, play: bool = False) -> ManagerBasedRlEnvCfg:
    terrain = (
        TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(ROUGH_TERRAINS_CFG),
            max_init_terrain_level=5,
        )
        if rough
        else TerrainEntityCfg(terrain_type="plane")
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=50 if play else 4096,
            env_spacing=2.5,
            terrain=terrain,
            entities={"robot": get_go2_x5_robot_cfg()},
            sensors=_contact_sensors(),
        ),
        observations=_observations(play=play),
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=list(LEG_JOINT_ORDER),
                scale={name: 0.25 for name in LEG_JOINT_ORDER},
                use_default_offset=True,
                preserve_order=True,
            ),
            "arm_pose": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=list(ARM_CONTROL_JOINT_ORDER),
                scale={name: 0.5 for name in ARM_CONTROL_JOINT_ORDER},
                use_default_offset=True,
                preserve_order=True,
            ),
        },
        commands=_commands(rough=rough, play=play),
        events=_events(rough=rough, play=play),
        rewards=_rewards(rough=rough),
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            "base_contact": TerminationTermCfg(
                func=mdp.illegal_contact,
                params={"sensor_name": "base_contact", "threshold": 0.5},
            ),
            "thigh_contact": TerminationTermCfg(
                func=mdp.illegal_contact,
                params={"sensor_name": "thigh_contact", "threshold": 0.5},
            ),
            "calf_contact": TerminationTermCfg(
                func=mdp.illegal_contact,
                params={"sensor_name": "calf_contact", "threshold": 0.5},
            ),
            "arm_contact": TerminationTermCfg(
                func=mdp.illegal_contact,
                params={"sensor_name": "arm_contact", "threshold": 0.5},
            ),
        },
        curriculum={
            "flat_ori_modify": CurriculumTermCfg(
                func=mdp.modify_reward_weight,
                params={"term_name": "flat_orientation_l2", "num_steps": 2000, "weight": -0.0},
            ),
            "flat_height_modify": CurriculumTermCfg(
                func=mdp.modify_reward_weight,
                params={"term_name": "height_reward", "num_steps": 4000, "weight": -1.0},
            ),
        },
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=GO2_X5_BASE_LINK,
            distance=3.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=64,
            njmax=1800,
            contact_sensor_maxmatch=512,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=20.0,
        is_finite_horizon=False,
        scale_rewards_by_dt=True,
        seed=42,
    )
    if rough and play and cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5
    return cfg
