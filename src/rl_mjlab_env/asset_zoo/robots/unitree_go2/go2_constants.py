"""Unitree Go2 constants and MJCF entity config."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

GO2_BASE_LINK = "base"
GO2_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GO2_FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")
GO2_THIGH_NAMES = ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")
GO2_CALF_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
GO2_JOINT_NAMES = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)

GO2_XML = Path(__file__).parent / "xmls" / "go2" / "go2.xml"
assert GO2_XML.exists(), GO2_XML


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, GO2_XML.parent / "assets", meshdir)
    return assets


def get_go2_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    spec.assets = get_assets(spec.meshdir)
    # The source MJCF contains motor actuators and a keyframe. The AMP task uses
    # mjlab-built position actuators and its own initial state, so keep the model
    # geometry/sensors but rebuild actuators and keyframes through EntityCfg.
    while spec.actuators:
        spec.delete(spec.actuators[-1])
    while spec.keys:
        spec.delete(spec.keys[-1])
    return spec


GO2_ACTUATORS = tuple(
    BuiltinPositionActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=25.0,
        damping=0.5,
        effort_limit=23.5,
        armature=0.01,
        frictionloss=0.0,
    )
    for joint_name in GO2_JOINT_NAMES
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=GO2_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

GO2_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.335),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "FL_hip_joint": 0.068,
        "FL_thigh_joint": 0.785,
        "FL_calf_joint": -1.44,
        "FR_hip_joint": -0.068,
        "FR_thigh_joint": 0.785,
        "FR_calf_joint": -1.44,
        "RL_hip_joint": 0.068,
        "RL_thigh_joint": 0.785,
        "RL_calf_joint": -1.44,
        "RR_hip_joint": -0.068,
        "RR_thigh_joint": 0.785,
        "RR_calf_joint": -1.44,
    },
    joint_vel={".*": 0.0},
)


def get_go2_robot_cfg() -> EntityCfg:
    """Return a fresh Go2 entity config."""
    return EntityCfg(
        init_state=GO2_INIT_STATE,
        spec_fn=get_go2_spec,
        articulation=GO2_ARTICULATION,
    )


GO2_ACTION_SCALE = {joint_name: 0.25 for joint_name in GO2_JOINT_NAMES}
