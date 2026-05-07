"""Unitree Go2 constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_XML: Path = Path(__file__).parent / "xmls" / "go2" / "go2.xml"
assert GO2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  asset_root = (GO2_XML.parent / meshdir).resolve()
  for asset_path in asset_root.rglob("*"):
    if asset_path.is_file():
      rel_path = asset_path.relative_to(asset_root).as_posix()
      assets[rel_path] = asset_path.read_bytes()
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_go2_spec() -> mujoco.MjSpec:
  return get_spec()


##
# Names.
##

GO2_BASE_LINK = "base_link"

GO2_HIP_JOINT_NAMES = (
  "FL_hip_joint",
  "FR_hip_joint",
  "RL_hip_joint",
  "RR_hip_joint",
)
GO2_THIGH_JOINT_NAMES = (
  "FL_thigh_joint",
  "FR_thigh_joint",
  "RL_thigh_joint",
  "RR_thigh_joint",
)
GO2_CALF_JOINT_NAMES = (
  "FL_calf_joint",
  "FR_calf_joint",
  "RL_calf_joint",
  "RR_calf_joint",
)
GO2_JOINT_NAMES = GO2_HIP_JOINT_NAMES + GO2_THIGH_JOINT_NAMES + GO2_CALF_JOINT_NAMES

GO2_THIGH_NAMES = (
  "FL_thigh",
  "FR_thigh",
  "RL_thigh",
  "RR_thigh",
)
GO2_CALF_NAMES = (
  "FL_calf",
  "FR_calf",
  "RL_calf",
  "RR_calf",
)

GO2_FOOT_NAMES = (
  "FL_foot",
  "FR_foot",
  "RL_foot",
  "RR_foot",
)
GO2_FOOT_SITE_NAMES = (
  "FL",
  "FR",
  "RL",
  "RR",
)
GO2_FOOT_GEOM_NAMES = (
  "FL_foot_collision",
  "FR_foot_collision",
  "RL_foot_collision",
  "RR_foot_collision",
)

GO2_BASE_COLLISION_GEOM_NAMES = (
  "base1_collision",
  "base2_collision",
  "base3_collision",
)
GO2_HIP_COLLISION_GEOM_NAMES = (
  "FL_hip_collision",
  "FR_hip_collision",
  "RL_hip_collision",
  "RR_hip_collision",
)
GO2_THIGH_COLLISION_GEOM_NAMES = (
  "FL_thigh_collision",
  "FR_thigh_collision",
  "RL_thigh_collision",
  "RR_thigh_collision",
)
GO2_CALF_COLLISION_GEOM_NAMES = (
  "FL_calf1_collision",
  "FL_calf2_collision",
  "FR_calf1_collision",
  "FR_calf2_collision",
  "RL_calf1_collision",
  "RL_calf2_collision",
  "RR_calf1_collision",
  "RR_calf2_collision",
)
GO2_COLLISION_GEOM_NAMES = (
  GO2_BASE_COLLISION_GEOM_NAMES
  + GO2_HIP_COLLISION_GEOM_NAMES
  + GO2_THIGH_COLLISION_GEOM_NAMES
  + GO2_CALF_COLLISION_GEOM_NAMES
  + GO2_FOOT_GEOM_NAMES
)

##
# Actuator config.
##

GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=GO2_HIP_JOINT_NAMES,
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=GO2_THIGH_JOINT_NAMES,
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=GO2_CALF_JOINT_NAMES,
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    "FL_hip_joint": -0.1,
    "FR_hip_joint": 0.1,
    "RL_hip_joint": -0.1,
    "RR_hip_joint": 0.1,
    "FL_thigh_joint": 0.9,
    "FR_thigh_joint": 0.9,
    "RL_thigh_joint": 0.9,
    "RR_thigh_joint": 0.9,
    "FL_calf_joint": -1.8,
    "FR_calf_joint": -1.8,
    "RL_calf_joint": -1.8,
    "RR_calf_joint": -1.8,
  },
  joint_vel={
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.0,
    "FR_thigh_joint": 0.0,
    "RL_thigh_joint": 0.0,
    "RR_thigh_joint": 0.0,
    "FL_calf_joint": 0.0,
    "FR_calf_joint": 0.0,
    "RL_calf_joint": 0.0,
    "RR_calf_joint": 0.0,
  },
)
GO2_INIT_STATE = INIT_STATE

##
# Collision config.
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=GO2_FOOT_GEOM_NAMES,
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=GO2_COLLISION_GEOM_NAMES,
  condim={
    "base1_collision": 1,
    "base2_collision": 1,
    "base3_collision": 1,
    "FL_hip_collision": 1,
    "FR_hip_collision": 1,
    "RL_hip_collision": 1,
    "RR_hip_collision": 1,
    "FL_thigh_collision": 1,
    "FR_thigh_collision": 1,
    "RL_thigh_collision": 1,
    "RR_thigh_collision": 1,
    "FL_calf1_collision": 1,
    "FL_calf2_collision": 1,
    "FR_calf1_collision": 1,
    "FR_calf2_collision": 1,
    "RL_calf1_collision": 1,
    "RL_calf2_collision": 1,
    "RR_calf1_collision": 1,
    "RR_calf2_collision": 1,
    "FL_foot_collision": 3,
    "FR_foot_collision": 3,
    "RL_foot_collision": 3,
    "RR_foot_collision": 3,
  },
  priority={
    "FL_foot_collision": 1,
    "FR_foot_collision": 1,
    "RL_foot_collision": 1,
    "RR_foot_collision": 1,
  },
  friction={
    "FL_foot_collision": (0.6,),
    "FR_foot_collision": (0.6,),
    "RL_foot_collision": (0.6,),
    "RR_foot_collision": (0.6,),
  },
  solimp={
    "FL_foot_collision": (0.9, 0.95, 0.023),
    "FR_foot_collision": (0.9, 0.95, 0.023),
    "RL_foot_collision": (0.9, 0.95, 0.023),
    "RR_foot_collision": (0.9, 0.95, 0.023),
  },
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 robot configuration instance."""
  return EntityCfg(
    init_state=GO2_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_go2_spec,
    articulation=GO2_ARTICULATION,
  )


GO2_ACTION_SCALE = {
  "FL_hip_joint": 0.25 * 23.5 / 20.0,
  "FR_hip_joint": 0.25 * 23.5 / 20.0,
  "RL_hip_joint": 0.25 * 23.5 / 20.0,
  "RR_hip_joint": 0.25 * 23.5 / 20.0,
  "FL_thigh_joint": 0.25 * 23.5 / 20.0,
  "FR_thigh_joint": 0.25 * 23.5 / 20.0,
  "RL_thigh_joint": 0.25 * 23.5 / 20.0,
  "RR_thigh_joint": 0.25 * 23.5 / 20.0,
  "FL_calf_joint": 0.25 * 45 / 40.0,
  "FR_calf_joint": 0.25 * 45 / 40.0,
  "RL_calf_joint": 0.25 * 45 / 40.0,
  "RR_calf_joint": 0.25 * 45 / 40.0,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())
  viewer.launch(robot.spec.compile())
