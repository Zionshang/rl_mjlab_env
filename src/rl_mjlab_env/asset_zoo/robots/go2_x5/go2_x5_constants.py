"""Unitree Go2 + X5 arm constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_X5_XML: Path = Path(__file__).parent / "xmls" / "go2_x5.xml"
assert GO2_X5_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2_X5_XML))


##
# Names and ordering.
##

GO2_X5_BASE_LINK = "base_link"
GO2_X5_LEG_JOINT_ORDER = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)
GO2_X5_ARM_JOINT_ORDER = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
  "joint7",
  "joint8",
)
GO2_X5_ARM_CONTROL_JOINT_ORDER = GO2_X5_ARM_JOINT_ORDER[:6]
GO2_X5_JOINT_ORDER = GO2_X5_LEG_JOINT_ORDER + GO2_X5_ARM_JOINT_ORDER
GO2_X5_CONTROL_JOINT_ORDER = GO2_X5_LEG_JOINT_ORDER + GO2_X5_ARM_CONTROL_JOINT_ORDER
GO2_X5_THIGH_NAMES = ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")
GO2_X5_CALF_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
GO2_X5_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GO2_X5_FOOT_SITE_NAMES = ("FL", "FR", "RL", "RR")
GO2_X5_FOOT_GEOM_NAMES = (
  "FL_foot_collision",
  "FR_foot_collision",
  "RL_foot_collision",
  "RR_foot_collision",
)
GO2_X5_ARM_BODY_NAMES = (
  "arm_base_link",
  "link1",
  "link2",
  "link3",
  "link4",
  "link5",
  "link6",
  "link7",
  "link8",
)
GO2_X5_GRIPPER_JOINT_NAMES = ("joint7", "joint8")
GO2_X5_GRIPPER_TIP_GEOM_NAMES = ("link7_tip_collision", "link8_tip_collision")

##
# Actuator config.
##

GO2_X5_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_X5_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_X5_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45.0,
  armature=0.02,
)

# X5 arm gains and nominal joint targets are aligned with RoboDuet's zarx_j1~j8
# configuration. Effort limits are taken from RoboDuet's arx5go2 URDF.
X5_ARMATURE = 0.005

X5_ACTUATOR_JOINT1 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint1",),
  stiffness=40.0,
  damping=3.0,
  effort_limit=15.0,
  armature=X5_ARMATURE,
  frictionloss=0.3,
)
X5_ACTUATOR_JOINT23 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint2", "joint3"),
  stiffness=70.0,
  damping=15.0,
  effort_limit=15.0,
  armature=X5_ARMATURE,
  frictionloss=0.3,
)
X5_ACTUATOR_JOINT456 = BuiltinPositionActuatorCfg(
  target_names_expr=("joint4", "joint5", "joint6"),
  stiffness=25.0,
  damping=2.0,
  effort_limit=3.0,
  armature=X5_ARMATURE,
  frictionloss=0.3,
)
X5_ACTUATOR_GRIPPER = BuiltinPositionActuatorCfg(
  target_names_expr=GO2_X5_GRIPPER_JOINT_NAMES,
  stiffness=50.0,
  damping=20.0,
  effort_limit=3.0,
  armature=X5_ARMATURE,
  frictionloss=0.0,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
    "joint8": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# All named collision geoms, including the X5 arm, are matched by the full-collision
# regex. Feet keep their higher-priority contact settings via the foot-specific regex.
GO2_X5_FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

GO2_X5_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2_X5_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_X5_ACTUATOR_HIP,
    GO2_X5_ACTUATOR_THIGH,
    GO2_X5_ACTUATOR_CALF,
    X5_ACTUATOR_JOINT1,
    X5_ACTUATOR_JOINT23,
    X5_ACTUATOR_JOINT456,
    X5_ACTUATOR_GRIPPER,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_x5_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 + X5 robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(GO2_X5_FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_X5_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_x5_robot_cfg())
  robot.spec.worldbody.add_geom(
    name="debug_ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(4.0, 4.0, 1.0),
  )
  robot.spec.worldbody.add_light(
    name="debug_sun",
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  viewer.launch(robot.spec.compile())
