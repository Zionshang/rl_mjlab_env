"""Go2 + X5 Go2Arm tasks."""

from local_rsl_rl.runners import OnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from rl_mjlab_env.envs import Go2ArmManagerBasedRlEnv
from rl_mjlab_env.tasks.env_classes import register_env_class
from rl_mjlab_env.tasks.go2_x5.config import go2arm_env_cfg, make_go2arm_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Go2Arm-Flat-Go2-X5",
    env_cfg=go2arm_env_cfg(rough=False, play=False),
    play_env_cfg=go2arm_env_cfg(rough=False, play=True),
    rl_cfg=make_go2arm_runner_cfg(rough=False),
    runner_cls=OnPolicyRunner,
)
register_env_class(task_id="Mjlab-Go2Arm-Flat-Go2-X5", env_cls=Go2ArmManagerBasedRlEnv)

register_mjlab_task(
    task_id="Mjlab-Go2Arm-Rough-Go2-X5",
    env_cfg=go2arm_env_cfg(rough=True, play=False),
    play_env_cfg=go2arm_env_cfg(rough=True, play=True),
    rl_cfg=make_go2arm_runner_cfg(rough=True),
    runner_cls=OnPolicyRunner,
)
register_env_class(task_id="Mjlab-Go2Arm-Rough-Go2-X5", env_cls=Go2ArmManagerBasedRlEnv)
