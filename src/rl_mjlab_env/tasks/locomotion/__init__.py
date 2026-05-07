"""Unitree Go2 locomotion tasks."""

from rsl_rl.runners import LocomotionOnPolicyRunner

from mjlab.tasks.registry import register_mjlab_task

from rl_mjlab_env.envs import AmpManagerBasedRlEnv
from rl_mjlab_env.tasks.env_classes import register_env_class
from rl_mjlab_env.tasks.locomotion.config.unitree_go2_rough import (
    make_go2_locomotion_runner_cfg,
    unitree_go2_locomotion_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Locomotion-Rough-Unitree-Go2",
    env_cfg=unitree_go2_locomotion_env_cfg(play=False),
    play_env_cfg=unitree_go2_locomotion_env_cfg(play=True),
    rl_cfg=make_go2_locomotion_runner_cfg(),
    runner_cls=LocomotionOnPolicyRunner,
)
register_env_class(task_id="Mjlab-Locomotion-Rough-Unitree-Go2", env_cls=AmpManagerBasedRlEnv)
