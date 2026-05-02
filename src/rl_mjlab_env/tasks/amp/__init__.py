"""Unitree Go2 AMP tasks."""

from mjlab.tasks.registry import register_mjlab_task
from rsl_rl.runners import AMPOnPolicyRunner

from rl_mjlab_env.envs import AmpManagerBasedRlEnv
from rl_mjlab_env.tasks.amp.config.unitree_go2_flat import (
    make_go2_amp_runner_cfg,
    unitree_go2_flat_amp_env_cfg,
)
from rl_mjlab_env.tasks.env_classes import register_env_class

TASK_ID = "Mjlab-AMP-Flat-Unitree-Go2"

register_mjlab_task(
    TASK_ID,
    env_cfg=unitree_go2_flat_amp_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_amp_env_cfg(play=True),
    rl_cfg=make_go2_amp_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)
register_env_class(TASK_ID, AmpManagerBasedRlEnv)
