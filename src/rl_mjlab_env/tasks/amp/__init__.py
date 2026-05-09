"""Unitree Go2 AMP tasks."""

from mjlab.tasks.registry import register_mjlab_task
from rsl_rl_local.runners import AMPOnPolicyRunner

from rl_mjlab_env.envs import AmpManagerBasedRlEnv
from rl_mjlab_env.tasks.amp.config.unitree_go2_flat import (
    make_go2_amp_runner_cfg,
    unitree_go2_flat_amp_env_cfg,
)
from rl_mjlab_env.tasks.env_classes import register_env_class

register_mjlab_task(
    task_id="Mjlab-AMP-Flat-Unitree-Go2",
    env_cfg=unitree_go2_flat_amp_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_amp_env_cfg(play=True),
    rl_cfg=make_go2_amp_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)
register_env_class(task_id="Mjlab-AMP-Flat-Unitree-Go2", env_cls=AmpManagerBasedRlEnv)
