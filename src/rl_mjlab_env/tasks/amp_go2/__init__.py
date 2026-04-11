"""Unitree Go2 AMP tasks."""

from rsl_rl.runners import AMPOnPolicyRunner

from rl_mjlab_env.envs import Go2AmpManagerBasedRlEnv
from rl_mjlab_env.rl import RslRlAmpOnPolicyRunnerCfg
from rl_mjlab_env.tasks.amp_go2.config.unitree_go2_flat import (
    unitree_go2_flat_amp_env_cfg,
)
from rl_mjlab_env.tasks.registry import register_mjlab_task

TASK_ID = "Mjlab-AMP-Flat-Unitree-Go2"

register_mjlab_task(
    TASK_ID,
    env_cfg=unitree_go2_flat_amp_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_amp_env_cfg(play=True),
    rl_cfg=RslRlAmpOnPolicyRunnerCfg(),
    runner_cls=AMPOnPolicyRunner,
    env_cls=Go2AmpManagerBasedRlEnv,
)

