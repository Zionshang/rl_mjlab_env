"""Unitree Go2 AMPVAE tasks."""

from rsl_rl.runners import AMPVAEOnPolicyRunner

from mjlab.tasks.registry import register_mjlab_task

from rl_mjlab_env.envs import AmpManagerBasedRlEnv
from rl_mjlab_env.tasks.env_classes import register_env_class
from rl_mjlab_env.tasks.ampvae.config.unitree_go2_rough import (
    make_go2_ampvae_runner_cfg,
    unitree_go2_ampvae_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-AMPVAE-Rough-Unitree-Go2",
    env_cfg=unitree_go2_ampvae_env_cfg(play=False),
    play_env_cfg=unitree_go2_ampvae_env_cfg(play=True),
    rl_cfg=make_go2_ampvae_runner_cfg(),
    runner_cls=AMPVAEOnPolicyRunner,
)
register_env_class(task_id="Mjlab-AMPVAE-Rough-Unitree-Go2", env_cls=AmpManagerBasedRlEnv)
