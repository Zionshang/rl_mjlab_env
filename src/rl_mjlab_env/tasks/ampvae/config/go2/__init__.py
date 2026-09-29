"""Unitree Go2 VAE+AMP task registration."""

from mjlab.tasks.registry import register_mjlab_task
from rsl_rl_local.runners import VaeAmpOnPolicyRunner

from rl_mjlab_env.envs import VaeAmpRlEnv
from rl_mjlab_env.tasks.env_classes import register_env_class

from .env_cfgs import unitree_go2_ampvae_env_cfg
from .rl_cfg import unitree_go2_ampvae_runner_cfg

TASK_ID = "Mjlab-AMPVAE-Rough-Unitree-Go2"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_go2_ampvae_env_cfg(),
  play_env_cfg=unitree_go2_ampvae_env_cfg(play=True),
  rl_cfg=unitree_go2_ampvae_runner_cfg(),
  runner_cls=VaeAmpOnPolicyRunner,
)
register_env_class(task_id=TASK_ID, env_cls=VaeAmpRlEnv)
