"""Configuration package for Go2 + X5 tasks."""

from rl_mjlab_env.tasks.go2_x5.config.go2arm_env_cfg import (
    go2arm_env_cfg,
    make_go2arm_runner_cfg,
)

__all__ = ["go2arm_env_cfg", "make_go2arm_runner_cfg"]
