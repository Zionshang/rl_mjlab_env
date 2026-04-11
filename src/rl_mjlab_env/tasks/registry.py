"""Small local task registry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg


@dataclass
class _TaskCfg:
    env_cfg: ManagerBasedRlEnvCfg
    play_env_cfg: ManagerBasedRlEnvCfg
    rl_cfg: Any
    runner_cls: type | None
    env_cls: type | None


_REGISTRY: dict[str, _TaskCfg] = {}


def register_mjlab_task(
    task_id: str,
    env_cfg: ManagerBasedRlEnvCfg,
    play_env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: Any,
    runner_cls: type | None = None,
    env_cls: type | None = None,
) -> None:
    if task_id in _REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered")
    _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls, env_cls)


def list_tasks() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _REGISTRY[task_name].play_env_cfg if play else _REGISTRY[task_name].env_cfg
    return deepcopy(cfg)


def load_rl_cfg(task_name: str) -> Any:
    return deepcopy(_REGISTRY[task_name].rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
    return _REGISTRY[task_name].runner_cls


def load_env_cls(task_name: str) -> type | None:
    return _REGISTRY[task_name].env_cls
