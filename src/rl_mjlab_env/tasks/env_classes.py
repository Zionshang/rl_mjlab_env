"""Thin task-to-environment-class mapping for custom MJLab tasks."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv

_ENV_CLASSES: dict[str, type[ManagerBasedRlEnv]] = {}


def register_env_class(task_id: str, env_cls: type[ManagerBasedRlEnv]) -> None:
    if task_id in _ENV_CLASSES:
        raise ValueError(f"Task '{task_id}' already has an environment class")
    _ENV_CLASSES[task_id] = env_cls


def load_env_class(task_id: str) -> type[ManagerBasedRlEnv]:
    try:
        return _ENV_CLASSES[task_id]
    except KeyError as exc:
        raise KeyError(f"Task '{task_id}' does not define a custom environment class") from exc
