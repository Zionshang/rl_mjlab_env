"""Play Go2 AMP tasks."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from rl_mjlab_env.rl import AmpVecEnvWrapper
from rl_mjlab_env.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_env_cls,
    load_rl_cfg,
    load_runner_cls,
)


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    num_envs: int | None = 16
    device: str | None = None
    viewer: Literal["native", "viser"] = "viser"


def run_play(task_id: str, cfg: PlayConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    env_cls = load_env_cls(task_id)
    if env_cls is None:
        raise RuntimeError(f"Task '{task_id}' does not define env_cls")
    env = AmpVecEnvWrapper(env_cls(cfg=env_cfg, device=device), clip_actions=100.0)

    if cfg.agent == "zero":
        policy = lambda obs: torch.zeros(env.num_envs, env.num_actions, device=device)
    elif cfg.agent == "random":
        policy = lambda obs: 2.0 * torch.rand(env.num_envs, env.num_actions, device=device) - 1.0
    else:
        if cfg.checkpoint_file is None:
            raise ValueError("checkpoint_file is required for trained play")
        runner_cls = load_runner_cls(task_id)
        if runner_cls is None:
            raise RuntimeError(f"Task '{task_id}' does not define runner_cls")
        runner = runner_cls(env, asdict(load_rl_cfg(task_id)), device=device)
        runner.load(str(Path(cfg.checkpoint_file)))

        def policy(obs):
            return runner.alg.actor_critic.act_inference(obs["actor_obs"].to(device))

    if cfg.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()
    env.close()


def main() -> None:
    import rl_mjlab_env.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    )
    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
