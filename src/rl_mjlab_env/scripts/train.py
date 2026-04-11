"""Train Go2 AMP tasks with the vendored rl_sim_env RSL-RL runner."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

from rl_mjlab_env.rl import AmpVecEnvWrapper, RslRlAmpOnPolicyRunnerCfg
from rl_mjlab_env.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_env_cls,
    load_rl_cfg,
    load_runner_cls,
)
from rl_mjlab_env.utils import AmpVideoRecorder


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlAmpOnPolicyRunnerCfg
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    upload_video_to_wandb: bool = True
    enable_nan_guard: bool = False
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, RslRlAmpOnPolicyRunnerCfg)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    configure_torch_backends()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if cuda_visible == "" else "cuda:0"
    seed = cfg.agent.seed
    env_cfg = cfg.env
    env_cfg.seed = seed
    if cfg.enable_nan_guard:
        env_cfg.sim.nan_guard.enabled = True
    print(f"[INFO] Training with: device={device}, seed={seed}, num_envs={env_cfg.scene.num_envs}")

    env_cls = load_env_cls(task_id)
    runner_cls = load_runner_cls(task_id)
    if env_cls is None or runner_cls is None:
        raise RuntimeError(f"Task '{task_id}' does not define env_cls/runner_cls")

    env = env_cls(cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None)
    if cfg.video:
        env = AmpVideoRecorder(
            env,
            video_folder=log_dir / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            upload_to_wandb=cfg.upload_video_to_wandb and cfg.agent.logger == "wandb",
            wandb_step_fn=lambda step: step // cfg.agent.num_steps_per_env,
        )
        print("[INFO] Recording videos during training.")
    env = AmpVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    runner = runner_cls(env, asdict(cfg.agent), str(log_dir), device)
    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)

    resume_path: Path | None = None
    if cfg.agent.resume:
        resume_path = get_checkpoint_path(
            log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
    runner.learn(cfg.agent.max_iterations, init_at_random_ep_len=True)
    env.close()


def launch_training(task_id: str, cfg: TrainConfig) -> None:
    log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.agent.run_name:
        log_dir_name += f"_{cfg.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
    if num_gpus > 1:
        raise NotImplementedError("This first AMP migration only wires single-process training.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"
    run_train(task_id, cfg, log_dir)


def main() -> None:
    import rl_mjlab_env.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    )
    launch_training(chosen_task, args)


if __name__ == "__main__":
    main()
