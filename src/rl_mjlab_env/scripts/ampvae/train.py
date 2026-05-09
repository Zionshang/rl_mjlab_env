"""Train AMPVAE tasks with the vendored RSL-RL AMPVAE runner."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scripts.train import TrainConfig as MjlabTrainConfig
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder

from rl_mjlab_env.rl import AmpVecEnvWrapper, AmpvaeRunnerCfg
from rl_mjlab_env.tasks.env_classes import load_env_class


def list_ampvae_tasks() -> list[str]:
    task_ids = []
    for task_id in list_tasks():
        if isinstance(load_rl_cfg(task_id), AmpvaeRunnerCfg):
            task_ids.append(task_id)
    return sorted(task_ids)


@dataclass(frozen=True)
class TrainConfig(MjlabTrainConfig):
    agent: AmpvaeRunnerCfg

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
        assert isinstance(agent_cfg, AmpvaeRunnerCfg)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    env_cfg = cfg.env
    env_cfg.seed = seed

    if cfg.enable_nan_guard:
        env_cfg.sim.nan_guard.enabled = True

    env_cls = load_env_class(task_id)
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        raise RuntimeError(f"Task '{task_id}' does not define runner_cls")

    train_video_dir = Path(log_dir) / "videos" / "train"
    env = env_cls(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if (cfg.video and rank == 0) else None,
    )

    resume_path: Path | None = None
    if cfg.agent.resume:
        if cfg.wandb_run_path is not None:
            resume_path, _ = get_wandb_checkpoint_path(
                log_dir.parent, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
            )
        else:
            resume_path = get_checkpoint_path(
                log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint
            )

    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=train_video_dir,
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = AmpVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = runner_cls(env, asdict(cfg.agent), str(log_dir), device)
    runner.video_dir = str(train_video_dir)
    runner.video_upload_enabled = rank == 0 and cfg.video and cfg.agent.logger == "wandb"
    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        runner.load(str(resume_path))

    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
        dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations,
        init_at_random_ep_len=True,
    )
    env.close()
    if rank == 0:
        runner.log_new_videos()
        if runner.writer is not None:
            runner.writer.stop()


def launch_training(task_id: str, cfg: TrainConfig | None = None):
    cfg = cfg or TrainConfig.from_task(task_id)

    log_root_path = Path("logs") / "rsl_rl_local" / cfg.agent.experiment_name
    log_root_path.resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.agent.run_name:
        log_dir_name += f"_{cfg.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        run_train(task_id, cfg, log_dir)
    else:
        import torchrunx

        logging.basicConfig(level=logging.INFO)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, cfg, log_dir)


def main() -> None:
    import rl_mjlab_env.tasks  # noqa: F401

    all_tasks = list_ampvae_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    del remaining_args
    launch_training(chosen_task, args)


if __name__ == "__main__":
    main()
