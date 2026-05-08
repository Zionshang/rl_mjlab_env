"""Play AMPVAE tasks."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import torch
import tyro
from mjlab.scripts.play import PlayConfig as MjlabPlayConfig
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from rl_mjlab_env.rl import AmpVecEnvWrapper, AmpvaeRunnerCfg
from rl_mjlab_env.tasks.env_classes import load_env_class


def list_ampvae_tasks() -> list[str]:
    task_ids = []
    for task_id in list_tasks():
        if isinstance(load_rl_cfg(task_id), AmpvaeRunnerCfg):
            task_ids.append(task_id)
    return sorted(task_ids)


@dataclass(frozen=True)
class PlayConfig(MjlabPlayConfig):
    pass


def _patch_zero_command_ranges_for_viser(env_cfg) -> None:
    for command_cfg in env_cfg.commands.values():
        if not isinstance(command_cfg, UniformVelocityCommandCfg):
            continue
        ranges = command_cfg.ranges
        if ranges.lin_vel_y == (0.0, 0.0):
            ranges.lin_vel_y = (-0.1, 0.1)
        if ranges.ang_vel_z == (0.0, 0.0):
            ranges.ang_vel_z = (-0.1, 0.1)


def run_play(task_id: str, cfg: PlayConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, AmpvaeRunnerCfg)

    dummy_mode = cfg.agent in {"zero", "random"}
    trained_mode = not dummy_mode

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if trained_mode:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError(
                    "`wandb_run_path` is required when `checkpoint_file` is not provided."
                )
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
            )
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "cached" if was_cached else "downloaded"
            print(
                f"[INFO]: Loading checkpoint: {checkpoint_name} "
                f"(run: {run_id}, {cached_str})"
            )
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    resolved_viewer = cfg.viewer
    if resolved_viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    if resolved_viewer == "viser":
        _patch_zero_command_ranges_for_viser(env_cfg)

    render_mode = "rgb_array" if (trained_mode and cfg.video) else None
    if cfg.video and dummy_mode:
        print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")

    env_cls = load_env_class(task_id)
    env = env_cls(cfg=env_cfg, device=device, render_mode=render_mode)

    if trained_mode and cfg.video:
        print("[INFO] Recording videos during play")
        assert log_dir is not None
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = AmpVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if dummy_mode:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id)
        if runner_cls is None:
            raise RuntimeError(f"Task '{task_id}' does not define runner_cls")
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path))

        def policy(obs):
            obs = obs.to(device)
            if getattr(runner.alg, "use_vae", False):
                obs = obs.clone()
                obs["estimator_out"] = runner.alg.vae.act_inference(obs["estimator_obs"])
            return runner.alg.actor_critic.act_inference(obs)

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


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
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    del remaining_args
    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
