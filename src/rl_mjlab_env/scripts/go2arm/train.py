"""Train Go2Arm tasks with the vendored local_rsl_rl runner."""

from __future__ import annotations

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
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from rl_mjlab_env.rl import Go2ArmRunnerCfg, Go2ArmVecEnvWrapper
from rl_mjlab_env.tasks.env_classes import load_env_class


def list_go2arm_tasks() -> list[str]:
    return sorted(task_id for task_id in list_tasks() if isinstance(load_rl_cfg(task_id), Go2ArmRunnerCfg))


@dataclass(frozen=True)
class TrainConfig(MjlabTrainConfig):
    agent: Go2ArmRunnerCfg

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
        assert isinstance(agent_cfg, Go2ArmRunnerCfg)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def _runner_dict(agent: Go2ArmRunnerCfg) -> dict:
    cfg = asdict(agent)
    cfg["algorithm"] = dict(cfg["algorithm"])
    cfg["policy"] = dict(cfg["policy"])
    return cfg


class Go2ArmVideoRecorder(VideoRecorder):
    """Video recorder variant for Go2Arm's split `(leg_reward, arm_reward)` step API."""

    def step(self, action):
        step_triggered = self.step_trigger is not None and self.step_trigger(self.step_count)
        episode_triggered = self.episode_trigger is not None and self.episode_trigger(
            self.episode_count
        )

        if (step_triggered or episode_triggered) and not self.is_recording:
            self.trigger_type = "step" if step_triggered else "episode"
            self._start_recording()

        obs, reward, arm_reward, terminated, truncated, info = self._wrapped_env.step(action)

        if terminated[0] or truncated[0]:
            self.episode_count += 1

        if self.is_recording:
            self._record_frame()
            if self.video_length is not None:
                should_stop = len(self.current_video_frames) >= self.video_length
            else:
                should_stop = terminated[0] or truncated[0]
            if should_stop:
                self._finish_recording()

        self.step_count += 1
        return obs, reward, arm_reward, terminated, truncated, info


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
    cfg.env.seed = seed

    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True

    env_cls = load_env_class(task_id)
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        raise RuntimeError(f"Task '{task_id}' does not define runner_cls")

    train_video_dir = Path(log_dir) / "videos" / "train"
    env = env_cls(
        cfg=cfg.env,
        device=device,
        render_mode="rgb_array" if (cfg.video and rank == 0) else None,
    )
    if cfg.video and rank == 0:
        env = Go2ArmVideoRecorder(
            env,
            video_folder=train_video_dir,
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording videos to: {train_video_dir}")

    env = Go2ArmVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = runner_cls(env, _runner_dict(cfg.agent), str(log_dir), device)
    runner.add_git_repo_to_log(__file__)

    if cfg.agent.resume:
        resume_path = get_checkpoint_path(log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint)
        runner.load(str(resume_path))

    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    env.close()
    if runner.writer is not None:
        runner.writer.close()


def launch_training(task_id: str, cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig.from_task(task_id)
    log_root_path = Path("logs") / "local_rsl_rl" / cfg.agent.experiment_name
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.agent.run_name:
        log_dir_name += f"_{cfg.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"
    if num_gpus > 1:
        raise NotImplementedError("Go2Arm local_rsl_rl training currently supports a single process.")
    run_train(task_id, cfg, log_dir)


def main() -> None:
    import rl_mjlab_env.tasks  # noqa: F401

    all_tasks = list_go2arm_tasks()
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
    launch_training(chosen_task, args)


if __name__ == "__main__":
    main()
