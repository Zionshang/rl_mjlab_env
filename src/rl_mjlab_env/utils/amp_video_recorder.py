"""Video recorder adapter for the AMP environment step signature."""

from __future__ import annotations

from typing import Any, Callable

import torch
from mjlab.utils.wrappers import VideoRecorder


class AmpVideoRecorder(VideoRecorder):
    """Use MJLab's recorder while accepting ``step(action, amp_out)``."""

    def __init__(
        self,
        *args,
        upload_to_wandb: bool = False,
        wandb_key: str = "Video/train",
        wandb_step_fn: Callable[[int], int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.upload_to_wandb = upload_to_wandb
        self.wandb_key = wandb_key
        self.wandb_step_fn = wandb_step_fn

    def step(self, action: torch.Tensor, amp_out: torch.Tensor | None = None) -> Any:
        step_triggered = self.step_trigger is not None and self.step_trigger(
            self.step_count
        )
        episode_triggered = self.episode_trigger is not None and self.episode_trigger(
            self.episode_count
        )

        if (step_triggered or episode_triggered) and not self.is_recording:
            self.trigger_type = "step" if step_triggered else "episode"
            self._start_recording()

        step_return = self._wrapped_env.step(action, amp_out)
        terminated, truncated = step_return[2], step_return[3]

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
        return step_return

    def _finish_recording(self) -> None:
        video_path = self.current_video_path
        fps = self._wrapped_env.metadata.get("render_fps", 30)
        wandb_step = self._wandb_step()
        super()._finish_recording()
        if self.upload_to_wandb and video_path is not None and video_path.exists():
            self._upload_video_to_wandb(video_path, fps=fps, step=wandb_step)

    def _wandb_step(self) -> int:
        if self.wandb_step_fn is None:
            return self.step_count
        return self.wandb_step_fn(self.step_count)

    def _upload_video_to_wandb(self, video_path, fps: int, step: int) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(
                {self.wandb_key: wandb.Video(str(video_path), fps=fps, format="mp4")},
                step=step,
            )
