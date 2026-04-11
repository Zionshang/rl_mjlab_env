"""Configuration dataclasses for the vendored rl_sim_env RSL-RL runner."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

from rl_mjlab_env.asset_zoo.robots.unitree_go2 import GO2_FOOT_NAMES, GO2_JOINT_NAMES

_DEFAULT_MOTION_DIR = (
    Path(__file__).resolve().parents[3] / "dataset" / "unitree_go2" / "trot" / "npz"
)


def go2_motion_files() -> list[str]:
    motion_dir = Path(os.environ.get("RL_MJLAB_GO2_MOTION_DIR", _DEFAULT_MOTION_DIR))
    return sorted(glob.glob(str(motion_dir / "*")))


def go2_amp_loader_cfg() -> dict:
    root_pos_size = 3
    root_rot_size = 4
    root_linear_vel_size = 3
    root_angular_vel_size = 3
    frame_pos_size = 12
    frame_vel_size = 12
    joint_pos_size = 12
    joint_vel_size = 12

    root_pos_start = 0
    root_pos_end = root_pos_start + root_pos_size
    root_rot_start = root_pos_end
    root_rot_end = root_rot_start + root_rot_size
    root_linear_vel_start = root_rot_end
    root_linear_vel_end = root_linear_vel_start + root_linear_vel_size
    root_angular_vel_start = root_linear_vel_end
    root_angular_vel_end = root_angular_vel_start + root_angular_vel_size
    frame_pos_start = root_angular_vel_end
    frame_pos_end = frame_pos_start + frame_pos_size
    frame_vel_start = frame_pos_end
    frame_vel_end = frame_vel_start + frame_vel_size
    joint_pos_start = frame_vel_end
    joint_pos_end = joint_pos_start + joint_pos_size
    joint_vel_start = joint_pos_end
    joint_vel_end = joint_vel_start + joint_vel_size

    return {
        "ROOT_POS_START_IDX": root_pos_start,
        "ROOT_POS_END_IDX": root_pos_end,
        "ROOT_ROT_START_IDX": root_rot_start,
        "ROOT_ROT_END_IDX": root_rot_end,
        "ROOT_LINEAR_VEL_START_IDX": root_linear_vel_start,
        "ROOT_LINEAR_VEL_END_IDX": root_linear_vel_end,
        "ROOT_ANGULAR_VEL_START_IDX": root_angular_vel_start,
        "ROOT_ANGULAR_VEL_END_IDX": root_angular_vel_end,
        "FRAME_POS_START_IDX": frame_pos_start,
        "FRAME_POS_END_IDX": frame_pos_end,
        "FRAME_VEL_START_IDX": frame_vel_start,
        "FRAME_VEL_END_IDX": frame_vel_end,
        "JOINT_POS_START_IDX": joint_pos_start,
        "JOINT_POS_END_IDX": joint_pos_end,
        "JOINT_VEL_START_IDX": joint_vel_start,
        "JOINT_VEL_END_IDX": joint_vel_end,
        "TOTAL_SIZE": joint_vel_end,
        "all_keys": [
            "root_position_world",
            "root_quaternion_wxyz",
            "root_linear_velocity_base",
            "root_angular_velocity_base",
        ]
        + [frame + "_position_base" for frame in GO2_FOOT_NAMES]
        + [frame + "_velocity_base" for frame in GO2_FOOT_NAMES]
        + [joint + "_q" for joint in GO2_JOINT_NAMES]
        + [joint + "_dq" for joint in GO2_JOINT_NAMES],
        "combined_indices": list(range(root_linear_vel_start, root_linear_vel_end - 1))
        + list(range(root_angular_vel_start + 2, root_angular_vel_end))
        + list(range(joint_pos_start, joint_pos_end))
        + list(range(joint_vel_start, joint_vel_end))
        + list(range(frame_pos_start, frame_pos_end)),
    }


@dataclass
class RslRlAmpOnPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 500
    experiment_name: str = "unitree_go2_flat_amp"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = 100.0
    logger: str = "wandb"
    wandb_project: str = "mjlab"
    wandb_tags: tuple[str, ...] = ("go2", "amp", "flat", "mjlab")
    policy: dict = field(
        default_factory=lambda: {
            "num_actor_obs": 45,
            "num_critic_obs": 48,
            "num_actions": 12,
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "min_normalized_std": [0.01, 0.01, 0.01] * 4,
        }
    )
    algorithm: dict = field(
        default_factory=lambda: {
            "value_loss_coef": 0.5,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 2.0e-5,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "amp_replay_buffer_size": 1000000,
            "amp_disc_grad_penalty": 5.0,
        }
    )
    amp: dict = field(
        default_factory=lambda: {
            "motion_files": go2_motion_files(),
            "num_preload_transitions": 2000000,
            "discr_hidden_dims": [1024, 512],
        }
    )
    amp_loader_cfg: dict = field(default_factory=go2_amp_loader_cfg)
