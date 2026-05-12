"""Generic configuration dataclasses for AMP runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AmpPolicyCfg:
    num_actor_obs: int
    num_critic_obs: int
    num_actions: int
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    min_normalized_std: tuple[float, ...] = ()


@dataclass
class AmpAlgorithmCfg:
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 2.0e-5
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False
    amp_replay_buffer_size: int = 1000000
    amp_disc_grad_penalty: float = 5.0


@dataclass
class AmpModuleCfg:
    motion_files: tuple[str, ...] = ()
    observation_schema: dict[str, Any] = field(default_factory=dict)
    num_preload_transitions: int = 2000000
    discr_hidden_dims: tuple[int, ...] = (1024, 512)


@dataclass
class AmpRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 500
    experiment_name: str = "amp_task"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = None
    logger: str = "wandb"
    wandb_project: str = "mjlab"
    wandb_tags: tuple[str, ...] = ()
    policy: AmpPolicyCfg = field(
        default_factory=lambda: AmpPolicyCfg(
            num_actor_obs=0,
            num_critic_obs=0,
            num_actions=0,
        )
    )
    algorithm: AmpAlgorithmCfg = field(default_factory=AmpAlgorithmCfg)
    amp: AmpModuleCfg = field(default_factory=AmpModuleCfg)


@dataclass
class AmpvaeRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 100000
    save_interval: int = 500
    experiment_name: str = ""
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = None
    logger: str = "wandb"
    wandb_project: str = "mjlab"
    wandb_tags: tuple[str, ...] = ()
    policy_type: dict[str, str] = field(default_factory=dict)
    training_type: str = "rl"
    module_cfg_dict: dict[str, Any] = field(default_factory=dict)
    train_cfg_dict: dict[str, Any] = field(default_factory=dict)


RslRlOnPolicyRunnerAMPCfg = AmpRunnerCfg


@dataclass
class Go2ArmRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 15000
    save_interval: int = 1000
    experiment_name: str = "unitree_go2_x5_flat"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = 100.0
    logger: str = "wandb"
    wandb_project: str = "mjlab"
    wandb_tags: tuple[str, ...] = ()
    empirical_normalization: bool = False
    policy: dict[str, Any] = field(
        default_factory=lambda: {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256],
            "critic_hidden_dims": [256],
            "activation": "elu",
            "activation_out": "elu",
            "leg_control_head_hidden_dims": [256, 128],
            "arm_control_head_hidden_dims": [256, 128],
            "critic_leg_control_head_hidden_dims": [256, 128, 64],
            "critic_arm_control_head_hidden_dims": [256, 128, 64],
            "priv_encoder_dims": [32, 18],
            "num_leg_actions": 12,
            "num_arm_actions": 6,
        }
    )
    algorithm: dict[str, Any] = field(
        default_factory=lambda: {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "dagger_update_freq": 20,
            "priv_reg_coef_schedual": [0, 0.1, 1500, 5000],
            "mixing_schedule": [1.0, 0, 4000],
            "eps": 1.0e-5,
        }
    )
