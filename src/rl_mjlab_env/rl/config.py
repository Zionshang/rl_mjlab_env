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


RslRlAmpOnPolicyRunnerCfg = AmpRunnerCfg
