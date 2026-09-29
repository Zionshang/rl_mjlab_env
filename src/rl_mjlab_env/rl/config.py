"""VAE+AMP runner configuration compatible with the original project."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VaeAmpActorCfg:
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  observation_normalization: bool = False


@dataclass
class VaeAmpPrivilegedEncoderCfg:
  hidden_dims: tuple[int, ...] = (64, 32)
  output_dim: int = 16


@dataclass
class VaeAmpHeightmapEncoderCfg:
  hidden_dims: tuple[int, ...] = (64, 32)
  output_dim: int = 32


@dataclass
class VaeAmpCriticCfg:
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  observation_normalization: bool = False


@dataclass
class VaeAmpPolicyCfg:
  class_name: str = "ActorCriticVae"
  actor: VaeAmpActorCfg = field(default_factory=VaeAmpActorCfg)
  privileged_encoder: VaeAmpPrivilegedEncoderCfg = field(
    default_factory=VaeAmpPrivilegedEncoderCfg
  )
  heightmap_encoder: VaeAmpHeightmapEncoderCfg = field(
    default_factory=VaeAmpHeightmapEncoderCfg
  )
  critic: VaeAmpCriticCfg = field(default_factory=VaeAmpCriticCfg)
  init_noise_std: float = 1.0
  noise_std_type: str = "scalar"
  activation: str = "elu"
  min_normalized_std: float | tuple[float, ...] = 0.01


@dataclass
class VaeCfg:
  class_name: str = "VAEBlind"
  enabled: bool = True
  encoder_hidden_dims: tuple[int, ...] = (128,)
  encoder_out_dim: int = 64
  encoder_head_dim_dict: dict[str, int] = field(
    default_factory=lambda: {
      "obs_vel": 3,
      "obs_com": 3,
      "obs_mass": 1,
      "obs_latent": 16,
    }
  )
  decoder_hidden_dims: tuple[int, ...] = (64, 128)
  activation: str = "elu"
  use_exclusive_optimizer: bool = False
  use_exclusive_lr: bool = False
  learning_rate: float = 1.0e-3
  beta_adaptive: bool = True
  beta_max_step: int = 5000
  beta: float = 1.0e-4
  beta_max: float = 0.01
  free_bits: float = 0.5
  use_adaboot: bool = True
  adaboot_max_step: int = 500


@dataclass
class AmpDataCfg:
  root_pos_size: int = 3
  root_rot_size: int = 4
  root_linear_vel_size: int = 3
  root_angular_vel_size: int = 3
  frame_pos_size: int = 12
  frame_vel_size: int = 12
  joint_pos_size: int = 12
  joint_vel_size: int = 12
  frame_keys: tuple[str, ...] = ()
  joint_keys: tuple[str, ...] = ()


@dataclass
class AmpCfg:
  enabled: bool = True
  hidden_dims: tuple[int, ...] = (1024, 512)
  replay_buffer_size: int = 1_000_000
  discriminator_grad_penalty: float = 5.0
  motion_files: tuple[str, ...] = ()
  num_preload_transitions: int = 2_000_000
  data: AmpDataCfg = field(default_factory=AmpDataCfg)


@dataclass
class VaeAmpPpoAlgorithmCfg:
  class_name: str = "VaeAmpPPO"
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


@dataclass
class VaeAmpOnPolicyRunnerCfg:
  class_name: str = "VaeAmpOnPolicyRunner"
  seed: int = 42
  device: str = "cuda:0"
  num_steps_per_env: int = 24
  max_iterations: int = 100_000
  save_interval: int = 500
  experiment_name: str = "vae_amp_go2"
  run_name: str = ""
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  empirical_normalization: bool = False
  clip_actions: float = 100.0
  logger: str = "wandb"
  wandb_project: str = "mjlab"
  wandb_tags: tuple[str, ...] = ("go2", "vae_amp", "mjlab")
  policy: VaeAmpPolicyCfg = field(default_factory=VaeAmpPolicyCfg)
  algorithm: VaeAmpPpoAlgorithmCfg = field(default_factory=VaeAmpPpoAlgorithmCfg)
  vae: VaeCfg = field(default_factory=VaeCfg)
  amp: AmpCfg = field(default_factory=AmpCfg)
