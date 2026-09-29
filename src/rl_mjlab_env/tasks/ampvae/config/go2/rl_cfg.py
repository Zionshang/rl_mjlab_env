"""RL configuration for the Go2 VAE+AMP task."""

from rl_mjlab_env.asset_zoo.robots.go2.go2_constants import (
  GO2_AMP_DATA_DIR,
  GO2_FOOT_NAMES,
  GO2_JOINT_ORDER,
)
from rl_mjlab_env.rl import AmpCfg, AmpDataCfg, VaeAmpOnPolicyRunnerCfg


def unitree_go2_ampvae_runner_cfg() -> VaeAmpOnPolicyRunnerCfg:
  """Create the legacy Go2 VAE+AMP runner config for the MJLab backend."""
  motion_files = tuple(str(path) for path in sorted(GO2_AMP_DATA_DIR.glob("*.npz")))
  return VaeAmpOnPolicyRunnerCfg(
    experiment_name="vae_amp_go2",
    amp=AmpCfg(
      motion_files=motion_files,
      data=AmpDataCfg(
        frame_keys=GO2_FOOT_NAMES,
        joint_keys=GO2_JOINT_ORDER,
      ),
    ),
  )
