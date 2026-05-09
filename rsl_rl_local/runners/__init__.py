# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP and AMPVAE runners."""

from .on_policy_runner_amp import OnPolicyRunnerAMP
from .on_policy_runner_ampvae import OnPolicyRunnerAMPVAE

__all__ = ["OnPolicyRunnerAMP", "OnPolicyRunnerAMPVAE"]
