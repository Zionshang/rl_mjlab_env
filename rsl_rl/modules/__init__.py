# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .amp_discriminator import AMPDiscriminator
from .actor_critic import ActorCritic
from .actor_critic_ampvae import ActorCriticEncoder
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .vae_blind import VAEBlind


__all__ = [
    "AMPDiscriminator",
    "ActorCritic",
    "ActorCriticEncoder",
    "ActorCriticRecurrent",
    "EmpiricalNormalization",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "VAEBlind",
]
