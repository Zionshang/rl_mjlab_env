"""Termination terms for Go2 + X5."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force_history is not None:
        force_mag = torch.norm(sensor.data.force_history, dim=-1)
        return (force_mag > threshold).any(dim=-1).any(dim=-1)
    assert sensor.data.force is not None
    return torch.any(torch.norm(sensor.data.force, dim=-1) > threshold, dim=-1)
