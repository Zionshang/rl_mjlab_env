"""Curriculum helpers for Go2 + X5."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv


def modify_reward_weight(
    env: ManagerBasedRlEnv,
    env_ids,
    term_name: str,
    num_steps: int,
    weight: float,
) -> dict[str, float]:
    del env_ids
    reward_cfg = env.reward_manager.get_term_cfg(term_name)
    active = env.common_step_counter >= num_steps
    if env.common_step_counter >= num_steps:
        reward_cfg.weight = weight
    return {
        "active": float(active),
        "step": float(env.common_step_counter),
        "target_step": float(num_steps),
        "weight": float(reward_cfg.weight),
    }
