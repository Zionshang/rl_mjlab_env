# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl_local.modules import AMPDiscriminator
from rsl_rl_local.storage import ReplayBuffer, RolloutStorageAmp


class PPOAMP:
    """Pure AMP PPO algorithm."""

    def __init__(
        self,
        actor_critic,
        amp_discriminator: AMPDiscriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size=100000,
        amp_disc_grad_penalty=5.0,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cuda:0",
        normalize_advantage_per_mini_batch=False,
        multi_gpu_cfg: dict | None = None,
    ):
        self.actor_critic = actor_critic.to(device)
        self.amp_discriminator = amp_discriminator.to(device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.amp_storage = ReplayBuffer(
            self.amp_discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_disc_grad_penalty = amp_disc_grad_penalty

        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.parameters(), "lr": learning_rate, "name": "actor_critic"},
                {
                    "params": self.amp_discriminator.trunk.parameters(),
                    "lr": learning_rate,
                    "weight_decay": 10e-4,
                    "name": "amp_trunk",
                },
                {
                    "params": self.amp_discriminator.amp_linear.parameters(),
                    "lr": learning_rate,
                    "weight_decay": 10e-2,
                    "name": "amp_head",
                },
            ]
        )

        self.storage: RolloutStorageAmp = None  # type: ignore
        self.transition = RolloutStorageAmp.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(self, num_envs, num_transitions_per_env, obs, action_shape, device):
        self.storage = RolloutStorageAmp(num_envs, num_transitions_per_env, obs, action_shape, device)

    def act(self, actor_obs, critic_obs):
        self.transition.actions = self.actor_critic.act(actor_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = {"actor_obs": actor_obs, "critic_obs": critic_obs}
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs, next_amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.amp_storage.insert(amp_obs, next_amp_obs)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, critic_obs):
        last_values = self.actor_critic.evaluate(critic_obs)
        self.storage.compute_returns(
            last_values.detach(), self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = sample

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            actor_obs_batch = obs_batch["actor_obs"]
            critic_obs_batch = obs_batch["critic_obs"]
            self.actor_critic.act(actor_obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            policy_state_unnorm = torch.clone(policy_state)
            expert_state_unnorm = torch.clone(expert_state)

            with torch.no_grad():
                policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            policy_d = self.amp_discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.amp_discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.amp_discriminator.compute_gradient_penalty(
                expert_state, expert_next_state, lambda_=self.amp_disc_grad_penalty
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + amp_loss
                + grad_pen_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            params = [p for p in self.optimizer.param_groups for p in p["params"] if p.grad is not None]
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self.optimizer.step()

            self.amp_normalizer.update(policy_state_unnorm.cpu().numpy())
            self.amp_normalizer.update(expert_state_unnorm.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        self.storage.clear()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "amp_loss": mean_amp_loss / num_updates,
            "grad_pen_loss": mean_grad_pen_loss / num_updates,
            "policy_pred": mean_policy_pred / num_updates,
            "expert_pred": mean_expert_pred / num_updates,
        }

    def broadcast_parameters(self):
        obj_list = [
            {
                "actor_critic": self.actor_critic.state_dict(),
                "amp_disc": self.amp_discriminator.state_dict(),
            }
        ]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        synced = obj_list[0]
        self.actor_critic.load_state_dict(synced["actor_critic"])
        self.amp_discriminator.load_state_dict(synced["amp_disc"])

    def reduce_parameters(self):
        grads = []
        for module in (self.actor_critic, self.amp_discriminator):
            for p in module.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
        if len(grads) == 0:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for module in (self.actor_critic, self.amp_discriminator):
            for p in module.parameters():
                if p.grad is not None:
                    numel = p.numel()
                    p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                    offset += numel
