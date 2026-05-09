# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from rl_mjlab_env.utils.amp_utils.motion_loader import AMPLoader
from rl_mjlab_env.utils.amp_utils.normalizer import Normalizer
from rsl_rl_local.algorithms import AMPPPO
from rsl_rl_local.env import VecEnv
from rsl_rl_local.modules import AMPDiscriminator, ActorCritic
from rsl_rl_local.utils import store_code_state


class AMPOnPolicyRunner:
    """On-policy runner for pure AMP training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cuda:0"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.amp_cfg = train_cfg["amp"]
        self.device = device
        self.env = env

        self._configure_multi_gpu()

        robot_data = self.env.unwrapped.scene["robot"].data
        joint_pos_limits = robot_data.default_joint_pos_limits
        dof_range = joint_pos_limits[0][:, 1] - joint_pos_limits[0][:, 0]
        policy_kwargs = dict(self.policy_cfg)
        min_normalized_std = policy_kwargs.pop("min_normalized_std")
        min_std = torch.tensor(min_normalized_std, device=self.device) * torch.abs(dof_range)

        actor_critic = ActorCritic(
            min_std=min_std,
            **policy_kwargs,
        ).to(self.device)

        amp_data = AMPLoader(
            device,
            observation_schema=self.amp_cfg["observation_schema"],
            time_between_frames=self.env.step_dt,
            num_preload_transitions=self.amp_cfg["num_preload_transitions"],
            motion_files=self.amp_cfg["motion_files"],
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        amp_discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2, self.amp_cfg["discr_hidden_dims"], device
        ).to(self.device)

        self.alg = AMPPPO(
            actor_critic=actor_critic,
            amp_discriminator=amp_discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        obs_dict = self.env.get_observations()
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            obs={
                "actor_obs": obs_dict["actor_obs"],
                "critic_obs": obs_dict["critic_obs"],
            },
            action_shape=[self.env.num_actions],
            device=self.device,
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [__file__]
        self.video_dir: str | None = None
        self.video_upload_enabled = False
        self.video_wandb_key = "Video/train"
        self.uploaded_video_paths: set[str] = set()
        _ = self.env.reset()

    def init_logger(self):
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            from rsl_rl_local.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        self.init_logger()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs_dict = self.env.get_observations()
        actor_obs = obs_dict["actor_obs"].to(self.device)
        critic_obs = obs_dict["critic_obs"].to(self.device)
        amp_obs = obs_dict["amp_obs"].to(self.device)
        next_amp_obs = amp_obs.clone()
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
                    amp_out = self.alg.amp_discriminator.discriminator_out(
                        amp_obs, next_amp_obs, normalizer=self.alg.amp_normalizer
                    )
                    amp_obs = torch.clone(next_amp_obs).detach()

                    obs_dict, rewards, dones, infos = self.env.step(actions.to(self.device), amp_out)

                    actor_obs = obs_dict["actor_obs"].to(self.device)
                    critic_obs = obs_dict["critic_obs"].to(self.device)
                    next_amp_obs = obs_dict["amp_obs"].to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    next_amp_obs_with_term = torch.clone(next_amp_obs).detach()
                    reset_env_ids = infos.get("reset_env_ids")
                    terminal_amp_states = infos.get("terminal_amp_states")
                    if terminal_amp_states is not None:
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states.detach().clone()

                    self.alg.process_env_step(rewards, dones, infos, amp_obs, next_amp_obs_with_term)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                self.log_new_videos()

            ep_infos.clear()
            if it == start_iter and not self.disable_logs and self.log_dir is not None:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.log_new_videos()

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "amp_discriminator_state_dict": self.alg.amp_discriminator.state_dict(),
                "amp_normalizer": self.alg.amp_normalizer,
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        resumed_training = self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=True)
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic, {}

    def train_mode(self):
        self.alg.actor_critic.train()
        self.alg.amp_discriminator.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        self.alg.amp_discriminator.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def log_new_videos(self):
        if not self.video_upload_enabled or self.disable_logs:
            return
        if self.writer is None:
            return
        if self.video_dir is None:
            return

        if not hasattr(self.writer, "log_video_directory"):
            return

        fps = int(round(self.env.unwrapped.metadata.get("render_fps", 30)))
        self.uploaded_video_paths = self.writer.log_video_directory(
            self.video_wandb_key,
            self.video_dir,
            self.uploaded_video_paths,
            fps=fps,
            step=self.current_learning_iteration,
        )

    def _configure_multi_gpu(self):
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank_offset = int(os.getenv("JAX_LOCAL_RANK", "0"))
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0")) + self.gpu_local_rank_offset
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        if (self.gpu_local_rank - self.gpu_local_rank_offset) >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)
