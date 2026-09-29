"""CNN encoders with spatial-softmax pooling for image observations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl_local.networks import MLP, EmpiricalNormalization
from rsl_rl_local.utils import resolve_nn_activation


def _as_list(value: Any, length: int) -> list[Any]:
    if isinstance(value, (tuple, list)):
        if len(value) != length:
            raise ValueError(f"Expected {length} values, got {len(value)}")
        return list(value)
    return [value] * length


class _CNN(nn.Sequential):
    """Small self-contained CNN used to avoid depending on upstream rsl-rl."""

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: tuple[int, ...] | list[int],
        kernel_size: int | tuple[int, ...] | list[int],
        stride: int | tuple[int, ...] | list[int] = 1,
        dilation: int | tuple[int, ...] | list[int] = 1,
        padding: str = "none",
        norm: str | tuple[str, ...] | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | tuple[bool, ...] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
    ) -> None:
        channels = list(output_channels)
        kernels = _as_list(kernel_size, len(channels))
        strides = _as_list(stride, len(channels))
        dilations = _as_list(dilation, len(channels))
        norms = _as_list(norm, len(channels))
        pools = _as_list(max_pool, len(channels))
        layers: list[nn.Module] = []
        height, width = input_dim
        in_channels = input_channels
        for out_channels, kernel, step, dilate, norm_name, pool in zip(
            channels, kernels, strides, dilations, norms, pools
        ):
            pad = ((dilate * (kernel - 1)) // 2) if padding != "none" else 0
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=step,
                    dilation=dilate,
                    padding=pad,
                    padding_mode=padding if padding != "none" else "zeros",
                )
            )
            height = (height + 2 * pad - dilate * (kernel - 1) - 1) // step + 1
            width = (width + 2 * pad - dilate * (kernel - 1) - 1) // step + 1
            if norm_name == "batch":
                layers.append(nn.BatchNorm2d(out_channels))
            elif norm_name == "layer":
                layers.append(nn.LayerNorm((out_channels, height, width)))
            elif norm_name != "none":
                raise ValueError(f"Unsupported normalization: {norm_name}")
            layers.append(resolve_nn_activation(activation))
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                height = (height + 1) // 2
                width = (width + 1) // 2
            in_channels = out_channels

        if global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
            height = width = 1
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            height = width = 1
        elif global_pool != "none":
            raise ValueError(f"Unsupported global pooling: {global_pool}")
        if flatten:
            layers.append(nn.Flatten(start_dim=1))

        super().__init__(*layers)
        self.output_channels = None if flatten else in_channels
        self.output_dim = (
            in_channels * height * width if flatten else (height, width)
        )


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax over feature maps."""

    def __init__(self, height: int, width: int, temperature: float = 1.0) -> None:
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        features = x.reshape(batch_size, channels, -1)
        weights = torch.softmax(features / self.temperature, dim=-1)
        expected_x = (weights * self.pos_x).sum(dim=-1)
        expected_y = (weights * self.pos_y).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(batch_size, channels * 2)


class SpatialSoftmaxCNN(nn.Module):
    """CNN encoder followed by spatial-softmax pooling."""

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        temperature: float = 1.0,
        **cnn_kwargs: Any,
    ) -> None:
        super().__init__()
        cnn_kwargs.pop("global_pool", None)
        cnn_kwargs.pop("flatten", None)
        self.cnn = _CNN(
            input_dim=input_dim,
            input_channels=input_channels,
            global_pool="none",
            flatten=False,
            **cnn_kwargs,
        )
        out_h, out_w = self.cnn.output_dim
        num_channels: int = self.cnn.output_channels
        self.spatial_softmax = SpatialSoftmax(out_h, out_w, temperature)
        self._output_dim = num_channels * 2

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def output_channels(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_softmax(self.cnn(x))


class SpatialSoftmaxCNNModel(nn.Module):
    """Self-contained spatial-softmax model compatible with the old API."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        active_groups = obs_groups[obs_set]
        self.obs_groups = [name for name in active_groups if obs[name].ndim == 2]
        self.obs_groups_2d = [name for name in active_groups if obs[name].ndim == 4]
        self.obs_dims_2d = [tuple(obs[name].shape[2:4]) for name in self.obs_groups_2d]
        self.obs_channels_2d = [obs[name].shape[1] for name in self.obs_groups_2d]
        self.obs_dim = sum(obs[name].shape[-1] for name in self.obs_groups)
        if not self.obs_groups_2d:
            raise ValueError("At least one 2D observation group is required.")
        if distribution_cfg is not None:
            raise ValueError("SpatialSoftmaxCNNModel only supports deterministic output.")

        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("The 2D observations must be identical for all models sharing CNN encoders.")
            _cnns = cnns
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of CNN configurations must match the number of 2D observation groups.")
            _cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                group_cfg = dict(cnn_cfg[obs_group])
                group_cfg.pop("spatial_softmax", None)
                temperature = group_cfg.pop("spatial_softmax_temperature", 1.0)
                _cnns[obs_group] = SpatialSoftmaxCNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx],
                    temperature=temperature,
                    **group_cfg,
                )

        self.cnn_latent_dim = 0
        for cnn in _cnns.values():
            if cnn.output_channels is not None:
                raise ValueError("The output of the CNN must be flattened before passing it to the MLP.")
            self.cnn_latent_dim += int(cnn.output_dim)

        self.obs_normalizer = (
            EmpiricalNormalization(self.obs_dim)
            if obs_normalization
            else nn.Identity()
        )
        self.mlp = MLP(
            self.obs_dim + self.cnn_latent_dim,
            output_dim,
            hidden_dims,
            activation,
        )

        if isinstance(_cnns, nn.ModuleDict):
            self.cnns = _cnns
        else:
            self.cnns = nn.ModuleDict(_cnns)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        latent_1d = self.obs_normalizer(
            torch.cat([obs[name] for name in self.obs_groups], dim=-1)
        )
        latent_2d = torch.cat(
            [self.cnns[name](obs[name]) for name in self.obs_groups_2d], dim=-1
        )
        return self.mlp(torch.cat([latent_1d, latent_2d], dim=-1))
