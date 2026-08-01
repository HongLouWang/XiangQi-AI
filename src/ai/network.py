from __future__ import annotations

import re

import torch
from torch import nn

from ai.encoding import ACTION_SIZE, INPUT_CHANNELS

_CUDA_DEVICE = re.compile(r"cuda(?::(?P<index>0|[1-9][0-9]*))?\Z")


class ResidualBlock(nn.Module):
    """保持棋盘尺寸和通道数不变的两层残差块。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


class PolicyValueNetwork(nn.Module):
    """AlphaZero 风格的共享残差主干与策略、价值双输出头。"""

    def __init__(self, channels: int = 64, residual_blocks: int = 4) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(
                INPUT_CHANNELS,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *(ResidualBlock(channels) for _ in range(residual_blocks)),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 10 * 9, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(inputs)
        return self.policy_head(features), self.value_head(features)


def configure_device(name: str, torch_threads: int) -> torch.device:
    """解析训练设备；显式 CUDA 请求不可满足时绝不静默回退。"""
    if type(torch_threads) is not int or torch_threads <= 0:
        raise ValueError("torch_threads 必须是大于 0 的整数")
    if not isinstance(name, str):
        raise TypeError("设备名称必须是字符串")

    torch.set_num_threads(torch_threads)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")

    match = _CUDA_DEVICE.fullmatch(name)
    if match is None:
        raise ValueError(f"不支持的设备: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"已请求 {name}，但 CUDA 不可用")

    index_text = match.group("index")
    if index_text is not None:
        index = int(index_text)
        device_count = torch.cuda.device_count()
        if index >= device_count:
            raise ValueError(
                f"CUDA 设备索引 {index} 不存在，当前共有 {device_count} 个设备"
            )
    return torch.device(name)
