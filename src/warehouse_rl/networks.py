from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.torch_utils import torch_device


class Conv(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(Conv, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 32 -> 16
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([8, 32, 32]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2: 16 -> 8
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([16, 16, 16]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3: 8 -> 4
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([32, 8, 8]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Flatten
            nn.Flatten(),
        )

        # Fully connected
        self.fc = Net(
            state_shape=32 * 4 * 4,
            action_shape=4,
            hidden_sizes=[1024, 1024, 512, 512, 256, 256, 128, 64],
            norm_layer=nn.LayerNorm,
            activation=nn.ReLU,
            dueling_param=(
                {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
                {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
            ),
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ):
        device = torch_device(self)
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
        x = self.features(obs)
        logits, _ = self.fc(x)
        return logits, state
