"""Local Continuity Module (LCM).

A small residual convolutional head applied to the predicted normal latent
before VAE decoding. It enforces local smoothness in latent space and is
trained jointly with the core predictor LoRA.
"""

import torch
import torch.nn as nn


class LocalContinuityModule(nn.Module):
    """Residual 2-layer conv head operating on raw VAE latents.

    Args:
        num_channels: Latent channel count (``transformer.in_channels // 4``,
            i.e. 32 for FLUX.2 [klein]).
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.lcm = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lcm_dtype = next(self.lcm.parameters()).dtype
        if x.dtype != lcm_dtype:
            x = x.to(dtype=lcm_dtype)
        return x + self.lcm(x)
