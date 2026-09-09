"""Geometric Refinement Module (GRM).

A lightweight (~0.4M param) edge-aware refinement head applied to the
VAE-decoded normal map. The VAE encoder/decoder bottleneck (8x downsample)
can introduce boundary-localized geometric errors; the GRM predicts a bounded
residual correction on top of an explicit anchor:

- opaque scenes use a full-RGB guided-filter anchor (He et al. 2010);
- transparent scenes use the identity anchor (no RGB-guided smoothing, since
  RGB edges behind glass do not correspond to geometry edges).

The predicted residual is scaled, norm-clipped using the configured threshold,
and added to the anchor before L2 normalization.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual conv block with optional dilation."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


def _build_edge_encoder(edge_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, edge_channels, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(edge_channels, edge_channels, 3, padding=1),
        nn.GELU(),
    )


def _resolve_dilations(num_blocks: int, dilations: Optional[List[int]]) -> List[int]:
    if dilations is None or len(dilations) == 0:
        return [1] * num_blocks
    if len(dilations) != num_blocks:
        raise ValueError(
            f"dilations length ({len(dilations)}) must equal num_blocks ({num_blocks})"
        )
    return [int(d) for d in dilations]


def _box_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    kernel = 2 * int(radius) + 1
    return F.avg_pool2d(
        x.float(), kernel_size=kernel, stride=1, padding=int(radius),
        count_include_pad=False,
    )


def guided_filter_torch(
    coarse_normal: torch.Tensor,
    rgb: torch.Tensor,
    radius: int = 4,
    eps: float = 0.01,
) -> torch.Tensor:
    """Full-RGB guided filter anchor for normal maps.

    Follows the multi-channel guided-filter solve used by OpenCV's ximgproc
    guidedFilter: a local linear model maps RGB guide values to each normal
    channel, then the filtered normal is L2-normalized.
    """
    if coarse_normal.shape[1] != 3 or rgb.shape[1] != 3:
        raise ValueError("guided_filter_torch expects 3-channel normal and RGB")

    out_dtype = coarse_normal.dtype
    I = rgb.float()
    p = coarse_normal.float()
    B, p_ch, H, W = p.shape
    i_ch = I.shape[1]

    mean_I = _box_mean(I, radius)
    mean_p = _box_mean(p, radius)

    cov_Ip = []
    for pc in range(p_ch):
        row = []
        for ic in range(i_ch):
            mean_Ip = _box_mean(I[:, ic:ic + 1] * p[:, pc:pc + 1], radius)
            row.append(mean_Ip - mean_I[:, ic:ic + 1] * mean_p[:, pc:pc + 1])
        cov_Ip.append(torch.cat(row, dim=1))
    cov_Ip = torch.stack(cov_Ip, dim=1)  # (B, 3 normal, 3 rgb, H, W)

    var_rows = []
    for i in range(i_ch):
        cols = []
        for j in range(i_ch):
            mean_II = _box_mean(I[:, i:i + 1] * I[:, j:j + 1], radius)
            cols.append(mean_II - mean_I[:, i:i + 1] * mean_I[:, j:j + 1])
        var_rows.append(torch.cat(cols, dim=1))
    var_I = torch.stack(var_rows, dim=1)  # (B, 3 row, 3 col, H, W)

    sigma = var_I.permute(0, 3, 4, 1, 2).reshape(-1, i_ch, i_ch)
    eye = torch.eye(i_ch, device=sigma.device, dtype=sigma.dtype).unsqueeze(0)
    sigma = sigma + float(eps) * eye
    rhs = cov_Ip.permute(0, 3, 4, 2, 1).reshape(-1, i_ch, p_ch)

    coeff = torch.linalg.solve(sigma, rhs)
    a = coeff.reshape(B, H, W, i_ch, p_ch).permute(0, 4, 3, 1, 2)
    b = mean_p - (a * mean_I.unsqueeze(1)).sum(dim=2)

    mean_a = _box_mean(a.reshape(B, p_ch * i_ch, H, W), radius)
    mean_a = mean_a.reshape(B, p_ch, i_ch, H, W)
    mean_b = _box_mean(b, radius)
    q = (mean_a * I.unsqueeze(1)).sum(dim=2) + mean_b
    q = F.normalize(q, p=2, dim=1)
    return q.to(dtype=out_dtype)


def _domain_mask(
    domain_is_transparent: Optional[Union[bool, torch.Tensor]],
    coarse_normal: torch.Tensor,
) -> torch.Tensor:
    B, _, H, W = coarse_normal.shape
    if domain_is_transparent is None:
        value = torch.zeros(B, 1, 1, 1, device=coarse_normal.device, dtype=coarse_normal.dtype)
    elif isinstance(domain_is_transparent, bool):
        value = torch.full(
            (B, 1, 1, 1), float(domain_is_transparent),
            device=coarse_normal.device, dtype=coarse_normal.dtype,
        )
    else:
        value = domain_is_transparent.to(device=coarse_normal.device, dtype=coarse_normal.dtype)
        if value.ndim == 0:
            value = value.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        elif value.ndim in (1, 2):
            value = value.view(B, 1, 1, 1)
        elif value.ndim == 3:
            value = value.unsqueeze(1)
        elif value.ndim == 4:
            value = value[:, :1]
        else:
            raise ValueError(f"Unsupported domain flag shape: {tuple(value.shape)}")
    return value.expand(B, 1, H, W)


class GRM(nn.Module):
    """Geometric Refinement Module (release inference path).

    Interface: ``refined = grm(coarse_normal, rgb, domain_is_transparent)``
    where ``coarse_normal`` is the VAE-decoded normal map in [-1, 1] and
    ``rgb`` is the input image in [-1, 1], both (B, 3, H, W). The output is a
    unit-normalized normal map in [-1, 1].

    The architecture is specified by the released ``config.json``.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        edge_channels: int = 32,
        num_blocks: int = 4,
        use_rgb: bool = True,
        dilations: Optional[List[int]] = None,
        anchor_radius: int = 4,
        anchor_eps: float = 0.01,
        residual_scale: float = 1.0,
        residual_clip_deg: float = 0.0,
        initial_alpha: float = 0.5,
    ):
        super().__init__()
        self.use_rgb = bool(use_rgb)
        self.dilations = _resolve_dilations(num_blocks, dilations)
        self.anchor_radius = int(anchor_radius)
        self.anchor_eps = float(anchor_eps)
        self.residual_scale = float(residual_scale)
        self.residual_clip_deg = float(residual_clip_deg)

        if self.use_rgb:
            self.edge_encoder = _build_edge_encoder(edge_channels)
            main_in_ch = 10  # coarse normal + anchor normal + RGB + domain flag
            combined_ch = hidden_channels + edge_channels
        else:
            self.edge_encoder = None
            main_in_ch = 7  # coarse normal + anchor normal + domain flag
            combined_ch = hidden_channels

        self.main_encoder = nn.Sequential(
            nn.Conv2d(main_in_ch, hidden_channels, 3, padding=1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()
        for d in self.dilations:
            self.projections.append(nn.Conv2d(combined_ch, hidden_channels, 1))
            self.blocks.append(ResBlock(hidden_channels, dilation=d))

        self.feature_proj = nn.Sequential(
            nn.Conv2d(combined_ch, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.delta_head = nn.Conv2d(hidden_channels, 3, 3, padding=1)
        self.alpha_head = nn.Conv2d(hidden_channels, 1, 1)

        # Zero-init residual head so an untrained module returns the anchor.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        init_alpha = min(max(float(initial_alpha), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, torch.logit(torch.tensor(init_alpha)).item())

    @torch.no_grad()
    def forward(
        self,
        coarse_normal: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        domain_is_transparent: Optional[Union[bool, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.use_rgb and rgb is None:
            raise ValueError("GRM requires the RGB input when use_rgb=True.")

        transparent_anchor = F.normalize(coarse_normal.float(), p=2, dim=1).to(coarse_normal.dtype)
        domain = _domain_mask(domain_is_transparent, coarse_normal)
        if self.use_rgb:
            opaque_anchor = guided_filter_torch(
                coarse_normal, rgb, radius=self.anchor_radius, eps=self.anchor_eps
            )
            anchor = torch.where(domain > 0.5, transparent_anchor, opaque_anchor)
        else:
            anchor = transparent_anchor

        edge_feat = self.edge_encoder(rgb) if self.use_rgb else None
        if self.use_rgb:
            x_in = torch.cat([coarse_normal, anchor, rgb, domain], dim=1)
        else:
            x_in = torch.cat([coarse_normal, anchor, domain], dim=1)
        x = self.main_encoder(x_in)

        for proj, block in zip(self.projections, self.blocks):
            x = proj(torch.cat([x, edge_feat], dim=1) if edge_feat is not None else x)
            x = block(x)

        feat = self.feature_proj(torch.cat([x, edge_feat], dim=1) if edge_feat is not None else x)
        delta = self.delta_head(feat)
        alpha = torch.sigmoid(self.alpha_head(feat))
        residual = self.residual_scale * alpha * delta

        if self.residual_clip_deg > 0:
            # Clip residual magnitude using tan(threshold in radians).
            max_norm = torch.tan(torch.deg2rad(torch.tensor(
                self.residual_clip_deg, device=residual.device, dtype=torch.float32
            )))
            residual_norm = torch.linalg.vector_norm(
                residual.float(), ord=2, dim=1, keepdim=True
            ).clamp_min(1e-6)
            clip = (max_norm / residual_norm).clamp(max=1.0).to(residual.dtype)
            residual = residual * clip

        refined = F.normalize(anchor + residual, p=2, dim=1)
        return refined

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
