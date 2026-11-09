"""
3D U-Net noise predictor for DDPM.

Layout follows Ho et al. (2020) with the refinements of Nichol & Dhariwal
(2021): GroupNorm + SiLU pre-activation residual blocks, timestep injected
by adaptive group-norm scale/shift, zero-initialised residual and output
convolutions, multi-head self-attention at low resolutions (computed with
torch's scaled_dot_product_attention, so flash / memory-efficient kernels
are used when available).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    groups = 32
    while channels % groups:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def zero_module(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv3d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, head_channels: int = 64):
        super().__init__()
        self.heads = max(1, channels // head_channels)
        self.norm = group_norm(channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, 1)
        self.proj = zero_module(nn.Conv3d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, d * h * w)
        q, k, v = (qkv[:, i].transpose(-1, -2) for i in range(3))  # (b, heads, n, ch)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, d, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, image_size: int = 64,
                 base_channels: int = 64, channel_mult: Sequence[int] = (1, 2, 2, 4),
                 num_res_blocks: int = 2, attention_resolutions: Sequence[int] = (16, 8),
                 dropout: float = 0.1, head_channels: int = 64):
        super().__init__()
        if image_size % (2 ** (len(channel_mult) - 1)):
            raise ValueError("image_size must be divisible by 2^(len(channel_mult)-1)")
        self.config = dict(in_channels=in_channels, out_channels=out_channels, image_size=image_size,
                           base_channels=base_channels, channel_mult=list(channel_mult),
                           num_res_blocks=num_res_blocks,
                           attention_resolutions=list(attention_resolutions),
                           dropout=dropout, head_channels=head_channels)
        emb_dim = 4 * base_channels
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.head = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.down = nn.ModuleList()
        skips = [base_channels]
        ch, res = base_channels, image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, base_channels * mult, emb_dim, dropout)]
                ch = base_channels * mult
                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, head_channels))
                self.down.append(nn.ModuleList(layers))
                skips.append(ch)
            if level != len(channel_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skips.append(ch)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout),
                                  AttentionBlock(ch, head_channels),
                                  ResBlock(ch, ch, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skips.pop(), base_channels * mult, emb_dim, dropout)]
                ch = base_channels * mult
                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, head_channels))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    res *= 2
                self.up.append(nn.ModuleList(layers))
        assert not skips

        self.out = nn.Sequential(group_norm(ch), nn.SiLU(),
                                 zero_module(nn.Conv3d(ch, out_channels, 3, padding=1)))

    @staticmethod
    def _run(layers, h, emb):
        for layer in layers:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        h = self.head(x)
        hs = [h]
        for layers in self.down:
            h = self._run(layers, h, emb)
            hs.append(h)
        h = self._run(self.mid, h, emb)
        for layers in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._run(layers, h, emb)
        return self.out(h)
