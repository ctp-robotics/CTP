from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ResBlock2d(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class Downsample2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class SingleHandSpatialEncoder(nn.Module):
    """[N, H, W, C] -> [N, H/4, W/4, hand_dim] (35x20 -> 9x5)."""

    def __init__(self, in_channels=3, hand_dim=48, hidden=128, mid=128, dropout=0.0):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.Conv2d(hidden, hidden, 1),
        )

        self.res1 = ResBlock2d(hidden, dropout)
        self.down1 = Downsample2d(hidden)

        self.res2 = ResBlock2d(hidden, dropout)
        self.proj2 = nn.Conv2d(hidden, mid, 3, padding=1)
        self.down2 = Downsample2d(mid)

        self.mid = nn.Sequential(
            ResBlock2d(mid, dropout),
            ResBlock2d(mid, dropout),
            ResBlock2d(mid, dropout),
        )

        self.out_norm = nn.GroupNorm(min(8, mid), mid, eps=1e-6)
        self.conv_out = nn.Conv2d(mid, hand_dim, 3, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        h = self.conv_in(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.proj2(h)
        h = self.down2(h)
        h = self.mid(h)
        h = F.silu(self.out_norm(h))
        h = self.conv_out(h)
        return h.permute(0, 2, 3, 1).contiguous()
