"""Compact UNet for top-down 2.5-D DLO perception.

Input  (B, 5, H, W): occupancy, z_mean, z_min, z_max, hand marker.
Output (B, 3, H, W): skeleton-mask logit, arc coord s in [0,1], height z.

Dense per-pixel supervision replaces sparse node regression; the ordered
chain is extracted by sorting mask pixels by predicted s.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNetSkeleton(nn.Module):
    def __init__(self, cin: int = 5, width: int = 16) -> None:
        super().__init__()
        w = width
        self.e1 = _block(cin, w)
        self.e2 = _block(w, w * 2)
        self.e3 = _block(w * 2, w * 4)
        self.e4 = _block(w * 4, w * 8)
        self.bott = _block(w * 8, w * 16)
        self.d4 = _block(w * 16 + w * 8, w * 8)
        self.d3 = _block(w * 8 + w * 4, w * 4)
        self.d2 = _block(w * 4 + w * 2, w * 2)
        self.d1 = _block(w * 2 + w, w)
        self.head = nn.Conv2d(w, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        b = self.bott(self.pool(x4))
        u = F.interpolate(b, scale_factor=2, mode="bilinear",
                          align_corners=False)
        u = self.d4(torch.cat([u, x4], 1))
        u = F.interpolate(u, scale_factor=2, mode="bilinear",
                          align_corners=False)
        u = self.d3(torch.cat([u, x3], 1))
        u = F.interpolate(u, scale_factor=2, mode="bilinear",
                          align_corners=False)
        u = self.d2(torch.cat([u, x2], 1))
        u = F.interpolate(u, scale_factor=2, mode="bilinear",
                          align_corners=False)
        u = self.d1(torch.cat([u, x1], 1))
        out = self.head(u)
        # channel 0: mask logit (raw); channel 1: s in [0,1]; channel 2: z
        return torch.cat(
            [out[:, :1], torch.sigmoid(out[:, 1:2]), out[:, 2:]], dim=1
        )
