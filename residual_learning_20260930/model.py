"""Small single-sided residual-grid predictor for frozen RopStitch outputs."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
        )


class PlainResidualGridNet(nn.Module):
    """Predict a bounded 13x13 displacement for the target-side mesh.

    Input channels are reference RGB, target RGB, their absolute difference,
    and the two validity masks.  The final layer is exactly zero-initialized so
    a fresh model reproduces the frozen RopStitch mesh.
    """

    def __init__(self, base_channels: int = 16, max_displacement: float = 32.0):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            ConvBlock(14, c),
            ConvBlock(c, 2 * c, stride=2),
            ConvBlock(2 * c, 4 * c, stride=2),
            ConvBlock(4 * c, 4 * c),
        )
        self.grid_pool = nn.AdaptiveAvgPool2d((13, 13))
        self.head = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(2 * c, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.max_displacement = float(max_displacement)

    @staticmethod
    def local_correlation(inputs: torch.Tensor, radius: int = 4) -> torch.Tensor:
        """Parameter-free local RGB correlation: expected dx/dy and peak score."""
        reference = F.avg_pool2d(inputs[:, :3], 4, 4)
        target = F.avg_pool2d(inputs[:, 3:6], 4, 4)
        target_mask = F.avg_pool2d(inputs[:, 10:11], 4, 4)
        reference = F.normalize(reference, dim=1, eps=1e-6)
        target = F.normalize(target, dim=1, eps=1e-6)
        kernel = 2 * radius + 1
        b, c, h, w = reference.shape
        candidates = F.unfold(target, kernel, padding=radius).reshape(b, c, kernel * kernel, h, w)
        valid = F.unfold(target_mask, kernel, padding=radius).reshape(b, kernel * kernel, h, w)
        score = (reference[:, :, None] * candidates).sum(1) * 5.0
        score = score.masked_fill(valid < .5, -20.0)
        probability = score.softmax(1)
        axis = torch.arange(-radius, radius + 1, device=inputs.device, dtype=inputs.dtype)
        dy, dx = torch.meshgrid(axis, axis, indexing="ij")
        expected_x = (probability * dx.reshape(1, -1, 1, 1)).sum(1, keepdim=True) / radius
        expected_y = (probability * dy.reshape(1, -1, 1, 1)).sum(1, keepdim=True) / radius
        peak = probability.max(1, keepdim=True).values
        correlation = torch.cat((expected_x, expected_y, peak), 1)
        return F.interpolate(correlation, inputs.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        augmented = torch.cat((inputs, self.local_correlation(inputs)), 1)
        encoded = self.grid_pool(self.encoder(augmented))
        # [B, 2, 13, 13] -> [B, 169, 2], in 512-input pixels.
        displacement = torch.tanh(self.head(encoded)) * self.max_displacement
        return displacement.permute(0, 2, 3, 1).reshape(inputs.shape[0], 169, 2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
