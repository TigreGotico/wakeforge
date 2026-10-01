"""Sub-Spectral Normalization for 2D spectrogram features.

Ported from Qualcomm AI Research's BC-ResNet implementation.
Reference: Kim et al., "Broadcasted Residual Learning for Efficient Keyword
Spotting", Interspeech 2021.

Normalizes frequency bins by dividing the spectrogram into spectral sub-groups
and applying BatchNorm (or InstanceNorm) independently to each group.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SubSpectralNorm(nn.Module):
    """Normalize along frequency bins by dividing into spectral sub-groups.

    Splits the frequency dimension into ``spec_groups`` equal parts and
    normalizes each sub-group independently.  This lets the model learn
    different statistics for different frequency bands.

    Args:
        num_features: Number of input channels.
        spec_groups: Number of spectral sub-groups along frequency dimension.
        affine: ``"Sub"`` uses BatchNorm's built-in affine (per sub-group),
                ``"All"`` adds a single learnable scale/bias across all groups,
                anything else disables affine parameters.
        batch: If True use BatchNorm2d, else InstanceNorm2d.
        dim: Frequency dimension index (2 = height, 3 or -1 = width).
    """

    def __init__(
        self,
        num_features: int,
        spec_groups: int = 16,
        affine: str = "Sub",
        batch: bool = True,
        dim: int = 2,
    ) -> None:
        super().__init__()
        self.spec_groups = spec_groups
        self.sub_dim = dim
        self.affine_all = affine == "All"

        use_affine = affine == "Sub"
        norm_cls = nn.BatchNorm2d if batch else nn.InstanceNorm2d
        self.ssnorm = norm_cls(num_features * spec_groups, affine=use_affine)

        if self.affine_all:
            self.weight = nn.Parameter(torch.ones(num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sub-spectral normalization.

        Args:
            x: Tensor of shape ``[B, C, H, W]``.

        Returns:
            Normalized tensor with the same shape.
        """
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()

        b, c, h, w = x.size()
        x = x.view(b, c * self.spec_groups, h // self.spec_groups, w)
        x = self.ssnorm(x)
        x = x.view(b, c, h, w)

        if self.affine_all:
            x = x * self.weight + self.bias

        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()

        return x
