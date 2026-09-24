"""
Learned Fourier encoder of the lat/lon/day-of-year coordinates, following
Li et al. (2021), "Learnable Fourier Features for Multi-Dimensional Spatial
Positional Encoding", NeurIPS:

    F(x) = 1/sqrt(M) [cos(W x), sin(W x)],   emb = MLP(F(x))

W (M frequencies) and the MLP are trained jointly with the UNet. Applied
per pixel, so W and the MLP are 1x1 convolutions. The input x is the raw
coordinate stack of coord_embeddings.build_raw_coord_channels (lon and DoY on
the unit circle), which keeps the embedding periodic in lon and DoY.
"""

import math

import torch
from torch import nn


class LearnedFourierCoordEncoder(nn.Module):
    def __init__(self, in_channels=5, num_frequencies=32, hidden_channels=64,
                 out_channels=16, sigma=8.0):
        """
        sigma: std of the initial frequencies W ~ N(0, sigma^2). Inputs lie in
            [-1, 1], so sigma=8 starts with a spread of scales from basin-wide
            down to a few degrees; training then moves them.
        """
        super().__init__()
        self.freqs = nn.Conv2d(in_channels, num_frequencies, kernel_size=1, bias=False)
        nn.init.normal_(self.freqs.weight, mean=0.0, std=sigma)
        self.scale = 1.0 / math.sqrt(num_frequencies)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * num_frequencies, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, coords):
        z = self.freqs(coords)
        features = torch.cat([torch.cos(z), torch.sin(z)], dim=1) * self.scale
        return self.mlp(features)
