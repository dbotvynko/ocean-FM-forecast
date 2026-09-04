"""
Flow-matching-compatible port of the "17M" UNet architecture from
dbotvynko/sst-multivar-forecast (src/models_UNet.py + src/parts.py).

The original there is a plain deterministic encoder-decoder -- forward(x)
only, no notion of a diffusion/flow timestep, trained directly with MSE +
a Sobel gradient loss. GenFlowLit (contrib/generative/models.py) instead
calls solver(xt=..., y=..., t=...) and needs the network to condition on
t to predict the correct velocity at each point along the noise->data
path, which that architecture has no mechanism for.

So the blocks below are vendored copies of that repo's StandardBlock/
ResBlock/Down/Up/OutConv (not a cross-repo import, since ocean-FM-forecast
and sst-multivar-forecast both define a top-level `contrib` package with
different contents, and vendoring avoids needing both repos on
sys.path/PYTHONPATH together at training time), with additive FiLM
time-conditioning added to each ResBlock -- injected the same way
contrib/generative/model_utils.py's ResNetBlock does it for the existing
unet.py: a sinusoidal positional embedding, projected per-block through a
small MLP, added to the feature map after the first conv. The very first
(StandardBlock) and last (OutConv) layers stay unconditioned, matching
unet.py's own convention (its initial_conv/output_conv also carry no time
embedding -- conditioning happens in the down/up blocks only).

Channel structure (64 -> 128 -> 256 -> 512 -> 1024/factor) and residual
scaling factors are unchanged from the original, so the parameter count
stays close to that "17M" architecture's.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from contrib.generative.model_utils import TransformerPositionalEmbedding


class StandardBlock(nn.Module):
    """Verbatim from sst-multivar-forecast/src/parts.py -- used unconditioned for the input stem."""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, dilation=1):
        super().__init__()
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResBlockTimeCond(nn.Module):
    """
    sst-multivar-forecast's ResBlock, split around its first conv so a
    time embedding can be added there (same injection point as
    contrib/generative/model_utils.py's ResNetBlock).
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, sf=1, time_emb_channels=None):
        super().__init__()
        self._scaling_factor = sf
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.time_projection = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_channels, mid_channels))
            if time_emb_channels
            else None
        )
        if in_channels != out_channels:
            self.projection_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, time_embedding):
        h = self.conv1(x)
        h = h + self.time_projection(time_embedding)[:, :, None, None]
        out = self.conv2(h)

        residual = self.projection_conv(x) if hasattr(self, "projection_conv") else x
        out = out * self._scaling_factor + residual
        return F.relu(out)


class DownTimeCond(nn.Module):
    """Downscaling with maxpool then a time-conditioned block."""

    def __init__(self, in_channels, out_channels, block=ResBlockTimeCond, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = block(in_channels, out_channels, **kwargs)

    def forward(self, x, time_embedding):
        x = self.pool(x)
        return self.conv(x, time_embedding)


class UpTimeCond(nn.Module):
    """Upscaling then a time-conditioned block."""

    def __init__(self, in_channels, out_channels, block=ResBlockTimeCond, bilinear=True, **kwargs):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = block(in_channels, out_channels, in_channels // 2, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = block(in_channels, out_channels, **kwargs)

    def forward(self, x1, x2, time_embedding):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, time_embedding)


class OutConv(nn.Module):
    """Verbatim from sst-multivar-forecast/src/parts.py -- unconditioned output projection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.out(x)


class UNet17M(nn.Module):
    """
    Flow-matching version of sst-multivar-forecast's 17M-parameter UNet
    (src/models_UNet.py:UNet). Same channel structure and residual
    scaling factors; forward(xt, y, t) instead of forward(x), matching
    GenFlowLit's calling convention (contrib/generative/models.py),
    with the FiLM time-conditioning described at the top of this file.
    """

    def __init__(self, input_channels=58, output_channels=29, bilinear=True, max_steps=100):
        super().__init__()
        factor = 2 if bilinear else 1
        time_emb_dim = 128 * 4

        # block-wise residual scaling factors, same as the original architecture
        sfs = 1 / torch.arange(1, 6).sqrt()

        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128, max_steps=max_steps),
            nn.Linear(128, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.inc = StandardBlock(input_channels, 64)
        self.down1 = DownTimeCond(64, 128, sf=sfs[1], time_emb_channels=time_emb_dim)
        self.down2 = DownTimeCond(128, 256, sf=sfs[2], time_emb_channels=time_emb_dim)
        self.down3 = DownTimeCond(256, 512, sf=sfs[3], time_emb_channels=time_emb_dim)
        self.down4 = DownTimeCond(512, 1024 // factor, sf=sfs[4], time_emb_channels=time_emb_dim)

        self.up1 = UpTimeCond(1024, 512 // factor, bilinear=bilinear, sf=sfs[4], time_emb_channels=time_emb_dim)
        self.up2 = UpTimeCond(512, 256 // factor, bilinear=bilinear, sf=sfs[3], time_emb_channels=time_emb_dim)
        self.up3 = UpTimeCond(256, 128 // factor, bilinear=bilinear, sf=sfs[2], time_emb_channels=time_emb_dim)
        self.up4 = UpTimeCond(128, 64, bilinear=bilinear, sf=sfs[1], time_emb_channels=time_emb_dim)
        self.outc = OutConv(64, output_channels)

    def forward(self, xt, y, t):
        input_tensor = torch.cat((xt, y), dim=1)
        time_embedding = self.positional_encoding(t)

        x1 = self.inc(input_tensor)
        x2 = self.down1(x1, time_embedding)
        x3 = self.down2(x2, time_embedding)
        x4 = self.down3(x3, time_embedding)
        x5 = self.down4(x4, time_embedding)

        x = self.up1(x5, x4, time_embedding)
        x = self.up2(x, x3, time_embedding)
        x = self.up3(x, x2, time_embedding)
        x = self.up4(x, x1, time_embedding)
        return self.outc(x)
