"""
NowcastNet model components for rainfall nowcasting.

Contains:
    NowcasnetGenerator    -- generator with noise spatial-size fix
    TemporalDiscriminator -- discriminator rebuilt for arbitrary img_size

Both are drop-in replacements for the originals in
src/models/nowcastnet/gan/model.py, corrected for 112x112 input
(original was hardcoded for 256x256).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as sn

from model.nowcastnet.gan.model import NowcasnetGenerator as _NowcasnetGenerator
from model.nowcastnet.gan.gan_parts import L2_Block


# ------------------------------------------------------------------------------
# GENERATOR
# Wraps _NowcasnetGenerator and fixes the noise spatial mismatch.
#
# Root cause: Generative_Encoder does 3x MaxPool2d(2) on 112x112 input,
# producing x_encoded at 14x14. PixelShuffle(4) can only output spatial
# sizes that are multiples of 4, so 14 is unreachable.
# Fix: F.adaptive_avg_pool2d resizes noise_feature to match x_encoded exactly.
# ------------------------------------------------------------------------------

class NowcasnetGenerator(nn.Module):
    """
    Generator for NowcastNet GAN.

    Args:
        channel_in  : total input channels = input_len + output_len  (e.g. 10)
        latent_dim  : noise latent dimension  (default 32)
        n_after     : number of frames to predict = output_len  (default 6)

    Input:
        x : (B, channel_in, H, W)  — cat(raw_context_frames, evo_prediction)
        z : (B, latent_dim, h_noise, h_noise)  — random noise

    Output:
        (B, n_after, H, W)  — predicted rainfall frames
    """
    def __init__(self, channel_in: int, latent_dim: int, n_after: int):
        super().__init__()
        self.inner = _NowcasnetGenerator(
            channel_in=channel_in,
            latent_dim=latent_dim,
            n_after=n_after,
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inner         = self.inner
        # Last n_after frames of x are the evolution prediction (evo signal).
        # Divide by 40 to normalise to roughly [0, 1] — matches lightning.py.
        pred          = x[:, -inner.n_after:, :, :] / 40.0       # (B, n_after, H, W)
        x_encoded     = inner.gen_encoder(x)                      # (B, ngf*8, H/8, W/8)
        noise_feature = inner.deeptospace(inner.noise_projector(z))
        # Resize noise to match x_encoded spatial dims (works for any img_size)
        noise_feature = F.adaptive_avg_pool2d(
            noise_feature, output_size=x_encoded.shape[-2:]
        )
        feature = torch.cat([x_encoded, noise_feature], dim=1)
        return inner.gen_decoder(feature, pred)                   # (B, n_after, H, W)

    # Delegate parameter/state methods to the wrapped inner model so that
    # optimisers and checkpointing work transparently.
    def parameters(self, recurse=True):
        return self.inner.parameters(recurse=recurse)

    def state_dict(self, **kw):
        return self.inner.state_dict(**kw)

    def load_state_dict(self, state_dict, strict=True):
        return self.inner.load_state_dict(state_dict, strict=strict)

    def train(self, mode=True):
        self.inner.train(mode)
        return self

    def eval(self):
        return self.train(False)


# ------------------------------------------------------------------------------
# DISCRIMINATOR
# Rebuilt from scratch to support arbitrary img_size.
#
# Root cause: model.py TemporalDiscriminator has `size=16` hardcoded,
# which assumes 256x256 input (Linear(256, 1)). For 112x112 the spatial
# size before Flatten is 7x7=49, causing a shape mismatch.
# Fix: measure flat_size via a dummy forward pass, build Linear dynamically.
#
# Architecture mirrors model.py exactly:
#   conv2d   : 2D spectral-norm conv  (B, C, H, W) -> (B, 64, H/2, W/2)
#   conv3d_1 : 3D spectral-norm conv  extracts short-range temporal features
#   conv3d_2 : 3D spectral-norm conv  extracts long-range temporal features
#   main2d_conv : 4x L2_Block + BN + Conv2d + LeakyReLU -> (B, 1, H', W')
#   fc       : spectral-norm Linear(flat_size, 1) -> raw logits (B, 1)
#
# The discriminator returns RAW LOGITS (no sigmoid).
# Use F.binary_cross_entropy_with_logits in the training loop.
# ------------------------------------------------------------------------------

class TemporalDiscriminator(nn.Module):
    """
    Temporal discriminator for NowcastNet GAN.

    Args:
        in_channel : total input channels = input_len + output_len  (e.g. 10)
        img_size   : spatial size of input frames  (default 112)

    Input:
        x : (B, in_channel, H, W)  — cat(raw_context, real_or_fake_target)

    Output:
        (B, 1)  — raw logits (use BCEWithLogitsLoss, not BCE)
    """
    def __init__(self, in_channel: int, img_size: int = 112):
        super().__init__()
        ndf          = 32
        kernel_size  = 4
        stride       = 2
        padding      = 1
        # Channel count after concatenating the three conv branches —
        # matches the `after_flatten` formula in model.py exactly.
        after_flatten = 64 + (in_channel - 3) * 4 + (in_channel // 4 + 1) * 8

        self.conv2d   = sn(nn.Conv2d(in_channel, 64, kernel_size=9, stride=2, padding=4))
        self.conv3d_1 = sn(nn.Conv3d(1, 4, kernel_size=(4, 9, 9),
                                     stride=(1, 2, 2), padding=(0, 4, 4)))
        self.conv3d_2 = sn(nn.Conv3d(1, 8, kernel_size=(4, 9, 9),
                                     stride=(4, 2, 2), padding=(2, 4, 4)))

        self.main2d_conv = nn.Sequential(
            L2_Block(after_flatten, ndf * 4,  kernel_size,     stride,     padding, bias=False),
            L2_Block(ndf * 4,       ndf * 8,  kernel_size,     stride,     padding, bias=False),
            L2_Block(ndf * 8,       ndf * 16, kernel_size,     stride,     padding, bias=False),
            L2_Block(ndf * 16,      ndf * 16, kernel_size - 1, stride - 1, padding, bias=False),
            nn.BatchNorm2d(ndf * 16),
            sn(nn.Conv2d(ndf * 16, 1, kernel_size=3, padding=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
        )

        # Measure the actual flat size for this img_size via a dummy forward
        with torch.no_grad():
            dummy     = torch.zeros(1, in_channel, img_size, img_size)
            flat_size = self._conv_forward(dummy).shape[1]

        self.fc = sn(nn.Linear(flat_size, 1))
        print(f"[discriminator] in_channel={in_channel}  "
              f"img_size={img_size}  flat_size={flat_size}")

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """All conv branches + flatten. Returns (B, flat_size)."""
        t3d  = x[:, None, :, :, :]                                      # (B,1,C,H,W)
        x2   = torch.flatten(self.conv3d_1(t3d), start_dim=1, end_dim=2)
        x3   = torch.flatten(self.conv3d_2(t3d), start_dim=1, end_dim=2)
        feat = self.main2d_conv(torch.cat([self.conv2d(x), x2, x3], dim=1))
        return feat.flatten(start_dim=1)                                 # (B, flat_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, 1). Use BCEWithLogitsLoss."""
        return self.fc(self._conv_forward(x))
