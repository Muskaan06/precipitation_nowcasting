"""
TUPANN model — independent of repo base classes.

Two-stage architecture:
  Stage 1 — AutoencoderKL: encodes input frames -> latent, decodes -> motion field + intensity
  Stage 2 — TUPANN: freezes autoencoder, uses MaxViT (Metnet) to evolve latent state
             then warps last observed frame with motion field + adds intensity correction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR


# ── Utility functions ─────────────────────────────────────────────────────────

def make_grid(input, device="cpu"):
    B, C, H, W = input.size()
    xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    return torch.cat((xx, yy), 1).float()


def warp(input, flow, grid, mode="bilinear", padding_mode="zeros", fill_value=0.0):
    B, C, H, W = input.size()
    vgrid = grid - flow
    vgrid[:, 0] = 2.0 * vgrid[:, 0].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(input - fill_value, vgrid, padding_mode=padding_mode,
                         mode=mode, align_corners=True) + fill_value


# ── Autoencoder building blocks ───────────────────────────────────────────────

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1   = Normalize(in_channels)
        self.conv1   = nn.Conv2d(in_channels,  out_channels, 3, 1, 1)
        self.norm2   = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.skip    = nn.Conv2d(in_channels,  out_channels, 1, 1, 0) if in_channels != out_channels else None
        self.act     = Swish()

    def forward(self, x, temb=None):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + (self.skip(x) if self.skip else x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm     = Normalize(in_channels)
        self.q        = nn.Conv2d(in_channels, in_channels, 1)
        self.k        = nn.Conv2d(in_channels, in_channels, 1)
        self.v        = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, h_, w_ = h.shape
        q = self.q(h).reshape(b, c, -1).permute(0, 2, 1)
        k = self.k(h).reshape(b, c, -1)
        v = self.v(h).reshape(b, c, -1)
        attn = F.softmax(torch.bmm(q, k) * (c ** -0.5), dim=2)
        out  = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, h_, w_)
        return x + self.proj_out(out)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 5, 4, 0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=4.0, mode="nearest"))

class Encoder(nn.Module):
    def __init__(self, ch, in_channels, resolution, z_channels, num_res_blocks,
                 ch_mult=(1,2,4,8), dropout=0.0, double_z=True):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks  = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        self.down = nn.ModuleList()
        block_in = ch
        for i, mult in enumerate(ch_mult):
            block_out = ch * mult
            blocks = nn.ModuleList([ResnetBlock(in_channels=block_in if j==0 else block_out,
                                                out_channels=block_out, dropout=dropout)
                                    for j in range(num_res_blocks)])
            down = nn.Module()
            down.block = blocks
            down.attn  = nn.ModuleList()
            if i != self.num_resolutions - 1:
                down.downsample = Downsample(block_out)
            self.down.append(down)
            block_in = block_out

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1  = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, 3, 1, 1)
        self.act      = Swish()

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i, down in enumerate(self.down):
            for block in down.block:
                hs.append(block(hs[-1]))
            if hasattr(down, 'downsample'):
                hs.append(down.downsample(hs[-1]))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1])))
        return self.conv_out(self.act(self.norm_out(h)))

class Decoder(nn.Module):
    def __init__(self, ch, out_ch, in_channels, resolution, z_channels,
                 num_res_blocks, ch_mult=(1,2,4,8), dropout=0.0):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks  = num_res_blocks
        block_in = ch * ch_mult[-1]

        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1  = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i]
            blocks = nn.ModuleList([ResnetBlock(in_channels=block_in if j==0 else block_out,
                                                out_channels=block_out, dropout=dropout)
                                    for j in range(num_res_blocks + 1)])
            up = nn.Module()
            up.block = blocks
            up.attn  = nn.ModuleList()
            if i != 0:
                up.upsample = Upsample(block_out)
            self.up.insert(0, up)
            block_in = block_out

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)
        self.act      = Swish()

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for i in reversed(range(self.num_resolutions)):
            for block in self.up[i].block:
                h = block(h)
            if hasattr(self.up[i], 'upsample'):
                h = self.up[i].upsample(h)
        return self.conv_out(self.act(self.norm_out(h)))


# ── Variational distribution ──────────────────────────────────────────────────

class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self):
        return 0.5 * torch.sum(self.mean**2 + torch.exp(self.logvar) - 1.0 - self.logvar, dim=[1,2,3])

    def mode(self):
        return self.mean


# ── AutoencoderKL ─────────────────────────────────────────────────────────────

class AutoencoderKL(nn.Module):
    """
    Variational autoencoder.
    Encodes input sequence -> latent, decodes -> (motion_field, intensity).
    input : (B, input_len, H, W)
    output: (B, 3, H, W) — channels: flow_x, flow_y, intensity
    """
    def __init__(self, input_len=4, img_size=112, embed_dim=6, reduc_factor=4,
                 channels=64, dropout=0.0, n_fields=1):
        super().__init__()
        self.embed_dim    = embed_dim
        self.reduc_factor = reduc_factor
        z_channels = 4
        ch_mult    = [2**i for i in range(int(np.log2(reduc_factor) // 2) + 1)]

        self.encoder = Encoder(ch=channels, in_channels=input_len, resolution=img_size,
                               z_channels=z_channels, num_res_blocks=2,
                               ch_mult=ch_mult, dropout=dropout)
        self.decoder = Decoder(ch=channels, out_ch=(2*n_fields + 1), in_channels=input_len,
                               resolution=img_size, z_channels=z_channels,
                               num_res_blocks=2, ch_mult=ch_mult, dropout=dropout)

        self.quant_conv      = nn.Conv2d(2*z_channels, 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim,    z_channels,  1)

    def encode(self, x):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        posterior = self.encode(x)
        z         = posterior.sample()
        return self.decode(z), posterior


# ── Simple MaxViT-style latent model (Metnet substitute) ─────────────────────

class MBConv(nn.Module):
    """Mobile inverted bottleneck conv block."""
    def __init__(self, dim, expansion=4, shrinkage=0.25, dropout=0.1):
        super().__init__()
        hidden = int(dim * expansion)
        se_dim = max(1, int(hidden * shrinkage))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.se = nn.Sequential(nn.Linear(dim, se_dim), nn.ReLU(),
                                nn.Linear(se_dim, dim), nn.Sigmoid())

    def forward(self, x):
        return x + self.net(x) * self.se(x.mean(dim=1, keepdim=True))

class WindowAttention(nn.Module):
    """Local window self-attention."""
    def __init__(self, dim, heads=4, window_size=8, dropout=0.1):
        super().__init__()
        self.heads       = heads
        self.window_size = window_size
        self.scale       = (dim // heads) ** -0.5
        self.to_qkv      = nn.Linear(dim, dim * 3, bias=False)
        self.to_out      = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.norm        = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        ws = min(self.window_size, H, W)
        x_ = self.norm(x)
        # partition into windows
        x_ = x_.reshape(B*T, H//ws, ws, W//ws, ws, C).permute(0,1,3,2,4,5)
        x_ = x_.reshape(-1, ws*ws, C)
        qkv = self.to_qkv(x_).chunk(3, dim=-1)
        q, k, v = [t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1,2) for t in qkv]
        attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) * self.scale, dim=-1)
        out  = torch.matmul(attn, v).transpose(1,2).reshape(x_.shape[0], ws*ws, C)
        out  = self.to_out(out)
        # unpartition
        out = out.reshape(B*T, H//ws, W//ws, ws, ws, C).permute(0,1,3,2,4,5)
        out = out.reshape(B, T, H, W, C)
        return x + out

class MaxViTBlock(nn.Module):
    """One MaxViT block: MBConv -> local window attn -> MBConv -> grid attn."""
    def __init__(self, dim, heads=4, window_size=4, dropout=0.1):
        super().__init__()
        self.mbconv1      = MBConv(dim, dropout=dropout)
        self.local_attn   = WindowAttention(dim, heads, window_size, dropout)
        self.mbconv2      = MBConv(dim, dropout=dropout)

    def forward(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x = x.reshape(B*T, H, W, C)
        x = self.mbconv1(x.reshape(B*T, H*W, C)).reshape(B*T, H, W, C)
        x = x.reshape(B, T, H, W, C)
        x = self.local_attn(x)
        B, T, H, W, C = x.shape
        x = x.reshape(B*T, H*W, C)
        x = self.mbconv2(x).reshape(B, T, H, W, C)
        return x

class LatentMaxViT(nn.Module):
    """
    Evolves latent representation across lead times.
    input : (B, embed_dim, h, w) — latent from autoencoder
    output: (B, embed_dim, h, w) — evolved latent for target lead time
    """
    def __init__(self, embed_dim, dim=16, depth=4, heads=4, window_size=4,
                 target_length=6, dropout=0.1):
        super().__init__()
        self.target_length = target_length
        self.proj_in  = nn.Linear(embed_dim, dim)
        self.blocks   = nn.ModuleList([MaxViTBlock(dim, heads, window_size, dropout)
                                       for _ in range(depth)])
        self.proj_out = nn.Linear(dim, embed_dim * target_length)
        self.norm     = nn.LayerNorm(dim)

    def forward(self, z, lead_time=None):
        # z: (B, C, h, w)
        B, C, h, w = z.shape
        x = z.permute(0, 2, 3, 1)              # (B, h, w, C)
        x = self.proj_in(x)                    # (B, h, w, dim)
        x = x.unsqueeze(1)                     # (B, 1, h, w, dim) — T=1
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).squeeze(1)            # (B, h, w, dim)
        x = self.proj_out(x)                   # (B, h, w, C*T)
        x = x.reshape(B, h, w, C, self.target_length)
        x = x.permute(0, 4, 3, 1, 2)          # (B, T, C, h, w)
        return x                               # one latent per lead time


# ── Full TUPANN model ─────────────────────────────────────────────────────────

class TUPANN(nn.Module):
    """
    Full TUPANN model.
    input_len    : number of context frames (4)
    target_length: number of frames to predict (6)
    img_size     : spatial size (112)

    Forward pass:
        1. Encode context frames -> latent
        2. Evolve latent with MaxViT -> one latent per lead time
        3. Decode each latent -> (motion field, intensity)
        4. Warp last observed frame + add intensity -> predicted frame
    """
    def __init__(self, input_len=4, target_length=6, img_size=112,
                 embed_dim=6, reduc_factor=4, channels=64, dropout=0.0,
                 maxvit_dim=16, maxvit_depth=4, maxvit_heads=4, window_size=4):
        super().__init__()
        self.target_length = target_length
        self.img_size      = img_size

        self.autoencoder = AutoencoderKL(
            input_len=input_len, img_size=img_size, embed_dim=embed_dim,
            reduc_factor=reduc_factor, channels=channels, dropout=dropout,
        )
        self.latent_model = LatentMaxViT(
            embed_dim=embed_dim, dim=maxvit_dim, depth=maxvit_depth,
            heads=maxvit_heads, window_size=window_size,
            target_length=target_length, dropout=dropout,
        )
        # grid registered as buffer so it moves with .to(device)
        sample = torch.zeros(1, 1, img_size, img_size)
        self.register_buffer("grid", make_grid(sample))

    def forward(self, x):
        """
        x   : (B, input_len, H, W) — rain channel context frames
        Returns: (B, target_length, H, W) — predicted rain frames
        """
        B = x.shape[0]
        grid = self.grid.repeat(B, 1, 1, 1)

        # encode context -> latent
        latent = self.autoencoder.encode(x).sample()   # (B, embed_dim, h, w)

        # evolve latent for all lead times
        future_latents = self.latent_model(latent)     # (B, T, embed_dim, h, w)

        current_frame = x[:, -1:]                      # last observed frame (B, 1, H, W)
        out = []

        for t in range(self.target_length):
            decoded = self.autoencoder.decode(future_latents[:, t])  # (B, 3, H, W)
            field_pred, intensity_pred = decoded[:, :2], decoded[:, 2:3]

            # warp last frame with motion field, add intensity correction
            pred_frame = warp(current_frame, field_pred, grid,
                              padding_mode="zeros", fill_value=0.0) + intensity_pred
            pred_frame = torch.clamp(pred_frame, min=0.0)  # rain is non-negative
            out.append(pred_frame)
            current_frame = pred_frame                 # autoregressive

        return torch.cat(out, dim=1)                   # (B, target_length, H, W)