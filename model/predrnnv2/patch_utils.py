import torch
import numpy as np


def reshape_patch(img_tensor, patch_size):
    """
    Torch version of PredRNN's patchify trick.
    [B, T, H, W, C] -> [B, T, H/p, W/p, p*p*C]
    """
    assert img_tensor.ndim == 5
    B, T, H, W, C = img_tensor.shape
    a = img_tensor.reshape(B, T, H // patch_size, patch_size, W // patch_size, patch_size, C)
    b = a.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return b.reshape(B, T, H // patch_size, W // patch_size, patch_size * patch_size * C)


def reshape_patch_back(patch_tensor, patch_size):
    """
    Inverse of reshape_patch.
    [B, T, h, w, p*p*C] -> [B, T, h*p, w*p, C]
    """
    assert patch_tensor.ndim == 5
    B, T, h, w, ch = patch_tensor.shape
    img_channels = ch // (patch_size * patch_size)
    a = patch_tensor.reshape(B, T, h, w, patch_size, patch_size, img_channels)
    b = a.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return b.reshape(B, T, h * patch_size, w * patch_size, img_channels)


def mmhr_to_intensity_with_normalize(x, R_max=60.0):
    """
    Forward: mm/hr -> [0, 1]
    """
    norm = torch.log1p(x) / np.log1p(R_max)
    return torch.clamp(norm, 0.0, 1.0)


def to_pixel_intensity(x_norm):
    """
    Converts normalized [0, 1] float tensor to [0, 255] uint8.
    """
    pixels = x_norm * 255.0
    return torch.clamp(pixels, 0, 255).to(torch.uint8)
