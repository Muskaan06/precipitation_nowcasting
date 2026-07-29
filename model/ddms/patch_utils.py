import torch
import numpy as np


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


def to_pm1(x):
    """[0, 1] -> [-1, 1], DDMS's internal diffusion-model convention."""
    return x * 2.0 - 1.0


def from_pm1(x):
    """[-1, 1] -> [0, 1]."""
    return (x + 1.0) * 0.5
