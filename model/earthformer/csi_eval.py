import torch
import torch.nn as nn


def soft_csi_loss(pred, target, threshold=1.0, smooth=1e-4):
    """
    Differentiable soft CSI loss for training.
    Uses sigmoid to approximate hard thresholding.
    CSI = TP / (TP + FP + FN), loss = 1 - CSI
    """
    # soft binarization via sigmoid
    pred_bin   = torch.sigmoid((pred   - threshold) * 10)
    target_bin = (target >= threshold).float()

    TP = (pred_bin * target_bin).sum()
    FP = (pred_bin * (1 - target_bin)).sum()
    FN = ((1 - pred_bin) * target_bin).sum()

    csi  = (TP + smooth) / (TP + FP + FN + smooth)
    return 1 - csi  # minimize loss → maximize CSI


def hard_csi(pred, target, threshold=1.0, smooth=1e-4):
    """
    Hard CSI for validation logging (non-differentiable).
    """
    pred_bin   = (pred   >= threshold).float()
    target_bin = (target >= threshold).float()

    TP = (pred_bin * target_bin).sum()
    FP = (pred_bin * (1 - target_bin)).sum()
    FN = ((1 - pred_bin) * target_bin).sum()

    return (TP + smooth) / (TP + FP + FN + smooth)