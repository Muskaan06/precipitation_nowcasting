import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
#  Soft / Hard CSI  (unchanged)
# ─────────────────────────────────────────────

def soft_csi_loss(pred, target, threshold=1.0, smooth=1e-4, steepness=50.0):
    """
    Differentiable soft CSI loss for training.
   
    Fixed: Increased 'steepness' to 50.0 to prevent sigmoid leakage
    from generating massive False Positives on empty frames.
    """
    # Use a higher steepness so sigmoid(-threshold * steepness) approaches 0
    # cleanly, preventing FP accumulation on empty frames.
    pred_bin   = torch.sigmoid((pred - threshold) * steepness)
    target_bin = (target >= threshold).float()

    TP = (pred_bin * target_bin).sum()
    FP = (pred_bin * (1 - target_bin)).sum()
    FN = ((1 - pred_bin) * target_bin).sum()

    # smooth in numerator is required here so that Loss = 0 when TP=FP=FN=0
    csi = (TP + smooth) / (TP + FP + FN + smooth)
    return 1.0 - csi


def hard_csi(pred, target, threshold=1.0, eps=1e-7):
    """
    Hard CSI for validation logging (non-differentiable).
   
    Fixed: Removed 'smooth' from the numerator to prevent True Negatives
    from artificially scoring a perfect 1.0. Returns None for empty frames.
    """
    pred_bin   = (pred   >= threshold).float()
    target_bin = (target >= threshold).float()

    TP = (pred_bin * target_bin).sum()
    FP = (pred_bin * (1 - target_bin)).sum()
    FN = ((1 - pred_bin) * target_bin).sum()

    denominator = TP + FP + FN
    # print(f"TP:{TP}, FP:{FP}, FN:{FN}")
    print(f"denom:{denominator}")
    # If there are no hits, no false alarms, and no misses, CSI is undefined.
    # Return None so your pipeline can skip it (just like your _ssim_2d function).
    if denominator == 0:
        return None

    return TP / (denominator + eps)

# ─────────────────────────────────────────────
#  Differentiable SSIM (PyTorch, for losses)
# ─────────────────────────────────────────────

def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5,
                     channels: int = 1) -> torch.Tensor:
    """Build a 2-D Gaussian kernel for SSIM computation."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g)                             # (k, k)
    return kernel.expand(channels, 1, kernel_size, kernel_size)


def torch_ssim(pred: torch.Tensor,
               target: torch.Tensor,
               kernel_size: int = 11,
               sigma: float = 1.5,
               C1: float = 0.01 ** 2,
               C2: float = 0.03 ** 2) -> torch.Tensor:
    """
    Differentiable SSIM between two (B, 1, H, W) or (B, H, W) tensors.
    Returns a scalar mean SSIM in [-1, 1]  (usually [0, 1] for non-negative data).
    """
    # Ensure 4-D  (B, 1, H, W)
    if pred.ndim == 3:
        pred   = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    B, C, H, W = pred.shape
    kernel = _gaussian_kernel(kernel_size, sigma, channels=C).to(pred.device)
    pad    = kernel_size // 2

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_x  = conv(pred)
    mu_y  = conv(target)

    mu_x2  = mu_x * mu_x
    mu_y2  = mu_y * mu_y
    mu_xy  = mu_x * mu_y

    sigma_x2  = conv(pred   * pred)   - mu_x2
    sigma_y2  = conv(target * target) - mu_y2
    sigma_xy  = conv(pred   * target) - mu_xy

    # Clamp numerical noise
    sigma_x2 = torch.clamp(sigma_x2, min=0)
    sigma_y2 = torch.clamp(sigma_y2, min=0)

    numerator   = (2 * mu_xy  + C1) * (2 * sigma_xy  + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = numerator / denominator          # (B, C, H, W)
    return ssim_map.mean()                      # scalar


# ─────────────────────────────────────────────
#  SSIM loss  (analogous to soft_csi_loss)
# ─────────────────────────────────────────────

def ssim_loss(pred: torch.Tensor,
              target: torch.Tensor,
              kernel_size: int = 11,
              sigma: float = 1.5) -> torch.Tensor:
    """
    Differentiable SSIM loss for training.
    loss = 1 - SSIM  →  minimising drives SSIM toward 1.
    """
    return 1.0 - torch_ssim(pred, target, kernel_size=kernel_size, sigma=sigma)


# ─────────────────────────────────────────────
#  SSIM metric  (numpy, per the formula in Eq. 7)
#
#  SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
#               ──────────────────────────────────────
#               (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)
#
#  x = true precipitation field
#  y = predicted values
#  mu_x, mu_y     : means of x and y
#  sigma_x^2, sigma_y^2 : variances of x and y
#  sigma_xy       : covariance between x and y
#  C1, C2         : stability constants to avoid division by zero
#  SSIM in [0, 1]; higher → better structural preservation
# ─────────────────────────────────────────────

def _ssim_2d(x: np.ndarray, y: np.ndarray,
             C1: float = (0.01 ** 2),
             C2: float = (0.03 ** 2)) -> float | None:
    """
    Compute SSIM for a single (H, W) frame using the formula from Eq. 7:

        SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
                     ──────────────────────────────────────
                     (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)

    Here x is the true precipitation field and y the predicted values.
    mu_x, mu_y are the means; sigma_x^2, sigma_y^2 the variances;
    sigma_xy the covariance; C1, C2 stability constants.
    SSIM ranges from 0 to 1 — higher indicates better preservation of
    the precipitation field's structure and spatial distribution.

    Returns None for degenerate frames (constant variance, all-zero rain
    fields, etc.) so callers can skip them — no 0-pollution in final score.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mu_x = x.mean()
    mu_y = y.mean()

    sigma_x2 = x.var()
    sigma_y2 = y.var()
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    # Degenerate: both fields are constant — SSIM is undefined
    if sigma_x2 == 0 and sigma_y2 == 0:
        return None

    numerator   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)

    score = numerator / denominator

    if np.isnan(score) or np.isinf(score):
        return None

    return float(score)


def _valid_mean(scores):
    """Average a list that may contain None entries, skipping them.
    Returns 0.0 only if *every* frame was degenerate (truly no valid data).
    """
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def compute_ssim(pred, actual):
    """
    Compute Structural Similarity Index (SSIM) using Eq. 7.

    SSIM is computed independently for each frame, then the results are
    averaged. Degenerate frames (constant variance, all-zero rain fields,
    etc.) are skipped and excluded from the average — they do NOT
    contribute a 0.

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Shape can be:
            (B, T, H, W)  — SSIM computed per (b, t) frame, then averaged
            (T, H, W)     — SSIM computed per t frame, then averaged
            (H, W)        — SSIM computed for the single frame
    actual : same shape as pred

    Returns
    -------
    float : mean SSIM score over valid frames, in [0, 1]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()

    # (B, T, H, W): compute SSIM per (b, t) frame, then average
    if pred.ndim == 4:
        B, T, H, W = pred.shape
        scores = [
            _ssim_2d(pred[b, t], actual[b, t])
            for b in range(B)
            for t in range(T)
        ]
        return _valid_mean(scores)

    # (T, H, W): compute SSIM per t frame, then average
    elif pred.ndim == 3:
        T = pred.shape[0]
        scores = [_ssim_2d(pred[t], actual[t]) for t in range(T)]
        return _valid_mean(scores)

    # (H, W): single frame
    elif pred.ndim == 2:
        score = _ssim_2d(pred, actual)
        return float(score) if score is not None else 0.0

    else:
        raise ValueError(f"Unsupported shape for SSIM computation: {pred.shape}")


# ─────────────────────────────────────────────
#  exp_weighted_temporal_ssim
# ─────────────────────────────────────────────

def exp_weighted_temporal_ssim(pred, actual):
    """
    Exponentially weighted temporal SSIM over a (B, T, H, W) batch,
    using the Eq. 7 SSIM formula.

    Later time-steps receive higher weight:
        w(t) = exp(t / T),   t in {1, ..., T}

    Degenerate frames (NaN / constant) are skipped per time-step; only
    valid batch scores contribute to each time-step mean, and only
    time-steps that have at least one valid score contribute to the
    weighted sum — so no 0s pollute the final average.

    Parameters
    ----------
    pred   : torch.Tensor or np.ndarray,  shape (B, T, H, W)
    actual : torch.Tensor or np.ndarray,  shape (B, T, H, W)

    Returns
    -------
    float : exponentially weighted mean SSIM over valid frames
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()

    B, T, H, W = pred.shape

    t_idx   = np.arange(1, T + 1)
    weights = np.exp(t_idx / T)            # shape (T,)

    total_score = 0.0
    total_weight = 0.0

    for t in range(T):
        batch_scores = [
            _ssim_2d(pred[b, t], actual[b, t])
            for b in range(B)
        ]
        # Only use valid (non-None) scores for this time-step
        valid = [s for s in batch_scores if s is not None]
        if not valid:
            # Entire time-step is degenerate — skip it completely
            continue

        mean_ssim_t   = float(np.mean(valid))
        total_score  += weights[t] * mean_ssim_t
        total_weight += weights[t]

    if total_weight == 0:
        return 0.0

    return float(total_score / total_weight)