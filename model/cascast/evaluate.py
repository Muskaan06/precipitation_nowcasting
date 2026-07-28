"""
evaluate.py
===========
Evaluation script for CasCast on satellite rainfall data.

Aligned to train.py (same pattern as ../tupann/evaluate.py is aligned to
../tupann/train.py, and ../earthformer/evaluate.py to ../earthformer/train.py):

  - NPZDataset         : one sample per .npz file, explicit 'array' key
  - preprocess_rain()  : clamp -> ln(1+R) / ln(1+128) -> [0, 1]
  - CasFormer          : DDIM sampling conditioned on a coarse prediction
                         (frozen Earthformer checkpoint, or persistence fallback)
  - Checkpoint format  : PyTorch Lightning .ckpt (key: "state_dict", prefix: "castformer.")

Metrics produced — identical set/formulas to ../tupann/evaluate.py, via the
shared eval_metrics.py (copied unchanged from tupann):
  MSE, MAE                         — normalised [0, 1] space
  Hard CSI / POD / FAR @ thresholds — mm/hr (physical)
  Soft CSI loss @ thresholds       — mm/hr
  SSIM per timestep + overall      — normalised [0, 1] space
  Exp-weighted temporal SSIM       — normalised [0, 1] space

Usage
-----
    python evaluate.py --ckpt ./checkpoint/cascast_best.ckpt --data_dir /path/to/data
    python evaluate.py --ckpt ./checkpoint/cascast_best.ckpt --coarse_ckpt none  # persistence baseline
"""

import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset

from model_arch import CasFormer
from diffusion_utils import (
    GaussianDiffusion,
    load_coarse_forecaster,
    run_coarse_forecaster,
    persistence_coarse,
    DEFAULT_COARSE_CKPT,
    EARTHFORMER_INPUT_LENGTH,
    EARTHFORMER_TARGET_LENGTH,
)

from eval_metrics import (
    hard_csi,
    hard_pod,
    hard_far,
    soft_csi_loss,
    compute_ssim,
    exp_weighted_temporal_ssim,
)


# =============================================================================
# Configuration  (override via CLI)
# =============================================================================

DEFAULTS = dict(
    data_dir            = "/media/kpdubey/8.0 TB Volume/Muskaan/nowcast/IMC_Combined",
    input_length        = EARTHFORMER_INPUT_LENGTH,
    target_length       = EARTHFORMER_TARGET_LENGTH,
    img_size            = 112,
    patch_size          = 4,
    hidden_size         = 384,
    enc_hidden_size     = 128,
    enc_depth           = 4,
    latent_depth        = 8,
    num_heads           = 6,
    diffusion_timesteps = 1000,
    num_inference_steps = 20,
    train_split         = 0.7,
    seed                = 42,
    batch_size          = 16,
    num_workers         = 0,
    ckpt                = "./checkpoint/cascast_best.ckpt",
    coarse_ckpt         = DEFAULT_COARSE_CKPT,
    out_dir             = "./eval_outputs",
    num_vis             = 5,
    device              = "cuda" if torch.cuda.is_available() else "cpu",
)

FALLBACK_CSI_THRESHOLDS_MMHR = [0.5, 1.0, 2.0, 5.0, 10.0]


# =============================================================================
# Preprocessing  (identical to train.py)
# =============================================================================

RAIN_MAX_MMHR: float = 128.0
_LOG_NORM_DENOM: float = float(np.log1p(RAIN_MAX_MMHR))


def preprocess_rain(x: np.ndarray) -> np.ndarray:
    """Raw mm/hr -> normalised [0, 1] log-rain space."""
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    x = np.clip(x, 0.0, RAIN_MAX_MMHR)
    x = np.log1p(x) / _LOG_NORM_DENOM
    return x


def denorm_to_mmhr(x_norm: torch.Tensor) -> torch.Tensor:
    """Inverse of preprocess_rain: normalised [0, 1] -> mm/hr."""
    r = torch.expm1(x_norm.clamp(0.0, 1.0) * _LOG_NORM_DENOM)
    return r.clamp(0.0, RAIN_MAX_MMHR)


def normalized_to_pixel(x) -> torch.Tensor:
    """Normalized [0, 1] log-rain space -> pixel intensity [0, 255] as float tensor."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.clamp(0.0, 1.0)
    return (x * 255.0).to(torch.uint8)


# =============================================================================
# Dataset  (matches NPZDataset in train.py exactly)
# =============================================================================

class NPZDataset(Dataset):
    def __init__(self, files, input_length=EARTHFORMER_INPUT_LENGTH, target_length=EARTHFORMER_TARGET_LENGTH):
        self.files = files
        self.input_length = input_length
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=True) as f:
            arr = np.array(f['array'], dtype=np.float32).copy()

        rain = arr[:, 0, :, :]
        rain = preprocess_rain(rain)

        x = torch.from_numpy(rain[:self.input_length]).contiguous()
        y = torch.from_numpy(rain[self.input_length:self.input_length + self.target_length]).contiguous()
        return x, y


# =============================================================================
# Model loading
# =============================================================================

def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    castformer_config = dict(
        input_size=DEFAULTS["img_size"],
        patch_size=DEFAULTS["patch_size"],
        in_channels=2,
        out_channels=DEFAULTS["target_length"],
        hidden_size=DEFAULTS["hidden_size"],
        enc_hidden_size=DEFAULTS["enc_hidden_size"],
        split_num=DEFAULTS["target_length"],
        enc_depth=DEFAULTS["enc_depth"],
        latent_depth=DEFAULTS["latent_depth"],
        num_heads=DEFAULTS["num_heads"],
    )
    model = CasFormer(arch="DiT-custom", config=castformer_config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    prefix = "castformer."
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        state = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        missing, unexpected = model.load_state_dict(state, strict=True)
        epoch = ckpt.get("epoch", "?")
        print(f"  [ckpt] loaded Lightning checkpoint <- {ckpt_path}  (epoch {epoch})")
        if missing:    print(f"  [ckpt] missing keys  : {missing}")
        if unexpected: print(f"  [ckpt] unexpected keys: {unexpected}")
    else:
        model.load_state_dict(ckpt)
        print(f"  [ckpt] loaded raw state-dict <- {ckpt_path}")

    model.eval()
    return model


# =============================================================================
# Inference  (DDIM sampling conditioned on the coarse prediction)
# =============================================================================

@torch.no_grad()
def collect_predictions(model, coarse_model, diffusion, loader, device):
    all_preds, all_targets = [], []

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        if coarse_model is not None:
            coarse = run_coarse_forecaster(coarse_model, x)
        else:
            coarse = persistence_coarse(x, DEFAULTS["target_length"])
        cond = coarse.unsqueeze(2)   # (B, T, 1, H, W)

        shape = (x.shape[0], DEFAULTS["target_length"], 1, x.shape[-1], x.shape[-1])
        pred = diffusion.ddim_sample_loop(
            model, cond, shape,
            num_inference_steps=DEFAULTS["num_inference_steps"], device=device,
        )
        pred = pred.squeeze(2).clamp(0.0, 1.0)   # (B, T, H, W)

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} done.")

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    print(f"\n[inference] pred   range: [{preds.min():.4f},   {preds.max():.4f}]")
    print(f"[inference] target range: [{targets.min():.4f}, {targets.max():.4f}]")
    return preds, targets


# =============================================================================
# Dynamic threshold computation
# =============================================================================

def compute_dynamic_thresholds(targets_pixel: torch.Tensor, num_thresholds: int = 5) -> list:
    flat = targets_pixel.numpy().flatten()
    flat = flat[flat > 0.0]

    if len(flat) == 0:
        print("  [WARNING] All targets are dry — using fallback thresholds.")
        return FALLBACK_CSI_THRESHOLDS_MMHR

    percentiles = np.linspace(10, 60, num_thresholds)
    thresholds = [float(np.round(np.percentile(flat, p), 2)) for p in percentiles]
    thresholds = sorted(set(thresholds))

    print("\nAuto-selected CSI thresholds (percentile-based, pixel [0-255]):")
    for p, t in zip(percentiles, thresholds):
        print(f"  {p:.1f}th percentile -> {t:.2f}")

    return thresholds


# =============================================================================
# Evaluation  — uses eval_metrics functions throughout (same as tupann)
# =============================================================================

def evaluate(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    results = {}
    N, T_out, H, W = preds.shape

    preds_pixel = normalized_to_pixel(preds)
    targets_pixel = normalized_to_pixel(targets)

    print(f"targets dtype : {targets_pixel.dtype}")
    print(f"targets range : {targets_pixel.min():.4f}  {targets_pixel.max():.4f}")
    csi_thresholds = compute_dynamic_thresholds(targets_pixel)

    # --------------------------------------------------
    # 1. MSE & MAE  (normalised space)
    # --------------------------------------------------
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    results["MSE"] = mse
    results["MAE"] = mae

    print(f"\n{'='*58}")
    print(f"  Regression metrics  (normalised [0,1] space)")
    print(f"  {'MSE':<15}: {mse:.6f}")
    print(f"  {'MAE':<15}: {mae:.6f}")
    print(f"\n  Per-lead-time MSE & MAE:")
    print(f"  {'Step':>6}  {'MSE':>12}  {'MAE':>12}")
    for t in range(T_out):
        mse_t = torch.mean((preds[:, t] - targets[:, t]) ** 2).item()
        mae_t = torch.mean(torch.abs(preds[:, t] - targets[:, t])).item()
        results[f"MSE@t={t+1}"] = mse_t
        results[f"MAE@t={t+1}"] = mae_t
        print(f"  {f't+{t+1}':>6}  {mse_t:>12.6f}  {mae_t:>12.6f}")

    # --------------------------------------------------
    # 2. Hard CSI
    # --------------------------------------------------
    print(f"\n{'='*58}")
    print(f"  Hard CSI  (pixel thresholds [0-255])")
    print(f"  {'Threshold':>20}  {'CSI':>10}")
    print(f"  {'-'*32}")
    for thr in csi_thresholds:
        csi_val = hard_csi(preds_pixel, targets_pixel, threshold=thr)
        if csi_val is None:
            results[f"hard_CSI@{thr}"] = float("nan")
            print(f"  {thr:>20.2f}  {'undefined':>10}")
        else:
            results[f"hard_CSI@{thr}"] = csi_val.item()
            print(f"  {thr:>20.2f}  {csi_val.item():>10.4f}")

    # --------------------------------------------------
    # 2b. Hard POD / FAR
    # --------------------------------------------------
    print(f"\n{'='*58}")
    print(f"  Hard POD / FAR  (pixel thresholds [0-255])")
    print(f"  {'Threshold':>20}  {'POD':>10}  {'FAR':>10}")
    print(f"  {'-'*44}")
    for thr in csi_thresholds:
        pod_val = hard_pod(preds_pixel, targets_pixel, threshold=thr)
        far_val = hard_far(preds_pixel, targets_pixel, threshold=thr)

        if pod_val is None:
            results[f"hard_POD@{thr}"] = float("nan")
            pod_str = "undefined"
        else:
            results[f"hard_POD@{thr}"] = pod_val.item()
            pod_str = f"{pod_val.item():.4f}"

        if far_val is None:
            results[f"hard_FAR@{thr}"] = float("nan")
            far_str = "undefined"
        else:
            results[f"hard_FAR@{thr}"] = far_val.item()
            far_str = f"{far_val.item():.4f}"

        print(f"  {thr:>20.2f}  {pod_str:>10}  {far_str:>10}")

    # --------------------------------------------------
    # 3. Soft CSI
    # --------------------------------------------------
    print(f"\n{'='*58}")
    print(f"  Soft CSI loss  (pixel thresholds [0-255])")
    print(f"  {'Threshold':>20}  {'Soft-CSI-Loss':>15}")
    print(f"  {'-'*37}")
    for thr in csi_thresholds:
        soft_val = soft_csi_loss(preds_pixel, targets_pixel, threshold=thr)
        results[f"soft_CSI_loss@{thr}"] = soft_val.item()
        print(f"  {thr:>20.2f}  {soft_val.item():>15.4f}")

    # --------------------------------------------------
    # 4. SSIM
    # --------------------------------------------------
    print(f"\n{'='*58}")
    print(f"  SSIM  (normalised [0,1] space)")
    print(f"  {'-'*45}")
    ssim_overall = compute_ssim(preds, targets)
    results["SSIM_overall"] = ssim_overall
    print(f"  {'SSIM_overall':>28}  {ssim_overall:>10.4f}")
    for t in range(T_out):
        ssim_t = compute_ssim(preds[:, t], targets[:, t])
        results[f"SSIM@t={t+1}"] = ssim_t
        print(f"  {f'SSIM@t={t+1}':>28}  {ssim_t:>10.4f}")

    # --------------------------------------------------
    # 5. Exp-weighted temporal SSIM
    # --------------------------------------------------
    print(f"\n{'='*58}")
    print(f"  Exponentially-weighted temporal SSIM")
    print(f"  {'-'*45}")
    tw_ssim = exp_weighted_temporal_ssim(preds, targets)
    results["TW_SSIM"] = float(tw_ssim)
    print(f"  {'TW_SSIM':>28}  {float(tw_ssim):>10.4f}")
    print(f"\n{'='*58}\n")

    return results


# =============================================================================
# Visualisation
# =============================================================================

def visualize_predictions(preds, targets, out_dir="./eval_outputs/plots", num_samples=5, input_length=EARTHFORMER_INPUT_LENGTH):
    os.makedirs(out_dir, exist_ok=True)
    T_out = preds.shape[1]
    n = min(num_samples, preds.shape[0])
    labels = [f"t+{i+1}" for i in range(T_out)]

    preds_mmhr = denorm_to_mmhr(preds).numpy()
    targets_mmhr = denorm_to_mmhr(targets).numpy()
    preds_norm = preds.numpy()
    targets_norm = targets.numpy()

    for i in range(n):
        row_data = [targets_norm[i], preds_norm[i], targets_mmhr[i], preds_mmhr[i]]
        row_labels = ["GT (normalised [0,1])", "Pred (normalised [0,1])", "GT (mm/hr)", "Pred (mm/hr)"]
        row_vlims = [
            (0.0, 1.0), (0.0, 1.0),
            (0.0, float(max(targets_mmhr[i].max(), 1e-3))),
            (0.0, float(max(targets_mmhr[i].max(), 1e-3))),
        ]

        fig = plt.figure(figsize=(T_out * 3.2, 14))
        fig.suptitle(
            f"CasCast — Sample {i+1}  |  Input={input_length} frames -> Predict={T_out} frames",
            fontsize=13, fontweight="bold", y=1.01,
        )
        gs = GridSpec(4, T_out + 1, figure=fig, width_ratios=[1]*T_out + [0.05], hspace=0.4, wspace=0.08)
        axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(T_out)] for r in range(4)]
        cbar_axes = [fig.add_subplot(gs[r, T_out]) for r in range(4)]

        for r in range(4):
            vlo, vhi = row_vlims[r]
            for t in range(T_out):
                ax = axes_grid[r][t]
                im = ax.imshow(row_data[r][t], cmap="viridis", vmin=vlo, vmax=vhi, origin="upper", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0: ax.set_title(labels[t], fontsize=9)
                if t == 0: ax.set_ylabel(row_labels[r], fontsize=9, labelpad=4)
            plt.colorbar(im, cax=cbar_axes[r])
            cbar_axes[r].set_ylabel("[0,1]" if r < 2 else "mm/hr", fontsize=8)

        fname = os.path.join(out_dir, f"cascast_sample_{i+1:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {fname}")

    print(f"Visualisation complete. Figures saved to '{out_dir}'.")


# =============================================================================
# Save results CSV
# =============================================================================

def save_results(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"\nResults saved -> {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CasCast")
    for key, val in DEFAULTS.items():
        if val is None:
            parser.add_argument(f"--{key}", default=val, type=str)
        elif isinstance(val, bool):
            parser.add_argument(f"--{key}", default=val, action="store_true")
        else:
            parser.add_argument(f"--{key}", default=val, type=type(val))
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    for key in DEFAULTS:
        DEFAULTS[key] = getattr(args, key)

    device = torch.device(DEFAULTS["device"])
    coarse_ckpt = None if str(DEFAULTS["coarse_ckpt"]).lower() == "none" else DEFAULTS["coarse_ckpt"]

    torch.manual_seed(DEFAULTS["seed"])
    np.random.seed(DEFAULTS["seed"])
    random.seed(DEFAULTS["seed"])

    os.makedirs(DEFAULTS["out_dir"], exist_ok=True)

    print(f"\nDevice     : {device}")
    print(f"Checkpoint : {DEFAULTS['ckpt']}")
    print(f"Coarse ckpt: {coarse_ckpt}  (None -> persistence baseline)")
    print(f"Data dir   : {DEFAULTS['data_dir']}\n")

    all_files = sorted([
        os.path.join(DEFAULTS["data_dir"], f)
        for f in os.listdir(DEFAULTS["data_dir"]) if f.endswith('.npz')
    ])
    random.shuffle(all_files)
    split_idx = int(len(all_files) * DEFAULTS["train_split"])
    val_files = all_files[split_idx:]

    val_loader = DataLoader(
        NPZDataset(val_files, DEFAULTS["input_length"], DEFAULTS["target_length"]),
        batch_size=DEFAULTS["batch_size"],
        shuffle=False,
        num_workers=DEFAULTS["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    print(f"Total .npz files : {len(all_files)}")
    print(f"Validation files : {len(val_files)}\n")

    model = load_model(DEFAULTS["ckpt"], device)
    coarse_model = load_coarse_forecaster(coarse_ckpt, device=device)
    diffusion = GaussianDiffusion(timesteps=DEFAULTS["diffusion_timesteps"], device=device)

    print("\nRunning inference...")
    preds, targets = collect_predictions(model, coarse_model, diffusion, val_loader, device)
    print(f"Predictions : {preds.shape}   (N, T_out, H, W)")
    print(f"Targets     : {targets.shape}")

    print("\nComputing metrics...")
    results = evaluate(preds, targets)

    csv_path = os.path.join(DEFAULTS["out_dir"], "cascast_results.csv")
    save_results(results, csv_path)

    plot_dir = os.path.join(DEFAULTS["out_dir"], "plots")
    print(f"\nGenerating {DEFAULTS['num_vis']} visualisation(s)...")
    visualize_predictions(preds, targets, out_dir=plot_dir,
                          num_samples=DEFAULTS["num_vis"],
                          input_length=DEFAULTS["input_length"])

    print("\nEvaluation complete.")
    print(f"  Results : {csv_path}")
    print(f"  Plots   : {plot_dir}/")


if __name__ == "__main__":
    main()
