"""
evaluate.py
===========
Evaluation script for TUPANN on 112×112 satellite rainfall data.

Checkpoint: 0--best-epoch=7.ckpt  (verified from state_dict inspection)
  - input_len   = 9      (encoder conv_in: (128, 9, 3, 3))
  - target_len  = 18     (target_shape_dict_val: (18, 256, 256))
  - channels    = 128    (autoencoder)
  - embed_dim   = 4      (quant_conv / post_quant_conv)
  - reduc_factor= 4      (single Downsample → latent = input/4)
  - maxvit_dim  = 64     (h_params.yaml confirmed)
  - maxvit_depth= 4
  - dropout     = 0.2
  - latent_model: Metnet (not simple MaxViT)

img_size = 112 (your data).  TUPANN internally pads each 112×112 frame to
256×256 before encoding (matching the training resolution), then crops outputs
back to 112×112.  This keeps the latent at 64×64, consistent with the trained
MaxViT window/grid attention sizes.

State-dict loading:
  - Keys are prefixed: autoencoder.*, latent_model.*
  - Grid buffer shape in checkpoint is (1,2,256,256); our model registers a
    (1,2,112,112) buffer.  We filter out shape-mismatched buffer keys before
    loading to avoid RuntimeError with strict=False.

Usage
-----
    python evaluate.py --stage 2 \\
        --ckpt /path/to/0--best-epoch=7.ckpt \\
        --data_dir /path/to/IMC_Combined

    python evaluate.py --stage 1 \\
        --ckpt /path/to/autoenc_best.ckpt \\
        --data_dir /path/to/IMC_Combined
"""

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
cv2.setNumThreads(0)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset

from model_arch import TUPANN, AutoencoderKL

from eval_metrics import (
    hard_csi,
    soft_csi_loss,
    compute_ssim,
    exp_weighted_temporal_ssim,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = dict(
    data_dir      ='/home/data_images/IMC_Combined',
    stage         = 2,

    # ── Data dimensions (your 112×112 patches) ──────────────────────────────
    img_size      = 112,
    input_length  = 9,           # confirmed: conv_in weight (128, 9, 3, 3)
    target_length = 18,          # confirmed: target_shape_dict_val [18, 256, 256]

    # ── Data split ──────────────────────────────────────────────────────────
    train_split   = 0.7,
    seed          = 42,
    batch_size    = 16,           # lower than training to fit memory with pad→256
    num_workers   = 0,

    # ── Model architecture (confirmed from checkpoint) ───────────────────────
    embed_dim     = 4,
    reduc_factor  = 4,
    channels      = 128,
    dropout       = 0.2,
    maxvit_dim    = 64,
    maxvit_depth  = 4,
    maxvit_heads  = 32,
    maxvit_head_dim = 16,
    window_size   = 8,

    # ── Internal padding: pad each frame to this size before encoding ────────
    # Must be a multiple of reduc_factor*window_size = 32.
    # 256 = training resolution → recommended for best accuracy.
    pad_size      = 256,

    ckpt          = "/home/tupann/model_weights/tupann/0--best-epoch=7.ckpt",
    autoenc_ckpt  = "/home/tupann/model_weights/tupann_autoenc/0--best-epoch=85.ckpt",
    out_dir       = "./eval_outputs",
    num_vis       = 5,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
)

FALLBACK_CSI_THRESHOLDS_MMHR = [0.5, 1.0, 2.0, 5.0, 10.0]


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing  (must match train.py)
# ═══════════════════════════════════════════════════════════════════════════════

RAIN_MAX_MMHR: float = 60.0
_LOG_DENOM: float = float(np.log1p(RAIN_MAX_MMHR))


def preprocess_rain(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    x = np.clip(x, 0.0, RAIN_MAX_MMHR)
    return np.log1p(x) / _LOG_DENOM


def denorm_to_mmhr(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x.clamp(0.0, 1.0) * _LOG_DENOM).clamp(0.0, RAIN_MAX_MMHR)


def normalized_to_pixel(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class NPZDataset(Dataset):
    def __init__(self, files, input_length=9, target_length=18):
        self.files         = files
        self.input_length  = input_length
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=True) as f:
            arr = np.array(f['array'], dtype=np.float32).copy()

        rain = arr[4:, 0, :, :]
        #print(f"Loaded '{self.files[idx]}' with shape {arr.shape} -> rain shape: {rain.shape}")
        rain = preprocess_rain(rain)

        needed = self.input_length + self.target_length
        if rain.shape[0] < needed:
            # Pad with zeros if sequence is too short (edge case)
            pad   = np.zeros((needed - rain.shape[0], *rain.shape[1:]), dtype=np.float32)
            rain  = np.concatenate([rain, pad], axis=0)

        x = torch.from_numpy(rain[:self.input_length]).contiguous()
        y = torch.from_numpy(
            rain[self.input_length:self.input_length + self.target_length]).contiguous()
        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_stub_modules():
    """Register stub modules so torch.load can unpickle the Lightning checkpoint."""
    import sys, types

    class LazyStub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('_'):
                raise AttributeError(name)
            cls = type(name, (object,), {'__init__': lambda s, *a, **k: None})
            setattr(self, name, cls)
            return cls

    for mod_name in [
        'src', 'src.models', 'src.models.metnet', 'src.models.metnet.metnet',
        'src.utils', 'src.utils.lightning_utils', 'src.utils.train_utils',
        'src.utils.metrics', 'src.utils.losses',
        'lightning', 'lightning.pytorch',
        'precipitation_nowcasting',
        'precipitation_nowcasting.model',
        'precipitation_nowcasting.model.tupann',
        'precipitation_nowcasting.model.tupann.autoenc_lightning',
        'precipitation_nowcasting.model.tupann.utils',
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = LazyStub(mod_name)


def load_model(ckpt_path: str, device: torch.device, cfg: dict) -> nn.Module:
    _load_stub_modules()
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if cfg["stage"] == 2:
        model = TUPANN(
            input_len       = cfg["input_length"],
            target_length   = cfg["target_length"],
            img_size        = cfg["img_size"],
            embed_dim       = cfg["embed_dim"],
            reduc_factor    = cfg["reduc_factor"],
            channels        = cfg["channels"],
            dropout         = cfg["dropout"],
            maxvit_dim      = cfg["maxvit_dim"],
            maxvit_depth    = cfg["maxvit_depth"],
            maxvit_heads    = cfg["maxvit_heads"],
            maxvit_head_dim = cfg["maxvit_head_dim"],
            window_size     = cfg["window_size"],
            pad_size        = cfg["pad_size"],
        ).to(device)

        # ── Step 1: load autoencoder weights first (if separate ckpt given) ──
        autoenc_path = cfg.get("autoenc_ckpt", "")
        if autoenc_path and os.path.isfile(autoenc_path):
            print(f"  [ckpt] Loading autoencoder from: {autoenc_path}")
            ae_ckpt = torch.load(autoenc_path, map_location=device, weights_only=False)
            ae_sd   = ae_ckpt["state_dict"] if "state_dict" in ae_ckpt else ae_ckpt

            # Strip 'autoencoder.' prefix if present, then re-add it for TUPANN
            ae_sd_reprefixed = {}
            for k, v in ae_sd.items():
                # If keys are bare (from standalone autoenc ckpt), prefix them
                new_k = k if k.startswith("autoencoder.") else f"autoencoder.{k}"
                ae_sd_reprefixed[new_k] = v

            model_sd = model.state_dict()
            filtered = {k: v for k, v in ae_sd_reprefixed.items()
                        if k in model_sd and model_sd[k].shape == v.shape}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(f"         Autoencoder loaded: {len(filtered)} tensors")
            print(f"         Missing: {len(missing)}  Unexpected: {len(unexpected)}")

        # ── Step 2: load main checkpoint (latent_model + possibly autoencoder) ──
        print(f"  [ckpt] Loading main model from: {ckpt_path}")

    else:
        model = AutoencoderKL(
            input_len    = cfg["input_length"],
            img_size     = cfg["img_size"],
            embed_dim    = cfg["embed_dim"],
            reduc_factor = cfg["reduc_factor"],
            channels     = cfg["channels"],
            dropout      = cfg["dropout"],
        ).to(device)

    # ── Load main checkpoint (always) ────────────────────────────────────────
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        sd = dict(raw_ckpt["state_dict"])

        model_sd = model.state_dict()
        filtered, skipped = {}, []
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                skipped.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}")
            else:
                filtered[k] = v

        if skipped:
            print(f"  [ckpt] Skipped {len(skipped)} shape-mismatched buffers:")
            for s in skipped:
                print(f"         {s}")

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        epoch = raw_ckpt.get("epoch", "?")
        print(f"  [ckpt] Loaded <- {ckpt_path}  (epoch {epoch})")
        print(f"         Missing   : {len(missing)}")
        print(f"         Unexpected: {len(unexpected)}")
        if missing:
            print(f"         First 10 missing: {missing[:10]}")
    else:
        model.load_state_dict(raw_ckpt, strict=False)

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(model, loader, cfg, device):
    all_preds, all_targets = [], []

    for i, (x, y) in enumerate(loader):
        x = x.to(device)

        if cfg["stage"] == 2:
            pred = model(x)[:, :11]                     # (B, target_length, H, W)
        else:
            decoded, _ = model(x)
            mid  = decoded.shape[1] // 2
            pred = decoded[:, mid:mid + 1].expand(-1, cfg["target_length"], -1, -1)

        pred = pred.clamp(0.0, 1.0)
        all_preds.append(pred.cpu())
        all_targets.append(y)

        if (i + 1) % 10 == 0:
            print(f"  Batch {i + 1}/{len(loader)}")

    preds   = torch.cat(all_preds,   dim=0)
    targets = torch.cat(all_targets, dim=0)
    print(f"\n[inference] pred   [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"[inference] target [{targets.min():.4f}, {targets.max():.4f}]")
    return preds, targets


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic CSI thresholds
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dynamic_thresholds(targets_pixel, num_thresholds=5):
    flat = targets_pixel.numpy().flatten().astype(np.float32)
    flat = flat[flat > 0.0]
    if len(flat) == 0:
        print("  [WARNING] All targets are dry — using fallback thresholds.")
        return FALLBACK_CSI_THRESHOLDS_MMHR
    pcts = np.linspace(10, 60, num_thresholds)
    thrs = sorted(set(float(np.round(np.percentile(flat, p), 2)) for p in pcts))
    print("\nAuto CSI thresholds (pixel [0-255]):")
    for t in thrs:
        print(f"  {t:.2f}")
    return thrs


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(preds, targets, cfg):
    results = {}
    N, T, H, W = preds.shape

    preds_px   = normalized_to_pixel(preds)
    targets_px = normalized_to_pixel(targets)
    thresholds = compute_dynamic_thresholds(targets_px)

    # 1. MSE / MAE
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    results["MSE"] = mse
    results["MAE"] = mae

    print(f"\n{'='*58}")
    print(f"  Regression  (normalised [0,1])")
    print(f"  MSE = {mse:.6f}   MAE = {mae:.6f}")
    print(f"\n  {'Step':>6}  {'MSE':>12}  {'MAE':>12}")
    for t in range(T):
        m = torch.mean((preds[:, t] - targets[:, t]) ** 2).item()
        a = torch.mean(torch.abs(preds[:, t] - targets[:, t])).item()
        results[f"MSE@t={t+1}"] = m
        results[f"MAE@t={t+1}"] = a
        print(f"  {f't+{t+1}':>6}  {m:>12.6f}  {a:>12.6f}")

    # 2. Hard CSI
    print(f"\n{'='*58}")
    print(f"  Hard CSI  (pixel [0-255])")
    for thr in thresholds:
        v = hard_csi(preds_px, targets_px, threshold=thr)
        if v is None:
            results[f"hard_CSI@{thr}"] = float("nan")
            print(f"  thr={thr:6.2f}  CSI=undefined")
        else:
            results[f"hard_CSI@{thr}"] = v.item()
            print(f"  thr={thr:6.2f}  CSI={v.item():.4f}")

    # 3. Soft CSI
    print(f"\n{'='*58}")
    print(f"  Soft CSI loss  (pixel [0-255])")
    for thr in thresholds:
        v = soft_csi_loss(preds_px, targets_px, threshold=thr)
        results[f"soft_CSI_loss@{thr}"] = v.item()
        print(f"  thr={thr:6.2f}  soft_CSI_loss={v.item():.4f}")

    # 4. SSIM
    print(f"\n{'='*58}")
    print(f"  SSIM  (normalised [0,1])")
    ssim_all = compute_ssim(preds, targets)
    results["SSIM_overall"] = ssim_all
    print(f"  overall = {ssim_all:.4f}")
    for t in range(T):
        s = compute_ssim(preds[:, t], targets[:, t])
        results[f"SSIM@t={t+1}"] = s
        print(f"  t+{t+1:02d}  = {s:.4f}")

    # 5. Exp-weighted SSIM
    print(f"\n{'='*58}")
    tw = exp_weighted_temporal_ssim(preds, targets)
    results["TW_SSIM"] = float(tw)
    print(f"  TW_SSIM = {float(tw):.4f}")
    print(f"{'='*58}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_predictions(preds, targets, out_dir, num_samples=5, input_length=9):
    os.makedirs(out_dir, exist_ok=True)
    T = preds.shape[1]
    n = min(num_samples, preds.shape[0])
    labels = [f"t+{i+1}" for i in range(T)]

    preds_mm   = denorm_to_mmhr(preds).numpy()
    targets_mm = denorm_to_mmhr(targets).numpy()

    for i in range(n):
        rows = [targets.numpy()[i], preds.numpy()[i],
                targets_mm[i], preds_mm[i]]
        rlabels = ["GT (norm)", "Pred (norm)", "GT (mm/hr)", "Pred (mm/hr)"]
        vlims   = [(0, 1), (0, 1),
                   (0, max(targets_mm[i].max(), 1e-3)),
                   (0, max(targets_mm[i].max(), 1e-3))]

        fig = plt.figure(figsize=(T * 3.2, 11))
        fig.suptitle(
            f"TUPANN — Sample {i+1}  |  Input={input_length} → Predict={T}",
            fontsize=13, fontweight="bold", y=1.01)
        gs = GridSpec(4, T + 1, figure=fig,
                      width_ratios=[1] * T + [0.05], hspace=0.4, wspace=0.08)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(T)] for r in range(4)]
        cbars = [fig.add_subplot(gs[r, T]) for r in range(4)]

        for r in range(4):
            vlo, vhi = vlims[r]
            for t in range(T):
                ax = axes[r][t]
                im = ax.imshow(rows[r][t], cmap="viridis",
                               vmin=vlo, vmax=vhi, origin="upper", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0: ax.set_title(labels[t], fontsize=9)
                if t == 0: ax.set_ylabel(rlabels[r], fontsize=9)
            plt.colorbar(im, cax=cbars[r])
            cbars[r].set_ylabel("[0,1]" if r < 2 else "mm/hr", fontsize=8)

        fname = os.path.join(out_dir, f"tupann_sample_{i+1:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {fname}")

    print(f"Plots saved to '{out_dir}'.")


# ═══════════════════════════════════════════════════════════════════════════════
# Save CSV
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in results.items():
            w.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"\nResults saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TUPANN / AutoencoderKL")
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true")
        else:
            p.add_argument(f"--{k}", default=v, type=t)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg  = {k: getattr(args, k) for k in DEFAULTS}

    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    print(f"\nDevice      : {device}")
    print(f"Stage       : {cfg['stage']}")
    print(f"Checkpoint  : {cfg['ckpt']}")
    print(f"Data dir    : {cfg['data_dir']}")
    print(f"img_size    : {cfg['img_size']}  pad_size: {cfg['pad_size']}")
    print(f"input_length: {cfg['input_length']}  target_length: {cfg['target_length']}")
    print(f"channels    : {cfg['channels']}  embed_dim: {cfg['embed_dim']}")
    print(f"maxvit_dim  : {cfg['maxvit_dim']}  maxvit_depth: {cfg['maxvit_depth']}\n")

    all_files = sorted([
        os.path.join(cfg["data_dir"], f)
        for f in os.listdir(cfg["data_dir"]) if f.endswith('.npz')
    ])
    random.shuffle(all_files)
    split  = int(len(all_files) * cfg["train_split"])
    val_files = all_files[split:]

    print(f"Total .npz  : {len(all_files)}")
    print(f"Validation  : {len(val_files)}\n")

    val_loader = DataLoader(
        NPZDataset(val_files, cfg["input_length"], cfg["target_length"]),
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        num_workers = cfg["num_workers"],
        pin_memory  = (device.type == "cuda"),
    )

    print("Loading model...")
    model = load_model(cfg["ckpt"], device, cfg)

    print("\nRunning inference...")
    preds, targets = collect_predictions(model, val_loader, cfg, device)
    preds = preds[:, :11]
    targets = targets[:, :11]
    print(f"Predictions : {tuple(preds.shape)}")
    print(f"Targets     : {tuple(targets.shape)}")

    print("\nComputing metrics...")
    results = evaluate(preds, targets, cfg)

    csv_path = os.path.join(cfg["out_dir"], "tupann_results.csv")
    save_results(results, csv_path)

    plot_dir = os.path.join(cfg["out_dir"], "plots")
    print(f"\nGenerating {cfg['num_vis']} visualisation(s)...")
    visualize_predictions(preds, targets, out_dir=plot_dir,
                          num_samples=cfg["num_vis"],
                          input_length=cfg["input_length"])

    print("\nEvaluation complete.")
    print(f"  Results : {csv_path}")
    print(f"  Plots   : {plot_dir}/")


if __name__ == "__main__":
    main()