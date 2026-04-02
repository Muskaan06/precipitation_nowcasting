import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from itertools import product

# ---- Import your modules (same as train.py) ----
from unet import UNet
from eval_metrics import soft_csi_loss, hard_csi
from eval_metrics import compute_ssim, exp_weighted_temporal_ssim

# ==========================================
# CONFIG — keep seed & split identical to train.py
# ==========================================
CONFIG = {
    "data_dir":       "/home/muskaan06/Desktop/Research/nowcasting/datasets/selected_sets_2024_SI_112",  # Path to your NPZ files
    "train_split":    0.7,
    "batch_size":     4,
    "num_workers":    0,
    "seed":           42,          # Same seed → same val/test split
    "checkpoint":     "/home/muskaan06/Desktop/Research/nowcasting/precipitation_nowcasting/checkpoints/unet-4_6_log1p.ckpt",   # ← change to your .pt path

    # --- Sweep ranges ---
    "csi_thresholds":  [0.5, 1.0, 2.0, 5.0],   # mm/h or whatever unit
}

# ==========================================
# DATASET  (identical to train.py)
# ==========================================
class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            arr = data[data.files[0]]             # (24, 3, 112, 112)

        arr = np.nan_to_num(arr, nan=0.0)

        x = torch.tensor(arr[:4],   dtype=torch.float32).reshape(12, 112, 112)
        y = torch.tensor(arr[4:10], dtype=torch.float32)
        y = torch.log1p(y)  # log(1 + x) transformation on targets
        y = y.reshape(18, 112, 112)
        return x, y


# ==========================================
# MODEL WRAPPER  (matches LightningModule forward)
# ==========================================
class UNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(channels_in=12, channels_out=18)

    def forward(self, x):
        return self.model(x)


# ==========================================
# LOAD MODEL
# ==========================================
def load_model(checkpoint_path, device):
    """
    Supports two checkpoint formats:
      1. Raw state-dict saved via torch.save(model.state_dict(), path)
      2. PyTorch Lightning checkpoint (has 'state_dict' key)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNetWrapper().to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Lightning checkpoint — strip the "model." prefix added by LightningModule
        state = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.model.load_state_dict(state)
        print("Loaded PyTorch Lightning checkpoint.")
    else:
        # Plain state-dict
        model.load_state_dict(checkpoint)
        print("Loaded plain state-dict checkpoint.")

    model.eval()
    return model


# ==========================================
# COLLECT PREDICTIONS
# ==========================================
@torch.no_grad()
def collect_predictions(model, loader, device):
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        pred = torch.expm1(pred)  # Inverse of log(1 + x) transformation
        all_preds.append(pred.cpu())
        y = torch.expm1(y)
        all_targets.append(y)

    preds   = torch.cat(all_preds,   dim=0)   # (N, 30, 112, 112)
    targets = torch.cat(all_targets, dim=0)   # (N, 30, 112, 112)
    return preds, targets


def compute_dynamic_thresholds(targets, num_thresholds=5):
    """
    Compute percentile-based thresholds from target data.
    Avoids extreme values that cause NaNs.
    """
    flat = targets.numpy().flatten()

    # Remove zeros (optional but recommended for rainfall-type data)
    flat = flat[flat > 0]

    if len(flat) == 0:
        raise ValueError("All targets are zero!")

    percentiles = np.linspace(50, 99, num_thresholds)
    thresholds = np.percentile(flat, percentiles)

    # Round for readability
    thresholds = [float(np.round(t, 2)) for t in thresholds]

    print("\n📊 Auto-selected thresholds (percentile-based):")
    for p, t in zip(percentiles, thresholds):
        print(f"  {p:.1f}th percentile → {t}")

    return thresholds


# ==========================================
# EVALUATION
# ==========================================
def evaluate(preds, targets):
    """
    preds, targets: (N, 30, 112, 112)  — on CPU
    Returns a dict of all metric results.
    """
    results = {}
    CONFIG["csi_thresholds"] = compute_dynamic_thresholds(targets)

    B, C, H, W = preds.shape                           # C == 30
    # Temporal view: (N, 10 timesteps, 3 channels, H, W)
    pred_seq = preds.view(B, 6, 3, H, W)[:, :, 0]    # (N, 10, H, W)  — channel 0
    y_seq    = targets.view(B, 6, 3, H, W)[:, :, 0]

    # data_range was previously passed to compute_ssim / exp_weighted_temporal_ssim
    # but is no longer a parameter in eval_metrics.py (Eq. 7 formula is scale-free).
    # It is retained here only for reference / future use.
    data_range = float(targets.max() - targets.min())
    if data_range == 0:
        data_range = 1.0

    # --------------------------------------------------
    # 1.  MSE  &  MAE  (over all channels & pixels)
    # --------------------------------------------------
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    results["MSE"] = mse
    results["MAE"] = mae
    print(f"\n{'='*55}")
    print(f"  MSE : {mse:.6f}")
    print(f"  MAE : {mae:.6f}")

    # --------------------------------------------------
    # 2.  Hard CSI at multiple thresholds
    #     hard_csi() returns a Tensor or None (when denominator == 0)
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Hard CSI across thresholds")
    print(f"  {'Threshold':>12}  {'CSI':>10}")
    print(f"  {'-'*25}")
    csi_results = {}
    for thr in CONFIG["csi_thresholds"]:
        csi_val = hard_csi(preds, targets, threshold=thr)
        # hard_csi returns None when denominator == 0 (undefined, not a zero score)
        if csi_val is None:
            csi_results[thr] = None
            results[f"hard_CSI@thr={thr}"] = float("nan")
            print(f"  {thr:>12.2f}  {'None':>10}")
        else:
            csi_scalar = csi_val.item()
            csi_results[thr] = csi_scalar
            results[f"hard_CSI@thr={thr}"] = csi_scalar
            print(f"  {thr:>12.2f}  {csi_scalar:>10.4f}")

    # --------------------------------------------------
    # 3.  Soft CSI loss at multiple thresholds
    #     soft_csi_loss() returns a Tensor — use .item() for logging
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Soft CSI loss across thresholds")
    print(f"  {'Threshold':>12}  {'Soft-CSI-Loss':>15}")
    print(f"  {'-'*30}")
    for thr in CONFIG["csi_thresholds"]:
        soft_val = soft_csi_loss(preds, targets, threshold=thr)
        soft_scalar = soft_val.item()
        results[f"soft_CSI_loss@thr={thr}"] = soft_scalar
        print(f"  {thr:>12.2f}  {soft_scalar:>15.4f}")

    # --------------------------------------------------
    # 4.  SSIM  (compute_ssim)
    #     data_range removed from call — Eq. 7 formula is scale-free.
    #     Overall  : pass full (B, T, H, W)
    #     Per-step : iterate over batch manually at each t → ndim==2 path
    #                (ndim==3 path collapses axis-0 as time, not batch)
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  SSIM (compute_ssim)")
    print(f"  {'Metric':>30}  {'Value':>10}")
    print(f"  {'-'*43}")

    # Overall: pass full (B, T, H, W) — compute_ssim averages time then batch
    ssim_overall = compute_ssim(pred_seq, y_seq)
    results["SSIM_overall"] = ssim_overall
    print(f"  {'SSIM_overall':>30}  {ssim_overall:>10.4f}")

    # Per-timestep SSIM — useful to spot temporal degradation
    pred_np = pred_seq.numpy() if isinstance(pred_seq, torch.Tensor) else pred_seq
    y_np    = y_seq.numpy()    if isinstance(y_seq,    torch.Tensor) else y_seq

    T = pred_np.shape[1]
    for t in range(T):
        # pred_np[:, t] is (B, H, W) — iterate over B so each call hits ndim==2
        scores_t = [
            compute_ssim(pred_np[b, t], y_np[b, t])
            for b in range(pred_np.shape[0])
        ]
        valid_t = [s for s in scores_t if s != 0.0]
        ssim_t  = float(np.mean(valid_t)) if valid_t else 0.0
        results[f"SSIM@t={t+1}"] = ssim_t
        print(f"  {f'SSIM@t={t+1}':>30}  {ssim_t:>10.4f}")

    # --------------------------------------------------
    # 5.  Exp-weighted temporal SSIM
    #     data_range removed from call — no longer a parameter in eval_metrics.py
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Exp-Weighted Temporal SSIM")
    print(f"  {'-'*43}")

    tw_ssim = exp_weighted_temporal_ssim(pred_seq, y_seq)
    tw_ssim = float(tw_ssim) if tw_ssim is not None else 0.0
    results["TW_SSIM"] = tw_ssim
    print(f"  {'TW_SSIM':>30}  {tw_ssim:>10.4f}")

    print(f"\n{'='*55}\n")
    return results


# ==========================================
# VISUALIZE PREDICTIONS
# ==========================================
def visualize_predictions(preds, targets, out_dir="precipitation_maps/unet_4_6_log", num_samples=5):
    """
    preds, targets : (N, 18, 112, 112) — CPU, linear scale
    Saves one PNG per sample: 2 rows (GT / Pred) x 6 timestep columns.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    T_OUT           = 6
    N_CH            = 3
    COLORMAP        = "viridis"
    TIMESTEP_LABELS = [f"t+{i*10} min" for i in range(1, T_OUT + 1)]

    os.makedirs(out_dir, exist_ok=True)
    n = min(num_samples, preds.shape[0])

    for i in range(n):
        pred_i = preds[i].view(T_OUT, N_CH, 112, 112)[:, 0].numpy()   # (6, 112, 112)
        gt_i   = targets[i].view(T_OUT, N_CH, 112, 112)[:, 0].numpy()

        vmax = 50
        vmin = 0.2

        fig = plt.figure(figsize=(T_OUT * 3.2, 7))
        fig.suptitle(f"UNet — Sample {i+1} | Precipitation (channel 0)",
                     fontsize=14, fontweight="bold", y=1.01)

        gs        = GridSpec(2, T_OUT + 1, figure=fig,
                             width_ratios=[1] * T_OUT + [0.05],
                             hspace=0.35, wspace=0.08)
        row_data   = [gt_i, pred_i]
        row_labels = ["Ground Truth", "Prediction"]
        axes_grid  = [[fig.add_subplot(gs[r, c]) for c in range(T_OUT)] for r in range(2)]
        cbar_axes  = [fig.add_subplot(gs[r, T_OUT]) for r in range(2)]

        for r in range(2):
            for t in range(T_OUT):
                ax = axes_grid[r][t]
                im = ax.imshow(row_data[r][t], cmap=COLORMAP,
                               vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(TIMESTEP_LABELS[t], fontsize=9)
                if t == 0:
                    ax.set_ylabel(row_labels[r], fontsize=9, labelpad=4)
            plt.colorbar(im, cax=cbar_axes[r])
            cbar_axes[r].set_ylabel("mm/h", fontsize=8)

        fname = os.path.join(out_dir, f"unet_sample{i+1:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")

    print(f"Visualization done. Figures saved to '{out_dir}'.")


# ==========================================
# SAVE RESULTS TO CSV
# ==========================================
def save_results(results, path="validation_results.csv"):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            # v can be float, float("nan"), or None — guard before formatting
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Results saved to {path}")


# ==========================================
# MAIN
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Reproduce exact same split as training ---
    if not os.path.exists(CONFIG["data_dir"]):
        raise FileNotFoundError(f"data_dir '{CONFIG['data_dir']}' not found.")

    all_files = sorted([
        os.path.join(CONFIG["data_dir"], f)
        for f in os.listdir(CONFIG["data_dir"])
        if f.endswith(".npz")
    ])

    random.seed(CONFIG["seed"])          # Same seed as train.py
    random.shuffle(all_files)

    split_idx  = int(len(all_files) * CONFIG["train_split"])
    val_files  = all_files[split_idx:]  # Same held-out set

    print(f"Total files : {len(all_files)}")
    print(f"Val files   : {len(val_files)}")

    val_loader = DataLoader(
        NPZDataset(val_files),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # --- Load model ---
    model = load_model(CONFIG["checkpoint"], device)

    # --- Run inference ---
    print("\nRunning inference...")
    preds, targets = collect_predictions(model, val_loader, device)
    print(f"Predictions shape : {preds.shape}")

    # --- Evaluate ---
    results = evaluate(preds, targets)

    # --- Save CSV ---
    save_results(results)

    # --- Visualize ---
    visualize_predictions(preds, targets)


if __name__ == "__main__":
    main()