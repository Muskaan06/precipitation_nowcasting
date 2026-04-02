import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_arch import EarthformerModel
from eval_metrics import soft_csi_loss, hard_csi
from eval_metrics import compute_ssim, exp_weighted_temporal_ssim

# ==========================================
# CONFIG — keep seed & split identical to train.py
# ==========================================
CONFIG = {
    "data_dir":    "/home/muskaan06/Desktop/Research/nowcasting/datasets/selected_sets_2024_SI_112",
    "train_split": 0.8,          # must match train.py
    "batch_size":  4,
    "num_workers": 2,
    "seed":        42,
    "checkpoint":  "/path/to/earthformer-best.ckpt",   # ← change to your .ckpt path

    # --- Sweep ranges (overwritten by compute_dynamic_thresholds) ---
    "csi_thresholds": [0.5, 1.0, 2.0, 5.0],
}

# ==========================================
# DATASET  (identical to train.py, with log1p on targets)
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

        # channels-last for Earthformer: (T, C, H, W) -> (T, H, W, C)
        x = torch.tensor(arr[:4],   dtype=torch.float32).permute(0, 2, 3, 1)  # (4, 112, 112, 3)
        y = torch.tensor(arr[4:10], dtype=torch.float32)
        y = torch.log1p(y)                                                      # log(1 + x)
        y = y.permute(0, 2, 3, 1)                                               # (6, 112, 112, 3)
        return x, y


# ==========================================
# LOAD MODEL
# ==========================================
def load_model(checkpoint_path, device):
    """
    Supports:
      1. PyTorch Lightning checkpoint (has 'state_dict' key, keys prefixed 'model.')
      2. Raw state-dict saved via torch.save(model.state_dict(), path)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = EarthformerModel().to(device)

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
        x = x.to(device)                           # (B, 4, 112, 112, 3)
        pred = model(x)                            # (B, 6, 112, 112, 3)

        # inverse log1p, then convert to channels-first flat for metrics
       # pred = torch.expm1(pred)                   # inverse of log(1 + x)
        pred = pred.permute(0, 1, 4, 2, 3)        # (B, 6, 3, 112, 112)
        pred = pred.reshape(pred.size(0), 18, 112, 112)

        all_preds.append(pred.cpu())
        all_targets.append(y)                      # (B, 6, 112, 112, 3) — still log-scaled

    preds   = torch.cat(all_preds,   dim=0)        # (N, 18, 112, 112)

    # convert targets to channels-first flat too (undo log1p for metric consistency)
    targets_raw = torch.cat(all_targets, dim=0)    # (N, 6, 112, 112, 3)
    targets_raw = torch.expm1(targets_raw)
    targets_raw = targets_raw.permute(0, 1, 4, 2, 3).reshape(-1, 18, 112, 112)

    return preds, targets_raw


def compute_dynamic_thresholds(targets, num_thresholds=5):
    flat = targets.numpy().flatten()
    flat = flat[flat > 0]

    if len(flat) == 0:
        raise ValueError("All targets are zero!")

    percentiles = np.linspace(50, 99, num_thresholds)
    thresholds  = np.percentile(flat, percentiles)
    thresholds  = [float(np.round(t, 2)) for t in thresholds]

    print("\n📊 Auto-selected thresholds (percentile-based):")
    for p, t in zip(percentiles, thresholds):
        print(f"  {p:.1f}th percentile → {t}")

    return thresholds


# ==========================================
# EVALUATION  (same pipeline as unet evaluate.py)
# ==========================================
def evaluate(preds, targets):
    """
    preds, targets: (N, 18, 112, 112)  — on CPU, linear scale
    """
    results = {}
    CONFIG["csi_thresholds"] = compute_dynamic_thresholds(targets)

    B, C, H, W = preds.shape                       # C == 18
    # temporal view: (N, 6 timesteps, 3 channels, H, W) → pick channel 0
    pred_seq = preds.view(B, 6, 3, H, W)[:, :, 0]  # (N, 6, H, W)
    y_seq    = targets.view(B, 6, 3, H, W)[:, :, 0]

    data_range = float(targets.max() - targets.min())
    if data_range == 0:
        data_range = 1.0

    # --------------------------------------------------
    # 1.  MSE  &  MAE
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
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Hard CSI across thresholds")
    print(f"  {'Threshold':>12}  {'CSI':>10}")
    print(f"  {'-'*25}")
    for thr in CONFIG["csi_thresholds"]:
        csi_val = hard_csi(preds, targets, threshold=thr)
        if csi_val is None:
            results[f"hard_CSI@thr={thr}"] = float("nan")
            print(f"  {thr:>12.2f}  {'None':>10}")
        else:
            csi_scalar = csi_val.item()
            results[f"hard_CSI@thr={thr}"] = csi_scalar
            print(f"  {thr:>12.2f}  {csi_scalar:>10.4f}")

    # --------------------------------------------------
    # 3.  Soft CSI loss at multiple thresholds
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Soft CSI loss across thresholds")
    print(f"  {'Threshold':>12}  {'Soft-CSI-Loss':>15}")
    print(f"  {'-'*30}")
    for thr in CONFIG["csi_thresholds"]:
        soft_val    = soft_csi_loss(preds, targets, threshold=thr)
        soft_scalar = soft_val.item()
        results[f"soft_CSI_loss@thr={thr}"] = soft_scalar
        print(f"  {thr:>12.2f}  {soft_scalar:>15.4f}")

    # --------------------------------------------------
    # 4.  SSIM  (compute_ssim — Eq. 7 formula)
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  SSIM (compute_ssim)")
    print(f"  {'Metric':>30}  {'Value':>10}")
    print(f"  {'-'*43}")

    ssim_overall = compute_ssim(pred_seq, y_seq)
    results["SSIM_overall"] = ssim_overall
    print(f"  {'SSIM_overall':>30}  {ssim_overall:>10.4f}")

    pred_np = pred_seq.numpy() if isinstance(pred_seq, torch.Tensor) else pred_seq
    y_np    = y_seq.numpy()    if isinstance(y_seq,    torch.Tensor) else y_seq

    T = pred_np.shape[1]
    for t in range(T):
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
# SAVE RESULTS TO CSV
# ==========================================
def save_results(results, path="validation_results_earthformer.csv"):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Results saved to {path}")


# ==========================================
# MAIN
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CONFIG["data_dir"]):
        raise FileNotFoundError(f"data_dir '{CONFIG['data_dir']}' not found.")

    all_files = sorted([
        os.path.join(CONFIG["data_dir"], f)
        for f in os.listdir(CONFIG["data_dir"])
        if f.endswith(".npz")
    ])

    random.seed(CONFIG["seed"])
    random.shuffle(all_files)

    split_idx = int(len(all_files) * CONFIG["train_split"])
    val_files = all_files[split_idx:]

    print(f"Total files : {len(all_files)}")
    print(f"Val files   : {len(val_files)}")

    val_loader = DataLoader(
        NPZDataset(val_files),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(CONFIG["checkpoint"], device)

    print("\nRunning inference...")
    preds, targets = collect_predictions(model, val_loader, device)
    print(f"Predictions shape : {preds.shape}")

    results = evaluate(preds, targets)
    save_results(results)


if __name__ == "__main__":
    main()