import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.colors as mcolors
from model_arch import EarthformerModel
from eval_metrics import soft_csi_loss, hard_csi
from eval_metrics import compute_ssim, exp_weighted_temporal_ssim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



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

CONFIG = {
    "data_dir":    "/mnt/sda1/Muskaan/nowcast/IMC_Combined",
    "train_split": 0.7,
    "batch_size":  16,
    "num_workers": 1,
    "seed":        42,
    "checkpoint":  "/mnt/sda1/Muskaan/nowcast/precipitation_nowcasting/model/earthformer/checkpoints/finetune_11/earthformer-finetune-best-epoch=4.ckpt",  # ← update to finetuned .ckpt path

    # --- Sweep ranges (overwritten by compute_dynamic_thresholds) ---
    "csi_thresholds": [0.5, 1.0, 2.0, 5.0],
}

# ==========================================
# DATASET — identical to 0-shot and finetune
# ==========================================
class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:,0,:,:], dtype=np.float32)

        arr = np.expand_dims(arr, axis=-1)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.concatenate(([arr[0]], arr))          # pad T by duplicating frame 0
        x_raw = torch.from_numpy(arr[:13]).float()     # T_in=13
        y_raw = torch.from_numpy(arr[13:25]).float()   # T_out=12

        return mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)

# ==========================================
# LOAD MODEL
# ==========================================
def load_model(checkpoint_path, device):
    """
    Loads finetuned Lightning .ckpt checkpoint.
    Mirrors load_model from evaluate_earthformer_0_shot.py exactly —
    only difference is the checkpoint is now a Lightning .ckpt (not plain .pt)
    so it always hits the 'state_dict' branch.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = EarthformerModel().to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Finetuned Lightning .ckpt — strip the "model." prefix
        state = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.model.load_state_dict(state)
        print("Loaded finetuned Lightning checkpoint.")
    else:
        # Fallback: plain state dict
        model.load_state_dict(checkpoint)
        print("Loaded plain state-dict checkpoint.")

    model.eval()
    return model


# ==========================================
# COLLECT PREDICTIONS — identical to 0-shot
# ==========================================
@torch.no_grad()
def collect_predictions(model, loader, device):
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        print(f"pred min: {pred.min().item():.4f}, max: {pred.max().item():.4f}, mean: {pred.mean().item():.4f}, std: {pred.std().item():.4f}, in [0,1]: {(pred >= 0).all() and (pred <= 1).all()}")

        pred = pred.permute(0, 1, 4, 2, 3)
        pred = pred.reshape(pred.size(0), 12, 112, 112)   # T_out=12, C=1
        pred = pred[:, :-1, :, :]

        all_preds.append(pred.cpu())
        all_targets.append(y)

    preds = torch.cat(all_preds, dim=0)

    targets_raw = torch.cat(all_targets, dim=0)
    targets_raw = targets_raw.permute(0, 1, 4, 2, 3).reshape(-1, 12, 112, 112)
    targets_raw = targets_raw[:, :-1, :, :]

    return preds, targets_raw


def compute_dynamic_thresholds(targets, num_thresholds=5):
    flat = targets.numpy().flatten()
    flat = flat[flat > 0]

    if len(flat) == 0:
        raise ValueError("All targets are zero!")

    percentiles = np.linspace(10, 60, num_thresholds)
    thresholds  = np.percentile(flat, percentiles)
    thresholds  = [float(np.round(t, 2)) for t in thresholds]

    print("\n📊 Auto-selected thresholds (percentile-based):")
    for p, t in zip(percentiles, thresholds):
        print(f"  {p:.1f}th percentile → {t}")

    return thresholds


# ==========================================
# EVALUATION — identical to 0-shot
# ==========================================
def evaluate(preds, targets):
    """
    preds, targets: (N, 12, 112, 112) — on CPU, normalized [0,1]
    """
    results = {}
    CONFIG["csi_thresholds"] = compute_dynamic_thresholds(to_pixel_intensity(targets))

    B, C, H, W = preds.shape                            # C == 12
    pred_seq = preds.view(B, 11, 1, H, W)[:, :, 0]     # (B, 12, H, W)
    y_seq    = targets.view(B, 11, 1, H, W)[:, :, 0]

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
        csi_val = hard_csi(to_pixel_intensity(preds), to_pixel_intensity(targets), threshold=thr)
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
        soft_val    = soft_csi_loss(to_pixel_intensity(preds), to_pixel_intensity(targets), threshold=thr)
        soft_scalar = soft_val.item()
        results[f"soft_CSI_loss@thr={thr}"] = soft_scalar
        print(f"  {thr:>12.2f}  {soft_scalar:>15.4f}")

    # --------------------------------------------------
    # 4.  SSIM
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  SSIM (compute_ssim)")
    print(f"  {'Metric':>30}  {'Value':>10}")
    print(f"  {'-'*43}")

    ssim_overall = compute_ssim(to_pixel_intensity(pred_seq), to_pixel_intensity(y_seq))
    results["SSIM_overall"] = ssim_overall
    print(f"  {'SSIM_overall':>30}  {ssim_overall:>10.4f}")

    pred_np = pred_seq.numpy() if isinstance(pred_seq, torch.Tensor) else pred_seq
    y_np    = y_seq.numpy()    if isinstance(y_seq,    torch.Tensor) else y_seq

    T = pred_np.shape[1]                                # T == 12
    for t in range(T):
        scores_t = [
            compute_ssim(to_pixel_intensity(torch.from_numpy(pred_np[b, t])),
                         to_pixel_intensity(torch.from_numpy(y_np[b, t])))
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

    tw_ssim = exp_weighted_temporal_ssim(to_pixel_intensity(pred_seq), to_pixel_intensity(y_seq))
    tw_ssim = float(tw_ssim) if tw_ssim is not None else 0.0
    results["TW_SSIM"] = tw_ssim
    print(f"  {'TW_SSIM':>30}  {tw_ssim:>10.4f}")

    print(f"\n{'='*55}\n")
    return results


def visualize_predictions(preds, targets, out_dir="./plottings_finetune_11", num_samples=5):

    T_OUT           = 11
    N_CH            = 1
    TIMESTEP_LABELS = [f"t+{i*10} min" for i in range(1, T_OUT + 1)]

    os.makedirs(out_dir, exist_ok=True)
    n = min(num_samples, preds.shape[0])

    for i in range(1000, 1000+n):
        pred_norm = preds[i].view(T_OUT, N_CH, 112, 112)[:, 0].numpy()
        gt_norm   = targets[i].view(T_OUT, N_CH, 112, 112)[:, 0].numpy()

        pred_mmhr = to_pixel_intensity(preds[i]).view(T_OUT, N_CH, 112, 112)[:, 0].numpy()
        gt_mmhr   = to_pixel_intensity(targets[i]).view(T_OUT, N_CH, 112, 112)[:, 0].numpy()

        fig = plt.figure(figsize=(T_OUT * 3.2, 14))
        fig.suptitle(f"EarthFormer Finetuned — Sample {i+1} | Precipitation (channel 0)",
                     fontsize=14, fontweight="bold", y=1.01)

        gs = GridSpec(4, T_OUT + 1, figure=fig,
                      width_ratios=[1] * T_OUT + [0.05],
                      hspace=0.4, wspace=0.08)

        row_data   = [gt_norm, pred_norm, gt_mmhr, pred_mmhr]
        row_labels = ["GT (normalized)", "Pred (normalized)", "GT (mm/hr)", "Pred (mm/hr)"]
        row_cmaps  = ["viridis", "viridis", "viridis", "viridis"]
        row_vlims  = [(0, 1), (0, 1), (0, None), (0, None)]

        axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(T_OUT)] for r in range(4)]
        cbar_axes = [fig.add_subplot(gs[r, T_OUT]) for r in range(4)]

        for r in range(4):
            vlo, vhi = row_vlims[r]
            vhi = vhi if vhi is not None else float(max(row_data[r].max(), 1e-6))
            for t in range(T_OUT):
                ax = axes_grid[r][t]
                im = ax.imshow(row_data[r][t], cmap=row_cmaps[r],
                               vmin=vlo, vmax=vhi, origin="upper", aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(TIMESTEP_LABELS[t], fontsize=9)
                if t == 0:
                    ax.set_ylabel(row_labels[r], fontsize=9, labelpad=4)
            plt.colorbar(im, cax=cbar_axes[r])
            cbar_axes[r].set_ylabel("[0,1]" if r < 2 else "mm/hr", fontsize=8)

        fname = os.path.join(out_dir, f"earthformer_finetune_sample{i+1:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")

    print(f"Visualization done. Figures saved to '{out_dir}'.")


# ==========================================
# SAVE RESULTS TO CSV
# ==========================================
def save_results(results, path="validation_results_earthformer_finetune_11.csv"):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Results saved to {path}")


# ==========================================
# MAIN — identical to 0-shot
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

    # --- Visualize ---
    visualize_predictions(preds, targets)


if __name__ == "__main__":
    main()