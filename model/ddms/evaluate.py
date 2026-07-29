"""
DDMS evaluation.

Unlike model/diffcast/, DDMS's diffusion component is genuinely trainable —
there's no fast "backbone-only" shortcut here. The only inference path is
GaussianDiffusion.sample(), which is autoregressive at the frame level: each
of the 8 output frames costs a full DDIM sampling loop (--sample_steps NFEs).
At the dev default (sample_steps=50) that's 8*50=400 sequential U-Net passes
per example; at the paper's sample_steps=200 it's 1600. --num_eval_samples
caps how many validation examples get scored, since a full val-set pass
would be impractically slow.
"""
import os
import random
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from model_arch import build_ddms_model
from patch_utils import mmhr_to_intensity_with_normalize, to_pixel_intensity, to_pm1, from_pm1
from eval_metrics import soft_csi_loss, hard_csi, hard_pod, hard_far
from eval_metrics import compute_ssim, exp_weighted_temporal_ssim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


T_CONTEXT = 8
T_OUT = 8

CONFIG = {
    "data_dir":    "/mnt/sda1/Muskaan/nowcast/IMC_Combined",
    "train_split": 0.7,          # must match train.py
    "seed":        42,
    "checkpoint":  "../../model_weights/ddms/ddms-best-epoch=3.ckpt",   # ← change to your .ckpt path
    "use_ema":     False,        # prefer the "ema" checkpoint key if present (raw DDMS format only)

    # DDMS architecture — must match the checkpoint being loaded
    "dim":               32,
    "dim_mults":          (1, 1, 2, 2, 4),
    "transform_dim_mults": (1, 2, 3, 4),
    "backbone":           "resnet",
    "timesteps":          500,

    # inference-only knobs — safe to override independent of the checkpoint
    "sample_steps": 50,
    "ddim": True,

    "num_eval_samples": 16,

    # --- Sweep ranges (overwritten by compute_dynamic_thresholds) ---
    "csi_thresholds": [0.5, 1.0, 2.0, 5.0],
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
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:, 0, :, :], dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.expand_dims(arr, axis=1)
        clip = torch.from_numpy(arr).float()

        return mmhr_to_intensity_with_normalize(clip)

# ==========================================
# LOAD MODEL
# ==========================================
def load_model(checkpoint_path, device):
    """
    Supports:
      1. PyTorch Lightning checkpoint (has 'state_dict' key, keys prefixed 'model.')
      2. Raw DDMS checkpoint ({"step", "model", "ema"}, keys possibly prefixed
         'module.' from DDP) — prefers "ema" if present and CONFIG["use_ema"].
      3. Plain state_dict saved via torch.save(model.state_dict(), path)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_ddms_model(
        dim=CONFIG["dim"], dim_mults=CONFIG["dim_mults"],
        transform_dim_mults=CONFIG["transform_dim_mults"], backbone=CONFIG["backbone"],
        timesteps=CONFIG["timesteps"], sample_steps=CONFIG["sample_steps"], ddim=CONFIG["ddim"],
    ).to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state)
        print("Loaded PyTorch Lightning checkpoint.")
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        key = "ema" if (CONFIG["use_ema"] and "ema" in checkpoint) else "model"
        raw_state = checkpoint[key]
        state = collections.OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in raw_state.items()
        )
        model.load_state_dict(state)
        print(f"Loaded raw DDMS checkpoint (key='{key}').")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded plain state-dict checkpoint.")

    model.eval()
    return model


# ==========================================
# COLLECT PREDICTIONS
# ==========================================
@torch.no_grad()
def collect_predictions(model, files, device, num_eval_samples=16):
    dataset = NPZDataset(files)
    n = min(num_eval_samples, len(dataset))
    print(f"Scoring {n} / {len(dataset)} val examples "
          f"(capped via CONFIG['num_eval_samples']; DDIM sampling is slow).")

    all_preds, all_targets = [], []
    for i in range(n):
        clip = dataset[i]                       # (16,1,H,W), [0,1]
        context = clip[:T_CONTEXT].unsqueeze(1).to(device)   # (8,1,1,H,W) time-major, B=1
        context = to_pm1(context)

        video, video_diff, video_mse = model.sample(context, num_of_frames=T_OUT)
        pred = from_pm1(video[T_CONTEXT:]).clamp(0, 1)        # (8,1,1,H,W)
        pred = pred.squeeze(1).squeeze(1)                     # (8,H,W)

        target = clip[T_CONTEXT:T_CONTEXT + T_OUT].squeeze(1)  # (8,H,W)

        all_preds.append(pred.unsqueeze(0).cpu())
        all_targets.append(target.unsqueeze(0))
        print(f"  sampled {i+1}/{n}")

    preds = torch.cat(all_preds, dim=0)     # (N,8,H,W)
    targets = torch.cat(all_targets, dim=0)
    return preds, targets


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
# EVALUATION
# ==========================================
def evaluate(preds, targets, T_out=8, label="DDMS"):
    """
    preds, targets: (N, T_out, 112, 112)  — on CPU, [0,1] normalized
    """
    results = {}
    CONFIG["csi_thresholds"] = compute_dynamic_thresholds(to_pixel_intensity(targets))

    B, C, H, W = preds.shape
    pred_seq = preds.view(B, T_out, 1, H, W)[:, :, 0]
    y_seq    = targets.view(B, T_out, 1, H, W)[:, :, 0]

    # --------------------------------------------------
    # 1.  MSE  &  MAE
    # --------------------------------------------------
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    results["MSE"] = mse
    results["MAE"] = mae
    print(f"\n{'='*55}")
    print(f"  [{label}]  MSE : {mse:.6f}")
    print(f"  [{label}]  MAE : {mae:.6f}")

    # --------------------------------------------------
    # 2.  Hard CSI / POD / FAR at multiple thresholds
    # --------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Hard CSI / POD / FAR across thresholds")
    print(f"  {'Threshold':>12}  {'CSI':>10}  {'POD':>10}  {'FAR':>10}")
    print(f"  {'-'*50}")
    for thr in CONFIG["csi_thresholds"]:
        pix_preds   = to_pixel_intensity(preds)
        pix_targets = to_pixel_intensity(targets)

        csi_val = hard_csi(pix_preds, pix_targets, threshold=thr)
        pod_val = hard_pod(pix_preds, pix_targets, threshold=thr)
        far_val = hard_far(pix_preds, pix_targets, threshold=thr)

        csi_str = f"{csi_val.item():.4f}" if csi_val is not None else "None"
        pod_str = f"{pod_val.item():.4f}" if pod_val is not None else "None"
        far_str = f"{far_val.item():.4f}" if far_val is not None else "None"

        results[f"hard_CSI@thr={thr}"] = csi_val.item() if csi_val is not None else float("nan")
        results[f"hard_POD@thr={thr}"] = pod_val.item() if pod_val is not None else float("nan")
        results[f"hard_FAR@thr={thr}"] = far_val.item() if far_val is not None else float("nan")

        print(f"  {thr:>12.2f}  {csi_str:>10}  {pod_str:>10}  {far_str:>10}")

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
    # 4.  SSIM  (compute_ssim — Eq. 7 formula)
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

    T = pred_np.shape[1]
    for t in range(T):
        scores_t = [
            compute_ssim(to_pixel_intensity(torch.from_numpy(pred_np[b, t])), to_pixel_intensity(torch.from_numpy(y_np[b, t])))
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


def visualize_predictions(preds, targets, T_out=8, out_dir="./plottings_ddms", num_samples=5, label="DDMS"):
    N_CH = 1
    TIMESTEP_LABELS = [f"t+{i*15} min" for i in range(1, T_out + 1)]

    os.makedirs(out_dir, exist_ok=True)
    n = min(num_samples, preds.shape[0])

    for i in range(n):
        pred_norm = preds[i].view(T_out, N_CH, 112, 112)[:, 0].numpy()
        gt_norm   = targets[i].view(T_out, N_CH, 112, 112)[:, 0].numpy()

        pred_mmhr = to_pixel_intensity(preds[i]).view(T_out, N_CH, 112, 112)[:, 0].numpy()
        gt_mmhr   = to_pixel_intensity(targets[i]).view(T_out, N_CH, 112, 112)[:, 0].numpy()

        fig = plt.figure(figsize=(T_out * 3.2, 14))
        fig.suptitle(f"{label} — Sample {i+1} | Precipitation (channel 0)",
                     fontsize=14, fontweight="bold", y=1.01)

        gs = GridSpec(4, T_out + 1, figure=fig,
                      width_ratios=[1] * T_out + [0.05],
                      hspace=0.4, wspace=0.08)

        row_data   = [gt_norm, pred_norm, gt_mmhr, pred_mmhr]
        row_labels = ["GT (normalized)", "Pred (normalized)", "GT (mm/hr)", "Pred (mm/hr)"]
        row_cmaps  = ["viridis", "viridis", "viridis", "viridis"]
        row_vlims  = [(0, 1), (0, 1), (0, None), (0, None)]

        axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(T_out)] for r in range(4)]
        cbar_axes = [fig.add_subplot(gs[r, T_out]) for r in range(4)]

        for r in range(4):
            vlo, vhi = row_vlims[r]
            vhi = vhi if vhi is not None else float(max(row_data[r].max(), 1e-6))
            for t in range(T_out):
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

        fname = os.path.join(out_dir, f"ddms_sample{i+1:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")

    print(f"Visualization done. Figures saved to '{out_dir}'.")


# ==========================================
# SAVE RESULTS TO CSV
# ==========================================
def save_results(results, path="validation_results_ddms.csv"):
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

    model = load_model(CONFIG["checkpoint"], device)

    print("\nRunning inference (DDIM sampling, slow)...")
    preds, targets = collect_predictions(model, val_files, device, num_eval_samples=CONFIG["num_eval_samples"])
    print(f"Predictions shape : {preds.shape}")
    results = evaluate(preds, targets, T_out=T_OUT, label="DDMS")
    save_results(results)

    # --- Visualize ---
    visualize_predictions(preds, targets, T_out=T_OUT)


if __name__ == "__main__":
    main()
