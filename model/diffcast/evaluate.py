"""
DiffCast evaluation — two modes:

Mode A (default): score a backbone-only checkpoint (from train.py). Fast —
one or a few forward passes per batch, same shape as the other backends'
evaluate.py scripts.

Mode B (--diffusion_checkpoint <path>): additionally run the full DiffCast
residual-diffusion sampling pipeline against an EXTERNALLY DOWNLOADED
pretrained checkpoint (the official repo never trains the diffusion stage —
see model/diffcast/diffusion.py's GaussianDiffusion.predict(), which raises
NotImplementedError on compute_loss=True, exactly matching upstream).

The only publicly available checkpoint (resources/diffcast_phydnet_sevir128.pt,
linked from the DiffCast README's Google Drive) was trained at T_in=5,
T_out=20, img_size=128, backbone=phydnet, dim=64 — NOT this repo's own
T_in=8/T_out=8/112px default. Mode B therefore uses its own independent CLI
config (--diff_*), resizes this repo's 112x112 frames up to 128x128 before
sampling, and resizes predictions back down to 112x112 for scoring. This is a
known caveat: VIL-vs-mm/hr domain mismatch and the resize round-trip both
affect fidelity of Mode B's numbers. Sampling is also slow (~250 DDIM steps x
T_out/T_in fragments = ~1000 sequential network passes per batch), so Mode B
caps how many validation samples it scores via --num_eval_samples.
"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model_arch import DiffCastBackbone
from patch_utils import mmhr_to_intensity_with_normalize, to_pixel_intensity
from eval_metrics import soft_csi_loss, hard_csi, hard_pod, hard_far
from eval_metrics import compute_ssim, exp_weighted_temporal_ssim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


CONFIG = {
    "data_dir":    "/mnt/sda1/Muskaan/nowcast/IMC_Combined",
    "train_split": 0.7,          # must match train.py
    "batch_size":  8,
    "num_workers": 1,
    "seed":        42,
    "backbone":    "phydnet",
    "T_in":        8,
    "T_out":       8,
    "img_width":   112,
    "checkpoint":  "../../model_weights/diffcast/diffcast-phydnet-best-epoch=3.ckpt",   # ← change to your .ckpt path

    # --- Sweep ranges (overwritten by compute_dynamic_thresholds) ---
    "csi_thresholds": [0.5, 1.0, 2.0, 5.0],

    # --- Mode B (optional) ---
    "diffusion_checkpoint": None,   # set to a downloaded .pt path to enable Mode B
    "diff_backbone":  "phydnet",
    "diff_T_in":      5,
    "diff_T_out":     20,
    "diff_img_size":  128,
    "diff_dim":       64,
    "diff_dim_mults": (1, 2, 4, 8),
    "diff_sampling_timesteps": 250,
    "num_eval_samples": 16,
}

# ==========================================
# DATASET  (identical to train.py)
# ==========================================
class NPZDataset(Dataset):
    def __init__(self, files, T_in=8, T_out=8):
        self.files = files
        self.T_in = T_in
        self.T_out = T_out

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"][:, 0, :, :], dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.expand_dims(arr, axis=1)  # (T_total, 1, H, W)
        x_raw = torch.from_numpy(arr[:self.T_in]).float()
        y_raw = torch.from_numpy(arr[self.T_in:self.T_in + self.T_out]).float()

        return mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)

# ==========================================
# LOAD MODEL (Mode A — backbone only)
# ==========================================
def load_model(checkpoint_path, device, backbone="phydnet", img_width=112, T_in=8, T_out=8):
    """
    Supports:
      1. PyTorch Lightning checkpoint (has 'state_dict' key, keys prefixed 'model.')
      2. Raw state-dict saved via torch.save(model.state_dict(), path)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DiffCastBackbone(backbone=backbone, C=1, H=img_width, W=img_width,
                             T_in=T_in, T_out=T_out, device=str(device)).to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state)
        print("Loaded PyTorch Lightning checkpoint.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded plain state-dict checkpoint.")

    model.eval()
    return model


# ==========================================
# COLLECT PREDICTIONS (Mode A)
# ==========================================
@torch.no_grad()
def collect_predictions(model, loader, device, T_out=8):
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred, _ = model.predict(x, compute_loss=False)
        print(f"pred min: {pred.min().item():.4f}, max: {pred.max().item():.4f}, mean: {pred.mean().item():.4f}, std: {pred.std().item():.4f}, in [0,1]: {(pred >= 0).all() and (pred <= 1).all()}")

        # (B, T_out, C=1, H, W) -> (B, T_out, H, W)
        pred = pred.reshape(pred.size(0), T_out, 112, 112)

        all_preds.append(pred.cpu())
        all_targets.append(y.reshape(y.size(0), T_out, 112, 112))
    preds = torch.cat(all_preds, dim=0)
    targets_raw = torch.cat(all_targets, dim=0)

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
# EVALUATION  (shared by Mode A and Mode B)
# ==========================================
def evaluate(preds, targets, T_out=8, label="DiffCast"):
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


def visualize_predictions(preds, targets, T_out=8, out_dir="./plottings_diffcast", num_samples=5, label="DiffCast"):
    N_CH = 1
    TIMESTEP_LABELS = [f"t+{i*10} min" for i in range(1, T_out + 1)]

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

        fname = os.path.join(out_dir, f"diffcast_sample{i+1:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")

    print(f"Visualization done. Figures saved to '{out_dir}'.")


# ==========================================
# SAVE RESULTS TO CSV
# ==========================================
def save_results(results, path="validation_results_diffcast.csv"):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Results saved to {path}")


# ==========================================
# MODE B — full diffusion sampling eval
# ==========================================
def run_mode_b(diffusion_checkpoint, val_files, device):
    """
    Runs the full residual-diffusion sampling pipeline against an externally
    downloaded pretrained checkpoint. SLOW (~1000 sequential network passes
    per batch at the official 250-DDIM-step / T_out=20,T_in=5 config) — caps
    total examples via CONFIG["num_eval_samples"].
    """
    import diffusion as diffusion_module

    diff_T_in = CONFIG["diff_T_in"]
    diff_T_out = CONFIG["diff_T_out"]
    diff_img_size = CONFIG["diff_img_size"]

    if diff_T_out % diff_T_in != 0:
        raise ValueError(
            f"GaussianDiffusion.sample() fragments the output in chunks of T_in "
            f"(loops range(T_out // T_in)), so T_out must be a multiple of T_in — "
            f"got diff_T_in={diff_T_in}, diff_T_out={diff_T_out}."
        )

    with np.load(val_files[0], allow_pickle=False) as probe:
        t_total = probe["array"].shape[0]
    if diff_T_in + diff_T_out > t_total:
        raise ValueError(
            f"Mode B config needs T_in+T_out={diff_T_in + diff_T_out} frames per clip "
            f"(the official checkpoint's T_in={diff_T_in}, T_out={diff_T_out}), but this "
            f"repo's .npz files only have {t_total} frames each. Either use .npz clips with "
            f"at least {diff_T_in + diff_T_out} frames, or lower --diff_T_out (results will no "
            f"longer match the official checkpoint's trained config)."
        )

    print(f"\n[Mode B] Building diffusion model: backbone={CONFIG['diff_backbone']}, "
          f"T_in={diff_T_in}, T_out={diff_T_out}, img_size={diff_img_size}, "
          f"dim={CONFIG['diff_dim']}, sampling_timesteps={CONFIG['diff_sampling_timesteps']}")

    backbone = DiffCastBackbone(
        backbone=CONFIG["diff_backbone"], C=1, H=diff_img_size, W=diff_img_size,
        T_in=diff_T_in, T_out=diff_T_out, device=str(device),
    ).net  # unwrap to the raw PhyDNet_Model/SimVP_Model the diffusion module expects

    diff_model = diffusion_module.get_model(
        img_channels=1, dim=CONFIG["diff_dim"], dim_mults=CONFIG["diff_dim_mults"],
        T_in=diff_T_in, T_out=diff_T_out,
        timesteps=1000, sampling_timesteps=CONFIG["diff_sampling_timesteps"],
    )
    diff_model.load_backbone(backbone)
    diff_model.to(device)

    print(f"[Mode B] Loading diffusion checkpoint: {diffusion_checkpoint}")
    ckpt = torch.load(diffusion_checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    diff_model.load_state_dict(state)
    diff_model.eval()

    val_dataset = NPZDataset(val_files, T_in=diff_T_in, T_out=diff_T_out)
    num_eval = min(CONFIG["num_eval_samples"], len(val_dataset))
    print(f"[Mode B] Scoring {num_eval} / {len(val_dataset)} val examples "
          f"(capped via --num_eval_samples; diffusion sampling is slow).")

    all_preds, all_targets = [], []
    for i in range(num_eval):
        x, y = val_dataset[i]
        x = x.unsqueeze(0).to(device)  # (1, T_in, 1, 112, 112)

        x_resized = F.interpolate(
            x.view(-1, 1, 112, 112), size=(diff_img_size, diff_img_size),
            mode="bilinear", align_corners=False,
        ).view(1, diff_T_in, 1, diff_img_size, diff_img_size)

        with torch.no_grad():
            pred, mu, ys = diff_model.sample(x_resized, T_out=diff_T_out)

        # resize prediction back down to 112x112 for metric comparability with GT / Mode A
        pred_112 = F.interpolate(
            pred.view(-1, 1, diff_img_size, diff_img_size), size=(112, 112),
            mode="bilinear", align_corners=False,
        ).view(1, diff_T_out, 112, 112)

        all_preds.append(pred_112.cpu())
        all_targets.append(y.view(1, diff_T_out, 112, 112))
        print(f"  [Mode B] sampled {i+1}/{num_eval}")

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return preds, targets, diff_T_out


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

    # ---------------- Mode A: backbone-only ----------------
    T_in, T_out = CONFIG["T_in"], CONFIG["T_out"]
    val_loader = DataLoader(
        NPZDataset(val_files, T_in=T_in, T_out=T_out),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(CONFIG["checkpoint"], device, backbone=CONFIG["backbone"],
                       img_width=CONFIG["img_width"], T_in=T_in, T_out=T_out)

    print("\n[Mode A] Running backbone-only inference...")
    preds, targets = collect_predictions(model, val_loader, device, T_out=T_out)
    print(f"Predictions shape : {preds.shape}")
    results = evaluate(preds, targets, T_out=T_out, label=f"DiffCast-{CONFIG['backbone']} (backbone-only)")
    save_results(results, path="validation_results_diffcast.csv")
    visualize_predictions(preds, targets, T_out=T_out, label=f"DiffCast-{CONFIG['backbone']}")

    # ---------------- Mode B: full diffusion (opt-in) ----------------
    if CONFIG["diffusion_checkpoint"]:
        diff_preds, diff_targets, diff_T_out = run_mode_b(CONFIG["diffusion_checkpoint"], val_files, device)
        print(f"[Mode B] Predictions shape : {diff_preds.shape}")
        diff_results = evaluate(diff_preds, diff_targets, T_out=diff_T_out, label="DiffCast (full diffusion)")
        save_results(diff_results, path="validation_results_diffcast_diffusion.csv")
        visualize_predictions(diff_preds, diff_targets, T_out=diff_T_out,
                              out_dir="./plottings_diffcast_diffusion", label="DiffCast (full diffusion)")
    else:
        print("\n[Mode B] Skipped (CONFIG['diffusion_checkpoint'] not set). "
              "Download resources/diffcast_phydnet_sevir128.pt from the DiffCast README's "
              "Google Drive link and set CONFIG['diffusion_checkpoint'] to enable it.")


if __name__ == "__main__":
    main()
