import os
import csv
import random
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# ── PySTEPS ──────────────────────────────────────────────────────────────────
import pysteps
from pysteps import nowcasts, motion
from pysteps.utils import transformation   # dBZ helpers (optional)
from pysteps.verification import detcatscores

# ── Borrow metric helpers from your existing eval_metrics module ─────────────
from eval_metrics import soft_csi_loss, hard_csi, compute_ssim, exp_weighted_temporal_ssim

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  — keep identical to evaluate.py where possible
# =============================================================================
CONFIG = {
    "data_dir":    "data_images/IMC_Combined",
    "train_split": 0.7,
    "seed":        42,

    # PySTEPS method: "steps" | "sprog" | "extrapolation" | "linda"
    # "extrapolation" is fast (deterministic), "steps" is full stochastic
    "method":      "steps",

    # STEPS ensemble size (ignored for deterministic methods)
    "n_ens_members": 24,

    # Number of cascade levels for STEPS / S-PROG
    "n_cascade_levels": 6,

    # Optical-flow estimator: "lucaskanade" | "vet" | "darts"
    "oflow_method": "lucaskanade",

    # How many validation samples to run (None = all)
    "max_samples":  None,

    # Output lead-times to evaluate (must be ≤ T_OUT)
    "T_IN":  13,    # input frames
    "T_OUT": 11,   # forecast frames

    # Spatial resolution in km/pixel. Required by STEPS velocity perturbation.
    # Compute as: domain_width_km / image_width_pixels (e.g. 560km/112px = 5.0)
    # Set to None to disable velocity perturbation (vel_pert_method=None).
    "kmperpixel":  8.0,         # 8 km/pixel → 112 × 8 = 896 km domain

    # Time step between input frames in minutes
    "timestep":    30.0,

    # mm/hr threshold below which rain is considered zero (masking)
    "zero_threshold": 0,

    # Evaluation thresholds (overwritten dynamically like evaluate.py)
    "csi_thresholds": [0.5, 1.0, 2.0, 5.0],

    # Visualisation
    "vis_dir":       "./plottings_pysteps",
    "n_vis_samples": 5,

    # Results CSV
    "results_csv":   "validation_results_pysteps.csv",
}

# =============================================================================
# HELPERS: shared with evaluate.py
# =============================================================================

def to_pixel_intensity(x_norm: torch.Tensor) -> torch.Tensor:
    """Converts [0,1] float → [0,255] uint8 (mirrors evaluate.py)."""
    pixels = x_norm * 255.0
    return torch.clamp(pixels, 0, 255).to(torch.uint8)


def mmhr_to_pixel(x_mmhr: np.ndarray, R_max: float = 60.0) -> torch.Tensor:
    """
    mm/hr numpy array → normalised [0,1] float tensor → uint8 pixel tensor.
    Matches the evaluate.py pipeline so metrics are on the same scale.
    """
    t = torch.from_numpy(x_mmhr).float()
    norm = torch.log1p(t) / np.log1p(R_max)
    norm = torch.clamp(norm, 0.0, 1.0)
    return to_pixel_intensity(norm)


def mmhr_to_float_norm(x_mmhr: np.ndarray, R_max: float = 60.0) -> torch.Tensor:
    """mm/hr → normalised [0,1] float tensor (for SSIM)."""
    t = torch.from_numpy(x_mmhr).float()
    norm = torch.log1p(t) / np.log1p(R_max)
    return torch.clamp(norm, 0.0, 1.0)

# =============================================================================
# DATA LOADING  (no normalisation; mm/hr is passed directly to PySTEPS)
# =============================================================================

def load_val_files():
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
    return val_files


def load_sample(path: str):
    """
    Returns:
      x_mmhr : (T_IN,  112, 112)  float32  mm/hr  — input sequence
      y_mmhr : (T_OUT, 112, 112)  float32  mm/hr  — target sequence
    """
    with np.load(path, allow_pickle=False) as data:
        arr = np.array(data["array"][:, 0, :, :], dtype=np.float32)   # (16, H, W)
    arr = np.nan_to_num(arr, nan=0.0)
    x_mmhr = arr[:CONFIG["T_IN"]]     # (6, 112, 112)
    y_mmhr = arr[CONFIG["T_IN"]: CONFIG["T_IN"] + CONFIG["T_OUT"]]    # (10, 112, 112)
    return x_mmhr, y_mmhr

# =============================================================================
# PYSTEPS  PREDICTION
# =============================================================================

def prepare_pysteps_input(x_mmhr: np.ndarray):
    """
    PySTEPS expects:
      R  : (T, H, W) float64, with np.nan where rain < threshold
      metadata : dict with at least 'unit', 'transform', 'threshold',
                 'zerovalue', 'accutime', 'cartesian_unit', 'x1','x2','y1','y2'
    We pass mm/hr directly without dBZ conversion so the scale matches
    what evaluate.py uses.
    """
    R = x_mmhr.astype(np.float64)

    # Mask near-zero values (PySTEPS requirement)
    R[R < CONFIG["zero_threshold"]] = np.nan

    metadata = {
        "unit":           "mm/h",
        "transform":      None,        # no dBZ transform
        "threshold":      CONFIG["zero_threshold"],
        "zerovalue":      0.0,
        "accutime":       30.0,        # minutes between frames
        "cartesian_unit": "m",
        "x1": 0.0, "x2": 112.0 * 8000,   # 8 km/pixel in metres
        "y1": 0.0, "y2": 112.0 * 8000,
        "yorigin":        "upper",
        "unit":           "mm/h",
        "institution":    "custom",
        "projection":     "+proj=laea",   # dummy
    }
    return R, metadata


def run_pysteps(x_mmhr: np.ndarray):
    """
    Runs the configured PySTEPS method on a single sample.

    Internally converts mm/hr → dBZ for cascade-based methods (steps, sprog),
    because PySTEPS noise / AR models assume log-transformed data.
    Converts back to mm/hr before returning.

    Returns:
      forecast : (T_OUT, H, W) float32  mm/hr
    """
    T_OUT    = CONFIG["T_OUT"]
    method   = CONFIG["method"]
    thr_mmhr = CONFIG["zero_threshold"]

    # ── 1. Prepare input in mm/hr with NaN mask ───────────────────────────────
    R_mmhr = x_mmhr.astype(np.float64)
    R_mmhr[R_mmhr < thr_mmhr] = np.nan

    # ── 2. Optical flow (done in mm/hr / linear space) ────────────────────────
    oflow_fn = motion.get_method(CONFIG["oflow_method"])
    V        = oflow_fn(R_mmhr)                         # (2, H, W)

    # ── 3. Nowcast ────────────────────────────────────────────────────────────
    if method == "extrapolation":
        # Extrapolation works fine in mm/hr
        nc_fn    = nowcasts.get_method("extrapolation")
        forecast = nc_fn(R_mmhr[-1], V, T_OUT)          # (T_OUT, H, W)
        forecast = np.where(np.isnan(forecast), 0.0, forecast)

    elif method in ("steps", "sprog"):
        # Convert to dBZ — required for cascade / noise models in PySTEPS
        R_db, _ = transformation.dB_transform(
            R_mmhr.copy(), threshold=thr_mmhr, zerovalue=-15.0
        )
        R_db[~np.isfinite(R_db)] = -15.0               # fill NaN / -inf
        # dBZ equivalent of the mm/hr threshold
        precip_thr_db = 10.0 * np.log10(thr_mmhr) if thr_mmhr > 0 else -15.0

        if method == "sprog":
            nc_fn    = nowcasts.get_method("sprog")
            forecast = nc_fn(
                R_db, V, T_OUT,
                n_cascade_levels = CONFIG["n_cascade_levels"],
                precip_thr       = precip_thr_db,
            )                                           # (T_OUT, H, W)

        else:  # steps
            nc_fn = nowcasts.get_method("steps")
            # kmperpixel and timestep are required when vel_pert_method is set.
            # If kmperpixel is unknown, disable velocity perturbation entirely.
            kmperpixel = CONFIG.get("kmperpixel")
            vel_pert   = "bps" if kmperpixel is not None else None
            mask_meth  = "incremental" if kmperpixel is not None else "sprog"
            steps_kwargs = dict(
                n_ens_members    = CONFIG["n_ens_members"],
                n_cascade_levels = CONFIG["n_cascade_levels"],
                precip_thr       = precip_thr_db,
                vel_pert_method  = vel_pert,
                mask_method      = mask_meth,
            )
            if kmperpixel is not None:
                steps_kwargs["kmperpixel"] = kmperpixel
                steps_kwargs["timestep"]   = CONFIG["timestep"]
            forecast = nc_fn(R_db, V, T_OUT, **steps_kwargs)  # (n_ens, T_OUT, H, W)
            forecast = np.nanmean(forecast, axis=0)            # ensemble mean

        # Back-transform dBZ → mm/hr
        forecast, _ = transformation.dB_transform(
            forecast, threshold=precip_thr_db, inverse=True
        )
        forecast = np.where(np.isnan(forecast), 0.0, forecast)

    elif method == "linda":
        nc_fn    = nowcasts.get_method("linda")
        forecast = nc_fn(R_mmhr, V, T_OUT)
        if forecast.ndim == 4:                           # (ens, T, H, W)
            forecast = np.nanmean(forecast, axis=0)
        forecast = np.where(np.isnan(forecast), 0.0, forecast)

    else:
        raise ValueError(f"Unknown method: {method}")

    forecast = np.clip(forecast, 0.0, None).astype(np.float32)
    return forecast                                      # (T_OUT, H, W)


# =============================================================================
# COLLECT ALL PREDICTIONS
# =============================================================================

def collect_predictions(val_files):
    """
    Returns:
      preds   : (N, T_OUT, 112, 112)  torch.float32  mm/hr
      targets : (N, T_OUT, 112, 112)  torch.float32  mm/hr
    """
    max_n   = CONFIG["max_samples"] or len(val_files)
    files   = val_files[:max_n]

    all_preds, all_targets = [], []
    failed = 0

    for path in tqdm(files, desc=f"PySTEPS [{CONFIG['method']}]"):
        try:
            x_mmhr, y_mmhr = load_sample(path)
            pred_mmhr       = run_pysteps(x_mmhr)       # (T_OUT, H, W)
            all_preds.append(torch.from_numpy(pred_mmhr))
            all_targets.append(torch.from_numpy(y_mmhr))
        except Exception as e:
            failed += 1
            print(f"\n  [WARN] Skipped {os.path.basename(path)}: {e}")

    print(f"\nSuccessfully processed : {len(all_preds)} / {len(files)}  ({failed} failed)")
    preds   = torch.stack(all_preds,   dim=0)   # (N, T_OUT, H, W)
    targets = torch.stack(all_targets, dim=0)   # (N, T_OUT, H, W)
    return preds, targets


# =============================================================================
# DYNAMIC THRESHOLD  (mirrors evaluate.py)
# =============================================================================

def compute_dynamic_thresholds(targets_px, num_thresholds=5):
    flat = targets_px.numpy().flatten().astype(float)
    flat = flat[flat > 0]
    if len(flat) == 0:
        raise ValueError("All targets are zero!")
    percentiles = np.linspace(10, 60, num_thresholds)
    thresholds  = [float(np.round(np.percentile(flat, p), 2)) for p in percentiles]
    print("\n📊 Auto-selected thresholds (percentile-based):")
    for p, t in zip(percentiles, thresholds):
        print(f"  {p:.1f}th percentile → {t}")
    return thresholds


# =============================================================================
# EVALUATE  (mirrors evaluate.py metric pipeline)
# =============================================================================

def evaluate(preds_mmhr, targets_mmhr):
    """
    preds_mmhr, targets_mmhr : (N, T_OUT, H, W)  float32  mm/hr

    MAE / MSE  → computed in normalised [0,1] space (same as evaluate.py)
    CSI        → computed in pixel intensity [0-255] (same as evaluate.py,
                 so thresholds are comparable across models)
    SSIM       → computed in pixel intensity [0-255]
    """
    results = {}
    B, T, H, W = preds_mmhr.shape

    # ── Raw mm/hr tensors (kept for reference, not used in metrics) ───────────
    preds_raw   = preds_mmhr.float()      # (N, T, H, W)  mm/hr
    targets_raw = targets_mmhr.float()    # (N, T, H, W)  mm/hr

    # ── Pixel-space [0-255] uint8 (for CSI, same scale as evaluate.py) ───────
    preds_px   = mmhr_to_pixel(preds_mmhr.numpy())     # (N, T, H, W) uint8
    targets_px = mmhr_to_pixel(targets_mmhr.numpy())

    # ── Normalised float [0,1] (for MSE, MAE, SSIM) ──────────────────────────
    preds_norm   = mmhr_to_float_norm(preds_mmhr.numpy())    # (N, T, H, W)
    targets_norm = mmhr_to_float_norm(targets_mmhr.numpy())

    CONFIG["csi_thresholds"] = compute_dynamic_thresholds(targets_px)

    # Temporal view for SSIM helpers
    pred_seq    = preds_norm.unsqueeze(2)[:, :, 0]     # (N, T, H, W)
    y_seq       = targets_norm.unsqueeze(2)[:, :, 0]
    pred_seq_px = preds_px.unsqueeze(2)[:, :, 0]
    y_seq_px    = targets_px.unsqueeze(2)[:, :, 0]

    # ------------------------------------------------------------------
    # 1. MSE & MAE  — in normalised [0,1] space (same scale as evaluate.py)
    # ------------------------------------------------------------------
    mse = torch.mean((preds_norm - targets_norm) ** 2).item()
    mae = torch.mean(torch.abs(preds_norm - targets_norm)).item()
    results["MSE"] = mse
    results["MAE"] = mae
    print(f"\n{'='*55}")
    print(f"  MSE (normalised) : {mse:.6f}")
    print(f"  MAE (normalised) : {mae:.6f}")

    # ------------------------------------------------------------------
    # 2. Hard CSI
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Hard CSI across thresholds")
    print(f"  {'Threshold':>12}  {'CSI':>10}")
    print(f"  {'-'*25}")
    for thr in CONFIG["csi_thresholds"]:
        csi_val = hard_csi(preds_px, targets_px, threshold=thr)
        if csi_val is None:
            results[f"hard_CSI@thr={thr}"] = float("nan")
            print(f"  {thr:>12.2f}  {'None':>10}")
        else:
            v = csi_val.item()
            results[f"hard_CSI@thr={thr}"] = v
            print(f"  {thr:>12.2f}  {v:>10.4f}")

    # ------------------------------------------------------------------
    # 3. Soft CSI loss
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Soft CSI loss across thresholds")
    print(f"  {'Threshold':>12}  {'Soft-CSI-Loss':>15}")
    print(f"  {'-'*30}")
    for thr in CONFIG["csi_thresholds"]:
        v = soft_csi_loss(preds_px, targets_px, threshold=thr).item()
        results[f"soft_CSI_loss@thr={thr}"] = v
        print(f"  {thr:>12.2f}  {v:>15.4f}")

    # ------------------------------------------------------------------
    # 4. SSIM per timestep
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  SSIM (compute_ssim)")
    print(f"  {'Metric':>30}  {'Value':>10}")
    print(f"  {'-'*43}")

    ssim_overall = compute_ssim(pred_seq_px, y_seq_px)
    results["SSIM_overall"] = ssim_overall
    print(f"  {'SSIM_overall':>30}  {ssim_overall:>10.4f}")

    pred_np = pred_seq.numpy()
    y_np    = y_seq.numpy()
    for t in range(T):
        scores_t = [
            compute_ssim(
                to_pixel_intensity(torch.from_numpy(pred_np[b, t])),
                to_pixel_intensity(torch.from_numpy(y_np[b, t]))
            )
            for b in range(B)
        ]
        valid_t = [s for s in scores_t if s != 0.0]
        ssim_t  = float(np.mean(valid_t)) if valid_t else 0.0
        results[f"SSIM@t={t+1}"] = ssim_t
        print(f"  {f'SSIM@t={t+1}':>30}  {ssim_t:>10.4f}")

    # ------------------------------------------------------------------
    # 5. Exp-weighted temporal SSIM
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Exp-Weighted Temporal SSIM")
    print(f"  {'-'*43}")
    tw_ssim = exp_weighted_temporal_ssim(pred_seq_px, y_seq_px)
    tw_ssim = float(tw_ssim) if tw_ssim is not None else 0.0
    results["TW_SSIM"] = tw_ssim
    print(f"  {'TW_SSIM':>30}  {tw_ssim:>10.4f}")
    print(f"\n{'='*55}\n")

    return results


# =============================================================================
# VISUALISE  (mirrors evaluate.py layout)
# =============================================================================

def visualize_predictions(preds_mmhr, targets_mmhr, num_samples=5):
    T_OUT           = CONFIG["T_OUT"]
    TIMESTEP_LABELS = [f"t+{i*10} min" for i in range(1, T_OUT + 1)]
    out_dir         = CONFIG["vis_dir"]
    os.makedirs(out_dir, exist_ok=True)

    n = min(num_samples, preds_mmhr.shape[0])

    for i in range(n):
        pred_mmhr = preds_mmhr[i].numpy()      # (T_OUT, H, W)
        gt_mmhr   = targets_mmhr[i].numpy()

        pred_norm = mmhr_to_float_norm(pred_mmhr).numpy()
        gt_norm   = mmhr_to_float_norm(gt_mmhr).numpy()

        fig = plt.figure(figsize=(T_OUT * 3.2, 14))
        fig.suptitle(
            f"PySTEPS [{CONFIG['method'].upper()}] — Sample {i+1} | Precipitation",
            fontsize=14, fontweight="bold", y=1.01
        )

        gs = GridSpec(4, T_OUT + 1, figure=fig,
                      width_ratios=[1] * T_OUT + [0.05],
                      hspace=0.4, wspace=0.08)

        row_data   = [gt_norm, pred_norm, gt_mmhr, pred_mmhr]
        row_labels = ["GT (normalized)", "Pred (normalized)", "GT (mm/hr)", "Pred (mm/hr)"]
        row_vlims  = [(0, 1), (0, 1), (0, None), (0, None)]

        axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(T_OUT)] for r in range(4)]
        cbar_axes = [fig.add_subplot(gs[r, T_OUT]) for r in range(4)]

        for r in range(4):
            vlo, vhi = row_vlims[r]
            vhi = vhi if vhi is not None else float(max(row_data[r].max(), 1e-6))
            for t in range(T_OUT):
                ax = axes_grid[r][t]
                im = ax.imshow(row_data[r][t], cmap="viridis",
                               vmin=vlo, vmax=vhi, origin="upper", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(TIMESTEP_LABELS[t], fontsize=9)
                if t == 0:
                    ax.set_ylabel(row_labels[r], fontsize=9, labelpad=4)
            plt.colorbar(im, cax=cbar_axes[r])
            cbar_axes[r].set_ylabel("[0,1]" if r < 2 else "mm/hr", fontsize=8)

        fname = os.path.join(out_dir, f"pysteps_{CONFIG['method']}_sample{i+1:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")

    print(f"Visualisations saved to '{out_dir}'.")


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results):
    path = CONFIG["results_csv"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Results saved to {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    #print(f"PySTEPS version : {pysteps.__version__}")
    print(f"Method          : {CONFIG['method']}")
    print(f"Optical flow    : {CONFIG['oflow_method']}")

    if not os.path.exists(CONFIG["data_dir"]):
        raise FileNotFoundError(f"data_dir '{CONFIG['data_dir']}' not found.")

    val_files = load_val_files()

    print("\nRunning PySTEPS inference...")
    preds, targets = collect_predictions(val_files)
    print(f"Predictions shape : {preds.shape}")
    print(f"Targets shape     : {targets.shape}")

    results = evaluate(preds, targets)
    save_results(results)
    visualize_predictions(preds, targets, num_samples=CONFIG["n_vis_samples"])


if __name__ == "__main__":
    main()