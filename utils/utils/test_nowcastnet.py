import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── NORMALIZATION FUNCTION (provided) ─────────────────────────────────────────

def mmhr_to_intensity_with_normalize(x: torch.Tensor, R_max: float = 60.0) -> torch.Tensor:
    """
    Forward: mm/hr -> [0, 1]
    Uses log1p normalization scaled by log1p(R_max).
    """
    norm = torch.log1p(x) / np.log1p(R_max)
    return torch.clamp(norm, 0.0, 1.0)


# ── PREPROCESSING ─────────────────────────────────────────────────────────────

def preprocess_rainfall(arr: np.ndarray):
    """
    Preprocessing pipeline for rainfall data.

    Input : arr — (3, 112, 112) float32, mm/hr
    Output:
        tensor : (112, 112) float32 — preprocessed, ready for model
        mask   : (112, 112) bool    — True where step-2 output was < 0

    Pipeline:
        Step 1 : mm/hr → [0,1]  via mmhr_to_intensity_with_normalize
        Step 2 : x/10 - 3       normalize as per authors
        Step 3 : mask creation (where < 0) + remove negatives (<0 → 0)
        Step 4 : Clip(0, 40)
    """
    # Extract rain channel (mm/hr) — no grayscale, direct mm/hr
    rain = arr[0, :, :].astype(np.float32)             # (112, 112)
    rain = np.nan_to_num(rain, nan=0.0)

    # Step 1: mm/hr → pixel intensity [0, 1] via log1p normalization
    rain_tensor = torch.from_numpy(rain)
    step1 = mmhr_to_intensity_with_normalize(rain_tensor)   # [0, 1]

    # Step 2: normalize using x/10 - 3  (input = step1 output)
    step2 = step1 / 10.0 - 3.0

    # Step 3: mask creation for <0, then remove negatives
    mask  = step2 < 0.0                                # True = no-rain pixel
    step3 = torch.clamp(step2, min=0.0)

    # Step 4: Clip(0, 40)
    step4 = torch.clamp(step3, 0.0, 40.0)

    return step4, mask


# ── DATASET ───────────────────────────────────────────────────────────────────

class RainfallNPZDataset(torch.utils.data.Dataset):
    """
    Each .npz contains shape (24, 3, 112, 112), mm/hr.

    Returns:
        X      : (input_len,  112, 112) float32 — normalized [0, 1]
        Y      : (output_len, 112, 112) float32 — normalized [0, 1]
        X_mask : (input_len,  112, 112) bool    — no-rain mask for context frames
        Y_mask : (output_len, 112, 112) bool    — no-rain mask for target frames
    """
    def __init__(self, file_paths, input_len=4, output_len=6, npz_key=None):
        self.file_paths = file_paths
        self.input_len  = input_len
        self.output_len = output_len
        s = np.load(file_paths[0])
        self.npz_key = npz_key or list(s.files)[0]
        s.close()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with np.load(self.file_paths[idx]) as n:
            arr = n[self.npz_key].copy()               # (24, 3, 112, 112)

        frames, masks = [], []
        for t in range(arr.shape[0]):
            f, m = preprocess_rainfall(arr[t])
            frames.append(f)
            masks.append(m)

        frames = torch.stack(frames)                   # (24, 112, 112)
        masks  = torch.stack(masks)                    # (24, 112, 112) bool

        X      = frames[:self.input_len]
        Y      = frames[self.input_len:self.input_len + self.output_len]
        X_mask = masks[:self.input_len]
        Y_mask = masks[self.input_len:self.input_len + self.output_len]

        return X, Y, X_mask, Y_mask


# ── LOSS FUNCTION ─────────────────────────────────────────────────────────────

def grid_cell_regularizer(
    generated_samples: torch.Tensor,   # (gen_steps, B, T, H, W) — model output [0,1]
    batch_targets:     torch.Tensor,   # (B, T, H, W)             — already [0,1]
    generation_steps:  int,
) -> torch.Tensor:
    """
    gt = gt / 40 is replaced by mmhr_to_intensity_with_normalize,
    which already maps [0,40] → [0,1] in the dataset.
    So NO extra division needed here — both pred and gt are in [0,1].
    """
    mp     = nn.MaxPool2d(kernel_size=5, stride=2)
    pooled = [mp(generated_samples[i]) for i in range(generation_steps)]
    x_pred = torch.mean(torch.stack(pooled, dim=0), dim=0)
    return torch.mean(torch.abs(x_pred - mp(batch_targets)))


# ── VISUALIZATION ─────────────────────────────────────────────────────────────

def visualize_preprocessing(npz_path: str, npz_key: str = None,
                             num_frames: int = 4, save_path: str = None):
    """
    Visualizes the full preprocessing pipeline for `num_frames` frames.

    Rows:
        Row 1 — RAW (mm/hr)       : original channel 0
        Row 2 — AFTER CLIP(0,40)  : negatives removed, clipped
        Row 3 — MASK (<0)         : no-rain pixel mask
        Row 4 — NORMALIZED [0,1]  : after mmhr_to_intensity_with_normalize
        Row 5 — HISTOGRAM         : distribution at each stage
    """
    with np.load(npz_path) as n:
        key = npz_key or list(n.files)[0]
        arr = n[key].copy().astype(np.float32)         # (24, 3, 112, 112)

    num_frames = min(num_frames, arr.shape[0])
    n_rows     = 5

    fig = plt.figure(figsize=(4.5 * num_frames, 5 * n_rows))
    fig.suptitle(
        "NowcastNet Preprocessing Pipeline\n"
        "mm/hr  →  Remove negatives + Clip(0,40)  →  log1p Normalize  →  [0,1]",
        fontsize=14, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(n_rows, num_frames, figure=fig,
                           hspace=0.55, wspace=0.35)

    row_labels = ["RAW\n(mm/hr)", "CLIPPED\n(0,40)", "MASK\n(<0 pixels)",
                  "NORMALIZED\n[0,1]", "HISTOGRAM"]

    for col, t in enumerate(range(num_frames)):
        raw_frame = arr[t]                             # (3, 112, 112)

        # ── Compute each stage ────────────────────────────────────────────
        raw      = np.nan_to_num(raw_frame[0].copy(), nan=0.0)   # mm/hr
        mask     = raw < 0.0
        clipped  = np.clip(raw, 0.0, 40.0)
        norm_out = mmhr_to_intensity_with_normalize(
                       torch.from_numpy(clipped)
                   ).numpy()                           # [0, 1]

        # ── Row 0: RAW ───────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(raw, cmap="viridis", vmin=0, vmax=80)
        ax.set_title(f"Frame {t} | RAW\nmin={raw.min():.1f}  max={raw.max():.1f}",
                     fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm/hr")

        # ── Row 1: CLIPPED ───────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(clipped, cmap="viridis", vmin=0, vmax=40)
        ax.set_title(f"Frame {t} | CLIPPED\nmin={clipped.min():.2f}  max={clipped.max():.2f}",
                     fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm/hr")

        # ── Row 2: MASK ──────────────────────────────────────────────────
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(mask, cmap="Reds", vmin=0, vmax=1)
        pct = 100.0 * mask.sum() / mask.size
        ax.set_title(f"Frame {t} | MASK\n{mask.sum()} px = {pct:.1f}% no-rain",
                     fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="1=masked")

        # ── Row 3: NORMALIZED ────────────────────────────────────────────
        ax = fig.add_subplot(gs[3, col])
        im = ax.imshow(norm_out, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Frame {t} | NORMALIZED\nmin={norm_out.min():.3f}  max={norm_out.max():.3f}",
                     fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="[0,1]")

        # ── Row 4: HISTOGRAM ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[4, col])
        ax.hist(raw.flatten(),      bins=50, alpha=0.4, color="steelblue",
                label="Raw",      density=True)
        ax.hist(clipped.flatten(),  bins=50, alpha=0.4, color="tomato",
                label="Clipped",  density=True)
        ax.hist(norm_out.flatten(), bins=50, alpha=0.4, color="seagreen",
                label="Norm",     density=True)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Frame {t} | Histogram", fontsize=8)
        ax.set_xlabel("value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    # Row labels on far left
    for r, label in enumerate(row_labels):
        fig.text(0.005, 0.88 - r * 0.185, label, va="center",
                 fontsize=9, fontweight="bold", color="dimgray", rotation=90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved → {save_path}")
    plt.show()


# ── SANITY CHECK ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob, os

    DATA_DIR  = "/home/muskaan06/Desktop/Research/nowcasting/datasets/selected_sets_2024_SI_112"
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

    visualize_preprocessing(
        npz_path   = npz_files[0],
        num_frames = 4,
        save_path  = "preprocessing_pipeline_viz.png"
    )
   