import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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


class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            arr = np.array(data["array"], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        x_raw = torch.from_numpy(arr[:4]).permute(0, 2, 3, 1)    # (4, 112, 112, 3)
        y_raw = torch.from_numpy(arr[4:10]).permute(0, 2, 3, 1)  # (6, 112, 112, 3)
        return x_raw, y_raw, mmhr_to_intensity_with_normalize(x_raw), mmhr_to_intensity_with_normalize(y_raw)


# ── Load ──────────────────────────────────────────────────────────────────────
val_file = "/mnt/sda1/Muskaan/nowcast/IMC_Combined/selected_sets_2021_7_Sisters_IMC_2021-05-31_22_45_to_2021-06-01_10_15.npz"

dataset = NPZDataset([val_file])
x_raw, y_raw, x_norm, y_norm = dataset[0]

y_np = to_pixel_intensity(y_norm)

# (6, 112, 112, 3) -> pick channel 0 -> (6, 112, 112)
without_pixel           = y_raw.permute(0, 3, 1, 2)[:, 0].numpy()
with_pixel              = y_norm.permute(0, 3, 1, 2)[:, 0].numpy()
with_pixel_and_normalize = y_np.permute(0, 3, 1, 2)[:, 0].numpy()

# ── Visualize ─────────────────────────────────────────────────────────────────
raw_vmax = max(float(without_pixel.max()), 1.0)

fig, axes = plt.subplots(3, 6, figsize=(20, 10))  # FIX: height 7→10 so row 3 isn't squished

for t in range(6):
    # Row 0: Raw mm/hr with Logarithmic Mapping
    im0 = axes[0, t].imshow(
        without_pixel[t],
        cmap="viridis",
        norm=colors.LogNorm(vmin=0.1, vmax=raw_vmax)
    )
    axes[0, t].set_title(f"t+{t+1} (Raw Log)", fontsize=11)
    axes[0, t].axis("off")
    plt.colorbar(im0, ax=axes[0, t], fraction=0.046, pad=0.04)

    # Row 1: Normalized [0, 1]
    im1 = axes[1, t].imshow(with_pixel[t], cmap="viridis", vmin=0, vmax=1.0)
    axes[1, t].set_title(f"t+{t+1} (Norm [0-1])", fontsize=11)
    axes[1, t].axis("off")
    plt.colorbar(im1, ax=axes[1, t], fraction=0.046, pad=0.04)

    # Row 2: Pixel [0, 255]                              FIX: was plotting im1 on axes[1] again
    im2 = axes[2, t].imshow(with_pixel_and_normalize[t], cmap="viridis", vmin=0, vmax=255)
    axes[2, t].set_title(f"t+{t+1} (Pixel [0-255])", fontsize=11)
    axes[2, t].axis("off")
    plt.colorbar(im2, ax=axes[2, t], fraction=0.046, pad=0.04)  # FIX: was im1 on axes[1]

# Row labels
axes[0, 0].set_ylabel("Raw mm/hr (Log Scale)", fontsize=12, labelpad=8)
axes[1, 0].set_ylabel("Normalized [0–1]", fontsize=12, labelpad=8)
axes[2, 0].set_ylabel("Pixel [0–255]", fontsize=12, labelpad=8)

plt.suptitle("Matching Visuals: Log-Scaled Raw vs. Normalized Pixels", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("visualization_matching.png", dpi=150, bbox_inches="tight")
plt.show()