import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Load data

data = np.load(
    "/mnt/sda1/Muskaan/nowcast/precipitation_nowcasting/dataset/small_data/northeastern/selected_sets_2021_7_Sisters_IMC_2021-05-31_22_45_to_2021-06-01_10_15.npz",
    allow_pickle=True
)

# array shape: (24, 3, 112, 112)
arr = data["array"]
print(arr.shape)

T = arr.shape[0]
cols = 6
rows = (T + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = axes.flatten()

norm = colors.SymLogNorm(linthresh=10, linscale=1, vmin=0, vmax=100)

for t in range(T):
    rain = arr[t, 0]          # (112, 112)
    lat  = arr[t, 1][:, 0]    # (112,)
    lon  = arr[t, 2][0, :]    # (112,)

    im = axes[t].pcolormesh(
        lon,
        lat,
        rain,
        shading="auto",
        cmap="viridis",
        norm=norm
    )
    axes[t].set_title(f"t={t}", fontsize=9)
    axes[t].set_aspect("equal", adjustable="box")
    axes[t].set_xticks([])
    axes[t].set_yticks([])

for i in range(t + 1, len(axes)):
    axes[i].axis("off")

plt.suptitle("Precipitation Time Series", fontsize=13)
plt.tight_layout(rect=[0, 0, 0.93, 1])  # leave right margin for colorbar

cbar_ax = fig.add_axes([0.95, 0.1, 0.015, 0.8])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Precipitation (mm/h)", fontsize=10)

plt.savefig("precipitation_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

