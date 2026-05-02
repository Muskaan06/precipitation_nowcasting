import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap


# 1. Load and clean data
file_path = "/home/muskaan06/Desktop/Research/nowcasting/datasets/3RIMG_L2B_IMC/2024/3RIMG_21JUL2024_1315_L2B_IMC_V01R00.h5"
with h5py.File(file_path, "r") as f:
    rain = np.squeeze(f["IMC"][:]).astype(np.float32)
    lat  = f["Latitude"][:].astype(np.float32) / 100.0
    lon  = f["Longitude"][:].astype(np.float32) / 100.0

# 2. Mask invalid coordinates and rain
valid = (
    (lat >= -90) & (lat <= 90) &
    (lon >= -180) & (lon <= 180) &
    (rain > 0.1)
)

rain = np.where(valid, rain, np.nan)

# 3. Ensure north-up orientation
if np.nanmean(lat[0, :]) > np.nanmean(lat[-1, :]):
    lat  = lat[::-1, :]
    lon  = lon[::-1, :]
    rain = rain[::-1, :]


# #for north eastern region
# lat_min, lat_max = 22, 30
# lon_min, lon_max = 90, 98

# #for eastern region (JOWB)    
# lat_min, lat_max = 18, 26
# lon_min, lon_max = 82, 90

# #for southern region   
lat_min, lat_max = 8, 14
lon_min, lon_max = 74, 80

# for himalayan region
# lat_min, lat_max = 26, 36
# lon_min, lon_max = 72.0, 82.0

# ~4 km resolution in degrees
res = 8.0 / 111.0   # ≈ 0.036°

# Create grid axes
lat_bins = np.arange(lat_min, lat_max + res, res)
lon_bins = np.arange(lon_min, lon_max + res, res)

# Force square grid
N = min(len(lat_bins), len(lon_bins))
lat_bins = lat_bins[:N]
lon_bins = lon_bins[:N]

# Initialize rainfall grid (mm/hr)
rain_grid = np.full((N, N), np.nan, dtype=np.float32)
count_grid = np.zeros((N, N), dtype=np.int32)

# Flatten original data
lat_f  = lat.flatten()
lon_f  = lon.flatten()
rain_f = rain.flatten()

valid = ~np.isnan(rain_f)

lat_f  = lat_f[valid]
lon_f  = lon_f[valid]
rain_f = rain_f[valid]

# Map to grid indices
i = ((lat_f - lat_min) / res).astype(int)
j = ((lon_f - lon_min) / res).astype(int)

mask = (i >= 0) & (i < N) & (j >= 0) & (j < N)
i, j, rain_f = i[mask], j[mask], rain_f[mask]

# Accumulate (average if multiple points fall in one cell)
for ii, jj, val in zip(i, j, rain_f):
    if np.isnan(rain_grid[ii, jj]):
        rain_grid[ii, jj] = val
    else:
        rain_grid[ii, jj] += val
    count_grid[ii, jj] += 1

rain_grid[count_grid > 0] /= count_grid[count_grid > 0]

print("Rainfall grid shape:", rain_grid.shape)
print("Grid resolution (deg):", res)

# ============================
# Save EVERYTHING in one file
# ============================

# np.savez(
#     "INSAT_IMC_20230625_1215_4km_grid.npz",
#     rain=rain_grid,     # (N, N) mm/hr
#     lat=lat_bins,       # (N,)
#     lon=lon_bins        # (N,)
# )

# print("Saved single file: INSAT_IMC_20230625_1215_4km_grid.npz")
# print("Rain grid shape:", rain_grid.shape)


## 4. Square FIGURE
fig, ax = plt.subplots(figsize=(8, 8))  # ✅ square canvas

# 5. Basemap
m = Basemap(
   projection="cyl",
   llcrnrlon=lon_min,
   urcrnrlon=lon_max,
   llcrnrlat=lat_min,
   urcrnrlat=lat_max,
   resolution="l",
   ax=ax
)

m.drawcoastlines(linewidth=1.2) 
m.drawcountries(linewidth=0.8) 
m.drawmapboundary(fill_color="lightcyan") 
m.drawstates(linewidth=0.8) 
m.fillcontinents(color="whitesmoke", lake_color="lightcyan")

# ✅ Force square map geometry
ax.set_aspect("equal", adjustable="box")

# 6. Plot rainfall
mesh = m.pcolormesh(
   lon_bins,
   lat_bins,
   rain_grid,
   latlon=True,
   shading="auto",
   cmap="viridis",
   norm=colors.LogNorm(vmin=0.1, vmax=50)
)

plt.colorbar(mesh, label="Rainfall Rate (mm/hr)", fraction=0.035, pad=0.02)
plt.title("INSAT-3DR IMC Rainfall (Square Map)")

plt.show()
# fig.savefig("./data_images/INSAT_IMC_20FEB2024_0915_square_map.png", dpi=300, bbox_inches="tight") 
