import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap

# 1. Load and clean data
file_path = "../datasets/3RIMG_L2B_IMC/2023/25JUN/3RIMG_25JUN2023_1215_L2B_IMC_V01R00.h5"
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

# 3. Ensure north-up orientation (2D-safe)
if np.nanmean(lat[0, :]) > np.nanmean(lat[-1, :]):
    lat  = lat[::-1, :]
    lon  = lon[::-1, :]
    rain = rain[::-1, :]

# 4. Setup Basemap
fig, ax = plt.subplots(figsize=(15, 10))

m = Basemap(
    projection="cyl",
    llcrnrlon=90,
    urcrnrlon=98,
    llcrnrlat=22,
    urcrnrlat=30,
    resolution="l",
    # round=True,
    ax=ax
)

m.drawcoastlines(linewidth=1.2)
m.drawcountries(linewidth=0.8)
m.drawmapboundary(fill_color="lightcyan")
m.drawstates(linewidth=0.8)
m.fillcontinents(color="whitesmoke", lake_color="lightcyan")

# 5. Plot with pcolormesh (2D lat/lon)
mesh = m.pcolormesh(
    lon,
    lat,
    rain,
    latlon=True,
    shading="auto",
    cmap="viridis",
    norm=colors.LogNorm(
        # boundaries=[0.1, 1, 5, 10, 20, 50],
        # ncolors=256
        vmin=0.1, vmax=50
    )
)

# rgba = mesh.to_rgba(rain)
# np.save("precipitation_mesh_rgba.npy", rgba)

plt.colorbar(mesh, label="Rainfall Rate (mm/hr)", fraction=0.02, pad=0.04)
plt.title("INSAT-3DR IMC Rainfall (Basemap, No Cartopy)")

plt.show()



# import numpy as np
# import matplotlib.pyplot as plt


# rgba = np.load("precipitation_mesh_rgba.npy")

# plt.figure(figsize=(14, 6))
# plt.imshow(rgba, origin="upper")
# plt.axis("off")
# plt.title("Restored Precipitation Mesh")
# plt.show()
