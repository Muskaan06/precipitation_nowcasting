import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Load data
data = np.load("INSAT_IMC_20230625_1215_4km_grid.npz")
rain = data["rain"]   # (N, N)
lat  = data["lat"]    # (N,)
lon  = data["lon"]    # (N,)

# Create 2D lat/lon grids
lon2d, lat2d = np.meshgrid(lon, lat)

# Plot
plt.figure(figsize=(8, 8))
plt.pcolormesh(
    lon2d,
    lat2d,
    rain,
    shading="auto",
    cmap="viridis",
    norm=colors.LogNorm(vmin=0.1, vmax=50)
)

plt.colorbar(label="Rainfall Rate (mm/hr)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("INSAT-3DR IMC Rainfall (4 km Grid)")
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
