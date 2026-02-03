import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Load data
data = np.load("data/set20-30.npz", allow_pickle=True)

print(data["array"].shape)
print(data["metadata"])

set0 = data["array"][6]
print("\nset shape: ", set0.shape)

#each file inside a set
file1 = set0[0]
# print(file1)
print(file1[0].shape)


rain = file1[0]   # (N, N)
lat  = file1[1][:, 0]    # (N,)
lon  = file1[2][0, :]    # (N,)
print(rain)
# ---- SHOW SHAPES ----
print("NPZ file contents and shapes:")
print(f"rain shape: {rain.shape}")
print(f"lat  shape: {lat.shape}")
print(f"lon  shape: {lon.shape}")
print("-" * 40)

# Create 2D lat/lon grids
lon2d, lat2d = np.meshgrid(lon, lat)

print(f"lon2d shape: {lon2d.shape}")
print(f"lat2d shape: {lat2d.shape}")

# Plot
plt.figure(figsize=(8, 8))
plt.pcolormesh(
    lon,
    lat,
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
