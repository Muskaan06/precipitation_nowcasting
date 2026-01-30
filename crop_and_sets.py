import h5py
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap

def extract_datetime_from_path(path):
    filename = Path(path).name

    # match: DDMMMYYYY_HHMM (e.g., 25JUN2023_1215)
    match = re.search(r'(\d{2}[A-Z]{3}\d{4})_(\d{4})', filename)

    if not match:
        raise ValueError("No valid date-time pattern found")

    date_str, time_str = match.groups()

    # parse to datetime
    dt = datetime.strptime(date_str + time_str, "%d%b%Y%H%M")

    d = dt.date().isoformat(),
    t = dt.time().strftime("%H:%M")

    return d,t


if __name__ == "__main__":

    year_dir_path = "../datasets/3RIMG_L2B_IMC/2023/" 

    sets_arr = []

    # for day in os.listdir(year_dir_path):
       
    day_path = year_dir_path + '25JUN'
    count = 1

    for time_file in sorted(os.listdir(day_path)):   #lists 48 files inside a day

        file_path = day_path + '/' + time_file
        d,t=extract_datetime_from_path(file_path)[0], extract_datetime_from_path(file_path)[1]
        print(d,"   ",t)
          
        with h5py.File(file_path, "r") as f:
            rain = np.squeeze(f["IMC"][:]).astype(np.float32)
            lat  = f["Latitude"][:].astype(np.float32) / 100.0
            lon  = f["Longitude"][:].astype(np.float32) / 100.0

        lat_min, lat_max = 22, 30.1
        lon_min, lon_max = 90.0, 98.0

        mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )

        # rows, cols = np.where(mask)
        # r0, r1 = rows.min(), rows.max() + 1
        # c0, c1 = cols.min(), cols.max() + 1

        # rain_crop = rain[r0:r1, c0:c1]
        # lat_crop  = lat[r0:r1, c0:c1]
        # lon_crop  = lon[r0:r1, c0:c1]

        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1

        h = r1 - r0
        w = c1 - c0
        side = min(h, w)

        r_center = (r0 + r1) // 2
        c_center = (c0 + c1) // 2

        r0_sq = r_center - side // 2
        r1_sq = r0_sq + side
        c0_sq = c_center - side // 2
        c1_sq = c0_sq + side

        rain_crop = rain[r0_sq:r1_sq, c0_sq:c1_sq]
        lat_crop  = lat[r0_sq:r1_sq, c0_sq:c1_sq]
        lon_crop  = lon[r0_sq:r1_sq, c0_sq:c1_sq]

        vmin = np.nanpercentile(rain_crop, 5)
        vmax = np.nanpercentile(rain_crop, 99)

        if t == "12:15" :
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect("equal", adjustable="box")
            plt.imshow(
                rain,
                origin="lower",
                cmap="viridis",
                norm=LogNorm(vmin=0.1, vmax=60),
                extent=[
                    lon_crop.min(), lon_crop.max(),
                    lat_crop.min(), lat_crop.max()
                ],
                aspect="auto"
            )
            plt.colorbar(label="Rainfall (mm/hr)")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Cropped Square Precipitation (Lat–Lon)")
            plt.show()
        
        count += 1
        img_data = [rain_crop, lat_crop, lon_crop]
        img_data = np.stack(img_data, axis=0)
        # print(img_data.shape)
        
        sets_arr.append(img_data)

    sets_arr = np.stack(sets_arr, axis=0)
    print(sets_arr.shape)

metadata = {
    "date":d,
    "start_time": "00:15",
    "end_time": "12:15"
}

np.savez(
    "set1_30JUN2023.npz",
    array=sets_arr[:25],
    metadata=metadata
)