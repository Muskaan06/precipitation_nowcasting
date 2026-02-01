import h5py
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime, timedelta

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

def is_30_min_apart(t1, t2):
    fmt = "%H:%M"
    return datetime.strptime(t2, fmt) - datetime.strptime(t1, fmt) == timedelta(minutes=30)

def cropping(file_path):
    with h5py.File(file_path, "r") as f:
        rain = np.squeeze(f["IMC"][:]).astype(np.float32)
        lat  = f["Latitude"][:].astype(np.float32) / 100.0
        lon  = f["Longitude"][:].astype(np.float32) / 100.0

    lat_min, lat_max = 22.0, 30.0
    lon_min, lon_max = 90.0, 98.0

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
        
    res = 8.0 / 111.0 

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
    # make lat/lon grids same shape as rain_grid
    lon_bins, lat_bins = np.meshgrid(lon_bins, lat_bins)
    # print(rain_grid.shape)
    # print(lat_bins.shape)

    img_data = [rain_grid, lat_bins, lon_bins]
    return img_data

if __name__ == "__main__":

    dir_path = "../datasets/3RIMG_L2B_IMC/2023/cont_set_1" 

    # for day in os.listdir(year_dir_path)[:3]:       #day = [cont_set_1, cont_set_2, cont_set_3]
       
    count = 0
    time_files = sorted(os.listdir(dir_path))   #[file1.h5,....]

    index = 0
    set_start_index = 0
    files_in_set = 0        #to keep count of number of file in set
    stack = []              #to keep track of previous file
    sets_arr = []           #current set keeping cropped files
    i=0
    multiset = []
    mulit_metadata = dict()

    while index < len(time_files):
        day = time_files[index]  
        file_path = dir_path + '/' + day        
        print("file name: ",file_path)
        print("current index: ", index)
        print("no of file in current set: ", files_in_set)
        print("start time: ",time_files[set_start_index])
        
        d,t=extract_datetime_from_path(day)[0], extract_datetime_from_path(day)[1]
        print(d,"   ",t)

        if stack:
            prev_file = stack.pop()
            prev_day, prev_time = extract_datetime_from_path(prev_file)
            if is_30_min_apart(prev_time, t):        #to check if current file is continuous of prvious file
                if files_in_set < 24:                   #to check if number of files in current set is < 20: if yes then continue appending else save the set
                    
                    cropped_file = cropping(file_path)
                    sets_arr.append(cropped_file)
                    files_in_set += 1

                else:

                    multiset.append(sets_arr)
                    sets_arr = np.stack(sets_arr, axis=0)
                    print(prev_time)
                    begin_day, begin_time = extract_datetime_from_path(time_files[set_start_index])[0], extract_datetime_from_path(time_files[set_start_index])[1]
                    metadata = {
                        "start_date":begin_day[0],
                        "end_date":prev_day[0],
                        "start_time": begin_time,
                        "end_time": prev_time
                    }
                    print(sets_arr.shape)
                    print(metadata)
                    mulit_metadata[f"set{i}"] = metadata
                    i += 1
                    #write saving code
                    if len(multiset) == 10:
                        multiset = np.stack(multiset, axis=0)
                        np.savez(
                            f"set{i-10}-{i}.npz",
                            array=multiset,
                            metadata=mulit_metadata
                        )
                        break

                    #setting new index
                    index = set_start_index + 1
                    set_start_index = index
                    files_in_set = 0
                    stack.clear()
                    sets_arr = []
                    print("\n\n\n")
                    continue
            
            else:   #if not 30 min apart
                set_start_index = index
                files_in_set = 0
                stack.clear()
                sets_arr = []
        
      
        stack.append(day)
        index += 1

        

    # metadata = {
    #     "date":d,
    #     "start_time": "00:15",
    #     "end_time": "12:15"
    # }

    # np.savez(
    #     "set1_30JUN2023.npz",
    #     array=sets_arr[:24],
    #     metadata=metadata
    # )

















    #  if t == "12:15" :
    #             plt.figure(figsize=(8, 8))
    #             plt.pcolormesh(
    #                 lon_bins,
    #                 lat_bins,
    #                 rain_grid,
    #                 shading="auto",
    #                 cmap="viridis",
    #                 norm=LogNorm(vmin=0.1, vmax=50)
    #             )

    #             plt.colorbar(label="Rainfall Rate (mm/hr)")
    #             plt.xlabel("Longitude")
    #             plt.ylabel("Latitude")
    #             plt.title("INSAT-3DR IMC Rainfall (4 km Grid)")
    #             plt.gca().set_aspect("equal", adjustable="box")
    #             plt.show()