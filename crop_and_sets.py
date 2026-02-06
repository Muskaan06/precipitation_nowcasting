import h5py
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import os
from datetime import datetime, timedelta
import time
from random_selection_code import compute_weighted_selection
# from random_selection_code import weighted_selection_without_replacement

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
    d1 = datetime.strptime(t1, fmt)
    d2 = datetime.strptime(t2, fmt)

    # handle midnight rollover
    if d2 < d1:
        d2 += timedelta(days=1)

    return d2 - d1 == timedelta(minutes=30)

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

def same_or_one_day_apart(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d1 - d2).days) <= 1

def saving_code(selected_indices):
    for idx in selected_indices:
        #file name: IMC_{start_date}_{start_time}_to_{end_date}_{end_time}.npz
        sdate = mulit_metadata[idx]["start_date"]
        stime = mulit_metadata[idx]["start_time"]
        edate = mulit_metadata[idx]["end_date"]
        etime = mulit_metadata[idx]["end_time"]
        f_name = f"IMC_{sdate}_{stime}_to_{edate}_{etime}.npz"
        print(f"Saving file: {f_name}")
        np.savez(
            f"../datasets/selected_sets/{f_name}",
            array=multiset[idx],
            metadata=mulit_metadata[idx]
        )
        
if __name__ == "__main__":

    dir_path = "../datasets/3RIMG_L2B_IMC/2023/cont_set_3" 

    # for day in os.listdir(year_dir_path)[:3]:       #day = [cont_set_1, cont_set_2, cont_set_3]
       
    count = 0
    # time_files = sorted(os.listdir(dir_path))   #[file1.h5,....]
    time_files = sorted(os.listdir(dir_path), key=lambda x: datetime.strptime(re.search(r'(\d{2}[A-Z]{3}\d{4}_\d{4})', x).group(1), "%d%b%Y_%H%M"))
    print(len(time_files))
    index = 0
    set_start_index = 0
    files_in_set = 0        #to keep count of number of file in set
    stack = []              #to keep track of previous file
    sets_arr = []           #current set keeping cropped files
    i=0                     #counts set no
    multiset = []           #store multiple sets together (default=10)
    mulit_metadata = [] #stores multiple metadatas

    while index < len(time_files):
        day = time_files[index]  
        file_path = dir_path + '/' + day      
        # print(f"{index} file name:{file_path} ")
        # print("current index: ", index)
        # print("no of file in current set: ", files_in_set)
        # print("start time: ",time_files[set_start_index])
        
        d,t=extract_datetime_from_path(day)[0], extract_datetime_from_path(day)[1]
        # print(d,"   ",t)

        if stack:
            prev_file = stack.pop()
            prev_day, prev_time = extract_datetime_from_path(prev_file)

            #check continuity
                #condition 1: 30 min apart and of same date or next day 
                #condition 2: 30 min apart but between last day of this month and next day of next month  
            if (is_30_min_apart(prev_time, t)) and (same_or_one_day_apart(d[0], prev_day[0])):        
                if files_in_set < 24:                  
                    print(f"{index} Adding file to current set: {file_path}")
                    cropped_file = cropping(file_path)
                    sets_arr.append(cropped_file)
                    files_in_set += 1

                else:       # create metadata, multiset and random selection
                    if len(sets_arr) == 24:
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
                        print("This is set no: ",i)
                        print(sets_arr.shape)
                        print(metadata)
                        mulit_metadata.append(metadata)
                        i += 1
                        #write saving code
                        if len(multiset) == 10:
                            multiset = np.stack(multiset, axis=0)
                            print(multiset.shape)
                            print(mulit_metadata)
                            temp_dict = {
                                "array": multiset,
                                "metadata": mulit_metadata}
                            # random selection
                            # selected_indices = weighted_selection_without_replacement(multiset, mulit_metadata, num_to_select=5)
                            selected_indices = compute_weighted_selection(temp_dict, num_to_select=10)
                            print(f"Selected Set Indices: {selected_indices}")
                            saving_code(selected_indices)
                            multiset = []
                            mulit_metadata = []
                            time.sleep(10)
                            
                            
                            # break

                    #setting new index
                    index = set_start_index + 1
                    set_start_index = index
                    files_in_set = 0
                    stack.clear()
                    sets_arr = []
                    print("\n\n\n")
                    continue
            
            else:   #if not 30 min apart or continuous dates
                #check if past set has 24 files, if yes create metadata and add to multiset
                if len(sets_arr) == 24:
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
                    print("This is set no: ",i)
                    print(sets_arr.shape)
                    print(metadata)
                    mulit_metadata.append(metadata)
                    i += 1
                    #setting new index
                    index = set_start_index + 1
                else:
                    set_start_index = index
                files_in_set = 0
                stack.clear()
                sets_arr = []
        
        else:
            print(f"Starting new set with file: {file_path}")
            cropped_file = cropping(file_path)
            sets_arr.append(cropped_file)
            files_in_set += 1
        stack.append(day)
        index += 1
    # a = is_30_min_apart('23:45','01:15') 
    # b = same_or_one_day_apart(  "2023-06-30", "2023-07-01")
    # print(a and b)
        
