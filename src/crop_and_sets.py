import h5py
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import os
from datetime import datetime, timedelta
import time
from random_selection_code import compute_weighted_selection

#function to extract date and time from file name
def extract_datetime_from_path(path):
    '''
    Extracts date and time from a file path based on the pattern DDMMMYYYY_HHMM.
    input: file path (str)
    output: date (str in YYYY-MM-DD format), time (str in HH:MM format)

    '''
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

#function to check if provide time (in str format) are 30 min apart or not
def is_30_min_apart(t1, t2):
    '''
    Checks if two time strings (in HH:MM format) are exactly 30 minutes apart, accounting for midnight rollover.
    input: t1, t2 (str in HH:MM format)
    output: True if 30 minutes apart, False otherwise
    '''
    fmt = "%H:%M"
    d1 = datetime.strptime(t1, fmt)
    d2 = datetime.strptime(t2, fmt)

    # handle midnight rollover
    if d2 < d1:
        d2 += timedelta(days=1)

    return d2 - d1 == timedelta(minutes=30)

#fucntion to crop the image and return cropped image along with lat and lon grid
def cropping(file_path):
    '''
    Crops the rainfall data from the given HDF5 file to a specific lat-lon region (currently 7 sisters) and returns the cropped rainfall grid along with corresponding latitude and longitude grids.
    input: file_path (str) - path to the HDF5 file containing rainfall data
    output: img_data (list) - [rain_grid, lat_bins, lon_bins]
    '''
    # try:
    with h5py.File(file_path, "r") as f:
        rain = np.squeeze(f["IMC"][:]).astype(np.float32)
        lat  = f["Latitude"][:].astype(np.float32) / 100.0
        lon  = f["Longitude"][:].astype(np.float32) / 100.0

    # except (OSError, EOFError) as e:
    #     print(f"File read error (possibly EOF or corrupted file): {e}")
    #     return None

    lat_min, lat_max = 26, 36
    lon_min, lon_max = 72, 82

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

#function to check if two dates are same or one day apart
def same_or_one_day_apart(d1, d2):
    '''
    Checks if two date strings (in YYYY-MM-DD format) are the same or one day apart.
    input: d1, d2 (str in YYYY-MM-DD format)
    output: True if same or one day apart, False otherwise
    '''
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d1 - d2).days) <= 1

#function to save the selected sets from the multiset (0-9)
def saving_code(selected_indices, multiset, multiset_metadata):
    '''
    save the selected indices from the multiset to the specified directory with appropriate file naming convention.
    input: selected_indices (list of int) - indices of the selected sets from the multiset
    multiset (numpy array) - the multiset containing all the sets of data
    multiset_metadata (list of dict) - metadata corresponding to each set in the multiset
    output: None (saves the selected sets as .npz files in the specified directory)
    '''
    for idx in selected_indices:
        #file name: IMC_{start_date}_{start_time}_to_{end_date}_{end_time}.npz
        sdate = multiset_metadata[idx]["start_date"]
        stime = multiset_metadata[idx]["start_time"]
        edate = multiset_metadata[idx]["end_date"]
        etime = multiset_metadata[idx]["end_time"]
        f_name = f"IMC_{sdate}_{stime}_to_{edate}_{etime}.npz"
        print(f"Saving file: {f_name}", "\n")
        np.savez(
            f"../datasets/selected_sets_himalayan/{f_name}",
            array=multiset[idx],
            metadata=multiset_metadata[idx]
        )
        
if __name__ == "__main__":

    dir_path = "../datasets/3RIMG_L2B_IMC/2024" 
    time_files = sorted(os.listdir(dir_path), key=lambda x: datetime.strptime(re.search(r'(\d{2}[A-Z]{3}\d{4}_\d{4})', x).group(1), "%d%b%Y_%H%M")) #list of file names sorted by date and time
    print(len(time_files))

    index = 513          #index of file from which you want to start (default=0)
    set_start_index = 0
    files_in_set = 0        #to keep count of number of file in set
    sets_arr = []           #current set keeping cropped files
    i=0                     #counts multiset no
    multiset = []           #store multiple sets together (default=10)
    multiset_metadata = []  #stores multiple metadatas

    while index < len(time_files):
        #current file
        day = time_files[index]  
        file_path = dir_path + '/' + day

        if not sets_arr:   #if starting new set
            print(f"Starting new set with file: {file_path}")
            cropped_file = cropping(file_path)
            if cropped_file is not None:
                sets_arr.append(cropped_file)
                set_start_index = index
                files_in_set = 1
            index += 1

        else:
            if len(sets_arr) >= 24:       #if current set already has 24 files, create metadata and add to multiset
                multiset.append(sets_arr)
                sets_arr = np.stack(sets_arr, axis=0)
                print("Current set has 24 files, adding to multiset!")
                print(f"len of multiset = {len(multiset)}")
                begin_day, begin_time = extract_datetime_from_path(time_files[set_start_index])[0], extract_datetime_from_path(time_files[set_start_index])[1]
                end_day, end_time = extract_datetime_from_path(time_files[index-1])[0], extract_datetime_from_path(time_files[index-1])[1]
                metadata = {
                    "start_date":begin_day[0],
                    "end_date":end_day[0],
                    "start_time": begin_time,
                    "end_time": end_time
                }
                print(sets_arr.shape)
                print(metadata,'\n')
                multiset_metadata.append(metadata)
                #re-initialization for new set
                sets_arr = []
                files_in_set = 0
                index = set_start_index + 1
                set_start_index = index 

            else: #if current set has less than 24 files, check if current file is continuous with previous file, if yes add to current set, if no create new set
                prev_day, prev_time = extract_datetime_from_path(time_files[index-1])[0], extract_datetime_from_path(time_files[index-1])[1]
                curr_day, curr_time = extract_datetime_from_path(day)[0], extract_datetime_from_path(day)[1]

                if (is_30_min_apart(prev_time, curr_time)) and (same_or_one_day_apart(curr_day[0], prev_day[0])):
                    print(f"Adding file to current set: {file_path}")
                    cropped_file = cropping(file_path)
                    if cropped_file is not None:
                        sets_arr.append(cropped_file)
                        files_in_set += 1
                    else:
                        print(f"Cropping failed for file: {file_path}, skipping this file.")

                else:
                    print(f"File {file_path} is not continuous with previous file, starting new set!")
                    sets_arr = []
                    sets_arr.append(cropping(file_path))
                    set_start_index = index
                    files_in_set = 1
                index += 1

            #write saving code
            if len(multiset) == 10:
                print(f"Multiset {i} ready for random selection!")
                i += 1
                multiset = np.stack(multiset, axis=0)
                print(multiset.shape)
                print(multiset_metadata)
                temp_dict = {
                    "array": multiset,
                    "metadata": multiset_metadata}
                # random selection
                selected_indices = compute_weighted_selection(temp_dict, num_to_select=10)
                print(f"Selected Set Indices: {selected_indices}")
                saving_code(selected_indices, multiset, multiset_metadata)
                multiset = []
                multiset_metadata = []
                time.sleep(10)

    #if there are remaining sets in the multiset after the loop, perform random selection
    if multiset:
        print(f"Multiset {i} ready for random selection!")
        multiset = np.stack(multiset, axis=0)
        temp_dict = {
                    "array": multiset,
                    "metadata": multiset_metadata}
        selected_indices = compute_weighted_selection(temp_dict, num_to_select=10)
        print(f"Selected Set Indices: {selected_indices}")
        saving_code(selected_indices, multiset, multiset_metadata)
