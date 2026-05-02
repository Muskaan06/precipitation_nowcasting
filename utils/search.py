import os
import re

# Define the directory path
directory_path = "/mnt/sda1/Muskaan/nowcast/IMC_Combined/"

# Define the regex pattern
# This looks for files containing '_08_15_to_' and ending with '_20_15.npz'
regex_pattern = r".*_08_15_to_.*_20_15\.npz$"

def find_matching_files(directory, pattern):
    matched_files = []
    
    try:
        # List all files in the directory
        files = os.listdir(directory)
        
        # Compile the regex for better performance
        regex = re.compile(pattern)
        
        for filename in files:
            if regex.search(filename):
                matched_files.append(filename)
                
    except FileNotFoundError:
        return "Directory not found. Please check the path."
    except PermissionError:
        return "Permission denied to access the directory."
        
    return matched_files

# Execute the search
results = find_matching_files(directory_path, regex_pattern)

# Print results
if isinstance(results, list):
    if results:
        print(f"Found {len(results)} matching file(s):")
        for file in results:
            print(f" - {file}")
    else:
        print("No files matched the specific timing criteria.")
else:
    print(results)