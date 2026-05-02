from pathlib import Path
import shutil

def flatten_day_folders(parent_dir):
    parent = Path(parent_dir)

    for day_folder in parent.iterdir():
        if day_folder.is_dir():
            for item in day_folder.iterdir():
                shutil.move(str(item), parent / item.name)
            day_folder.rmdir()

flatten_day_folders('../datasets/3RIMG_L2B_IMC/2023')