from pathlib import Path
from datetime import datetime, timedelta
import shutil

def group_continuous_date_folders(root_dir):
    root = Path(root_dir)

    folders = []
    BASE_YEAR = 2024  # leap-safe reference year

    for p in root.iterdir():
        if p.is_dir():
            try:
                dt = datetime.strptime(f"{p.name} {BASE_YEAR}", "%d %b %Y")
                folders.append((dt, p))
            except ValueError:
                pass

    if not folders:
        return

    folders.sort(key=lambda x: x[0])

    groups = []
    current = [folders[0]]

    for (prev_dt, _), (curr_dt, curr_path) in zip(folders, folders[1:]):
        if curr_dt - prev_dt == timedelta(days=1):
            current.append((curr_dt, curr_path))
        else:
            groups.append(current)
            current = [(curr_dt, curr_path)]

    groups.append(current)

    # create group folders and move data
    for idx, group in enumerate(groups, start=1):
        group_dir = root / f"group_{idx:02d}"
        group_dir.mkdir(exist_ok=True)

        for _, folder_path in group:
            shutil.move(str(folder_path), group_dir / folder_path.name)
            # use shutil.copytree(...) instead if you want to copy

# usage
group_continuous_date_folders("../datasets/test_folder")
