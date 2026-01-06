import shutil
import re
from pathlib import Path
import numpy as np
import os

# Configuration
BASE_DIR = Path('/home/tidmad/bliu/resum-xenon')
SOURCE_HF = BASE_DIR / 'temp_new_data/hf'
SOURCE_LF = BASE_DIR / 'temp_new_data/lf'

DEST_BASE = BASE_DIR / 'src/xenon/in/data/new_both'
DEST_TRAIN_HF = DEST_BASE / 'training/hf'
DEST_VAL_HF = DEST_BASE / 'validation/hf'
DEST_TRAIN_LF = DEST_BASE / 'training/lf'
DEST_VAL_LF = DEST_BASE / 'validation/lf'

# Regex to extract coordinates for sorting
# Matches X...Y... pattern in filenames like sim_X10_Y80_ALL.csv or sim_X0_Y81_task0.csv
PAT_COORDS = re.compile(r'X(\d+)_?Y(\d+)')

def get_sorted_files(directory):
    """Returns list of (path, x, y) sorted by X, then Y."""
    files = []
    for p in directory.glob('*.csv'):
        m = PAT_COORDS.search(p.name)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            files.append((p, x, y))
        else:
            print(f"Warning: Could not parse coordinates from {p.name}")
    
    # Sort by X, then Y
    files.sort(key=lambda item: (item[1], item[2]))
    return files

def move_files(file_list, dest_dir, dry_run=False):
    """Moves files to destination directory."""
    if not dest_dir.exists():
        print(f"Creating directory: {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
    for p, x, y in file_list:
        dest_path = dest_dir / p.name
        print(f"Moving {p.name} -> {dest_path}")
        if not dry_run:
            shutil.move(str(p), str(dest_path))

def main():
    print("--- Moving Data Files ---")
    
    # 1. Process HF
    print("\nProcessing HF Data...")
    hf_files = get_sorted_files(SOURCE_HF)
    if not hf_files:
        print("No HF files found!")
    else:
        print(f"Found {len(hf_files)} HF files.")
        
        # Select 6 evenly distributed files for training
        # np.linspace generates evenly spaced numbers over a specified interval
        indices = np.linspace(0, len(hf_files) - 1, 6, dtype=int)
        # Use a set for fast lookup
        train_indices = set(indices)
        
        train_files = []
        val_files = []
        
        for i, item in enumerate(hf_files):
            if i in train_indices:
                train_files.append(item)
            else:
                val_files.append(item)
        
        print(f"Selected {len(train_files)} files for Training HF")
        print(f"Selected {len(val_files)} files for Validation HF")
        
        move_files(train_files, DEST_TRAIN_HF)
        move_files(val_files, DEST_VAL_HF)

    # 2. Process LF
    print("\nProcessing LF Data...")
    lf_files = get_sorted_files(SOURCE_LF)
    if not lf_files:
        print("No LF files found!")
    else:
        print(f"Found {len(lf_files)} LF files.")
        
        # Select 10% for validation (every 10th file)
        val_files = lf_files[::10]
        # The rest are training
        # We can use set difference or just iterate and check if not in val
        val_paths = {f[0] for f in val_files}
        train_files = [f for f in lf_files if f[0] not in val_paths]
        
        print(f"Selected {len(val_files)} files for Validation LF")
        print(f"Selected {len(train_files)} files for Training LF")
        
        move_files(val_files, DEST_VAL_LF)
        move_files(train_files, DEST_TRAIN_LF)

    print("\nDone!")

if __name__ == "__main__":
    main()
