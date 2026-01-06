import shutil
from pathlib import Path
import re
import os

def extract_coordinates_hf(filename):
    """Extract coordinates from HF files: sim_X{x}_Y{y}_ALL.csv"""
    m = re.search(r'sim_X(\d+)_Y(\d+)_ALL\.csv', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return -1, -1

def extract_coordinates_lf(filename):
    """Extract coordinates from LF files: sim_X{x}_Y{y}_task0.csv"""
    m = re.search(r'sim_X(\d+)_Y(\d+)_task0\.csv', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return -1, -1

def split_hf_data():
    """
    Split HF data: 10 evenly distributed files to training, rest to validation
    """
    print("=" * 60)
    print("Processing HF data...")
    print("=" * 60)
    
    src_dir = Path('/home/tidmad/bliu/resum-xenon/temp_new_data/hf')
    train_dir = Path('/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/training/hf')
    val_dir = Path('/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/validation/hf')
    
    # Ensure directories exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all HF files sorted by coordinates
    files = sorted(src_dir.glob('*.csv'), key=lambda p: extract_coordinates_hf(p.name))
    
    print(f"Found {len(files)} HF files in {src_dir}")
    
    # Select 10 evenly distributed files for training
    total_files = len(files)
    if total_files < 10:
        print(f"Warning: Only {total_files} files found, expected at least 10")
        train_count = total_files
    else:
        train_count = 10
    
    # Calculate indices for evenly distributed selection
    train_indices = set()
    if train_count > 0:
        step = total_files / train_count
        for i in range(train_count):
            idx = int(i * step)
            train_indices.add(idx)
    
    print(f"Selected {len(train_indices)} files for training (indices: {sorted(train_indices)})")
    print(f"Remaining {total_files - len(train_indices)} files will go to validation")
    
    # Move files
    train_files = []
    val_files = []
    
    for i, f in enumerate(files):
        if i in train_indices:
            train_files.append(f)
            dest = train_dir / f.name
            print(f"  Training: {f.name}")
            shutil.copy2(str(f), str(dest))
        else:
            val_files.append(f)
            dest = val_dir / f.name
            shutil.copy2(str(f), str(dest))
    
    print(f"\nHF Summary:")
    print(f"  Training: {len(train_files)} files -> {train_dir}")
    print(f"  Validation: {len(val_files)} files -> {val_dir}")

def split_lf_data():
    """
    Split LF data: 10% evenly distributed to validation, 90% to training
    """
    print("\n" + "=" * 60)
    print("Processing LF data...")
    print("=" * 60)
    
    src_dir = Path('/home/tidmad/bliu/resum-xenon/temp_new_data/lf')
    train_dir = Path('/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/training/lf')
    val_dir = Path('/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/validation/lf')
    
    # Ensure directories exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all LF files sorted by coordinates
    files = sorted(src_dir.glob('*.csv'), key=lambda p: extract_coordinates_lf(p.name))
    
    print(f"Found {len(files)} LF files in {src_dir}")
    
    # Calculate 10% for validation (evenly distributed)
    total_files = len(files)
    val_count = max(1, int(total_files * 0.1))  # At least 1 file
    
    # Calculate indices for evenly distributed selection
    val_indices = set()
    if val_count > 0:
        step = total_files / val_count
        for i in range(val_count):
            idx = int(i * step)
            val_indices.add(idx)
    
    print(f"Selected {len(val_indices)} files for validation (~10%)")
    print(f"Remaining {total_files - len(val_indices)} files will go to training (~90%)")
    
    # Move files
    train_files = []
    val_files = []
    
    for i, f in enumerate(files):
        if i in val_indices:
            val_files.append(f)
            dest = val_dir / f.name
            if i < 5 or i >= total_files - 2:  # Show first few and last few
                print(f"  Validation: {f.name}")
            elif i == 5:
                print(f"  ... ({len(val_indices) - 7} more validation files) ...")
            shutil.copy2(str(f), str(dest))
        else:
            train_files.append(f)
            dest = train_dir / f.name
            shutil.copy2(str(f), str(dest))
    
    print(f"\nLF Summary:")
    print(f"  Training: {len(train_files)} files ({len(train_files)/total_files*100:.1f}%) -> {train_dir}")
    print(f"  Validation: {len(val_files)} files ({len(val_files)/total_files*100:.1f}%) -> {val_dir}")

def main():
    print("\n" + "=" * 60)
    print("DATA SPLITTING SCRIPT")
    print("=" * 60)
    print("This script will:")
    print("  1. HF: Move 10 evenly distributed files to training")
    print("         Move remaining files to validation")
    print("  2. LF: Move 10% evenly distributed files to validation")
    print("         Move 90% files to training")
    print("=" * 60 + "\n")
    
    # Process HF data
    split_hf_data()
    
    # Process LF data
    split_lf_data()
    
    print("\n" + "=" * 60)
    print("DONE! All files have been split successfully.")
    print("=" * 60)

if __name__ == '__main__':
    main()
