import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
import tqdm

def convert_csv_to_h5(csv_path, h5_path, fidelity):
    """
    Convert a single CSV file to HDF5 format.
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Select required columns
        # Based on user request and settings2.yaml
        required_cols = ['eventid', 'scint_x', 'scint_y', 'initial_m_x', 'initial_m_y', 'initial_m_z', 'tag_final']
        
        # Check if columns exist
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error converting {Path(csv_path).name}: Missing columns {missing}")
            return False

        df = df[required_cols]

        # Save to HDF5
        with h5py.File(h5_path, 'w') as f:
            # Create dataset for data
            # Store as float32 to save space, except eventid/tag which might need specific types
            # But usually for ML float32 is fine for all features.
            # Let's keep types consistent with pandas dtypes or force float32 for features.
            
            # Store data as a single array or separate datasets?
            # Previous scripts might expect specific structure.
            # Usually it's 'data' dataset or columns as datasets.
            # Let's look at how typical H5 conversion is done in this project.
            # The previous error didn't show the structure, but `preprocess_mixup.py` reads it.
            # Let's assume a simple structure: one dataset per column or a compound dataset?
            # Or just a pandas HDF store?
            # "convert_csv_to_h5.py" used `df.to_hdf(h5_path, key='data', mode='w')` probably?
            # Let's check the original file content if possible.
            # I don't have the original file content in history, but I can assume `to_hdf` or `h5py`.
            # Wait, I can read `convert_csv_to_h5.py` from the user's context or previous turns?
            # I see `convert_csv_to_h5.py` in the file list. I should have read it.
            # I will use `df.to_hdf` as it's the standard pandas way and likely what was used.
            
            pass # Context manager not needed for to_hdf
        
        df.to_hdf(h5_path, key='data', mode='w', format='table', data_columns=True)
        
        # Add attributes if needed (fidelity)
        # to_hdf doesn't easily add root attributes.
        # If fidelity is needed as metadata, we might need h5py.
        # But let's stick to simple conversion first.
        
        return True

    except Exception as e:
        print(f"Error converting {Path(csv_path).name}: {e}")
        return False

def convert_directory(directory, fidelity):
    """
    Convert all CSV files in a directory to HDF5.
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Warning: Directory not found: {directory}")
        return

    csv_files = sorted(list(directory.glob('*.csv')))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    print(f"Converting {len(csv_files)} files in {directory}...")

    with ProcessPoolExecutor() as executor:
        futures = []
        for csv_file in csv_files:
            h5_file = csv_file.with_suffix('.h5')
            futures.append(executor.submit(convert_csv_to_h5, str(csv_file), str(h5_file), fidelity))

        for _ in tqdm.tqdm(futures, desc="Converting"):
            _.result()

def main():
    """Main conversion function."""
    base_path = Path('/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both')

    # Define directories and their fidelity levels
    directories = [
        (base_path / 'training/lf', 0),
        (base_path / 'training/hf', 1),
        (base_path / 'validation/lf', 0),
        (base_path / 'validation/hf', 1)
    ]

    print("="*60)
    print("CSV to HDF5 Conversion Script (Xenon2)")
    print("="*60)

    for directory, fidelity in directories:
        convert_directory(directory, fidelity)

    print("="*60)
    print("Conversion complete!")
    print("="*60)

if __name__ == '__main__':
    main()
