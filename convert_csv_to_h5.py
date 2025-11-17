#!/usr/bin/env python3
"""
Standalone script to convert CSV files to HDF5 format.
Matches the structure of the reference HDF5 file.
"""

import os
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_csv_to_h5(csv_path, h5_path, fidelity_value):
    """
    Convert a single CSV file to HDF5 format.

    Args:
        csv_path: Path to input CSV file
        h5_path: Path to output HDF5 file
        fidelity_value: 0 for low fidelity, 1 for high fidelity
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Define column mappings based on the settings.yaml structure
    theta_cols = ['scint_x', 'scint_y']
    phi_cols = ['x_init_mm', 'y_init_mm', 'z_init_mm', 'x_final_mm', 'y_final_mm', 'z_final_mm', 'energy_keV', 'time_ns']
    target_cols = ['tag_final']

    # Extract theta values (should be constant for all rows in a file)
    theta = df[theta_cols].iloc[0].values.astype(np.float64)

    # Extract phi data (event-specific parameters)
    phi = df[phi_cols].values.astype(np.float64)

    # Extract target data
    target = df[target_cols].values.astype(np.int64)

    # Create weights (default to 1 if not in CSV)
    if 'weights' in df.columns:
        weights = df[['weights']].values.astype(np.int64)
    else:
        weights = np.ones((len(df), 1), dtype=np.int64)

    # Create fidelity array
    fidelity = np.full((len(df), 1), fidelity_value, dtype=np.float64)

    # Write to HDF5
    with h5py.File(h5_path, 'w') as hf:
        # Theta (design parameters) - 1D array
        hf.create_dataset('theta', data=theta, compression='gzip')
        hf.create_dataset('theta_headers',
                         data=np.array(theta_cols, dtype='S'),
                         compression='gzip')

        # Phi (event-specific parameters) - 2D array
        hf.create_dataset('phi', data=phi, compression='gzip')
        hf.create_dataset('phi_labels',
                         data=np.array(phi_cols, dtype='S'),
                         compression='gzip')

        # Target (outcomes) - 2D array
        hf.create_dataset('target', data=target, compression='gzip')
        hf.create_dataset('target_headers',
                         data=np.array(target_cols, dtype='S'),
                         compression='gzip')

        # Weights - 2D array
        hf.create_dataset('weights', data=weights, compression='gzip')
        hf.create_dataset('weights_labels',
                         data=np.array(['weights'], dtype='S'),
                         compression='gzip')

        # Fidelity - 2D array
        hf.create_dataset('fidelity', data=fidelity, compression='gzip')

def convert_directory(input_dir, fidelity_value):
    """
    Convert all CSV files in a directory to HDF5.

    Args:
        input_dir: Directory containing CSV files
        fidelity_value: 0 for low fidelity, 1 for high fidelity
    """
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"\nConverting {len(csv_files)} CSV files in {input_dir}")
    print(f"Fidelity level: {fidelity_value} ({'LF' if fidelity_value == 0 else 'HF'})")

    for csv_file in tqdm(csv_files, desc="Converting"):
        h5_file = csv_file.with_suffix('.h5')
        try:
            convert_csv_to_h5(str(csv_file), str(h5_file), fidelity_value)
        except Exception as e:
            print(f"\nError converting {csv_file.name}: {e}")

def main():
    """Main conversion function."""
    base_path = Path('/home/tidmad/bliu/resum-xenon/src/xenon/in/data/only2')

    # Define directories and their fidelity levels
    directories = [
        (base_path / 'training' / 'lf', 0),
        (base_path / 'training' / 'hf', 1),
        (base_path / 'validation' / 'lf', 0),
    ]

    print("=" * 60)
    print("CSV to HDF5 Conversion Script")
    print("=" * 60)

    for directory, fidelity in directories:
        if directory.exists():
            convert_directory(str(directory), fidelity)
        else:
            print(f"\nWarning: Directory not found: {directory}")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
