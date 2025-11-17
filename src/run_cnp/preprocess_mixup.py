#!/usr/bin/env python3
"""
Preprocessing script to apply mixup data augmentation to HDF5 files.

This script should be run BEFORE training to avoid file locking issues
when using multiprocessing in the DataLoader.

Usage:
    python preprocess_mixup.py
    # or with nohup:
    nohup python preprocess_mixup.py > mixup_output.log 2>&1 &
"""

import yaml
import os
import sys
from tqdm import tqdm
import h5py
import numpy as np
import re

# Add the parent directory to the path to import resum
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from resum.utilities import utilities as utils

def parse_condition(condition_str, columns):
    """
    Parses condition strings like 'BBH Events==1' or 'some name>=value'
    and returns (column index, condition lambda).
    """
    # Supported operators, ordered by length to match longest first
    operators = ['==', '!=', '>=', '<=', '>', '<']

    # Try each operator and see if it's in the string
    for op in operators:
        if op in condition_str:
            parts = condition_str.split(op)
            if len(parts) != 2:
                raise ValueError(f"Invalid condition format: {condition_str}")
            column_name = parts[0].strip()
            value_str = parts[1].strip()
            break
    else:
        raise ValueError(f"No valid operator found in: {condition_str}")

    if column_name not in columns:
        raise ValueError(f"Column '{column_name}' not found in target!")

    column_idx = columns.index(column_name)

    # Try to convert value to number
    try:
        value = float(value_str) if '.' in value_str else int(value_str)
    except ValueError:
        value = f'"{value_str}"'  # Quote string for eval

    # Return column index and lambda condition
    return column_idx, lambda x: eval(f"x {op} {value}", {"x": x})


def mixup_augment_data(filename, use_beta, condition_strings, seed=42):
    """
    Augments an imbalanced dataset using the "mixup" method for HDF5 files.

    Each background event is combined with a randomly drawn signal event using a weighted sum.
    The ratio is drawn from either a uniform distribution or a beta distribution.

    Args:
        filename (str): Path to the HDF5 file.
        use_beta (list or None): Distribution from which the ratio is drawn.
            - `None`: Uniform distribution in [0,1].
            - `[z1, z2]`: Beta distribution B(z1, z2).
        condition_strings (list): List of condition strings to identify signal events.
        seed (int): Random seed for reproducibility.

    Returns:
        None: Updates the existing HDF5 file with new datasets.
    """
    np.random.seed(seed)  # Set the seed for reproducibility
    with h5py.File(filename, "a") as f:  # Open in append mode
        # Check if mixup datasets already exist
        if "phi_mixedup" in f and "target_mixedup" in f:
            if "signal_condition" in f:
                existing_conditions = [s.decode("utf-8") for s in f["signal_condition"][:]]
                if existing_conditions == condition_strings:
                    print(f"Skipping {os.path.basename(filename)} - mixup already applied with same conditions")
                    return

        phi = np.array(f["phi"])  # Feature data
        target = np.array(f["target"])  # Labels
        has_weights = "weights" in f  # Check if "weights" dataset exists
        weights = np.array(f["weights"]) if has_weights else None

        # Identify background (0) and signal (1) indices
        all_target_names = f["target_headers"][:]
        all_target_names = [label.decode("utf-8") for label in all_target_names]

        # Convert conditions to apply on NumPy target array
        conditions = np.ones(target.shape[0], dtype=bool)  # Start with all True

        for cond_str in condition_strings:
            col_idx, cond_func = parse_condition(cond_str, all_target_names)  # Get index and condition

            if np.ndim(target) > 1:
                conditions &= cond_func(target[:, col_idx])  # Apply condition to the correct dimension
            else:
                conditions &= cond_func(target[:])

        # Find matching indices
        signal_indices = np.where(conditions)[0]

        # All indices in the dataset
        all_indices = np.arange(target.shape[0])

        # Background indices are those NOT in signal_indices
        background_indices = np.setdiff1d(all_indices, signal_indices)

        if len(background_indices) == 0 or len(signal_indices) == 0:
            print(f"Skipping {os.path.basename(filename)} - no signal samples (signals: {len(signal_indices)}, background: {len(background_indices)})")
            # Create empty mixedup datasets so the file can still be loaded during training
            # Use empty arrays with correct shape
            empty_shape = (0,) + phi.shape[1:]
            if "phi_mixedup" in f:
                del f["phi_mixedup"]
            f.create_dataset("phi_mixedup", shape=empty_shape, dtype=phi.dtype, compression="gzip")

            empty_target_shape = (0,) + target.shape[1:] if target.ndim > 1 else (0,)
            if "target_mixedup" in f:
                del f["target_mixedup"]
            f.create_dataset("target_mixedup", shape=empty_target_shape, dtype=target.dtype, compression="gzip")

            if has_weights:
                empty_weights_shape = (0,) + weights.shape[1:] if weights.ndim > 1 else (0,)
                if "weights_mixedup" in f:
                    del f["weights_mixedup"]
                f.create_dataset("weights_mixedup", shape=empty_weights_shape, dtype=weights.dtype, compression="gzip")

            if "signal_condition" in f:
                del f["signal_condition"]
            f.create_dataset("signal_condition", data=np.array(condition_strings, dtype="S"))
            return  # Skip this file

        # Randomly pair each background sample with a signal sample
        sampled_signal_indices = np.random.choice(signal_indices, size=len(background_indices), replace=True)

        # Generate mixup ratios
        if use_beta and isinstance(use_beta, (list, tuple)) and len(use_beta) == 2:
            alpha = np.random.beta(use_beta[0], use_beta[1], size=(len(background_indices), 1))
        else:
            alpha = np.random.uniform(0, 1, size=(len(background_indices), 1))

        # Perform mixup augmentation
        phi_mixedup = alpha * phi[sampled_signal_indices] + (1 - alpha) * phi[background_indices]
        target_mixedup = alpha * target[sampled_signal_indices] + (1 - alpha) * target[background_indices]

        # Apply mixup to weights if they exist
        weights_mixedup = None
        if has_weights:
            weights_mixedup = alpha * weights[sampled_signal_indices] + (1 - alpha) * weights[background_indices]

        # Store new datasets in the same file
        if "phi_mixedup" in f:
            del f["phi_mixedup"]
        f.create_dataset("phi_mixedup", data=phi_mixedup, compression="gzip")

        if "target_mixedup" in f:
            del f["target_mixedup"]
        f.create_dataset("target_mixedup", data=target_mixedup, compression="gzip")

        if has_weights:
            if "weights_mixedup" in f:
                del f["weights_mixedup"]
            f.create_dataset("weights_mixedup", data=weights_mixedup, compression="gzip")

        if "signal_condition" in f:
            del f["signal_condition"]
        f.create_dataset("signal_condition", data=np.array(condition_strings, dtype="S"))


def main():
    """Main preprocessing function."""
    print("=" * 80)
    print("MIXUP DATA AUGMENTATION PREPROCESSING")
    print("=" * 80)

    # Load configuration
    config_path = "../xenon/settings.yaml"
    print(f"\nLoading configuration from: {config_path}")

    with open(config_path, "r") as f:
        config_file = yaml.safe_load(f)

    # Check if mixup is enabled
    use_data_augmentation = config_file["cnp_settings"]["use_data_augmentation"]

    if use_data_augmentation != "mixup":
        print(f"\nWARNING: use_data_augmentation is set to '{use_data_augmentation}', not 'mixup'")
        print("This script is specifically for mixup augmentation.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            sys.exit(0)

    # Get settings
    path_to_files = config_file["path_settings"]["path_to_files_train"]
    use_beta = config_file["cnp_settings"]["use_beta"]
    signal_condition = config_file["simulation_settings"]["signal_condition"]

    print(f"\nSettings:")
    print(f"  Data path: {path_to_files}")
    print(f"  Beta distribution: {use_beta}")
    print(f"  Signal condition: {signal_condition}")

    # Convert CSV to HDF5 if needed
    if not any(f.endswith(".h5") for f in os.listdir(path_to_files)):
        print("\nNo HDF5 files found. Converting CSV to HDF5...")
        utils.convert_all_csv_to_hdf5(config_file)

    # Get all HDF5 files
    files = sorted([os.path.join(path_to_files, f) for f in os.listdir(path_to_files) if f.endswith(".h5")])

    print(f"\nFound {len(files)} HDF5 files")
    print("\nStarting mixup augmentation...")

    # Apply mixup to all files
    errors = []
    processed = 0
    for file in tqdm(files, desc="Applying mixup"):
        try:
            mixup_augment_data(file, use_beta, signal_condition)
            # If function completes without error, it was processed successfully
            processed += 1
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(file)}: {e}"
            errors.append(error_msg)
            tqdm.write(error_msg)

    skipped = len(files) - processed - len(errors)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    print(f"\nTotal files: {len(files)}")
    print(f"  Successfully processed: {processed}")
    if skipped > 0:
        print(f"  Skipped (no signal/background): {skipped}")
    if errors:
        print(f"  Errors: {len(errors)}")

    if errors:
        print(f"\nEncountered {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\nYou can now run training with multiprocessing enabled:")
        print("  number_of_walkers: 1 (or higher)")
        print("\nRun training with:")
        print("  python cnp_training.py")
        print("  # or with nohup:")
        print("  nohup python cnp_training.py > training_output.log 2>&1 &")


if __name__ == "__main__":
    main()
