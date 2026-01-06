#!/usr/bin/env python3
"""
Preprocessing script to apply mixup data augmentation to HDF5 files (Xenon2 format).

This script should be run BEFORE training to avoid file locking issues
when using multiprocessing in the DataLoader.

Usage:
    python preprocess_mixup_xenon2.py
    # or with nohup:
    nohup python preprocess_mixup_xenon2.py > mixup_output.log 2>&1 &
"""

import yaml
import os
import sys
from tqdm import tqdm
import h5py
import numpy as np

# Add the parent directory to the path to import resum
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from resum.utilities import utilities as utils

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
    
    try:
        with h5py.File(filename, "a") as f:  # Open in append mode
            # Read data from the new structure (compound dataset 'data/table')
            if "data/table" not in f:
                print(f"Skipping {os.path.basename(filename)} - 'data/table' not found")
                return
                
            data = f["data/table"]
            
            # --- COMPATIBILITY LAYER: Create legacy datasets if missing ---
            
            # 1. PHI (Features)
            phi_cols = ["initial_m_x", "initial_m_y", "initial_m_z"]
            if "phi" not in f:
                # Check if columns exist
                for col in phi_cols:
                    if col not in data.dtype.names:
                        print(f"Skipping {os.path.basename(filename)} - Column '{col}' not found in data")
                        return
                # Stack columns to create phi array (N, 3)
                phi = np.column_stack([data[col] for col in phi_cols])
                f.create_dataset("phi", data=phi, compression="gzip")
            else:
                phi = np.array(f["phi"])

            if "phi_labels" not in f:
                f.create_dataset("phi_labels", data=np.array(phi_cols, dtype="S"), compression="gzip")

            # 2. THETA (Design Parameters)
            theta_cols = ["scint_x", "scint_y"]
            if "theta" not in f:
                # Check if columns exist
                for col in theta_cols:
                    if col not in data.dtype.names:
                        print(f"Skipping {os.path.basename(filename)} - Column '{col}' not found in data")
                        return
                # Stack columns to create theta array (N, 2)
                theta = np.column_stack([data[col] for col in theta_cols])
                f.create_dataset("theta", data=theta, compression="gzip")
            
            if "theta_headers" not in f:
                f.create_dataset("theta_headers", data=np.array(theta_cols, dtype="S"), compression="gzip")

            # 3. TARGET (Labels)
            target_col = "tag_final"
            if "target" not in f:
                if target_col not in data.dtype.names:
                    print(f"Skipping {os.path.basename(filename)} - Column '{target_col}' not found in data")
                    return
                target = data[target_col]
                f.create_dataset("target", data=target, compression="gzip")
            else:
                target = np.array(f["target"])

            if "target_headers" not in f:
                f.create_dataset("target_headers", data=np.array([target_col], dtype="S"), compression="gzip")

            # 4. WEIGHTS
            has_weights = "weights" in data.dtype.names
            weights = None
            if has_weights:
                if "weights" not in f:
                    weights = data["weights"]
                    f.create_dataset("weights", data=weights, compression="gzip")
                else:
                    weights = np.array(f["weights"])
                
                if "weights_labels" not in f:
                    f.create_dataset("weights_labels", data=np.array(["weights"], dtype="S"), compression="gzip")
            
            # --- END COMPATIBILITY LAYER ---

            # Check if mixup datasets already exist
            if "phi_mixedup" in f and "target_mixedup" in f:
                if "signal_condition" in f:
                    existing_conditions = [s.decode("utf-8") for s in f["signal_condition"][:]]
                    if existing_conditions == condition_strings:
                        # print(f"Skipping {os.path.basename(filename)} - mixup already applied with same conditions")
                        return

            # Identify background (0) and signal (1) indices
            # For the new format, we can check the columns directly.
            # The condition strings refer to column names.
            
            # Convert conditions to apply on the data
            conditions = np.ones(len(data), dtype=bool)  # Start with all True

            for cond_str in condition_strings:
                # Parse condition to get column name and value
                # We need to handle this slightly differently than the old script
                # because we don't have a separate target array with headers.
                # We have a compound dataset where we can access columns by name.
                
                # Simple parsing for now, assuming simple conditions like "tag_final==1"
                operators = ['==', '!=', '>=', '<=', '>', '<']
                for op in operators:
                    if op in cond_str:
                        parts = cond_str.split(op)
                        col_name = parts[0].strip()
                        val_str = parts[1].strip()
                        
                        if col_name not in data.dtype.names:
                             raise ValueError(f"Column '{col_name}' not found in data table!")
                        
                        # Get column data
                        col_data = data[col_name]
                        
                        # Evaluate condition
                        # Use eval carefully or just implement simple logic
                        try:
                            val = float(val_str) if '.' in val_str else int(val_str)
                        except ValueError:
                            val = f'"{val_str}"'
                            
                        # Apply condition
                        # We can use numpy boolean indexing
                        # Construct the expression
                        expr = f"col_data {op} {val}"
                        conditions &= eval(expr, {"col_data": col_data})
                        break
                else:
                     raise ValueError(f"No valid operator found in: {cond_str}")

            # Find matching indices
            signal_indices = np.where(conditions)[0]

            # All indices in the dataset
            all_indices = np.arange(len(data))

            # Background indices are those NOT in signal_indices
            background_indices = np.setdiff1d(all_indices, signal_indices)

            if len(background_indices) == 0 or len(signal_indices) == 0:
                print(f"Skipping {os.path.basename(filename)} - no signal samples (signals: {len(signal_indices)}, background: {len(background_indices)})")
                # Create empty mixedup datasets
                empty_shape = (0,) + phi.shape[1:]
                if "phi_mixedup" in f:
                    del f["phi_mixedup"]
                f.create_dataset("phi_mixedup", shape=empty_shape, dtype=phi.dtype, compression="gzip")

                # Target shape
                target_shape = (0,) + (target.shape[1:] if target.ndim > 1 else ())
                if "target_mixedup" in f:
                    del f["target_mixedup"]
                f.create_dataset("target_mixedup", shape=target_shape, dtype=target.dtype, compression="gzip")

                if has_weights:
                    empty_weights_shape = (0,) + (weights.shape[1:] if weights.ndim > 1 else ())
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
            # phi is (N, 3), alpha is (N_bg, 1)
            phi_mixedup = alpha * phi[sampled_signal_indices] + (1 - alpha) * phi[background_indices]
            
            # target might be 1D or 2D
            target_signal = target[sampled_signal_indices]
            target_bg = target[background_indices]
            
            if target.ndim == 1:
                # Reshape for broadcasting if needed, or alpha handles it if (N,1) * (N,) -> (N,N) which is wrong
                # alpha is (N, 1). target is (N,).
                # We need target to be (N, 1) or alpha to be (N,)
                target_signal = target_signal.reshape(-1, 1)
                target_bg = target_bg.reshape(-1, 1)
                
            target_mixedup = alpha * target_signal + (1 - alpha) * target_bg
            
            if target.ndim == 1:
                 target_mixedup = target_mixedup.flatten()

            # Apply mixup to weights if they exist
            weights_mixedup = None
            if has_weights:
                w_signal = weights[sampled_signal_indices]
                w_bg = weights[background_indices]
                if weights.ndim == 1:
                    w_signal = w_signal.reshape(-1, 1)
                    w_bg = w_bg.reshape(-1, 1)
                
                weights_mixedup = alpha * w_signal + (1 - alpha) * w_bg
                if weights.ndim == 1:
                    weights_mixedup = weights_mixedup.flatten()

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
            
    except Exception as e:
        raise RuntimeError(f"Failed to process {filename}: {e}")


def main():
    """Main preprocessing function."""
    print("=" * 80)
    print("MIXUP DATA AUGMENTATION PREPROCESSING (XENON2)")
    print("=" * 80)

    # Load configuration
    config_path = "../xenon/settings2.yaml"
    print(f"\nLoading configuration from: {config_path}")

    with open(config_path, "r") as f:
        config_file = yaml.safe_load(f)

    # Check if mixup is enabled
    use_data_augmentation = config_file["cnp_settings"]["use_data_augmentation"]

    if use_data_augmentation != "mixup":
        print(f"\nWARNING: use_data_augmentation is set to '{use_data_augmentation}', not 'mixup'")
        print("This script is specifically for mixup augmentation.")
        # response = input("Do you want to continue anyway? (y/n): ")
        # if response.lower() != 'y':
        #     print("Exiting.")
        #     sys.exit(0)
        print("Continuing anyway as requested...")

    # Get settings
    if len(sys.argv) > 1:
        path_to_files = sys.argv[1]
        print(f"Overriding data path with argument: {path_to_files}")
    else:
        path_to_files = config_file["path_settings"]["path_to_files_train"]
    
    use_beta = config_file["cnp_settings"]["use_beta"]
    signal_condition = config_file["simulation_settings"]["signal_condition"]

    print(f"\nSettings:")
    print(f"  Data path: {path_to_files}")
    print(f"  Beta distribution: {use_beta}")
    print(f"  Signal condition: {signal_condition}")

    # Convert CSV to HDF5 if needed
    if not os.path.exists(path_to_files):
        print(f"Error: Path {path_to_files} does not exist.")
        sys.exit(1)

    # Get all HDF5 files
    files = sorted([os.path.join(path_to_files, f) for f in os.listdir(path_to_files) if f.endswith(".h5")])

    if not files:
        print(f"\nNo HDF5 files found in {path_to_files}")
        sys.exit(1)

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
