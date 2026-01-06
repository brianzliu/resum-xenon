"""
Multi-Fidelity Gaussian Process Analyzer

A comprehensive automated analysis pipeline for multi-fidelity Gaussian process predictions.
Handles CSV file processing, prediction generation, uncertainty visualization, and coverage analysis.

Author: Generated for MFGP analysis pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os
from pathlib import Path
from collections import defaultdict
import seaborn as sns
import yaml
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from matplotlib.lines import Line2D

# Import required modules for MFGP
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays


with open("../xenon/settings2.yaml", "r") as f:
    config_file = yaml.safe_load(f)

PLOT_AFTER = int(config_file["cnp_settings"]["plot_after"])
FILES_PER_BATCH = config_file["cnp_settings"]["files_per_batch_predict"]
target_range = config_file["simulation_settings"]["target_range"]
is_binary = target_range[0] >= 0 and target_range[1] <= 1

path_out  = config_file["path_settings"]["path_out_cnp"]
version   = config_file["path_settings"]["version"]
iteration = config_file["path_settings"]["iteration"]
fidelity  = config_file["path_settings"]["fidelity"]


class MFGPAnalyzer:
    """
    Automated analysis pipeline for multi-fidelity Gaussian process predictions.
    
    This class provides comprehensive tools for:
    - Processing multiple CSV files with theta combinations
    - Generating predictions with uncertainty quantification
    - Creating uncertainty band visualizations
    - Calculating coverage statistics
    - Generating contour maps and summary plots
    """
    
    def __init__(self, mf_model, x_labels, y_label_sim='y_raw', output_dir=None):
        """
        Initialize the MFGPAnalyzer.
        
        Parameters:
        -----------
        mf_model : GPyMultiOutputWrapper
            The trained multi-fidelity Gaussian process model
        x_labels : list
            List of parameter names (e.g., ['water_shielding_mm', 'veto_thickness_mm'])
        y_label_sim : str
            Name of the target variable column (default: 'y_raw')
        output_dir : str or Path
            Directory to save output plots and results (default: current directory)
        """
        self.mf_model = mf_model
        self.x_labels = x_labels
        self.y_label_sim = y_label_sim
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize default bounds
        self.theta_min = [0, 0]
        self.theta_max = [95, 102] # Default fallback values
        
        # specific logic to auto-load settings for bounds if available
        try:
            # Assuming standard project structure: src/run_mfgp/mfgp_visualizations.py -> src/xenon/settings2.yaml
            # Go up two levels from this file's location
            current_file = Path(__file__)
            project_src = current_file.parent.parent 
            settings_path = project_src / "xenon" / "settings2.yaml"
            
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if 'simulation_settings' in config:
                        self.theta_min = config['simulation_settings'].get('theta_min', self.theta_min)
                        self.theta_max = config['simulation_settings'].get('theta_max', self.theta_max)
                        print(f"  Loaded simulation bounds from {settings_path}")
        except Exception as e:
            print(f"  Could not auto-load settings.yaml for bounds: {e}")
        
        print(f"MFGPAnalyzer initialized:")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Parameter labels: {self.x_labels}")
        print(f"  Target variable: {self.y_label_sim}")
        
    def is_point_valid(self, x, y):
        """
        Check if a point (x, y) satisfies the geometric constraints.
        Valid regions:
        1. X < 35.65 AND Y > 75.6
        2. 35.65 <= X < 89.35 AND Y > 65.0
        3. X >= 89.35
        """
        c_x1, c_x2 = 35.65, 89.35
        c_y1, c_y2 = 75.6, 65.0
        
        if x < c_x1:
            return y > c_y1
        elif x < c_x2:
            return y > c_y2
        else:
            return True # X >= 89.35 is always valid in the plotting range logic
            
    def load_and_process_csv_files(self, file_patterns, fidelity_filter=1.0, iteration_filter=0):
        """
        Load and process multiple CSV files to extract unique theta combinations.
        
        Parameters:
        -----------
        file_patterns : list or str
            List of file patterns or single pattern to match CSV files
            Supports glob patterns like 'data/*.csv' or specific file paths
        fidelity_filter : float
            Fidelity level to filter (default: 1.0 for high fidelity)
        iteration_filter : int
            Iteration number to filter (default: 0)
            
        Returns:
        --------
        dict: Dictionary with file names as keys and processed data as values
        """
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
            
        all_files = []
        for pattern in file_patterns:
            matched_files = glob.glob(pattern)
            all_files.extend(matched_files)
            
        if not all_files:
            print(f"Warning: No files found matching patterns: {file_patterns}")
            return {}
            
        processed_data = {}
        
        for file_path in all_files:
            print(f"Processing: {file_path}")
            try:
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                required_cols = self.x_labels + [self.y_label_sim, 'fidelity', 'iteration']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"  Warning: Missing columns {missing_cols}, skipping file")
                    continue
                
                # Filter data
                filtered_df = df[(df['fidelity'] == fidelity_filter) & 
                               (df['iteration'] == iteration_filter)]
                
                if filtered_df.empty:
                    print(f"  Warning: No data found with fidelity={fidelity_filter}, iteration={iteration_filter}")
                    continue
                
                # Group by unique theta combinations
                theta_groups = {}
                unique_combinations = filtered_df[self.x_labels].drop_duplicates()
                
                for _, combo in unique_combinations.iterrows():
                    combo_key = tuple(combo.values)
                    mask = True
                    for i, label in enumerate(self.x_labels):
                        mask &= (filtered_df[label] == combo.iloc[i])
                    
                    group_data = filtered_df[mask]
                    theta_groups[combo_key] = {
                        'x_values': combo.values,
                        'y_values': group_data[self.y_label_sim].values,
                        'raw_data': group_data
                    }
                
                processed_data[Path(file_path).name] = {
                    'theta_groups': theta_groups,
                    'full_data': filtered_df,
                    'file_path': file_path
                }
                
                print(f"  Found {len(theta_groups)} unique theta combinations")
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                
        return processed_data
    
    def predict_for_theta_groups(self, processed_data):
        """
        Generate predictions for all theta combinations in processed data.
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary returned by load_and_process_csv_files()
            
        Returns:
        --------
        dict: Nested dictionary with predictions for each file and theta combination
        """
        predictions = {}
        
        for file_name, file_data in processed_data.items():
            print(f"\nGenerating predictions for {file_name}")
            file_predictions = {}
            
            for combo_key, group_data in file_data['theta_groups'].items():
                # Prepare prediction input (add fidelity=1 for high-fidelity prediction)
                x_pred = np.array([list(combo_key) + [1.0]])  # Add fidelity indicator
                
                # Get model predictions
                mean_pred, var_pred = self.mf_model.predict(x_pred)
                std_pred = np.sqrt(var_pred)
                
                file_predictions[combo_key] = {
                    'x_values': group_data['x_values'],
                    'y_true': group_data['y_values'],
                    'y_pred_mean': mean_pred[0, 0],
                    'y_pred_std': std_pred[0, 0],
                    'raw_data': group_data['raw_data']
                }
                
                print(f"  Theta {combo_key}: mean={mean_pred[0, 0]:.6f}, std={std_pred[0, 0]:.6f}")
            
            predictions[file_name] = file_predictions
            
        return predictions
    
    def calculate_coverage_statistics(self, predictions):
        """
        Calculate coverage statistics for all predictions.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary returned by predict_for_theta_groups()
            
        Returns:
        --------
        dict: Coverage statistics for each file
        """
        coverage_stats = {}
        
        for file_name, file_preds in predictions.items():
            print(f"\nCalculating coverage for {file_name}")
            
            all_deviations = []
            sigma_bands = [1, 2, 3]
            coverage_counts = {sigma: 0 for sigma in sigma_bands}
            total_points = 0
            
            for combo_key, pred_data in file_preds.items():
                y_true = pred_data['y_true']
                y_pred_mean = pred_data['y_pred_mean']
                y_pred_std = pred_data['y_pred_std']
                
                # Calculate deviations for each data point
                deviations = np.abs(y_true - y_pred_mean) / y_pred_std
                all_deviations.extend(deviations)
                
                # Count coverage for different sigma bands
                for sigma in sigma_bands:
                    coverage_counts[sigma] += np.sum(deviations <= sigma)
                
                total_points += len(y_true)
            # Calculate percentages
            coverage_percentages = {sigma: 100 * count / total_points 
                                  for sigma, count in coverage_counts.items()}
            
            coverage_stats[file_name] = {
                'coverage_counts': coverage_counts,
                'coverage_percentages': coverage_percentages,
                'total_points': total_points,
                'all_deviations': np.array(all_deviations)
            }
            
            print(f"  Total points: {total_points}")
            for sigma in sigma_bands:
                print(f"  ±{sigma}σ: {coverage_counts[sigma]}/{total_points} ({coverage_percentages[sigma]:.1f}%)")
                
        return coverage_stats

    def plot_uncertainty_bands_for_theta_group(self, combo_key, pred_data, file_name, save_plots=True):
        """
        Plot uncertainty bands for a specific theta combination with coverage statistics.
        
        Parameters:
        -----------
        combo_key : tuple
            Theta combination values
        pred_data : dict
            Prediction data for this theta combination
        file_name : str
            Name of the source file
        save_plots : bool
            Whether to save the plot to disk
        """
        y_true = pred_data['y_true']
        y_pred_mean = pred_data['y_pred_mean']
        y_pred_std = pred_data['y_pred_std']
        
        # Create index for plotting
        idx = np.arange(len(y_true))
        
        plt.figure(figsize=(12, 4))
        
        # Plot uncertainty bands
        plt.fill_between(idx, y_pred_mean - 3*y_pred_std, y_pred_mean + 3*y_pred_std, 
                        facecolor='r', alpha=0.1, label='±3σ')
        plt.fill_between(idx, y_pred_mean - 2*y_pred_std, y_pred_mean + 2*y_pred_std, 
                        facecolor='y', alpha=0.15, label='±2σ')
        plt.fill_between(idx, y_pred_mean - 1*y_pred_std, y_pred_mean + 1*y_pred_std, 
                        facecolor='g', alpha=0.2, label='RESuM ±1σ')
        
        # Plot actual data points
        plt.scatter(idx, y_true, color='k', s=20, label='Validation Data', zorder=5)
        
        # Plot mean prediction line
        plt.axhline(y=y_pred_mean, color='red', linestyle='--', alpha=0.8, label='RESuM Mean')
        
        plt.xlabel('Sample Index')
        plt.ylabel(f'{self.y_label_sim}')
        plt.title(f'Theta: {combo_key} | File: {file_name}')
        
        # Custom legend order
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [4, 3, 2, 1, 0]  # Reorder to put validation data first
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
                  ncol=5, loc='upper right', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        
        # Calculate and add coverage statistics as text box
        deviations = np.abs(y_true - y_pred_mean) / y_pred_std
        coverage_text = []
        for sigma in [1, 2, 3]:
            within_sigma = np.sum(deviations <= sigma)
            percentage = 100 * within_sigma / len(y_true)
            coverage_text.append(f"±{sigma}σ: {within_sigma}/{len(y_true)} ({percentage:.1f}%)")
        
        # Add coverage statistics as text box
        text_str = "Coverage Statistics:\n" + "\n".join(coverage_text)
        plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        if save_plots:
            combo_str = '_'.join([f'{val:.1f}' for val in combo_key])
            filename = f'uncertainty_bands_{Path(file_name).stem}_theta_{combo_str}.png'
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {save_path}")
            
        plt.show()
        
        # Also print coverage statistics to console
        print(f"    Coverage statistics for Theta {combo_key}:")
        for sigma in [1, 2, 3]:
            within_sigma = np.sum(deviations <= sigma)
            percentage = 100 * within_sigma / len(y_true)
            print(f"    ±{sigma}σ: {within_sigma}/{len(y_true)} ({percentage:.1f}%)")
            
    def plot_coverage_summary(self, coverage_stats, save_plots=True):
        """
        Create summary plots of coverage statistics across all files.
        
        Parameters:
        -----------
        coverage_stats : dict
            Dictionary returned by calculate_coverage_statistics()
        save_plots : bool
            Whether to save the plot to disk
        """
        # Collect data for plotting
        files = list(coverage_stats.keys())
        sigma_levels = [1, 2, 3]
        
        # Coverage percentages plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of coverage percentages
        x_pos = np.arange(len(files))
        width = 0.25
        
        for i, sigma in enumerate(sigma_levels):
            percentages = [coverage_stats[f]['coverage_percentages'][sigma] for f in files]
            ax1.bar(x_pos + i*width, percentages, width, 
                   label=f'±{sigma}σ', alpha=0.8)
        
        ax1.set_xlabel('Files')
        ax1.set_ylabel('Coverage Percentage (%)')
        ax1.set_title('Coverage Statistics by File')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels([Path(f).stem for f in files], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add theoretical lines for normal distribution
        theoretical_coverage = {1: 68.27, 2: 95.45, 3: 99.73}
        for sigma, theoretical in theoretical_coverage.items():
            ax1.axhline(y=theoretical, color='gray', linestyle='--', alpha=0.7)
            ax1.text(len(files)-0.5, theoretical+1, f'{theoretical:.1f}% (theory)', 
                    fontsize=9, alpha=0.7)
        
        # Histogram of all deviations
        all_deviations = []
        for stats in coverage_stats.values():
            all_deviations.extend(stats['all_deviations'])
        
        ax2.hist(all_deviations, bins=50, alpha=0.7, density=True, 
                label='Observed Deviations')
        
        # Overlay theoretical normal distribution
        x_theory = np.linspace(0, max(all_deviations), 100)
        y_theory = 2 * np.exp(-0.5 * x_theory**2) / np.sqrt(2*np.pi)  # Half-normal (absolute values)
        ax2.plot(x_theory, y_theory, 'r--', label='Theoretical (Half-Normal)', linewidth=2)
        
        ax2.set_xlabel('|Deviation| (in σ units)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Prediction Deviations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for 1, 2, 3 sigma
        for sigma in [1, 2, 3]:
            ax2.axvline(x=sigma, color='gray', linestyle=':', alpha=0.7)
            ax2.text(sigma, ax2.get_ylim()[1]*0.9, f'{sigma}σ', 
                    rotation=90, verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        
        if save_plots:
            save_path = self.output_dir / f'{version}_coverage_summary.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Coverage summary saved: {save_path}")
            
        plt.show()
        
        # Print overall statistics
        total_points = sum(stats['total_points'] for stats in coverage_stats.values())
        print(f"\nOverall Statistics ({len(files)} files, {total_points} total points):")
        
        for sigma in sigma_levels:
            total_within = sum(stats['coverage_counts'][sigma] for stats in coverage_stats.values())
            percentage = 100 * total_within / total_points
            theoretical = theoretical_coverage[sigma]
            print(f"±{sigma}σ: {total_within}/{total_points} ({percentage:.1f}%) | " +
                  f"Theoretical: {theoretical:.1f}% | Diff: {percentage-theoretical:+.1f}%")

    def create_enhanced_contour_plots(self, processed_data, grid_steps=50, levels=25, save_plots=True, show_hf_training=True, hf_training_data_file=None, hf_validation_dir=None):
        """
        Create enhanced contour plots showing mean prediction and uncertainty with training data overlaid.
        The grid is determined by the X values found in validation CSV files and evenly spaced Y values.
        
        Parameters:
        -----------
        processed_data : dict
            Dictionary returned by load_and_process_csv_files()
        grid_steps : int
            Number of grid points for Y dimension (X is determined by files)
        levels : int
            Number of contour levels
        save_plots : bool
            Whether to save plots to disk
        hf_validation_dir : str, optional
            Directory containing validation CSV files to determine X-grid
            
        Returns:
        --------
        matplotlib.figure.Figure: The generated figure
        """
        # Determine grid from validation files
        if hf_validation_dir:
            val_dir = Path(hf_validation_dir)
        else:
            val_dir = Path("/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/validation/hf")
            
        # Force grid to use configuration bounds for consistent axes
        if self.theta_max:
             x_max = self.theta_max[0]
             y_max = self.theta_max[1]
             x_min = self.theta_min[0] if self.theta_min else 0
             y_min = self.theta_min[1] if self.theta_min else 0
        else:
             x_max = 95
             y_max = 102
             x_min = 0
             y_min = 0
             
        x_grid_vals = np.linspace(x_min, x_max, grid_steps)
        y_grid_vals = np.linspace(y_min, y_max, grid_steps)

        # Create meshgrid: X-axis = scint_x, Y-axis = scint_y
        Xg, Yg = np.meshgrid(x_grid_vals, y_grid_vals)
        
        # Prepare points for prediction: [scint_x, scint_y]
        # Xg contains scint_x values, Yg contains scint_y values
        points = np.column_stack([Xg.ravel(), Yg.ravel()])
        
        fidelity_col = np.ones((len(points), 1))
        points_with_fidelity = np.hstack([points, fidelity_col])
        
        # Get predictions
        mean_pred, var_pred = self.mf_model.predict(points_with_fidelity)
        std_pred = np.sqrt(var_pred)
        
        # Automatically determine scale factor based on magnitude
        max_mean = np.max(np.abs(mean_pred))
        if max_mean > 0:
            exponent = int(np.floor(np.log10(max_mean)))
            scale_factor = 10 ** (-exponent)
        else:
            exponent = 0
            scale_factor = 1
        
        Z_mean = mean_pred.reshape(Xg.shape) * scale_factor
        Z_std = std_pred.reshape(Xg.shape) * scale_factor
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        
        # 1. Mean Prediction Plot
        contour1 = ax1.contourf(Xg, Yg, Z_mean, levels=levels, cmap='viridis')
        cbar1 = fig.colorbar(contour1, ax=ax1)
        cbar1.set_label(rf"Predicted $y_{{\rm{{raw}}}}$ (mean) [$\times10^{{{exponent}}}$]", fontsize=12)
        ax1.contour(Xg, Yg, Z_mean, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # Plot LF Training Data
        lf_train_dir = Path("../xenon/in/data/original_vars/training/lf")
        lf_x_coords = []
        lf_y_coords = []
        if lf_train_dir.exists():
            import re
            for lf_file in lf_train_dir.glob('*.csv'):
                match = re.search(r'sim_X(\d+)_Y(\d+)', lf_file.name)
                if match:
                    # Match X to X-axis (scint_x), Y to Y-axis (scint_y)
                    lf_x_coords.append(int(match.group(1))) 
                    lf_y_coords.append(int(match.group(2)))
            
            if lf_x_coords and lf_y_coords:
                ax1.scatter(lf_x_coords, lf_y_coords, c='orange', s=20,
                           marker='^', edgecolors='white', linewidth=0.8,
                           alpha=0.7, zorder=4, label='LF Training Data')

        # Plot HF Training Data
        hf_x = []
        hf_y = []
        
        if show_hf_training:
            # 1. Scan the specific training directory (Highest Priority as per user request)
            # This ensures we overlay the coordinates designated by the X and Y in filenames from the training dir
            try:
                training_dir = "/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/training/hf"
                if os.path.exists(training_dir):
                    print(f"  Scanning HF training dir: {training_dir}")
                    sim_files = glob.glob(os.path.join(training_dir, "sim_*.csv"))
                    existing_points = set(zip(hf_x, hf_y))
                    
                    for f in sim_files:
                        # filename format: sim_X{val}_Y{val}.csv or sim_X{val}_Y{val}_ALL.csv
                        match = re.search(r"sim_X(\d+)_Y(\d+)", os.path.basename(f))
                        if match:
                            x_val = int(match.group(1))
                            y_val = int(match.group(2))
                            if (x_val, y_val) not in existing_points:
                                hf_x.append(x_val)
                                hf_y.append(y_val)
                                existing_points.add((x_val, y_val))
                                
                    print(f"  Found {len(hf_x)} HF training points from directory.")
            except Exception as e:
                print(f"  Warning: Failed to scan HF training directory: {e}")

            # 2. If no data found yet, try loading from explicit file if provided
            if not hf_x and hf_training_data_file:
                try:
                    df_all = pd.read_csv(hf_training_data_file)
                    # Use provided file to filter for fidelity 1.0
                    hf_df = df_all[(df_all['fidelity'] == 1.0) & (df_all['iteration'] == 0)]
                    if not hf_df.empty:
                        hf_unique = hf_df[self.x_labels].drop_duplicates()
                        hf_x = [row[self.x_labels[0]] for _, row in hf_unique.iterrows()]
                        hf_y = [row[self.x_labels[1]] for _, row in hf_unique.iterrows()]
                except Exception as e:
                    print(f"  Warning: Failed to load provided HF training file: {e}")

            # 3. Fallback: Try to infer from processed_data (likely just current batch)
            if not hf_x and processed_data:
                try:
                    # Logic to extract unique x/y from processed_data if available
                    # This is a weak fallback but better than nothing
                    pass 
                except Exception:
                    pass

            if hf_x:
                try:
                    # Increased marker size and specific styling for better visibility
                    ax1.scatter(hf_x, hf_y, c='dodgerblue', s=80,
                                marker='^', edgecolors='white', linewidth=1.5,
                                label='HF Training Data', alpha=1.0, zorder=10)
                except Exception as e:
                    print(f"  Could not plot HF training data: {e}")

        # Label axes: X is now scint_x (x_labels[0]), Y is now scint_y (x_labels[1])
        ax1.set_xlabel(self.x_labels[0], fontsize=12) 
        ax1.set_ylabel(self.x_labels[1], fontsize=12)
        ax1.set_title('Mean Prediction', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Uncertainty Plot
        contour2 = ax2.contourf(Xg, Yg, Z_std, levels=levels, cmap='Reds')
        cbar2 = fig.colorbar(contour2, ax=ax2)
        cbar2.set_label(rf"Uncertainty ($\sigma$) [$\times10^{{{exponent}}}$]", fontsize=12)
        ax2.contour(Xg, Yg, Z_std, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # Plot LF Training Data on Uncertainty Plot
        if lf_x_coords and lf_y_coords:
            ax2.scatter(lf_x_coords, lf_y_coords, c='orange', s=20,
                       marker='^', edgecolors='white', linewidth=0.8,
                       alpha=0.7, zorder=4, label='LF Training Data')

        # Plot HF Training Data on Uncertainty Plot
        if show_hf_training and hf_x:
             ax2.scatter(hf_x, hf_y, c='dodgerblue', s=80,
                        marker='^', edgecolors='white', linewidth=1.5,
                        label='HF Training Data', alpha=1.0, zorder=10)

        ax2.set_xlabel(self.x_labels[0], fontsize=12)
        ax2.set_ylabel(self.x_labels[1], fontsize=12)
        ax2.set_title('Prediction Uncertainty', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # --- Overlay Constraints ---
        def apply_constraints_overlay(ax):
            # Constraint Constants
            c_x1, c_x2 = 35.65, 89.35
            c_y1, c_y2 = 75.6, 65.0
            
            # Determine plot limits to ensure fills cover everything
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # 1. Fill Forbidden Region (Red) - DISABLED per user request
            # Region A: X < 35.65, Y < 75.6
            # rect1 = Rectangle((xlim[0], ylim[0]), c_x1 - xlim[0], c_y1 - ylim[0],
            #                 facecolor='red', alpha=0.1, zorder=1)
            # ax.add_patch(rect1)
            
            # Region B: 35.65 <= X < 89.35, Y < 65.0
            # rect2 = Rectangle((c_x1, ylim[0]), c_x2 - c_x1, c_y2 - ylim[0],
            #                 facecolor='red', alpha=0.1, zorder=1)
            # ax.add_patch(rect2)
            
            # 2. Fill Allowed Region (Green) - DISABLED per user request
            # Region C: X < 35.65, Y > 75.6
            # rect3 = Rectangle((xlim[0], c_y1), c_x1 - xlim[0], ylim[1] - c_y1,
            #                 facecolor='green', alpha=0.1, zorder=1)
            # ax.add_patch(rect3)
            
            # Region D: 35.65 <= X < 89.35, Y > 65.0
            # rect4 = Rectangle((c_x1, c_y2), c_x2 - c_x1, ylim[1] - c_y2,
            #                 facecolor='green', alpha=0.1, zorder=1)
            # ax.add_patch(rect4)
            
            # Region E: X >= 89.35 (All valid)
            # rect5 = Rectangle((c_x2, ylim[0]), xlim[1] - c_x2, ylim[1] - ylim[0],
            #                 facecolor='green', alpha=0.1, zorder=1)
            # ax.add_patch(rect5)
            
            # 3. Draw Boundaries
            # Vertical dashed lines
            ax.axvline(c_x1, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(c_x2, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Blue boundary line
            line_x = [xlim[0], c_x1, c_x1, c_x2, c_x2, xlim[1]]
            line_y = [c_y1, c_y1, c_y2, c_y2, ylim[0], ylim[0]]
            ax.plot(line_x, line_y, color='blue', linewidth=1.5, alpha=0.8, zorder=2)
            
            # 4. Add Text Labels
            # Allowed
            font_props = dict(fontsize=11, fontweight='bold', alpha=0.6)
            ax.text(c_x1 + 5, c_y1 + 5, "ALLOWED", color='green', **font_props)
            
            # Forbidden (center in the largest forbidden block)
            ax.text(c_x1 + 5, c_y2 - 20, "FORBIDDEN", color='darkred', **font_props)
            
            # Ensure limits didn't change due to plotting elements
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        apply_constraints_overlay(ax1)
        apply_constraints_overlay(ax2)
        
        if save_plots:
            save_path = self.output_dir / f'{version}_enhanced_contour_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced contour analysis saved: {save_path}")
        plt.show()
        return fig
    
    def plot_prediction_vs_true(self, predictions, file_name=None, save_plot=True):
        """
        Create a scatter plot of predicted vs true values aggregated per unique theta.

        This updated version first aggregates (averages) the true observed values for each
        unique theta configuration (matching the averaging approach used in
        plot_uncertainty_bands_across_thetas) and then compares those per-theta
        averages against the model's predicted mean with corresponding predictive
        standard deviation as error bars.

        Parameters:
        -----------
        predictions : dict
            Dictionary returned by predict_for_theta_groups()
        file_name : str, optional
            Specific file to plot (default: first file)
        save_plot : bool
            Whether to save the plot to disk
        """
        if file_name is None:
            file_name = list(predictions.keys())[0]
        
        file_preds = predictions[file_name]
        
        # Aggregate per theta: compute mean of true values for each theta
        all_true_mean = []      # per-theta averaged true value
        all_pred_mean = []      # model predicted mean (already per-theta)
        all_pred_std = []       # model predicted std (already per-theta)
        theta_labels = []       # string labels for optional future use / debugging
        
        for combo_key, pred_data in file_preds.items():
            y_true_vals = pred_data['y_true']
            y_true_avg = np.mean(y_true_vals)
            all_true_mean.append(y_true_avg)
            all_pred_mean.append(pred_data['y_pred_mean'])
            all_pred_std.append(pred_data['y_pred_std'])
            theta_labels.append(str(combo_key))
        
        all_true_mean = np.array(all_true_mean)
        all_pred_mean = np.array(all_pred_mean)
        all_pred_std = np.array(all_pred_std)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Scatter with error bars (prediction uncertainty on y-axis)
        plt.errorbar(all_true_mean, all_pred_mean, yerr=all_pred_std,
                     fmt='o', alpha=0.8, markersize=6, capsize=3, label='Per-Theta Mean')
        
        # Perfect prediction line
        min_val = min(all_true_mean.min(), all_pred_mean.min())
        max_val = max(all_true_mean.max(), all_pred_mean.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--',
                 label='Perfect Prediction', linewidth=2)
        
        # Global average std for bands (gives visual sense of typical uncertainty)
        avg_sigma = np.mean(all_pred_std)
        x_line = np.linspace(min_val, max_val, 200)
        for i, sigma_mult in enumerate([1, 2, 3]):
            alpha_val = 0.25 - i * 0.07
            plt.fill_between(x_line, x_line - sigma_mult * avg_sigma,
                              x_line + sigma_mult * avg_sigma,
                              alpha=alpha_val, label=f'±{sigma_mult}σ (avg)')
        
        plt.xlabel(f'True {self.y_label_sim} (per-theta mean)')
        plt.ylabel(f'Predicted {self.y_label_sim}')
        plt.title(f'Predicted vs True (Per-Theta Aggregated)\nFile: {Path(file_name).stem}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics based on aggregated points
        correlation = np.corrcoef(all_true_mean, all_pred_mean)[0, 1] if len(all_true_mean) > 1 else np.nan
        mae = np.mean(np.abs(all_true_mean - all_pred_mean))
        rmse = np.sqrt(np.mean((all_true_mean - all_pred_mean)**2))
        
        stats_text = (f'Points (unique thetas): {len(all_true_mean)}\n'
                      f'Correlation: {correlation:.3f}\n'
                      f'MAE: {mae:.6f}\nRMSE: {rmse:.6f}')
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        plt.tight_layout()
        
        if save_plot:
            save_path = self.output_dir / f'predicted_vs_true_{Path(file_name).stem}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction vs true (aggregated) plot saved: {save_path}")
        
        plt.show()
    
    def plot_uncertainty_bands_across_thetas(self, predictions, processed_data, file_name=None, save_plot=True, include_hf_training=True, hf_training_data_file=None, validation_cnp_file=None, hf_validation_dir=None, lf_validation_dir=None):
        """
        Plot uncertainty bands across all theta values for a given file.

        Compact layout with condensed tick labels (ws|vt). Fixed overlapping x-axis labels
        by simplifying xlabel and removing bottom annotation.
        
        Parameters:
        -----------
        validation_cnp_file : str, optional
            Path to validation CNP output file with HF mean data (for black dots) - Deprecated in favor of hf_validation_dir
        hf_validation_dir : str, optional
            Directory containing raw HF validation CSV files to calculate means from
        lf_validation_dir : str, optional
            Directory containing raw LF validation CSV files to calculate means from (default: /home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/validation/lf)
        """
        if file_name is None:
            file_name = list(predictions.keys())[0]
        file_preds = predictions[file_name]
        file_proc_data = processed_data[file_name]
        sorted_thetas = sorted(file_preds.keys())
        y_lf_means = []  # LF validation means (grey dots)
        y_hf_means = []  # HF validation means from CNP file (black dots)
        y_pred_means = []
        y_pred_stds = []
        x_coords = []  # Actual x coordinates (scint_x) for plotting
        ws_vals = []
        vt_vals = []
        
        # Load LF validation data from raw CSV files
        lf_cnp_data = {}
        # Default to known path if not provided
        if lf_validation_dir is None:
            lf_validation_dir = "/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/validation/lf"
            
        try:
            lf_dir = Path(lf_validation_dir)
            if lf_dir.exists():
                print(f"Loading LF validation data from {lf_dir}...")
                import re
                for csv_file in lf_dir.glob("*.csv"):
                    try:
                        # Extract theta from filename for consistency
                        match = re.search(r'sim_X(\d+)_Y(\d+)', csv_file.name)
                        if match:
                            scint_x = int(match.group(1))
                            scint_y = int(match.group(2))
                            theta_key = (scint_x, scint_y)
                            
                            # Read CSV for target value
                            df = pd.read_csv(csv_file)
                            # Check for tag_final first (standard for raw files), then fallback to y_label_sim
                            if 'tag_final' in df.columns:
                                mean_val = df['tag_final'].mean()
                                lf_cnp_data[theta_key] = mean_val
                            elif self.y_label_sim in df.columns:
                                mean_val = df[self.y_label_sim].mean()
                                lf_cnp_data[theta_key] = mean_val
                    except Exception as e:
                        print(f"  Error reading {csv_file.name}: {e}")
                print(f"Loaded LF validation data for {len(lf_cnp_data)} theta combinations")
            else:
                print(f"Warning: LF validation directory not found: {lf_validation_dir}")
        except Exception as e:
            print(f"Warning: Could not load LF validation data from directory: {e}")

        # Load HF validation data from raw CSV files if provided
        hf_cnp_data = {}
        if hf_validation_dir:
            try:
                hf_dir = Path(hf_validation_dir)
                if hf_dir.exists():
                    print(f"Loading HF validation data from {hf_dir}...")
                    import re
                    for csv_file in hf_dir.glob("*.csv"):
                        try:
                            # Extract theta from filename for consistency
                            match = re.search(r'sim_X(\d+)_Y(\d+)', csv_file.name)
                            if match:
                                scint_x = int(match.group(1))
                                scint_y = int(match.group(2))
                                theta_key = (scint_x, scint_y)
                                
                                # Read CSV for target value
                                df = pd.read_csv(csv_file)
                                if 'tag_final' in df.columns:
                                    mean_tag = df['tag_final'].mean()
                                    hf_cnp_data[theta_key] = mean_tag
                        except Exception as e:
                            print(f"  Error reading {csv_file.name}: {e}")
                    print(f"Loaded HF validation data for {len(hf_cnp_data)} theta combinations")
                else:
                    print(f"Warning: HF validation directory not found: {hf_validation_dir}")
            except Exception as e:
                print(f"Warning: Could not load HF validation data from directory: {e}")
        # Fallback to CNP file if directory not provided/failed but file is provided
        elif validation_cnp_file and not hf_cnp_data:
            try:
                df_hf = pd.read_csv(validation_cnp_file)
                hf_df = df_hf[df_hf['fidelity'] == 1.0]
                # Group by theta combination and get mean y_cnp
                for _, row in hf_df.iterrows():
                    theta_key = (row['scint_x'], row['scint_y'])
                    if theta_key not in hf_cnp_data:
                        hf_cnp_data[theta_key] = []
                    hf_cnp_data[theta_key].append(row['y_cnp'])
                # Average y_cnp for each theta
                for theta_key in hf_cnp_data:
                    hf_cnp_data[theta_key] = np.mean(hf_cnp_data[theta_key])
                print(f"Loaded HF validation data for {len(hf_cnp_data)} theta combinations from {validation_cnp_file}")
            except Exception as e:
                print(f"Warning: Could not load HF validation data: {e}")
        
        # Use UNION of keys to ensure both predictions (bands) and HF/LF data (dots) are shown
        all_keys = set(file_preds.keys())
        if hf_cnp_data:
            all_keys.update(hf_cnp_data.keys())
        if lf_cnp_data:
            all_keys.update(lf_cnp_data.keys())
            
        # Sort key logic: theta is (scint_x, scint_y)
        # We want to sort primarily by scint_x, then scint_y
        sorted_thetas = sorted(list(all_keys), key=lambda t: (t[0], t[1]))
        
        print(f"Plotting for {len(sorted_thetas)} theta combinations (Union of Prediction & HF Data)")
        
        for theta in sorted_thetas:
            x_coords.append(theta[0])  # scint_x coordinate for X-axis
            ws_vals.append(theta[0])  # scint_x
            vt_vals.append(theta[1])  # scint_y
            
            # Get LF validation data if available
            if theta in lf_cnp_data:
                y_lf_means.append(lf_cnp_data[theta])
            elif theta in file_proc_data['theta_groups']:
                y_lf_values = file_proc_data['theta_groups'][theta]['y_values']
                y_lf_means.append(np.mean(y_lf_values))
            else:
                y_lf_means.append(np.nan)
            
            # Get HF mean if available
            if theta in hf_cnp_data:
                y_hf_means.append(hf_cnp_data[theta])
            else:
                y_hf_means.append(np.nan)
            
            # Get Prediction data
            if theta in file_preds:
                y_pred_means.append(file_preds[theta]['y_pred_mean'])
                y_pred_stds.append(file_preds[theta]['y_pred_std'])
            else:
                # Generate prediction on the spot if missing from initial batch
                # Construct input vector: [scint_x, scint_y, fidelity=1.0]
                x_pred = np.array([[theta[0], theta[1], 1.0]])
                mean_pred, var_pred = self.mf_model.predict(x_pred)
                std_pred = np.sqrt(var_pred)
                y_pred_means.append(mean_pred[0, 0])
                y_pred_stds.append(std_pred[0, 0])
                
        y_pred_means = np.array(y_pred_means)
        y_pred_stds = np.array(y_pred_stds)
        y_lf_means = np.array(y_lf_means)
        y_hf_means = np.array(y_hf_means)
        
        n_thetas = len(sorted_thetas)
        
        # Use simple integer indexing for the x-axis to evenly space the groups
        plot_indices = np.arange(n_thetas)
        
        # Prepare fine-grained data for smoother bands
        fine_indices = []
        fine_pred_means = []
        fine_pred_stds = []
        
        steps_per_interval = 50  # Number of intermediate points
        
        for i in range(n_thetas):
            # 1. Add the exact point i
            fine_indices.append(float(i))
            fine_pred_means.append(y_pred_means[i])
            fine_pred_stds.append(y_pred_stds[i])
            
            # 2. Interpolate between i and i+1 if in the same scint_x group
            if i < n_thetas - 1:
                t1 = sorted_thetas[i]
                t2 = sorted_thetas[i+1]
                
                if t1[0] == t2[0]: # Same scint_x
                    # Interpolate scint_y
                    y_start = t1[1]
                    y_end = t2[1]
                    
                    # Generate intermediate y values (excluding endpoints)
                    y_interps = np.linspace(y_start, y_end, steps_per_interval + 2)[1:-1]
                    
                    # Prepare input for prediction: [scint_x, y_interp, fidelity=1.0]
                    X_new = np.column_stack([
                        np.full(len(y_interps), t1[0]), # scint_x constant
                        y_interps,                      # scint_y varying
                        np.ones(len(y_interps))         # fidelity HF
                    ])
                    
                    # Batch prediction
                    m_new, v_new = self.mf_model.predict(X_new)
                    std_new = np.sqrt(v_new).flatten()
                    mean_new = m_new.flatten()
                    
                    # Calculate fractional indices
                    idx_interps = np.linspace(float(i), float(i+1), steps_per_interval + 2)[1:-1]
                    
                    fine_indices.extend(idx_interps)
                    fine_pred_means.extend(mean_new)
                    fine_pred_stds.extend(std_new)
        
        fine_indices = np.array(fine_indices)
        fine_pred_means = np.array(fine_pred_means)
        fine_pred_stds = np.array(fine_pred_stds)

        # Adjust figure width based on number of points
        fig_width = min(20, max(10, 2 + 0.25 * n_thetas))
        fig_height = 5.5
        
        # --- 1. Main Plot ---
        plt.figure(figsize=(fig_width, fig_height))
        ax = plt.gca()
        
        # Plot bands using FINE indices
        ax.fill_between(fine_indices, fine_pred_means - 3 * fine_pred_stds, fine_pred_means + 3 * fine_pred_stds,
                         facecolor='r', alpha=0.1, label='±3σ')
        ax.fill_between(fine_indices, fine_pred_means - 2 * fine_pred_stds, fine_pred_means + 2 * fine_pred_stds,
                         facecolor='y', alpha=0.15, label='±2σ')
        ax.fill_between(fine_indices, fine_pred_means - 1 * fine_pred_stds, fine_pred_means + 1 * fine_pred_stds,
                         facecolor='g', alpha=0.2, label='RESuM ±1σ')
                         
        # Plot dots using ORIGINAL integer indices (Validation Data)
        if len(y_hf_means) > 0 and not np.all(np.isnan(y_hf_means)):
            ax.scatter(plot_indices, y_hf_means, color='black', linewidth=0.6,
                       s=28, label='HF Validation Mean', zorder=5)
        ax.scatter(plot_indices, y_lf_means, color='grey', linewidth=0.6,
                   s=28, label='LF Validation Mean', zorder=4)
                   
        # Setup labels
        def choose_fmt(vals):
            arr = np.asarray(vals)
            if len(arr) == 0: return '{:.1f}'
            if np.all(np.abs(arr - np.round(arr)) < 1e-6):
                return '{:.0f}'
            return '{:.1f}'
            
        ws_fmt = choose_fmt(ws_vals)
        vt_fmt = choose_fmt(vt_vals)
        # Construct label strings "x, y"
        base_labels = [f"{ws_fmt.format(ws)}, {vt_fmt.format(vt)}" for ws, vt in zip(ws_vals, vt_vals)]
        
        
        # Determine step for sparse labeling if too many points - NOW DISABLED to show all
        step = 1
            
        display_labels = [lab if (i % step == 0) else '' for i, lab in enumerate(base_labels)]
        
        # Set ticks at integer indices
        ax.set_xticks(plot_indices)
        ax.set_xticklabels(display_labels, rotation=90, ha='center', fontsize=6)
        ax.set_xlabel(f"{self.x_labels[0]}, {self.x_labels[1]}")
        ax.set_ylabel(f'Average {self.y_label_sim}')

        handles, labels = ax.get_legend_handles_labels()
        desired_order = ['HF Validation Mean', 'LF Validation Mean', 'RESuM ±1σ', '±2σ', '±3σ']
        order_map = {label: i for i, label in enumerate(desired_order)}
        try:
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: order_map.get(x[1], 99))
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)
            ax.legend(sorted_handles, sorted_labels, ncol=4, loc='upper right', fontsize=9)
        except Exception:
            ax.legend(ncol=4, loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add vertical separators for scint_x changes
        prev_x = sorted_thetas[0][0]
        for i in range(1, n_thetas):
            curr_x = sorted_thetas[i][0]
            if curr_x != prev_x:
                midpoint = i - 0.5
                ax.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
                # Ensure the label for the first point of the new group is shown if sparse labeling was used (not currently)
                prev_x = curr_x
        
        # Y-limits
        all_y_values = [y for y in y_lf_means if not np.isnan(y)]
        all_y_values.extend([y for y in (y_pred_means - 3 * y_pred_stds) if not np.isnan(y)])
        all_y_values.extend([y for y in (y_pred_means + 3 * y_pred_stds) if not np.isnan(y)])
        if len(y_hf_means) > 0:
            all_y_values.extend([y for y in y_hf_means if not np.isnan(y)])
        if len(all_y_values) == 0:
            all_y_values = [0, 1] 
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min if y_max > y_min else 1.0
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.08 * y_range)
        
        # Calculation coverage text (using HF means)
        coverage_text = []
        # Filter for valid HF data
        valid_indices = [i for i in range(len(y_hf_means)) if not np.isnan(y_hf_means[i])]
        total_count = len(valid_indices)
        
        for sigma in [1, 2, 3]:
            within_sigma = 0
            for i in valid_indices:
                lower = y_pred_means[i] - sigma * y_pred_stds[i]
                upper = y_pred_means[i] + sigma * y_pred_stds[i]
                if lower <= y_hf_means[i] <= upper:
                    within_sigma += 1
            pct = 100 * within_sigma / total_count if total_count else 0.0
            coverage_text.append(f"±{sigma}σ: {within_sigma}/{total_count} ({pct:.1f}%)")
        text_str = "Coverage (vs HF):\n" + "\n".join(coverage_text)
        ax.text(0.01, 0.99, text_str, transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=9, family='monospace')

        plt.tight_layout()
        if save_plot:
            filename = f'uncertainty_bands_across_thetas_{Path(file_name).stem}.png'
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved plot: {save_path}")
        plt.show()

        # --- 2. Zoomed-in Plot ---
        plt.figure(figsize=(fig_width, fig_height))
        ax = plt.gca()
        # Use FINE indices for bands
        ax.fill_between(fine_indices, fine_pred_means - 3 * fine_pred_stds, fine_pred_means + 3 * fine_pred_stds,
                         facecolor='r', alpha=0.1, label='±3σ')
        ax.fill_between(fine_indices, fine_pred_means - 2 * fine_pred_stds, fine_pred_means + 2 * fine_pred_stds,
                         facecolor='y', alpha=0.15, label='±2σ')
        ax.fill_between(fine_indices, fine_pred_means - 1 * fine_pred_stds, fine_pred_means + 1 * fine_pred_stds,
                         facecolor='g', alpha=0.2, label='RESuM ±1σ')
        
        if len(y_hf_means) > 0 and not np.all(np.isnan(y_hf_means)):
            ax.scatter(plot_indices, y_hf_means, color='black', linewidth=0.6,
                       s=28, label='HF Validation Mean', zorder=5)
        ax.scatter(plot_indices, y_lf_means, color='grey', linewidth=0.6,
                   s=28, label='LF Validation Mean', zorder=4)
                   
        ax.set_xticks(plot_indices)
        ax.set_xticklabels(display_labels, rotation=90, ha='center', fontsize=6)
        ax.set_xlabel(f"{self.x_labels[0]}, {self.x_labels[1]}")
        ax.set_ylabel(f'Average {self.y_label_sim}')

        # Re-use legend logic
        handles, labels = ax.get_legend_handles_labels()
        # Same order
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: order_map.get(x[1], 99))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        ax.legend(sorted_handles, sorted_labels, ncol=4, loc='upper right', fontsize=9)
        
        ax.grid(True, alpha=0.3)
        
        # Add vertical separators for scint_x changes (Zoomed plot)
        prev_x = sorted_thetas[0][0]
        for i in range(1, n_thetas):
            curr_x = sorted_thetas[i][0]
            if curr_x != prev_x:
                midpoint = i - 0.5
                ax.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
                prev_x = curr_x
        
        # Optimize y-limits for bands
        bands_y_values = [y for y in (y_pred_means - 3 * y_pred_stds) if not np.isnan(y)]
        bands_y_values.extend([y for y in (y_pred_means + 3 * y_pred_stds) if not np.isnan(y)])
        if len(bands_y_values) == 0:
            bands_y_values = [0, 1] 
        y_min_zoomed = min(bands_y_values)
        y_max_zoomed = max(bands_y_values)
        y_range_zoomed = y_max_zoomed - y_min_zoomed if y_max_zoomed > y_min_zoomed else 1.0
        
        ax.set_ylim(y_min_zoomed - 0.1 * y_range_zoomed, y_max_zoomed + 0.1 * y_range_zoomed)
        ax.text(0.01, 0.99, text_str, transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=9, family='monospace')
        
        plt.tight_layout()
        if save_plot:
            filename_zoomed = f'uncertainty_bands_across_thetas_zoomed_{Path(file_name).stem}.png'
            save_path_zoomed = self.output_dir / filename_zoomed
            plt.savefig(save_path_zoomed, dpi=300, bbox_inches='tight')
            print(f"    Saved zoomed plot: {save_path_zoomed}")
        plt.show()

    def run_complete_analysis(self, file_patterns, fidelity_filter=1.0, iteration_filter=0, 
                            plot_individual_groups=True, save_all_plots=True, 
                            show_hf_training=True, include_hf_training=True,
                            hf_training_data_file=None, validation_cnp_file=None, hf_validation_dir=None, lf_validation_dir=None):
        """
        Run the complete automated analysis pipeline.
        
        Parameters:
        -----------
        file_patterns : list or str
            File patterns to process
        fidelity_filter : float
            Fidelity level to filter
        iteration_filter : int
            Iteration number to filter
        plot_individual_groups : bool
            Whether to plot individual theta combinations
        save_all_plots : bool
            Whether to save all plots to disk
        show_hf_training : bool
            Whether to show HF training data in contour plots (default: True)
        include_hf_training : bool
            Whether to include HF training data in across-theta plots (default: True)
        hf_training_data_file : str, optional
            Path to a separate file containing HF training data (default: None)
        validation_cnp_file : str, optional
            Path to validation CNP output file with HF mean data (default: None) - Deprecated
        hf_validation_dir : str, optional
            Directory containing raw HF validation CSV files to calculate means from
        lf_validation_dir : str, optional
            Directory containing raw LF validation CSV files to calculate means from
            
        Returns:
        --------
        dict: Complete analysis results
        """
        print("="*80)
        print("AUTOMATED MFGP ANALYSIS PIPELINE")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        
        # Step 1: Load and process CSV files
        print("\n1. Loading and processing CSV files...")
        processed_data = self.load_and_process_csv_files(file_patterns, fidelity_filter, iteration_filter)
        
        if not processed_data:
            print("No data found! Check your file patterns and filters.")
            return None
            
        # Step 2: Generate predictions
        print("\n2. Generating predictions for all theta combinations...")
        predictions = self.predict_for_theta_groups(processed_data)
        
        # Step 3: Calculate coverage statistics
        print("\n3. Calculating coverage statistics...")
        coverage_stats = self.calculate_coverage_statistics(predictions)
        
        # Step 4: Plot individual groups (if requested)
        if plot_individual_groups:
            print("\n4. Generating individual uncertainty band plots...")
            for file_name, file_preds in predictions.items():
                print(f"\nPlotting groups for {file_name}:")
                for combo_key, pred_data in file_preds.items():
                    print(f"  Theta: {combo_key}")
                    self.plot_uncertainty_bands_for_theta_group(
                        combo_key, pred_data, file_name, save_all_plots)
        
        # Step 5: Create coverage summary
        print("\n5. Creating coverage summary plots...")
        self.plot_coverage_summary(coverage_stats, save_all_plots)
        
        # Step 6: Create enhanced contour plots
        print("\n6. Creating enhanced contour analysis...")
        self.create_enhanced_contour_plots(processed_data, save_plots=save_all_plots, 
                                          show_hf_training=show_hf_training,
                                          hf_training_data_file=hf_training_data_file,
                                          hf_validation_dir=hf_validation_dir)
        
        # Step 7: Create prediction vs true plots
        print("\n7. Creating prediction vs true value plots...")
        for file_name in predictions.keys():
            self.plot_prediction_vs_true(predictions, file_name, save_all_plots)
        
        # Step 8: Create plot across all theta values
        print("\n8. Creating plot across all theta values...")
        for file_name in predictions.keys():
            self.plot_uncertainty_bands_across_thetas(predictions, processed_data, file_name, 
                                                     save_all_plots, include_hf_training=include_hf_training,
                                                     hf_training_data_file=hf_training_data_file,
                                                     validation_cnp_file=validation_cnp_file,
                                                     hf_validation_dir=hf_validation_dir,
                                                     lf_validation_dir=lf_validation_dir)



        # Step 9: Find and print highest valid average
        print("\n9. Finding highest valid average y_raw...")
        
        # Aggregate all known averages
        all_averages = {} # Theta -> Mean Value
        
        # 1. From Predictions (Batch Data)
        for fname, fpreds in predictions.items():
            for theta, pdata in fpreds.items():
                if theta not in all_averages:
                    # predictions['y_true'] are the actual values from the input CSV
                    all_averages[theta] = np.mean(pdata['y_true'])
        
        # 2. From HF Validation Dir
        if hf_validation_dir and Path(hf_validation_dir).exists():
            import re
            for csv_file in Path(hf_validation_dir).glob("*.csv"):
                match = re.search(r'sim_X(\d+)_Y(\d+)', csv_file.name)
                if match:
                    theta = (int(match.group(1)), int(match.group(2)))
                    try:
                        df = pd.read_csv(csv_file)
                        val = df['tag_final'].mean() if 'tag_final' in df.columns else (df[self.y_label_sim].mean() if self.y_label_sim in df.columns else None)
                        if val is not None:
                            all_averages[theta] = val
                    except: pass

        # 3. From LF Validation Dir
        # Default if not provided
        if lf_validation_dir is None:
             lf_validation_dir = "/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/validation/lf"
        
        if lf_validation_dir and Path(lf_validation_dir).exists():
            import re
            for csv_file in Path(lf_validation_dir).glob("*.csv"):
                match = re.search(r'sim_X(\d+)_Y(\d+)', csv_file.name)
                if match:
                    theta = (int(match.group(1)), int(match.group(2)))
                    # Prefer HF data if we already found it, but if it's new (LF only), add it
                    if theta not in all_averages:
                        try:
                            df = pd.read_csv(csv_file)
                            val = df['tag_final'].mean() if 'tag_final' in df.columns else (df[self.y_label_sim].mean() if self.y_label_sim in df.columns else None)
                            if val is not None:
                                all_averages[theta] = val
                                all_averages[theta] = val
                        except: pass
                        
        # 4. From Training Directories (Auto-scan)
        training_dirs = [
            "/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/training/hf",
            "/home/tidmad/bliu/resum-xenon/src/xenon/in/data/new_both/training/lf"
        ]
        for t_dir in training_dirs:
            if Path(t_dir).exists():
                for csv_file in Path(t_dir).glob("*.csv"):
                    match = re.search(r'sim_X(\d+)_Y(\d+)', csv_file.name)
                    if match:
                        theta = (int(match.group(1)), int(match.group(2)))
                        if theta not in all_averages:
                            try:
                                df = pd.read_csv(csv_file, usecols=['tag_final'])
                                if 'tag_final' in df.columns:
                                    all_averages[theta] = df['tag_final'].mean()
                            except: pass
        
        # Filter and Find Max
        valid_points = []
        for theta, avg_val in all_averages.items():
            x, y = theta[0], theta[1]
            if self.is_point_valid(x, y):
                valid_points.append((theta, avg_val))
                
        if valid_points:
            # Sort by value descending
            valid_points.sort(key=lambda x: x[1], reverse=True)
            max_theta, max_val = valid_points[0]
            
            print("\n" + "="*60)
            print("HIGHEST VALID AVERAGE y_raw")
            print("="*60)
            print(f"Theta (X, Y): {max_theta}")
            print(f"Average Value: {max_val:.8f}")
            print(f"Number of valid points checked: {len(valid_points)}")
            
            print("\nTop 5 Valid Predictions:")
            for t, v in valid_points[:5]:
                print(f"  Theta {t}: {v:.8f}")
        else:
            print("No valid points found satisfying the constraints.")

        # Step 10: Grid Search for Highest Predicted Value
        print("\n10. Grid Search for Highest Predicted Value (Continuous Domain)...")
        # Define grid based on bounds
        x_min, x_max = (0, 100)
        y_min, y_max = (0, 100)
        
        # Try to use instance bounds if they look reasonable (not default 0,0)
        if hasattr(self, 'theta_min') and hasattr(self, 'theta_max'):
             if self.theta_max[0] > 0:
                 x_min, y_min = self.theta_min[0], self.theta_min[1]
                 x_max, y_max = self.theta_max[0], self.theta_max[1]
        
        grid_res = 100 # 100x100 = 10,000 points
        x_grid = np.linspace(x_min, x_max, grid_res)
        y_grid = np.linspace(y_min, y_max, grid_res)
        
        # Create mesh
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
        
        # Predict
        # Add fidelity column
        grid_points_w_fid = np.hstack([grid_points, np.ones((len(grid_points), 1))])
        
        try:
            mean_pred, var_pred = self.mf_model.predict(grid_points_w_fid)
            std_pred = np.sqrt(var_pred)
            
            # Filter and find max
            valid_preds = []
            for i in range(len(grid_points)):
                x, y = grid_points[i]
                if self.is_point_valid(x, y):
                    valid_preds.append({
                        'theta': (x, y),
                        'mean': mean_pred[i, 0],
                        'std': std_pred[i, 0]
                    })
            
            if valid_preds:
                valid_preds.sort(key=lambda x: x['mean'], reverse=True)
                best = valid_preds[0]
                
                print("\n" + "="*60)
                print("HIGHEST PREDICTED VALUE (MODEL GRID SEARCH)")
                print("="*60)
                print(f"Grid Resolution: {grid_res}x{grid_res} ({len(grid_points)} points)")
                print(f"Valid Points: {len(valid_preds)}")
                print(f"Theta (X, Y): ({best['theta'][0]:.4f}, {best['theta'][1]:.4f})")
                print(f"Predicted Mean: {best['mean']:.8f}")
                print(f"Predicted Std:  {best['std']:.8f}")
                
                print("\nTop 5 Valid Predictions:")
                for p in valid_preds[:5]:
                    print(f"  Theta ({p['theta'][0]:.2f}, {p['theta'][1]:.2f}): {p['mean']:.8f}")
            else:
                 print("No valid points found in grid search.")
                 
        except Exception as e:
            print(f"Error during grid search: {e}")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        if save_all_plots:
            print(f"All plots saved to: {self.output_dir}")
            
        return {
            'processed_data': processed_data,
            'predictions': predictions,
            'coverage_stats': coverage_stats
        }


def explore_predictions(predictions, file_name=None):
    """
    Utility function to explore prediction results interactively.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary returned by MFGPAnalyzer.predict_for_theta_groups()
    file_name : str, optional
        Specific file to explore (default: first file)
    """
    if file_name is None:
        file_name = list(predictions.keys())[0]
        
    print(f"Exploring predictions for: {file_name}")
    print("-" * 50)
    
    file_preds = predictions[file_name]
    
    for i, (combo_key, pred_data) in enumerate(file_preds.items()):
        print(f"\n{i+1}. Theta: {combo_key}")
        print(f"   Predicted mean: {pred_data['y_pred_mean']:.6f}")
        print(f"   Predicted std:  {pred_data['y_pred_std']:.6f}")
        print(f"   True values:    {len(pred_data['y_true'])} samples")
        print(f"   Value range:    [{np.min(pred_data['y_true']):.6f}, {np.max(pred_data['y_true']):.6f}]")
        
        # Calculate simple metrics
        residuals = pred_data['y_true'] - pred_data['y_pred_mean']
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        normalized_rmse = rmse / pred_data['y_pred_std']
        
        print(f"   MAE:            {mae:.6f}")
        print(f"   RMSE:           {rmse:.6f}")
        print(f"   Normalized RMSE: {normalized_rmse:.3f}σ")
