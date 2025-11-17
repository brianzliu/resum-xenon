#!/usr/bin/env python3
"""
Extract the theta and standard deviation for the highest predicted value from MFGP analysis.
"""

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, '/home/tidmad/bliu/resum-xenon/src')

# Load configuration
with open("/home/tidmad/bliu/resum-xenon/src/xenon/settings.yaml", "r") as f:
    config_file = yaml.safe_load(f)

# Extract settings
version = config_file["path_settings"]["version"]
path_out_cnp = config_file["path_settings"]["path_out_cnp"]
path_out_mfgp = config_file["path_settings"]["path_out_mfgp"]
x_labels = config_file["simulation_settings"]["theta_headers"]
y_label_sim = 'y_raw'

# Convert to absolute paths
base_path = Path("/home/tidmad/bliu/resum-xenon/src/xenon")
if not Path(path_out_cnp).is_absolute():
    file_in = str(base_path / path_out_cnp / f'cnp_{version}_output_15epochs.csv')
else:
    file_in = f'{path_out_cnp}/cnp_{version}_output_15epochs.csv'

print("Configuration loaded:")
print(f"  Version: {version}")
print(f"  Parameter labels: {x_labels}")
print(f"  Input file: {file_in}")
print()

# Load data and setup
np.random.seed(42)
data = pd.read_csv(file_in)

# Get noise values for low-fidelity and high-fidelity
y_err_label_cnp = 'y_cnp_err'
LF_cnp_noise = np.mean(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_err_label_cnp].to_numpy())
HF_sim_noise = np.std(data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_label_sim].to_numpy())

# Filter for high-fidelity data
filtered_data = data.loc[(data['fidelity']==1.) & (data['iteration']==0)]

# Extract training data
x_train_hf_sim = filtered_data[x_labels].to_numpy().tolist()
y_train_hf_sim = filtered_data[y_label_sim].to_numpy().tolist()

# Extract low-fidelity training data
x_train_lf_cnp = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][x_labels].to_numpy().tolist()
y_train_lf_cnp = data.loc[(data['fidelity']==0.) & (data['iteration']==0)]['y_cnp'].to_numpy().tolist()

trainings_data = {"lf": [x_train_lf_cnp, y_train_lf_cnp], "hf": [x_train_hf_sim, y_train_hf_sim]}
noise = {"lf": LF_cnp_noise, "hf": HF_sim_noise * 0.001}

print(f"Training data loaded:")
print(f"  LF samples: {len(x_train_lf_cnp)}")
print(f"  HF samples: {len(x_train_hf_sim)}")
print()

# Now import and setup the MFGP model
import GPy
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays

# Prepare training data
x_train = []
y_train = []
for fidelity in ["lf", "hf"]:
    x_tmp = np.atleast_2d(trainings_data[fidelity][0])
    y_tmp = np.atleast_2d(trainings_data[fidelity][1]).T
    x_train.append(x_tmp)
    y_train.append(y_tmp)

X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)

# Build the model
num_fidelities = 2
kernels = [GPy.kern.Matern32(input_dim=X_train[0].shape[0] - 1),
           GPy.kern.Matern32(input_dim=X_train[0].shape[0] - 1)]

linear_mf_kernel = LinearMultiFidelityKernel(kernels)
gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, linear_mf_kernel, n_fidelities=num_fidelities)

# Set noise
gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(noise['lf'])
gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(noise['hf'])

# Set hyperparameter bounds
gpy_linear_mf_model['multifidelity.Mat32.variance'].constrain_bounded(1e-6, 1e2)
gpy_linear_mf_model['multifidelity.Mat32.lengthscale'].constrain_bounded(1e-2, 1e3)
gpy_linear_mf_model['multifidelity.Mat32_1.variance'].constrain_bounded(1e-6, 1e2)
gpy_linear_mf_model['multifidelity.Mat32_1.lengthscale'].constrain_bounded(1e-2, 1e3)
gpy_linear_mf_model['multifidelity.scale'].constrain_bounded(1e-3, 1e2)

if hasattr(gpy_linear_mf_model.mixed_noise.Gaussian_noise, 'unfix'):
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.unfix()
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.unfix()

    gpy_linear_mf_model.mixed_noise.Gaussian_noise.constrain_bounded(
        noise['lf'] * 0.15, noise['lf'] * 1.85)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.constrain_bounded(
        noise['hf'] * 0.15, noise['hf'] * 1.85)

# Wrap and optimize
print("Training the MFGP model (10 restarts)...")
lin_mf_model = GPyMultiOutputWrapper(gpy_linear_mf_model, num_fidelities,
                                     n_optimization_restarts=10, verbose_optimization=False)
lin_mf_model.optimize()
print("Model training complete!")
print()

# Generate continuous predictions on a grid (search across whole space)
print("Generating predictions on continuous theta grid (searching entire space)...")

# Get theta bounds from settings
theta_min = config_file["simulation_settings"]["theta_min"]
theta_max = config_file["simulation_settings"]["theta_max"]

# Create a fine grid for continuous predictions
# Use higher resolution for better accuracy
scint_x_vals = np.linspace(theta_min[0], theta_max[0], 50)
scint_y_vals = np.linspace(theta_min[1], theta_max[1], 50)  # Full range

print(f"  scint_x range: [{theta_min[0]}, {theta_max[0]}] ({len(scint_x_vals)} points)")
print(f"  scint_y range: [{theta_min[1]}, {theta_max[1]}] ({len(scint_y_vals)} points)")

predictions = {}
all_points = []
all_means = []
all_stds = []

for scint_y in scint_y_vals:
    for scint_x in scint_x_vals:
        theta = np.array([scint_x, scint_y])

        # Prepare prediction input (add fidelity=1 for high-fidelity prediction)
        x_pred = np.array([list(theta) + [1.0]])

        # Get model predictions
        mean_pred, var_pred = lin_mf_model.predict(x_pred)
        std_pred = np.sqrt(var_pred)

        theta_key = tuple(np.round(theta, 6))  # Round for key to avoid floating point issues
        predictions[theta_key] = {
            'theta': theta,
            'mean': mean_pred[0, 0],
            'std': std_pred[0, 0]
        }

        all_points.append(theta)
        all_means.append(mean_pred[0, 0])
        all_stds.append(std_pred[0, 0])

all_means = np.array(all_means)
all_stds = np.array(all_stds)

print(f"Generated predictions for {len(predictions)} continuous theta combinations")
print()

# Find the theta with the highest predicted mean
max_pred = max(predictions.items(), key=lambda x: x[1]['mean'])
max_theta_key = max_pred[0]
max_pred_data = max_pred[1]

print("="*70)
print("HIGHEST PROBABILITY PREDICTION (CONTINUOUS GRID SEARCH)")
print("="*70)
print(f"\nTheta ({x_labels[0]}, {x_labels[1]}): ({max_pred_data['theta'][0]:.4f}, {max_pred_data['theta'][1]:.4f})")
print(f"Predicted Mean: {max_pred_data['mean']:.8f}")
print(f"Predicted Std Dev (σ): {max_pred_data['std']:.8f}")
print(f"95% Confidence Interval: [{max_pred_data['mean'] - 1.96*max_pred_data['std']:.8f}, {max_pred_data['mean'] + 1.96*max_pred_data['std']:.8f}]")
print()

# Show grid statistics
print("="*70)
print("CONTINUOUS GRID STATISTICS")
print("="*70)
print(f"Grid resolution: {len(scint_x_vals)} x {len(scint_y_vals)} = {len(predictions)} points")
print(f"Mean predictions range: [{np.min(all_means):.8f}, {np.max(all_means):.8f}]")
print(f"Std predictions range: [{np.min(all_stds):.8f}, {np.max(all_stds):.8f}]")
print(f"Mean of all predictions: {np.mean(all_means):.8f}")
print(f"Std of all predictions: {np.std(all_means):.8f}")
print()

# Show top 10 predictions for context
print("="*70)
print("TOP 10 PREDICTIONS (CONTINUOUS GRID)")
print("="*70)
sorted_preds = sorted(predictions.items(), key=lambda x: x[1]['mean'], reverse=True)
for i, (theta_key, pred_data) in enumerate(sorted_preds[:10], 1):
    print(f"\n{i}. Theta (scint_x={pred_data['theta'][0]:.4f}, scint_y={pred_data['theta'][1]:.4f})")
    print(f"   Mean: {pred_data['mean']:.8f}")
    print(f"   Std:  {pred_data['std']:.8f}")

print("\n" + "="*70)
