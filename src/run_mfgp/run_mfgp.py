# %% [markdown]
# ## Followed this notebook: https://github.com/EmuKit/emukit/blob/main/notebooks/Emukit-tutorial-multi-fidelity.ipynb

# %%
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import importlib.util
import random
random.seed(42)

import GPy
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

# %%
with open("../xenon/settings.yaml", "r") as f:
    config_file = yaml.safe_load(f)

version       = config_file["path_settings"]["version"]
path_out_cnp  = config_file["path_settings"]["path_out_cnp"]
path_out_mfgp = config_file["path_settings"]["path_out_mfgp"]
file_in=f'{path_out_cnp}/cnp_{version}_output_20epochs.csv'

# %%
# data processing/setup
np.random.seed(42)

if not os.path.exists(path_out_mfgp):
   os.makedirs(path_out_mfgp)

# Set parameter name/x_labels -> needs to be consistent with data input file
x_labels        = config_file["simulation_settings"]["theta_headers"]
y_label_cnp     = 'y_cnp'
y_err_label_cnp = 'y_cnp_err'
y_label_sim     = 'y_raw'

# %%
data=pd.read_csv(file_in)

LF_cnp_noise=np.mean(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_err_label_cnp].to_numpy())
HF_cnp_noise=np.mean(data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_err_label_cnp].to_numpy())
LF_sim_noise=np.std(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_label_sim].to_numpy())
HF_sim_noise=np.std(data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_label_sim].to_numpy())

# Get the filtered dataframe first
filtered_data = data.loc[(data['fidelity']==1.) & (data['iteration']==0)]

# # Get unique combinations of x_label values to select diverse training points
# unique_x_combinations = filtered_data[x_labels].drop_duplicates()

# # Select up to 3 diverse training points based on different x_label combinations
# if len(unique_x_combinations) >= 3:
#     # Select 3 points with most diverse x values
#     # Get min, max, and a middle point for each x dimension
#     x_data = unique_x_combinations.values
    
#     # Find points with min/max values for first dimension (water_shielding_mm)
#     min_x1_idx = np.argmin(x_data[:, 0])
#     max_x1_idx = np.argmax(x_data[:, 0])
    
#     # Find a point with different x2 value (veto_thickness_mm) that's not min/max x1
#     remaining_indices = [i for i in range(len(x_data)) if i not in [min_x1_idx, max_x1_idx]]
#     if remaining_indices:
#         # Select point with most different x2 value from the min/max x1 points
#         x1_values = x_data[[min_x1_idx, max_x1_idx], 1]
#         mid_x2_idx = remaining_indices[np.argmax([abs(x_data[i, 1] - np.mean(x1_values)) for i in remaining_indices])]
#     else:
#         mid_x2_idx = min_x1_idx if min_x1_idx != max_x1_idx else 0
    
#     selected_combinations = unique_x_combinations.iloc[[min_x1_idx, max_x1_idx, mid_x2_idx]]
# else:
#     # If fewer than 3 unique combinations, use all available
#     selected_combinations = unique_x_combinations

# # Find the indices in filtered_data that match these selected combinations
# train_indices = []
# for _, combo in selected_combinations.iterrows():
#     # Find first occurrence of this combination in filtered_data
#     mask = (filtered_data[x_labels[0]] == combo[x_labels[0]]) & (filtered_data[x_labels[1]] == combo[x_labels[1]])
#     matching_indices = filtered_data[mask].index.tolist()
#     if matching_indices:
#         train_indices.append(matching_indices[0])  # Take first occurrence

# test_indices = filtered_data.index.difference(train_indices)



# filtered_data = data.loc[(data['fidelity']==1.) & (data['iteration']==0)]

# unique_x_combinations = filtered_data[x_labels].drop_duplicates().values

# combination_1 = []
# combination_2 = []
# combination_3 = []

# samples_with_combination_1 = filtered_data.loc[filtered_data[x_labels].values==unique_x_combinations[0]]
# combination_1.extend(list(set(samples_with_combination_1.index.to_list())))
# samples_with_combination_2 = filtered_data.loc[filtered_data[x_labels].values==unique_x_combinations[1]]
# combination_2.extend(list(set(samples_with_combination_2.index.to_list())))
# samples_with_combination_3 = filtered_data.loc[filtered_data[x_labels].values==unique_x_combinations[2]]
# combination_3.extend(list(set(samples_with_combination_3.index.to_list())))

# random.shuffle(combination_1)
# random.shuffle(combination_2)
# random.shuffle(combination_3)

# combination_1_70 = combination_1[:int(len(combination_1) // (10/9))]
# combination_1_30 = combination_1[int(len(combination_1) // (10/9)):]
# combination_2_70 = combination_2[:int(len(combination_2) // (10/9))]
# combination_2_30 = combination_2[int(len(combination_2) // (10/9)):]
# combination_3_70 = combination_3[:int(len(combination_3) // (10/9))]
# combination_3_30 = combination_3[int(len(combination_3) // (10/9)):]


# # Extract training data
# x_train_hf_sim = filtered_data.loc[train_indices][x_labels].to_numpy().tolist()
# y_train_hf_sim = filtered_data.loc[train_indices][y_label_sim].to_numpy().tolist()

# # Extract testing data
# x_test_hf_sim = filtered_data.loc[test_indices][x_labels].to_numpy().tolist()
# y_test_hf_sim = filtered_data.loc[test_indices][y_label_sim].to_numpy().tolist()

# Approach 2
# Extract training data
x_train_hf_sim = filtered_data[x_labels].to_numpy().tolist()
y_train_hf_sim = filtered_data[y_label_sim].to_numpy().tolist()

# # Extract testing data
# x_test_hf_sim = filtered_data[x_labels].to_numpy().tolist()
# y_test_hf_sim = filtered_data[y_label_sim].to_numpy().tolist()

# ## Approach 3
# # Extract training data
# x_train_hf_sim = filtered_data.loc[combination_1_70][x_labels].to_numpy().tolist()
# x_train_hf_sim.extend(filtered_data.loc[combination_2_70][x_labels].to_numpy().tolist())
# x_train_hf_sim.extend(filtered_data.loc[combination_3_70][x_labels].to_numpy().tolist())
# y_train_hf_sim = filtered_data.loc[combination_1_70][y_label_sim].to_numpy().tolist()
# y_train_hf_sim.extend(filtered_data.loc[combination_2_70][y_label_sim].to_numpy().tolist())
# y_train_hf_sim.extend(filtered_data.loc[combination_3_70][y_label_sim].to_numpy().tolist())
# combined_train_hf_sim = list(zip(x_train_hf_sim, y_train_hf_sim))
# random.shuffle(combined_train_hf_sim)
# x_train_hf_sim, y_train_hf_sim = zip(*combined_train_hf_sim)
# x_train_hf_sim = list(x_train_hf_sim)
# y_train_hf_sim = list(y_train_hf_sim)

# # Extract testing data
# x_test_hf_sim = filtered_data.loc[combination_1_30][x_labels].to_numpy().tolist()
# x_test_hf_sim.extend(filtered_data.loc[combination_2_30][x_labels].to_numpy().tolist())
# x_test_hf_sim.extend(filtered_data.loc[combination_3_30][x_labels].to_numpy().tolist())
# y_test_hf_sim = filtered_data.loc[combination_1_30][y_label_sim].to_numpy().tolist()
# y_test_hf_sim.extend(filtered_data.loc[combination_2_30][y_label_sim].to_numpy().tolist())
# y_test_hf_sim.extend(filtered_data.loc[combination_3_30][y_label_sim].to_numpy().tolist())
# combined_test_hf_sim = list(zip(x_test_hf_sim, y_test_hf_sim))
# random.shuffle(combined_test_hf_sim)
# x_test_hf_sim, y_test_hf_sim = zip(*combined_test_hf_sim)
# x_test_hf_sim = list(x_test_hf_sim)
# y_test_hf_sim = list(y_test_hf_sim)

x_train_lf_cnp = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][x_labels].to_numpy().tolist()
y_train_lf_cnp = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_label_cnp].to_numpy().tolist()


trainings_data = {"lf": [x_train_lf_cnp,y_train_lf_cnp], "hf": [x_train_hf_sim,y_train_hf_sim]}#, } "mf": [x_train_hf_cnp,y_train_hf_cnp]
noise = {"lf": LF_cnp_noise, "hf": HF_sim_noise*0.001}#, "hf": 0.0}  # why were mf and hf noise originally set to 0?
# noise = {"lf": 1.7e-6, "hf": 1.7e-6}

# %%
fidelities = list(trainings_data.keys())
nfidelities = len(fidelities)

# %%
x_train = []
y_train = []
for fidelity in fidelities:
    x_tmp=np.atleast_2d(trainings_data[fidelity][0])
    y_tmp=np.atleast_2d(trainings_data[fidelity][1]).T
    x_train.append(x_tmp)
    y_train.append(y_tmp)

X_train, Y_train = convert_xy_lists_to_arrays(x_train, y_train)

# %%
num_fidelities = 2  # just lf and hf for now
kernels = [GPy.kern.Matern32(input_dim=X_train[0].shape[0] - 1), GPy.kern.Matern32(input_dim=X_train[0].shape[0] - 1)]  # since there are two theta parameters, input_dim is 2
# kernels = [GPy.kern.RBF(input_dim=X_train[0].shape[0] - 1), GPy.kern.RBF(input_dim=X_train[0].shape[0] - 1)]

linear_mf_kernel = LinearMultiFidelityKernel(kernels)
gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, linear_mf_kernel, n_fidelities = num_fidelities)

# set noise
gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(noise['lf'])  # lf noise
gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(noise['hf'])  # mf/hf noise

# %%
# SET KERNEL HYPERPARAMETER BOUNDS
# Low-fidelity kernel (Mat32)
gpy_linear_mf_model['multifidelity.Mat32.variance'].constrain_bounded(1e-6, 1e2)
gpy_linear_mf_model['multifidelity.Mat32.lengthscale'].constrain_bounded(1e-2, 1e3)

# High-fidelity kernel (Mat32_1) 
gpy_linear_mf_model['multifidelity.Mat32_1.variance'].constrain_bounded(1e-6, 1e2)
gpy_linear_mf_model['multifidelity.Mat32_1.lengthscale'].constrain_bounded(1e-2, 1e3)

# # Low-fidelity kernel (RBF)
# gpy_linear_mf_model['multifidelity.rbf.variance'].constrain_bounded(1e-6, 1e2)
# gpy_linear_mf_model['multifidelity.rbf.lengthscale'].constrain_bounded(1e-2, 1e3)

# # High-fidelity kernel (RBF_1) 
# gpy_linear_mf_model['multifidelity.rbf_1.variance'].constrain_bounded(1e-6, 1e2)
# gpy_linear_mf_model['multifidelity.rbf_1.lengthscale'].constrain_bounded(1e-2, 1e3)

# Scale parameter (correlation between fidelities)
gpy_linear_mf_model['multifidelity.scale'].constrain_bounded(1e-3, 1e1)

# If you can unfix noise, increase it slightly
if hasattr(gpy_linear_mf_model.mixed_noise.Gaussian_noise, 'unfix'):
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.unfix()
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.unfix()
    
    # Set to 1.2x your original noise values with some bounds
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.constrain_bounded(
        noise['lf'] * 0.99, noise['lf'] * 1.01)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.constrain_bounded(
        noise['hf'] * 0.99, noise['hf'] * 1.01)

# %%
'''# More aggressive bounds that encourage higher uncertainty
gpy_linear_mf_model['multifidelity.Mat32.variance'].constrain_bounded(1e-5, 1e3)
gpy_linear_mf_model['multifidelity.Mat32_1.variance'].constrain_bounded(1e-5, 1e3)
gpy_linear_mf_model['multifidelity.Mat32.lengthscale'].constrain_bounded(1e-4, 1e2)
gpy_linear_mf_model['multifidelity.Mat32_1.lengthscale'].constrain_bounded(1e-4, 1e2)

# Allow the scale parameter more freedom
gpy_linear_mf_model['multifidelity.scale'].constrain_bounded(1e-2, 1e2)

# If you can unfix noise, increase it slightly
if hasattr(gpy_linear_mf_model.mixed_noise.Gaussian_noise, 'unfix'):
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.unfix()
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.unfix()
    
    # Set to 1.2x your original noise values with some bounds
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.constrain_bounded(
        noise['lf'] * 0.85, noise['lf'] * 1.15)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.constrain_bounded(
        noise['hf'] * 0.85, noise['hf'] * 1.15)'''

# %%
## Wrap the model using the given 'GPyMultiOutputWrapper'
lin_mf_model = GPyMultiOutputWrapper(gpy_linear_mf_model, num_fidelities, n_optimization_restarts=10, verbose_optimization=True)

## Fit the model
lin_mf_model.optimize()

# %%
import importlib
import sys

if 'mfgp_analyzer' in sys.modules:
    del sys.modules['mfgp_analyzer']

# Import the automated analysis pipeline
from mfgp_visualizations import MFGPAnalyzer, explore_predictions

print("MFGPAnalyzer imported successfully")
print("All analysis results will be saved to the output folder")

# %%
# Initialize the automated analyzer
# All plots and results will be saved to path_out_mfgp
analyzer = MFGPAnalyzer(
    mf_model=lin_mf_model,
    x_labels=x_labels,
    y_label_sim=y_label_sim,
    output_dir=path_out_mfgp  # This ensures all results are saved to the configured output folder
)

print("MFGPAnalyzer initialized successfully")
print(f"Output directory: {analyzer.output_dir}")
print(f"Parameter labels: {analyzer.x_labels}")
print(f"Target variable: {analyzer.y_label_sim}")
print("All plots and analysis results will be automatically saved")

# %%
current_file_results = analyzer.run_complete_analysis(
    file_patterns=["../xenon/out/cnp/cnp_v1.0_output_validation_20epochs.csv"],  # The file we used for training
    fidelity_filter=0.0,     # High fidelity data
    iteration_filter=0,      # First iteration
    plot_individual_groups=True,  # Plot each theta combination
    save_all_plots=True      # 💾 Save ALL generated plots to output folder
)

print("Complete. Check your output folder for all saved plots and results.")


