# RESuM for XENON Experiment
## File Overview
### Data
The data is stored somewhat weirdly; in resum-xenon/src/xenon/in/data you'll find 4 different experiments.
- only1 - Set preprocessed y value to be 1 when one raw neutron event's rows contains y value 1 (but not 2), otherwise 0 (out of raw y values 0, 1, or 2); this corresponds to neutron depositing energy in scintillator but not reaching TPC
- only2 - Set preprocessed y value to be 1 when one raw neutron event's rows contains y value 2 (but not 1), otherwise 0. This corresponds to neutron entering TPC, but not reaching scintillator.
- original_vars - Set preprocessed y value to be 1 when one raw neutron event's rows contains both 1 and 2, otherwise 0. This corresponds to neutron both reaching scintillator and TPC (the metric we most ultimately care about!)
(more on the meaning of 0, 1, 2 in Instructions/Data Preprocessing)

Inside each of these folders you'll find:
- training/hf
- training/lf
- validation/lf
For validation, since we don't have enough training data for validation (only 3 configurations), I allocated ~10% of LF configurations across a wide range of the design space (theta) for validation instead.

Each hf/lf subfolder contains many csv files—1 csv file per configuration/simulation. Each csv file corresponds with a .h5 file, which is needed to run the CNP. 

### CNP
Inside src/run_cnp find the training and prediction Jupyter Notebooks. Running these notebooks alone will train and predict the CNP, and you don't need to run anything else. 

However, if CNP training takes a while, you can run the command "nohup python cnp_training.py > output.log 2>&1 &" instead to run it in the background (you can even close VSCode). If you choose this route, please also run "python preprocess_mixup.py", as this is not automated with cnp_training.py. After training, run conditional_neural_process_predict_xenon.ipynb as usual. Results are saved to src/xenon/out/cnp.

### MFGP
Inside src/run_mfgp find the main file, mfgp_xenon.ipynb, which "trains" the MFGP and returns RESuM's final results and visualizations. Similar to CNP, if training takes a while you can run run_mfgp.py. Results will be saved to src/xenon/out/mfgp.

The graphs generated in mfgp_xenon.ipynb are from using mfgp_visualizations.py—feel free to modify as appropriate. 

To get the theta/configuration RESuM's highest predicted y_raw (along with the predicted y_raw itself, stdevs, thetas with next highest predicted y_raws), run extract_highest_prediction.py. 


## Instructions
### Data Preprocessing
After importing the (original) XENON dataset, you'll find the "XENON" folder with contents looking something like this:
- ScintorHF
- ScintorLF
- TPCHF
- TPCLF
- Data_introduce.py

Feel free to explore the data. Every unique event id represents one simulated neutron. There are multiple timesteps (time_ns), each with initial and ending position. (This is represented either as ('pre_x_mm', 'pre_y_mm', 'pre_z_mm', 'post_x_mm', 'post_y_mm', 'post_z_mm') or ('vrt_x_mm', 'vrt_y_mm', 'vrt_z_mm', 'x_mm', 'y_mm', 'z_mm') respectively) depending on the file. We also have kinetic energy (energy_keV), and the raw y value ('creatpro' or 'tag' depending on the file).

To preprocess the data, run process_xenon_original_vars.py and modify the line
"cases = ['only1', 'only2', 'both']" 
depending on whether you want to process only1, only2, and/or (1 and 2). (For the definitions of these see File Overview/Data).

After process_xenon_original_vars.py, run convert_csv_to_h5.py. Hardcode the path to only1 / only2 / original_vars (1 and 2) folder inside convert_csv_to_h5.py accordingly. 

### CNP Training/Prediction
For the CNP, make sure before training you are referencing the correct file path to train on; depending on the dataset you're using, change
path_to_files_train: "../xenon/in/data/[dataset]/training/lf/"
in settings.yaml. Model weights will be saved in src/xenon/out/cnp, as mentioned in File Overview.

You have to run prediction 2 times: 1 for the MFGP training data, 1 for the MFGP validation data (the ~10% separately allocated LF). I.e. set
path_to_files_predict: ["../xenon/in/data/[dataset]/training/lf", "../xenon/in/data/[dataset]/training/hf/"]
and run prediction once. Then set
path_to_files_predict: ["../xenon/in/data/[dataset]/validation/lf"] 
and run prediction once more. I put code in conditional_neural_process_predict.xenon.ipynb that saves the filename depending on which subset you are predicting on. The graphs generated for both subsets will also be saved in src/xenon/out/cnp.

### MFGP Fitting
For the MFGP, run mfgp_xenon.ipynb. Running all the cells in order should do the trick. Just be mindful that some of the files' filenames being imported/saved are hardcoded (particularly at the very beginning and at the end when using mfgp_visualizations.py), so change the file paths as appropriate. 

## Post scriptum
- For the lab PC's tidmad user, there's a Miniconda environment called "coherent" already in place with the necessary Python packages; feel free to use that instead of creating from scratch using xenon_environment.yml.

- If you get some error about GBLICXX not being found when running .py file in your terminal, please paste and run the following code: export LD_LIBRARY_PATH=~/miniconda3/envs/coherent/lib:\${LD_LIBRARY_PATH:-}
(or replace "coherent" with the name of your own conda env)

- The first cell in some files that imports sklearn will sometimes fail the first time, just re-run the cell again—don't know why it does that.

- The file paths in the Python files are slightly hardcoded (especially the suffix e.g. "_15epochs"), as I found it easy to do that to keep track of running the same experiment with different hyperparameters.


