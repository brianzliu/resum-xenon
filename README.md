# RESuM for XENON Experiment

## File Overview

### Data
The data is stored somewhat weirdly; in `resum-xenon/src/xenon/in/data` you'll find 4 different experiments. Although the old data directories are noted below, **the rest of the `README.md` will focus on instructions for processing the new data.** Find that the old preprocessing scripts are included for redundancy (i.e. those without the suffix "2"), while the main scripts like `conditional_neural_process_training_xenon.ipynb` are modified for the new data.

#### Old Data
- `only1` - Set preprocessed y value to be 1 when one raw neutron event's rows contains y value 1 (but not 2), otherwise 0 (out of raw y values 0, 1, or 2); this corresponds to neutron depositing energy in scintillator but not reaching TPC

- `only2` - Set preprocessed y value to be 1 when one raw neutron event's rows contains y value 2 (but not 1), otherwise 0. This corresponds to neutron entering TPC, but not reaching scintillator.

- `both` - Set preprocessed y value to be 1 when one raw neutron event's rows contains both 1 and 2, otherwise 0. This corresponds to neutron both reaching scintillator and TPC (the metric we most ultimately care about!)

#### New Data
- **`new_both`** - Set preprocessed y value to be 1 when one raw neutron event's rows contains both 1 and 2, otherwise 0. This corresponds to neutron both reaching scintillator and TPC (the metric we most ultimately care about!)

(more on the meaning of 0, 1, 2 in Instructions/Data Preprocessing)

Inside each of these folders you'll find:
- `training/hf`
- `training/lf`
- `validation/hf` (for the **new data**)
- `validation/lf`

For validation, since we don't have enough HF data for validation (only 3 configurations), I allocated ~10% of LF configurations across a wide range of the design space (theta) for validation instead.

Each HF/LF subfolder contains many csv files—1 csv file per configuration/simulation. Each csv file corresponds with a `.h5` file, which is needed to run the CNP. 

### CNP
Inside `src/run_cnp` find the training and prediction Jupyter Notebooks. Running these notebooks alone will train and predict the CNP, and you don't need to run anything else. 

However, if CNP training takes a while, you can run the command below instead to run it in the background (you can even close VSCode):

```
nohup python cnp_training.py > output.log 2>&1 &
```

If you choose this route, please also run `python preprocess_mixup_xenon2.py` before `cnp_training.py`, as this is not automated with `cnp_training.py`. After training, run `conditional_neural_process_predict_xenon.ipynb` as usual. Results are saved to `src/xenon/out/cnp`.

### MFGP
Inside `src/run_mfgp` find the main file, `mfgp_xenon.ipynb`, which "trains" the MFGP and returns RESuM's final results and visualizations. Similar to CNP, if training takes a while you can run `run_mfgp.py`. Results will be saved to `src/xenon/out/mfgp`.

The graphs generated in `mfgp_xenon.ipynb` are from using `mfgp_visualizations.py`—feel free to modify as appropriate. 
To get the theta/configuration RESuM's highest predicted y_raw (along with the predicted y_raw itself, stdevs, thetas with next highest predicted y_raws), run `extract_highest_prediction.py`. 


## Instructions

### Data Preprocessing
After importing the XENON dataset, you'll find the "XENON" folder with contents looking something like this:
- `ScintorHF`
- `ScintorLF`
- `TPCHF`
- `TPCLF`

Feel free to explore the data. The dataset is actually comprised of both `ReSUME2v2.tar.gz` and `ScintorHF.tar.gz` in the shared OneDrive, so be sure to merge them.

- Every unique event id represents one simulated neutron. 
- The rest of the columns represent the neutron's momentum at its generation (initial_m_(x,y,z)), the point it passes through the scintillator (second_m_(x,y,z)), and the point it passes through the TPC (third_m_(x,y,z)). Through this interpretation, the co-existence of second_m_(x,y,z) and third_m_(x,y,z) for a given neutron signifies the rare event that we are looking for.

To preprocess the data, run `process_xenon2.py`. The resulting columns are `eventid`, `scint_x`, `scint_y`, `initial_m_x`, `initial_m_y`, `initial_m_z`, and `tag_final`, corresponding to event ID, scintillator pos., neutron gun pos., initial momentum (X, Y, Z), and the y variable. The LF and HF files should show up in `temp_new_data/lf` and `temp_new_data/hf`, respectively. Then, run `split_data.py` to perform a training/validation split, which simultaneously copies the CSVs to `src/xenon/in/data/new_both/`.  

After `split_data.py`, run `convert_csv_to_h5_xenon2.py`. Hardcode the path to the `new_both` folder inside `convert_csv_to_h5_xenon2.py` accordingly. 

### CNP Training/Prediction
For the CNP, make sure before training you are referencing the correct file path to train on; depending on the dataset you're using, change the following block in `settings2.yaml`:

```
path_to_files_train: "../xenon/in/data/[dataset]/training/lf/"
```

Model weights will be saved in `src/xenon/out/cnp`, as mentioned in File Overview.

You have to run prediction 2 times: 1 for the MFGP training data, 1 for the MFGP validation data (the ~10% separately allocated LF). I.e. set

```
path_to_files_predict: ["../xenon/in/data/[dataset]/training/lf", "../xenon/in/data/[dataset]/training/hf/"]
```

and run prediction once. Then set

```
path_to_files_predict: ["../xenon/in/data/[dataset]/validation/lf"]
```

and run prediction once more. I put code in `conditional_neural_process_predict.xenon.ipynb` that saves the filename depending on which subset you are predicting on. The graphs generated for both subsets will also be saved in `src/xenon/out/cnp`.

### MFGP Fitting
For the MFGP, run `mfgp_xenon.ipynb`. Running all the cells in order should do the trick. Just be mindful that some of the files' filenames being imported/saved are hardcoded (particularly at the very beginning and at the end when using `mfgp_visualizations.py`), so change the file paths as appropriate. 

## Post scriptum
- For the lab PC's tidmad user, there's a Miniconda environment called "coherent" already in place with the necessary Python packages; feel free to use that instead of creating from scratch using `xenon_environment.yml`.

- If you get some error about GBLICXX not being found when running a `.py` file in your terminal, please paste and run the following code:

```
export LD_LIBRARY_PATH=~/miniconda3/envs/coherent/lib:${LD_LIBRARY_PATH:-}
```

(or replace "coherent" with the name of your own conda env)

- The first cell in some files that imports `sklearn` will sometimes fail the first time, just re-run the cell again—don't know why it does that.

- The file paths in the Python files are slightly hardcoded (especially the suffix e.g. `_15epochs`), as I found it easy to do that to keep track of running the same experiment with different hyperparameters.


