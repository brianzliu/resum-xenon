# CNP Training with Mixup Preprocessing

This guide explains how to run CNP training with mixup data augmentation using multiprocessing.

## Problem

When running `cnp_training.py` with `nohup` and multiprocessing enabled (`number_of_walkers > 0`), you may encounter:

```
BlockingIOError: [Errno 11] Unable to synchronously open file
(unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')
```

This happens because:
- Mixup augmentation opens HDF5 files in append mode during training initialization
- DataLoader worker processes try to read the same files simultaneously
- HDF5 file locking prevents concurrent access → error

## Solution: Two-Step Workflow

### Step 1: Run Mixup Preprocessing (One Time)

Run the preprocessing script to apply mixup augmentation to all HDF5 files:

```bash
cd /home/tidmad/bliu/resum-xenon/src/run_cnp

# Interactive
python preprocess_mixup.py

# Or with nohup
nohup python preprocess_mixup.py > mixup_output.log 2>&1 &
```

**What it does:**
- Reads settings from `../xenon/settings.yaml`
- Applies mixup augmentation to all training HDF5 files
- Creates `phi_mixedup`, `target_mixedup`, and `weights_mixedup` datasets
- Skips files that already have mixup applied with the same settings

**This only needs to be run once**, or whenever you:
- Change the `signal_condition` in settings.yaml
- Change the `use_beta` distribution parameters
- Add new training data files

### Step 2: Run Training with Multiprocessing

After preprocessing completes, you can run training with multiprocessing enabled:

```bash
# Make sure number_of_walkers is set to 1 or higher in settings.yaml
# number_of_walkers: 1

# Interactive
python cnp_training.py

# Or with nohup
nohup python cnp_training.py > training_output.log 2>&1 &
```

**What happens:**
- Training script detects that mixup datasets already exist
- Skips re-running mixup augmentation (fast startup!)
- Uses the pre-generated augmented data
- DataLoader workers can safely read the HDF5 files in parallel

## Settings

In `../xenon/settings.yaml`:

```yaml
cnp_settings:
  number_of_walkers: 1  # 0 = no multiprocessing, 1+ = parallel data loading
  use_data_augmentation: "mixup"
  use_beta: [0.1, 0.1]  # Beta distribution parameters for mixup ratio

simulation_settings:
  signal_condition: ["tag_final==1"]  # Condition to identify signal events
```

## Why This Works

**Without preprocessing:**
```
Training script starts
  → Runs mixup (opens files in 'a' mode)
  → Immediately creates DataLoader with workers
  → Workers try to open same files in 'r' mode
  → FILE LOCK CONFLICT ❌
```

**With preprocessing:**
```
Preprocessing script runs
  → Runs mixup once (opens files in 'a' mode)
  → Completes and closes all files
  → Exits

Training script starts (later)
  → Detects mixup already done, skips it
  → Creates DataLoader with workers
  → Workers open files in 'r' mode
  → No conflicts, all files already closed ✅
```

## Troubleshooting

### Error: `prefetch_factor option could only be specified in multiprocessing`

This is fixed automatically in the code. If you still see it, make sure you're using the updated `data_generator.py`.

### Training still shows "Data Augmentation in Progress"

If the training script still runs mixup augmentation, it means:
- Mixup datasets don't exist in the HDF5 files, OR
- Signal conditions changed

Solution: Run `preprocess_mixup.py` again.

### Want to re-run mixup with different parameters

1. Update `use_beta` or `signal_condition` in settings.yaml
2. Run `preprocess_mixup.py` - it will detect the changed settings and re-apply mixup

## Alternative: No Multiprocessing

If you don't need multiprocessing, simply set:

```yaml
number_of_walkers: 0
```

Then you can run training directly without preprocessing:

```bash
python cnp_training.py
```

Mixup will run during training initialization (single process, no conflicts).
