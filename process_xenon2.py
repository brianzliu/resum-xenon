import pandas as pd
import numpy as np
from pathlib import Path
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ------------- Global config / perf knobs -------------
MAX_WORKERS = max(1, os.cpu_count() or 1)
READ_ENGINE = "pyarrow"
PD_CSV_KW = dict(engine=READ_ENGINE)
SHUFFLE_SEED_DEFAULT = None
# ------------------------------------------------------

# ------------- Precompiled regexes --------------------
PAT_TPCX = re.compile(r'(?:TPC)?X(\d+)_?Y(\d+)')
PAT_SIM  = re.compile(r'sim_X(\d+)_Y(\d+)_task')
# Match both old format (ScintorHFX##Y##) and new format (X##_Y##)
PAT_HF_DIR_OLD = re.compile(r'ScintorHFX(\d+)Y(\d+)')
PAT_HF_DIR_NEW = re.compile(r'X(\d+)_Y(\d+)')
PAT_SIMXY = re.compile(r'sim_X(\d+)_Y(\d+)')
# ------------------------------------------------------

def extract_coordinates_from_filename(filename: str):
    """Extract X and Y from filename."""
    m = PAT_TPCX.search(filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = PAT_SIM.search(filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Could not extract coordinates from {filename}")

def _read_csv_needed(path: Path):
    """
    Read only necessary columns: eventid, initial_m_{x,y,z}, second_m_x, third_m_x.
    """
    usecols = ['eventid', 'initial_m_x', 'initial_m_y', 'initial_m_z', 'second_m_x', 'third_m_x']
    
    # dtypes
    dtype_map = {
        'eventid': 'int64',
        'initial_m_x': 'float32',
        'initial_m_y': 'float32',
        'initial_m_z': 'float32',
        'second_m_x': 'float32',
        'third_m_x': 'float32'
    }

    try:
        # We might encounter files where some columns are missing (e.g. if no second/third interaction at all?)
        # But based on 'head', the columns exist even if empty (NaN).
        # However, if the file is truly empty or headers are different, we should handle it.
        # For now, assume headers are consistent as per 'head' check.
        df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype=dtype_map, **PD_CSV_KW)
    except:
        df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype=dtype_map)

    return df

def _aggregate_events(df: pd.DataFrame, scint_x: int, scint_y: int) -> pd.DataFrame:
    """
    Process events:
      - Keep eventid, initial_m_{x,y,z}
      - Compute tag_final: 1 if BOTH second_m_x and third_m_x are not NaN, else 0.
    """
    # Create output dataframe
    # We assume 1 row per eventid in this new dataset format
    out = df[['eventid', 'initial_m_x', 'initial_m_y', 'initial_m_z']].copy()
    
    # Compute tag_final
    # Check if second_m_x and third_m_x are not null
    has_second = df['second_m_x'].notna()
    has_third = df['third_m_x'].notna()
    
    out['tag_final'] = (has_second & has_third).astype('int8')

    # Add scint coords
    out.insert(1, 'scint_x', scint_x)
    out.insert(2, 'scint_y', scint_y)

    return out[['eventid', 'scint_x', 'scint_y', 'initial_m_x', 'initial_m_y', 'initial_m_z', 'tag_final']]

def _process_single_csv(path: Path, coord_from_name=True):
    """Read/process one CSV and return DataFrame + info."""
    if coord_from_name:
        scint_x, scint_y = extract_coordinates_from_filename(path.name)
    else:
        # For ScintorHF we parse from parent dir name - handle both old and new formats
        m = PAT_HF_DIR_NEW.search(path.parent.name)
        if not m:
            m = PAT_HF_DIR_OLD.search(path.parent.name)
        if not m:
            raise ValueError(f"Could not extract XY from dir {path.parent}")
        scint_x, scint_y = int(m.group(1)), int(m.group(2))

    df = _read_csv_needed(path)
    if df.empty:
        return pd.DataFrame(), 0

    out = _aggregate_events(df, scint_x, scint_y)
    total_signals = int(out['tag_final'].sum())
    return out, total_signals

# Module-level worker functions for multiprocessing
def _work_scintorlf(args):
    """Worker for ScintorLF processing."""
    p, output_dir = args
    out, signals = _process_single_csv(p, coord_from_name=True)
    if not out.empty:
        out.to_csv(output_dir / p.name, index=False)
    return p.name, len(out), signals

def _work_tpclf(args):
    """Worker for TPCLF processing."""
    p, output_dir = args
    out, signals = _process_single_csv(p, coord_from_name=True)
    if not out.empty:
        out.to_csv(output_dir / p.name, index=False)
    return p.name, len(out), signals

def _work_tpchf(args):
    """Worker for TPCHF processing."""
    p, output_dir = args
    out, signals = _process_single_csv(p, coord_from_name=True)
    if not out.empty:
        out.to_csv(output_dir / p.name, index=False)
    return p.name, len(out), signals

def _work_scintorhf(args):
    """Worker for ScintorHF processing."""
    p, x, y = args
    df = _read_csv_needed(p)
    if df.empty:
        return p.name, None, 0
    out = _aggregate_events(df, x, y)
    signals = int(out['tag_final'].sum())
    return p.name, out, signals

def process_scintorlf():
    base_input = Path('/home/tidmad/bliu/ReSUM2/ScintillatorLF')
    base_output = Path('/home/tidmad/bliu/resum-xenon/temp_new_data')
    lf_output = base_output / 'lf'
    lf_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nScintillatorLF\n{'='*70}")

    csv_files = sorted(base_input.glob('*.csv'))
    if not csv_files:
        print("No files found.")
        return

    args_list = [(p, lf_output) for p in csv_files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_work_scintorlf, args): args[0] for args in args_list}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                name, nrows, sig = fut.result()
                if i <= 3 or i % 50 == 0:
                    print(f"  {name}: {nrows} events (signals: {sig})")
            except Exception as e:
                print(f"  Error {futures[fut].name}: {e}")
                raise

def process_tpclf():
    base_input = Path('/home/tidmad/bliu/ReSUM2/TPCLF')
    base_output = Path('/home/tidmad/bliu/resum-xenon/temp_new_data')
    temp_output = base_output / '.tpclf_temp'
    temp_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nTPCLF\n{'='*70}")

    csv_files = sorted(base_input.glob('*.csv'))
    if not csv_files:
        print("No files found.")
        return

    args_list = [(p, temp_output) for p in csv_files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_work_tpclf, args): args[0] for args in args_list}
        for fut in as_completed(futures):
            try:
                name, nrows, sig = fut.result()
                print(f"  {name}: {nrows} events (signals: {sig})")
            except Exception as e:
                print(f"  Error {futures[fut].name}: {e}")
                raise

def process_scintorhf():
    base_input = Path('/home/tidmad/bliu/ReSUM2/ScintillatorHF')
    base_output = Path('/home/tidmad/bliu/resum-xenon/temp_new_data')
    hf_output = base_output / 'hf_temp_scintor'
    hf_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nScintillatorHF\n{'='*70}")

    targets = []
    # Look for all subdirectories - both old format (ScintorHFX##Y##) and new format (X##_Y##)
    for xy_dir in sorted(base_input.iterdir()):
        if not xy_dir.is_dir():
            continue
        
        # Try new format first (X##_Y##)
        m = PAT_HF_DIR_NEW.search(xy_dir.name)
        if not m:
            # Try old format (ScintorHFX##Y##)
            m = PAT_HF_DIR_OLD.search(xy_dir.name)
        if not m:
            continue
        
        scint_x, scint_y = int(m.group(1)), int(m.group(2))
        for csv_file in xy_dir.glob('*.csv'):
            targets.append((csv_file, scint_x, scint_y))

    if not targets:
        print("No files found.")
        return

    args_list = [(p, x, y) for (p, x, y) in targets]

    grouped_out = {}  # key=(x,y) -> list of partial dfs
    signals_by_xy = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_work_scintorhf, args): args[:3] for args in args_list}
        for fut in as_completed(futures):
            p, x, y = futures[fut]
            try:
                name, out, sig = fut.result()
                if out is None:
                    continue
                grouped_out.setdefault((x, y), []).append(out)
                signals_by_xy[(x, y)] = signals_by_xy.get((x, y), 0) + sig
            except Exception as e:
                print(f"    Error {p.name}: {e}")
                raise

    # Write one combined file per (x,y)
    for (x, y), parts in sorted(grouped_out.items()):
        combined = pd.concat(parts, ignore_index=True)
        out_path = hf_output / f'sim_X{x}_Y{y}_task0_combined.csv'
        combined.to_csv(out_path, index=False)
        print(f"  X={x}, Y={y}: {len(combined)} events (signals: {signals_by_xy[(x,y)]})")

def process_tpchf():
    base_input = Path('/home/tidmad/bliu/ReSUM2/TPCHF')
    base_output = Path('/home/tidmad/bliu/resum-xenon/temp_new_data')
    temp_output = base_output / '.tpchf_temp'
    temp_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nTPCHF\n{'='*70}")

    csv_files = sorted(base_input.glob('*.csv'))
    if not csv_files:
        print("No files found.")
        return

    args_list = [(p, temp_output) for p in csv_files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_work_tpchf, args): args[0] for args in args_list}
        for fut in as_completed(futures):
            try:
                name, nrows, sig = fut.result()
                print(f"  {name}: {nrows} events (signals: {sig})")
            except Exception as e:
                print(f"  Error {futures[fut].name}: {e}")
                raise

def combine_lf_data(shuffle=True, seed=SHUFFLE_SEED_DEFAULT):
    """Combine TPCLF + ScintorLF (already processed), optionally shuffle, and save."""
    base_output = Path('/home/tidmad/bliu/resum-xenon/temp_new_data')
    lf_output = base_output / 'lf'
    temp_tpclf = base_output / '.tpclf_temp'

    print(f"\n{'='*70}\nCombining LF Data\n{'='*70}")

    if not temp_tpclf.exists():
        return

    tpclf_files = sorted(temp_tpclf.glob('*.csv'))
    for tpclf_file in tpclf_files:
        scint_x, scint_y = extract_coordinates_from_filename(tpclf_file.name)

        scintlf_file = lf_output / tpclf_file.name
        tpclf_df = pd.read_csv(tpclf_file, **PD_CSV_KW)

        if scintlf_file.exists():
            scintlf_df = pd.read_csv(scintlf_file, **PD_CSV_KW)
            combined_df = pd.concat([scintlf_df, tpclf_df], ignore_index=True)
        else:
            combined_df = tpclf_df

        if shuffle:
            combined_df = combined_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        combined_df.to_csv(scintlf_file, index=False)
        print(f"  {tpclf_file.name}: {len(combined_df)} rows {'[shuffled]' if shuffle else ''}")

    shutil.rmtree(temp_tpclf, ignore_errors=True)

def combine_hf_data(shuffle=True, seed=SHUFFLE_SEED_DEFAULT):
    """Combine ScintorHF + TPCHF and save to final location."""
    base_output = Path('/home/tidmad/bliu/resum-xenon/temp_new_data')
    hf_output = base_output / 'hf'
    hf_temp_scintor = base_output / 'hf_temp_scintor'
    hf_temp_tpchf = base_output / '.tpchf_temp'
    hf_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nCombining HF Data\n{'='*70}")

    # Index the temp files by (x,y)
    scintor_files = {}
    if hf_temp_scintor.exists():
        for f in sorted(hf_temp_scintor.glob('*.csv')):
            m = PAT_SIMXY.search(f.name)
            if m:
                scintor_files[(int(m.group(1)), int(m.group(2)))] = f

    tpchf_files = {}
    if hf_temp_tpchf.exists():
        for f in sorted(hf_temp_tpchf.glob('*.csv')):
            m = PAT_TPCX.search(f.name)
            if m:
                tpchf_files[(int(m.group(1)), int(m.group(2)))] = f

    all_keys = sorted(set(scintor_files) | set(tpchf_files))

    for x, y in all_keys:
        frames = []
        n_tpchf = n_scintor = 0
        if (x, y) in tpchf_files:
            df = pd.read_csv(tpchf_files[(x, y)], **PD_CSV_KW)
            n_tpchf = len(df)
            frames.append(df)
        if (x, y) in scintor_files:
            df = pd.read_csv(scintor_files[(x, y)], **PD_CSV_KW)
            n_scintor = len(df)
            frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            if shuffle:
                combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            out = hf_output / f'sim_X{x}_Y{y}_ALL.csv'
            combined.to_csv(out, index=False)
            tag = []
            if n_tpchf > 0: tag.append(f"{n_tpchf} (TPCHF)")
            if n_scintor > 0: tag.append(f"{n_scintor} (ScintillatorHF)")
            tag_str = " + ".join(tag) if tag else "0"
            suff = " [shuffled]" if shuffle else ""
            print(f"  X={x}, Y={y}: {tag_str} = {len(combined)}{suff}")

    # Cleanup
    shutil.rmtree(hf_temp_scintor, ignore_errors=True)
    shutil.rmtree(hf_temp_tpchf, ignore_errors=True)


def main():
    print("\n" + "="*70)
    print("XENON ReSUM2 Processing Pipeline")
    print("Processing with new logic: tag_final = 1 if (second_m & third_m) else 0")
    print("="*70)
 
    # Process all raw data in parallel where possible
    process_scintorlf()
    process_tpclf()
    process_scintorhf()
    process_tpchf()

    # Combine and shuffle
    combine_lf_data(shuffle=True, seed=SHUFFLE_SEED_DEFAULT)
    combine_hf_data(shuffle=True, seed=SHUFFLE_SEED_DEFAULT)

    print("\n" + "="*70)
    print("All processing complete!")
    print("="*70)

if __name__ == '__main__':
    main()
