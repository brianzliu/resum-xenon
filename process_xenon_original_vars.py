import pandas as pd
import numpy as np
from pathlib import Path
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os

# ------------- Global config / perf knobs -------------
MAX_WORKERS = max(1, os.cpu_count() or 1)
READ_ENGINE = "pyarrow"  # fallback to C if pyarrow is missing; pandas will handle
PD_CSV_KW = dict(engine=READ_ENGINE)  # merged into read_csv calls
SHUFFLE_SEED_DEFAULT = None  # set in combine_* if desired
# ------------------------------------------------------

# ------------- Precompiled regexes --------------------
PAT_TPCX = re.compile(r'(?:TPC)?X(\d+)Y(\d+)')
PAT_SIM  = re.compile(r'sim_X(\d+)_Y(\d+)_task')
PAT_HF_DIR = re.compile(r'X(\d+)_Y(\d+)')
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

def _detect_columns(sample_cols):
    """
    Decide which pre/post columns exist (Scintor vs TPC variants).
    Returns (pre_x, pre_y, pre_z, post_x, post_y, post_z).
    """
    if 'pre_x_mm' in sample_cols:
        return ('pre_x_mm', 'pre_y_mm', 'pre_z_mm', 'post_x_mm', 'post_y_mm', 'post_z_mm')
    else:
        return ('vrt_x_mm', 'vrt_y_mm', 'vrt_z_mm', 'x_mm', 'y_mm', 'z_mm')

def _needed_columns(tag_col, sample_cols):
    """Compute the minimal set of columns to read."""
    pre_x, pre_y, pre_z, post_x, post_y, post_z = _detect_columns(sample_cols)
    cols = {
        'eventid', 'energy_keV', 'time_ns', tag_col,
        pre_x, pre_y, pre_z, post_x, post_y, post_z
    }
    return list(cols)

def _read_csv_needed(path: Path, tag_col: str):
    """
    Read only necessary columns; detect columns with a tiny sniff (nrows=1).
    Auto-detects which tag column exists ('tag' or 'creatpro').
    """
    # quick sniff (use default engine since pyarrow doesn't support nrows)
    sniff = pd.read_csv(path, nrows=1)

    # Check if first column is unnamed (like in TPCHF files)
    has_unnamed_index = (len(sniff.columns) > 0 and
                        (sniff.columns[0] == 'Unnamed: 0' or sniff.columns[0] == ''))

    # Determine which columns to read
    if has_unnamed_index:
        # For files with unnamed index, read with index_col=0 and use remaining columns
        sniff_indexed = pd.read_csv(path, nrows=1, index_col=0)
        cols_to_check = sniff_indexed.columns
        index_col_param = 0
    else:
        # For normal files, don't use index_col
        cols_to_check = sniff.columns
        index_col_param = None

    # Auto-detect which tag column exists in this file
    actual_tag_col = tag_col
    if tag_col not in cols_to_check:
        # Try alternative tag columns
        if tag_col == 'tag' and 'creatpro' in cols_to_check:
            actual_tag_col = 'creatpro'
        elif tag_col == 'creatpro' and 'tag' in cols_to_check:
            actual_tag_col = 'tag'

    usecols = _needed_columns(actual_tag_col, cols_to_check)

    # dtypes help: all coords/energy are float, eventid int, tag_col small int
    dtype_map = {c: 'float32' for c in usecols if c.endswith('_mm') or c.endswith('_keV') or c.endswith('_ns')}
    if 'energy_keV' in usecols: dtype_map['energy_keV'] = 'float32'
    if 'time_ns' in usecols: dtype_map['time_ns'] = 'float32'
    if 'eventid' in usecols: dtype_map['eventid'] = 'int64'
    if actual_tag_col in usecols: dtype_map[actual_tag_col] = 'int8'

    # Read full file
    if index_col_param is not None:
        # When using index_col, don't use usecols to avoid complications
        # Just read all columns with appropriate dtypes and filter afterward
        df = pd.read_csv(path, dtype=dtype_map, index_col=index_col_param)
        # Keep only the columns we need
        cols_to_keep = [c for c in usecols if c in df.columns]
        df = df[cols_to_keep]
    else:
        # Try pyarrow for better performance, fallback to default engine
        try:
            df = pd.read_csv(path, usecols=usecols, dtype=dtype_map, **PD_CSV_KW)
        except:
            df = pd.read_csv(path, usecols=usecols, dtype=dtype_map)

    # Rename tag column to standard name if it was different
    if actual_tag_col != tag_col and actual_tag_col in df.columns:
        df = df.rename(columns={actual_tag_col: tag_col})

    return df

def _aggregate_events(df: pd.DataFrame, tag_col: str, scint_x: int, scint_y: int, case: str = 'both') -> pd.DataFrame:
    """
    Extract first row per eventid with original position variables:
      - grab the first row per eventid for initial xyz, final xyz, energy, and time
      - compute tag_final based on case:
        'both': tag_final = 1 if BOTH tag 1 AND tag 2 exist for this eventid
        'only1': tag_final = 1 if ONLY tag 1 exists (not tag 2)
        'only2': tag_final = 1 if ONLY tag 2 exists (not tag 1)
    """
    pre_x, pre_y, pre_z, post_x, post_y, post_z = _detect_columns(df.columns)

    # First rows per event
    # Using drop_duplicates is very fast; keep='first' preserves order
    firsts = df.drop_duplicates('eventid', keep='first')[[
        'eventid', pre_x, pre_y, pre_z, post_x, post_y, post_z, 'energy_keV', 'time_ns'
    ]].copy()

    # Rename for output schema (standardized names)
    firsts = firsts.rename(columns={
        pre_x: 'x_init_mm',
        pre_y: 'y_init_mm',
        pre_z: 'z_init_mm',
        post_x: 'x_final_mm',
        post_y: 'y_final_mm',
        post_z: 'z_final_mm'
    })

    # Tag aggregation per eventid
    if tag_col not in df.columns:
        raise ValueError(f"Column '{tag_col}' not found in DataFrame.")

    tags = df[['eventid', tag_col]].copy()
    tags['_is1'] = (tags[tag_col] == 1)
    tags['_is2'] = (tags[tag_col] == 2)
    ag = tags.groupby('eventid', sort=False)[['_is1', '_is2']].any().reset_index()
    
    # Compute tag_final based on case
    if case == 'both':
        # tag_final = 1 only if BOTH tag 1 AND tag 2 exist for this eventid
        ag['tag_final'] = (ag['_is1'] & ag['_is2']).astype('int8')
    elif case == 'only1':
        # tag_final = 1 if ONLY tag 1 exists (has 1 but not 2)
        ag['tag_final'] = (ag['_is1'] & ~ag['_is2']).astype('int8')
    elif case == 'only2':
        # tag_final = 1 if ONLY tag 2 exists (has 2 but not 1)
        ag['tag_final'] = (~ag['_is1'] & ag['_is2']).astype('int8')
    else:
        raise ValueError(f"Unknown case: {case}")
    
    ag = ag[['eventid', 'tag_final']]

    # Merge first rows with tag aggregation
    out = firsts.merge(ag, on='eventid', how='left')
    out['tag_final'] = out['tag_final'].fillna(0).astype('int8')

    # Add scint coords
    out.insert(1, 'scint_x', scint_x)
    out.insert(2, 'scint_y', scint_y)

    # Order columns
    out = out[['eventid', 'scint_x', 'scint_y',
               'x_init_mm', 'y_init_mm', 'z_init_mm',
               'x_final_mm', 'y_final_mm', 'z_final_mm',
               'energy_keV', 'time_ns', 'tag_final']]

    return out

def _process_single_csv(path: Path, tag_col: str, case: str = 'both', coord_from_name=True):
    """Read/process one CSV and return DataFrame + info."""
    if coord_from_name:
        scint_x, scint_y = extract_coordinates_from_filename(path.name)
    else:
        # For ScintorHF we parse from parent dir name X##_Y##
        m = PAT_HF_DIR.search(path.parent.name)
        if not m:
            raise ValueError(f"Could not extract XY from dir {path.parent}")
        scint_x, scint_y = int(m.group(1)), int(m.group(2))

    df = _read_csv_needed(path, tag_col)
    if df.empty:
        return pd.DataFrame(), 0

    out = _aggregate_events(df, tag_col, scint_x, scint_y, case=case)
    total_signals = int(out['tag_final'].sum())
    return out, total_signals

# Module-level worker functions for multiprocessing
def _work_scintorlf(args):
    """Worker for ScintorLF processing."""
    p, tag_col, case, output_dir = args
    out, signals = _process_single_csv(p, tag_col, case=case, coord_from_name=True)
    if not out.empty:
        out.to_csv(output_dir / p.name, index=False)
    return p.name, len(out), signals

def _work_tpclf(args):
    """Worker for TPCLF processing."""
    p, tag_col, case, output_dir = args
    out, signals = _process_single_csv(p, tag_col, case=case, coord_from_name=True)
    if not out.empty:
        out.to_csv(output_dir / p.name, index=False)
    return p.name, len(out), signals

def _work_tpchf(args):
    """Worker for TPCHF processing."""
    p, tag_col, case, output_dir = args
    out, signals = _process_single_csv(p, tag_col, case=case, coord_from_name=True)
    if not out.empty:
        out.to_csv(output_dir / p.name, index=False)
    return p.name, len(out), signals

def _work_scintorhf(args):
    """Worker for ScintorHF processing."""
    p, x, y, tag_col, case = args
    df = _read_csv_needed(p, tag_col)
    if df.empty:
        return p.name, None, 0
    out = _aggregate_events(df, tag_col, x, y, case=case)
    signals = int(out['tag_final'].sum())
    return p.name, out, signals

def process_scintorlf(case='both'):
    case_name = case
    base_input = Path('/home/tidmad/bliu/XENON/ScintorLF')
    base_output = Path(f'/home/tidmad/bliu/resum-xenon/temp_new_data/{case_name}')
    lf_output = base_output / 'lf'
    lf_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nScintorLF - Case {case}\n{'='*70}")

    csv_files = sorted(base_input.glob('*.csv'))
    if not csv_files:
        return

    tag_col = 'creatpro'
    args_list = [(p, tag_col, case, lf_output) for p in csv_files]

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

def process_tpclf(case='both'):
    case_name = case
    base_input = Path('/home/tidmad/bliu/XENON/TPCLF')
    base_output = Path(f'/home/tidmad/bliu/resum-xenon/temp_new_data/{case_name}')
    temp_output = base_output / '.tpclf_temp'
    temp_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nTPCLF - Case {case}\n{'='*70}")

    csv_files = sorted(base_input.glob('*.csv'))
    if not csv_files:
        return

    tag_col = 'creatpro'
    args_list = [(p, tag_col, case, temp_output) for p in csv_files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_work_tpclf, args): args[0] for args in args_list}
        for fut in as_completed(futures):
            try:
                name, nrows, sig = fut.result()
                print(f"  {name}: {nrows} events (signals: {sig})")
            except Exception as e:
                print(f"  Error {futures[fut].name}: {e}")
                raise

def process_scintorhf(case='both'):
    case_name = case
    base_input = Path('/home/tidmad/bliu/XENON/ScintorHF')
    base_output = Path(f'/home/tidmad/bliu/resum-xenon/temp_new_data/{case_name}')
    hf_output = base_output / 'hf_temp_scintor'
    hf_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nScintorHF - Case {case}\n{'='*70}")

    # Each subdir X##_Y## contains many CSVs. We process per-file in parallel.
    targets = []
    for xy_dir in sorted(base_input.glob('X*_Y*')):
        if not xy_dir.is_dir():
            continue
        m = PAT_HF_DIR.search(xy_dir.name)
        if not m:
            continue
        scint_x, scint_y = int(m.group(1)), int(m.group(2))
        for csv_file in xy_dir.glob('*.csv'):
            targets.append((csv_file, scint_x, scint_y))

    if not targets:
        return

    tag_col = 'creatpro'
    args_list = [(p, x, y, tag_col, case) for (p, x, y) in targets]

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

def process_tpchf(case='both'):
    case_name = case
    base_input = Path('/home/tidmad/bliu/XENON/TPCHF')
    base_output = Path(f'/home/tidmad/bliu/resum-xenon/temp_new_data/{case_name}')
    temp_output = base_output / '.tpchf_temp'
    temp_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nTPCHF - Case {case}\n{'='*70}")

    csv_files = sorted(base_input.glob('*.csv'))
    if not csv_files:
        return

    tag_col = 'tag'
    args_list = [(p, tag_col, case, temp_output) for p in csv_files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_work_tpchf, args): args[0] for args in args_list}
        for fut in as_completed(futures):
            try:
                name, nrows, sig = fut.result()
                print(f"  {name}: {nrows} events (signals: {sig})")
            except Exception as e:
                print(f"  Error {futures[fut].name}: {e}")
                raise

def combine_lf_data(case='both', shuffle=True, seed=SHUFFLE_SEED_DEFAULT):
    """Combine TPCLF + ScintorLF (already processed), optionally shuffle, and save."""
    case_name = case
    base_output = Path(f'/home/tidmad/bliu/resum-xenon/temp_new_data/{case_name}')
    lf_output = base_output / 'lf'
    temp_tpclf = base_output / '.tpclf_temp'

    print(f"\n{'='*70}\nCombining LF Data - Case {case}\n{'='*70}")

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

def combine_hf_data(case='both', shuffle=True, seed=SHUFFLE_SEED_DEFAULT):
    """Combine ScintorHF + TPCHF and save to final location."""
    case_name = case
    base_output = Path(f'/home/tidmad/bliu/resum-xenon/temp_new_data/{case_name}')
    hf_output = base_output / 'hf'
    hf_temp_scintor = base_output / 'hf_temp_scintor'
    hf_temp_tpchf = base_output / '.tpchf_temp'
    hf_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nCombining HF Data - Case {case}\n{'='*70}")

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
            if n_scintor > 0: tag.append(f"{n_scintor} (ScintorHF)")
            tag_str = " + ".join(tag) if tag else "0"
            suff = " [shuffled]" if shuffle else ""
            print(f"  X={x}, Y={y}: {tag_str} = {len(combined)}{suff}")

    # Cleanup
    shutil.rmtree(hf_temp_scintor, ignore_errors=True)
    shutil.rmtree(hf_temp_tpchf, ignore_errors=True)


def main():
    print("\n" + "="*70)
    print("XENON Original Variables Preprocessing Pipeline")
    print("Processing cases: both (1 AND 2), only1, only2")
    print("="*70)
 
    cases = ['only1', 'only2', 'both'] 
    
    for case in cases:
        print(f"\n\n{'#'*70}\nCASE: {case}\n{'#'*70}")

        # Process all raw data in parallel where possible
        process_scintorlf(case)
        process_tpclf(case)
        process_scintorhf(case)
        process_tpchf(case)

        # Combine and shuffle
        combine_lf_data(case, shuffle=True, seed=SHUFFLE_SEED_DEFAULT)
        combine_hf_data(case, shuffle=True, seed=SHUFFLE_SEED_DEFAULT)

    print("\n" + "="*70)
    print("All processing complete!")
    print("="*70)

if __name__ == '__main__':
    main()
