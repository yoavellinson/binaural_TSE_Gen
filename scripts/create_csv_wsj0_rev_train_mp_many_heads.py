# parallel_dataset_builder.py
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import pickle
import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ---- your imports ----
from reverbration_simulator.create_setup import run_single_simulation

# ----------------------
# Globals set in worker processes via initializer()
_SPEAKER_TO_WAVS = None
_SOFA_DB_TO_FILES = None
ADD_STR='_1k_mp_m'
def _init_worker(speaker_to_wavs, sofa_db_to_files, seed_base):
    """
    Per-process initializer: set module-level globals so jobs
    don't carry huge pickled payloads repeatedly.
    Also set an independent RNG seed per process.
    """
    global _SPEAKER_TO_WAVS, _SOFA_DB_TO_FILES
    _SPEAKER_TO_WAVS = speaker_to_wavs
    _SOFA_DB_TO_FILES = sofa_db_to_files
    # Make each process RNG independent
    rnd = random.Random()
    # A bit of entropy: pid + provided base
    rnd.seed(seed_base + os.getpid())
    # store as process-local
    random._inst = rnd  # not necessary, but keeps intent clear

def load_db(path: str):
    data = torch.load(path)
    return {'patches': data['patches'], 'pos': data['pos']}

def _pick_two_distinct(seq):
    """Pick two distinct items from a non-empty sequence."""
    a = random.choice(seq)
    b = random.choice(seq)
    while b == a:
        b = random.choice(seq)
    return a, b

def _select_angles(valid_pos_org,valid_pos_other_head):
    """
    Choose (az1, elev1, az2, elev2) respecting your constraints.
    This mirrors your logic but avoids while-loops that can spin too long
    by retrying a bounded number of times.
    """

    
    for _ in range(64):  # bounded attempts
        az1 = float(random.choice(valid_pos[:, 0]))
        valid_az = [float(az) for az in valid_pos[:, 0] if abs(float(az) - az1) >= 30.0]
        if not valid_az:
            continue
        az2 = float(random.choice(valid_az))

        elev1_candidates = [float(e) for e in valid_pos[valid_pos[:, 0] == az1][:, 1] if abs(float(e)) < 30.0]
        if not elev1_candidates:
            continue
        elev2_candidates = [float(e) for e in valid_pos[valid_pos[:, 0] == az2][:, 1] if abs(float(e)) < 20.0]
        if not elev2_candidates:
            continue

        elev1 = float(random.choice(elev1_candidates))
        elev2 = float(random.choice(elev2_candidates))

        # format to 3 decimals, then ensure float
        az1  = float(f"{az1:.3f}")
        az2  = float(f"{az2:.3f}")
        elev1 = float(f"{elev1:.3f}")
        elev2 = float(f"{elev2:.3f}")
        return az1, elev1, az2, elev2
    # If we failed to find a combo, signal failure
    return None

def _worker_job(task):
    """
    Worker function: builds one row (or returns None on failure).
    task: (j, root_sofas_path, root_pts_path)
    """
    j, root_sofas_path, root_pts_path = task
    try:
        # Choose speakers
        speaker_names = list(_SPEAKER_TO_WAVS.keys())
        if len(speaker_names) < 2:
            return None  # not enough speakers

        spk1, spk2 = _pick_two_distinct(speaker_names)
        s1 = random.choice(_SPEAKER_TO_WAVS[spk1])
        s2 = random.choice(_SPEAKER_TO_WAVS[spk2])

        # Choose a SOFA
        sofa_db = random.choice(list(_SOFA_DB_TO_FILES.keys()))
        sofa_path = random.choice(_SOFA_DB_TO_FILES[sofa_db])

        # Map to .pt
        pt_path = sofa_path.replace(str(root_sofas_path), str(root_pts_path)).replace('.sofa', '.pt').replace('/train_set','').replace('/test_set','')

        # Load PT for valid positions
        d = load_db(pt_path)
        valid_pos = d['pos']
        sel = _select_angles(valid_pos)
        if sel is None:
            return None
        az1, elev1, az2, elev2 = sel

        sir = float(np.random.uniform(-5, 5, 1)[0])

        # Run simulation (kept exactly as in your code)
        h1_rev, h1_zero, h2_rev, h2_zero, rt_60 = run_single_simulation(
            j=j,
            sofa_path=sofa_path,
            az1=az1, elev1=elev1,
            az2=az2, elev2=elev2,
            dir_add_str=ADD_STR
        )

        row = {
            "sofa_path": sofa_path,
            "pt_path": pt_path,
            "speaker_1": s1,
            "az_1": az1,
            "elev_1": elev1,
            "hrir_rev_1_path": str(h1_rev),
            "hrir_zero_1_path": str(h1_zero),
            "speaker_2": s2,
            "az_2": az2,
            "elev_2": elev2,
            "hrir_rev_2_path": str(h2_rev),
            "hrir_zero_2_path": str(h2_zero),
            "rt_60": float(rt_60),
            "sir": sir,
        }
        return row
    except Exception:
        # Swallow and return None so the parent can continue
        return None

def build_parallel_dataset(
    root_path="/dsi/gannot-lab/gannot-lab1/datasets/sharon_db/wsj0/Test/",
    root_sofas_path="/home/workspace/yoavellinson/binaural_TSE_Gen/sofas/test_set",
    root_pts_path="/home/workspace/yoavellinson/binaural_TSE_Gen/pts",
    output_csv=f"/home/workspace/yoavellinson/binaural_TSE_Gen/csvs/HRTF_test_many_heads_wsj0{ADD_STR}.csv",
    max_samples=1000,
    max_workers=None,
    write_every=200
):
    root_path = Path(root_path)
    root_sofas_path = Path(root_sofas_path)
    root_pts_path = Path(root_pts_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) Index speakers → wavs (only once)
    speakers_list = sorted([p for p in root_path.glob("*") if p.is_dir()])
    speaker_to_wavs = {}
    for spkdir in speakers_list:
        wavs = glob(str(spkdir / "*.wav"))
        if len(wavs) > 1:
            random.shuffle(wavs)
            speaker_to_wavs[spkdir.name] = wavs

    # 2) Index sofa dbs → sofa files (only once)
    sofa_db_to_files = {}
    for dbdir in root_sofas_path.glob("*"):
        if not dbdir.is_dir():
            continue
        sofas = glob(str(dbdir / "*.sofa"))
        if len(sofas) > 0:
            random.shuffle(sofas)
            sofa_db_to_files[dbdir.name] = sofas

    if len(speaker_to_wavs) < 2 or len(sofa_db_to_files) == 0:
        raise RuntimeError("Not enough speakers or SOFA files indexed.")

    # CSV header (write once)
    columns = ["sofa_path","pt_path","speaker_1","az_1","elev_1",
               "hrir_rev_1_path","hrir_zero_1_path",
               "speaker_2","az_2","elev_2",
               "hrir_rev_2_path","hrir_zero_2_path","rt_60","sir"]
    if not output_csv.exists():
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

    # 3) Build tasks
    tasks = [(j, str(root_sofas_path), str(root_pts_path)) for j in range(max_samples)]

    # 4) Parallel execution
    if max_workers is None:
        # Heuristic: sofamyroom is heavy → don't oversubscribe
        # Start with min(8, cpu_count//2 or >=1)
        cw = max(1, os.cpu_count() // 2 if os.cpu_count() else 2)
        max_workers = min(8, cw)

    # Optional: cap threads of any MKL/OpenMP libs in children
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    rows_buffer = []
    written = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(speaker_to_wavs, sofa_db_to_files, random.randrange(1_000_000_000))
    ) as ex:
        futures = [ex.submit(_worker_job, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Simulating"):
            row = fut.result()
            if row is not None:
                rows_buffer.append(row)

            # Flush in batches
            if len(rows_buffer) >= write_every:
                df = pd.DataFrame.from_records(rows_buffer, columns=columns)
                # Append without header
                df.to_csv(output_csv, mode="a", header=False, index=False)
                written += len(rows_buffer)
                rows_buffer.clear()

    # Final flush
    if rows_buffer:
        df = pd.DataFrame.from_records(rows_buffer, columns=columns)
        df.to_csv(output_csv, mode="a", header=False, index=False)
        written += len(rows_buffer)

    print(f"Done. Wrote {written} rows to {output_csv}")

if __name__ == "__main__":
    # Adjust as needed
    build_parallel_dataset(
        max_samples=1000,
        max_workers=1,       # try 4; tune up/down per machine
        write_every=1      # smaller batches → more frequent appends
    )
