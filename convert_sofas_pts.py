#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import concurrent.futures as cf
import traceback

# Cap per-process BLAS/OMP threads (prevents oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
# CHANGE THIS IMPORT to wherever your SOFA reader lives
from hrtf_convolve import SOFA_HRTF_db


def process_one(sofa_path_str: str, root_in_str: str, root_out_str: str, overwrite: bool) -> str:
    """
    Runs in a separate process. Don't share objects; pass only strings/ints.
    """
    sofa_path = Path(sofa_path_str)
    root_in   = Path(root_in_str)
    root_out  = Path(root_out_str) if root_out_str else None
    try:
        # Mirror directory structure under output root (or save alongside)
        if root_out:
            rel      = sofa_path.resolve().relative_to(root_in.resolve())
            out_path = (root_out / rel).with_suffix(".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = sofa_path.with_suffix(".pt")

        if out_path.exists() and not overwrite:
            return f"[skip] {out_path}"

        # Open/read in this process (important for HDF5 thread-safety)
        db = SOFA_HRTF_db(str(sofa_path))
        patches, pos = db.get_positions()

        patches = torch.as_tensor(patches).detach().cpu().contiguous()
        pos     = torch.as_tensor(pos).detach().cpu().contiguous()

        torch.save({"patches": patches, "pos": pos}, out_path, _use_new_zipfile_serialization=True)
        return f"[ok] {out_path}"
    except Exception as e:
        return f"[error] {sofa_path}: {e}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r","--root_in",default='/home/workspace/yoavellinson/binaural_TSE_Gen/sofas', type=Path, help="Root directory to scan for .sofa files")
    ap.add_argument("-o", "--root_out", type=Path, default='/home/workspace/yoavellinson/binaural_TSE_Gen/pts', help="Output root; mirrors structure")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="Processes to use")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root_in = args.root_in.resolve()
    root_out = args.root_out.resolve() if args.root_out else None

    sofa_files = list(root_in.rglob("*.sofa"))
    print(f"Found {len(sofa_files)} SOFA files under {root_in}")

    if not sofa_files:
        return

    # On some platforms you may need spawn:
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    with cf.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = [
            ex.submit(process_one, str(p), str(root_in), str(root_out) if root_out else "", args.overwrite)
            for p in sofa_files
        ]
        for f in cf.as_completed(futures):
            msg = f.result()
            print(msg)

if __name__ == "__main__":
    main()
