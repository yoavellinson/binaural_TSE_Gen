#!/usr/bin/env python3
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

def check_pt_files(root):
    root = Path(root)
    bad = []

    for path in root.rglob("*.pt"):
        try:
            # Force load to CPU to avoid CUDA initialization issues
            torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[BAD] {path}")
            print(f"      Error: {e}")
            bad.append(path)

    print("\n=== SUMMARY ===")
    print(f"Total .pt files checked: {len(list(root.rglob('*.pt')))}")
    print(f"Bad files: {len(bad)}")

    if bad:
        print("\nList of bad files:")
        for b in bad:
            print(b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check .pt files in a directory tree.")
    parser.add_argument("directory", type=str, default='/home/workspace/yoavellinson/binaural_TSE_Gen/ae_res' , help="Root directory to scan")
    args = parser.parse_args()

    check_pt_files(args.directory)
