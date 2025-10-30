from pathlib import Path
import shutil
import random

src_root = Path("/home/workspace/yoavellinson/binaural_TSE_Gen/sofas")   # directory with subfolders
dst_root = Path("/home/workspace/yoavellinson/binaural_TSE_Gen/test_set")  # where to move files
dst_root.mkdir(exist_ok=True)

# loop over each subdirectory
for subdir in src_root.iterdir():
    if subdir.is_dir():
        files = list(subdir.glob("*"))
        if not files:
            continue
        # choose one file randomly (or pick files[0] if you prefer)
        file_to_move = random.choice(files)
        # create matching subdir in target
        target_subdir = dst_root / subdir.name
        target_subdir.mkdir(exist_ok=True)
        # move the file
        shutil.move(str(file_to_move), target_subdir / file_to_move.name)
        print(f"Moved {file_to_move} → {target_subdir / file_to_move.name}")
