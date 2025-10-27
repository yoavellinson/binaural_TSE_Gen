from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class PatchDBDataset(Dataset):
    def __init__(self, hp,train=True):
        df = hp.dataset.train_csv_path if train else hp.dataset.test_csv_path
        self.paths = pd.read_csv(df)

    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        d = load_db(self.paths.iloc[idx].path)
        patches = d['patches']    # [N, 4, 513] or [N, 2, 513] — adapt here
        pos     = d['pos']        # [N, 2]
        # Ensure [C,N,T]
        # if patches.dim()==3 and patches.shape[1] in (2,4):   # [N,C,T]
        #     patches = patches.permute(1,0,2)                 # [C,N,T]
        # elif patches.dim()==4 and patches.shape[0] in (2,4): # already [C,N,T]?
        #     patches = patches.squeeze(0)
        C, N, T = patches.shape
        kpm = torch.zeros(N, dtype=torch.bool)
        return {"patches": patches, "pos": pos, "kpm": kpm}

def load_db(path):
    data = torch.load(path)
    return {'patches':data['patches'],'pos':data['pos']}

def collate_simple(batch):
    # same N per sample? then stack directly; else pad to max N and build kpm
    Cs = {b["patches"].shape[1] for b in batch}
    Ts = {b["patches"].shape[-1] for b in batch}
    assert len(Cs)==1 and len(Ts)==1, "Mixed C/T not supported in this simple collate"
    C = next(iter(Cs)); T = next(iter(Ts))
    Ns = [b["patches"].shape[0] for b in batch]
    Nmax = max(Ns)
    B = len(batch)
    x = torch.zeros(B, Nmax,C, T)
    xy= torch.zeros(B, Nmax, 2)
    kpm = torch.ones(B, Nmax, dtype=torch.bool)
    for i,b in enumerate(batch):
        N = b["patches"].shape[0]
        x[i, :N, :]  = b["patches"]
        xy[i, :N]    = b["pos"]
        kpm[i, :N]   = False
    return {"patches": x, "pos": xy, "kpm": kpm}
