from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from hrtf_convolve import SOFA_HRTF_wrapper
import torchaudio.functional as F
import soundfile as sf
from scipy import signal
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

class JoinedDataset(Dataset):
    """
    Combines two datasets sharing the same CSV/row order.
    Left side (db_) is dict from PatchDBDataset.
    Right side (mix_) is tuple (Mix, Y1, Y2) from ExtractionDatasetRevVAE, converted to dict.
    """
    def __init__(self, ds_db: Dataset, ds_mix: Dataset, prefix_db='db_', prefix_mix='mix_'):
        assert len(ds_db) == len(ds_mix), "Datasets must have equal length for index-wise join"
        self.ds_db = ds_db
        self.ds_mix = ds_mix
        self.prefix_db = prefix_db
        self.prefix_mix = prefix_mix

    def __len__(self):
        return len(self.ds_db)

    def __getitem__(self, i):
        A = self.ds_db[i]                       # dict: patches/pos/kpm
        B = self.ds_mix[i]                      # tuple: (Mix, Y1, Y2) or dict
        if not isinstance(B, dict):
            Mix, Y1, Y2 = B
            B = {"mix": Mix, "y1": Y1, "y2": Y2}
        A = {self.prefix_db + k: v for k, v in A.items()}
        B = {self.prefix_mix + k: v for k, v in B.items()}
        return {**A, **B}

def collate_joined(batch):
    # Split into db_ and mix_ sub-dicts
    db_batch  = [{k[3:]: v for k, v in b.items() if k.startswith("db_")}  for b in batch]
    mix_batch = [{k[4:]: v for k, v in b.items() if k.startswith("mix_")} for b in batch]

    db_part  = collate_simple(db_batch)                             # padded [B, Nmax, C, T], etc.
    mix_part = default_collate(mix_batch)                           # handles complex tensors too

    # Re-namespace keys for clarity
    out = {f"db_{k}": v for k, v in db_part.items()}
    out.update({f"mix_{k}": v for k, v in mix_part.items()})
    return out

class PatchDBDataset(Dataset):
    def __init__(self, hp, train=True):
        df = hp.dataset.train_csv_path if train else hp.dataset.test_csv_path
        self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)  # <- fix: was len(self.paths)

    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        d = load_db(line['pt_path'])
        patches = d['patches']    # expect [N, C, T]
        pos     = d['pos']        # [N, 2]
        N, C, T = patches.shape 
        kpm = torch.zeros(N, dtype=torch.bool)

        #get hrtfs from az/elev
        az1 = line['az_1']
        az2 = line['az_2']

        elev1 = line['elev_1']
        elev2 = line['elev_2']

        idx1 = int(((pos[:, 0] == az1) & (pos[:, 1] == elev1)).nonzero(as_tuple=True)[0])
        idx2 = int(((pos[:, 0] == az2) & (pos[:, 1] == elev2)).nonzero(as_tuple=True)[0])

        hrtf1 = patches[idx1]
        hrtf2 = patches[idx2]

        return {"patches": patches, "pos": pos, "kpm": kpm,'hrtf1':hrtf1,'hrtf2':hrtf2}

def load_db(path):
    data = torch.load(path)
    return {'patches': data['patches'], 'pos': data['pos']}

def collate_simple(batch):
    """
    batch[i] = {
      "patches": [N_i, C, T],
      "pos":     [N_i, 2],
      "kpm":     [N_i] (bool),
      "hrtf1":   [2*C, T] (real-concat),
      "hrtf2":   [2*C, T] (real-concat),
    }
    Returns:
      patches: [B, Nmax, C, T]
      pos:     [B, Nmax, 2]
      kpm:     [B, Nmax] (True = padded)
      hrtf1:   [B, C, T] (complex64)
      hrtf2:   [B, C, T] (complex64)
    """
    # --- sanity / common shapes ---
    Cs = {b["patches"].shape[1] for b in batch}
    Ts = {b["patches"].shape[2] for b in batch}
    assert len(Cs) == 1 and len(Ts) == 1, f"Mixed C/T not supported"
    C = next(iter(Cs)); T = next(iter(Ts))

    Ns   = [b["patches"].shape[0] for b in batch]
    Nmax = max(Ns)
    B    = len(batch)

    dtype_x  = batch[0]["patches"].dtype
    dtype_xy = batch[0]["pos"].dtype
    device   = batch[0]["patches"].device  # usually CPU inside DataLoader

    # --- allocate padded buffers ---
    patches = torch.zeros(B, Nmax, C, T, dtype=dtype_x, device=device)
    pos     = torch.zeros(B, Nmax, 2,    dtype=dtype_xy, device=device)
    kpm     = torch.ones( B, Nmax,       dtype=torch.bool, device=device)  # True=padded

    # per-sample (complex)
    hrtf1 = torch.empty(B, C//2, T, dtype=torch.complex64, device=device)
    hrtf2 = torch.empty(B, C//2, T, dtype=torch.complex64, device=device)

    # --- copy and restore ---
    for i, b in enumerate(batch):
        N = b["patches"].shape[0]
        patches[i, :N] = b["patches"]
        pos[i, :N]     = b["pos"]
        kpm[i, :N]     = False

        # restore complex HRTFs
        H1 = b["hrtf1"]; H2 = b["hrtf2"]
        C2 = H1.shape[0] // 2
        Hr1, Hi1 = H1[:C2], H1[C2:]
        Hr2, Hi2 = H2[:C2], H2[C2:]
        hrtf1[i] = torch.complex(Hr1, Hi1)
        hrtf2[i] = torch.complex(Hr2, Hi2)

    return {
        "patches": patches,   # [B, Nmax, C, T]
        "pos":     pos,       # [B, Nmax, 2]
        "kpm":     kpm,       # [B, Nmax] True=padded
        "hrtf1":   hrtf1,     # [B, C, T] complex64
        "hrtf2":   hrtf2,     # [B, C, T] complex64
    }

class ExtractionDatasetRevVAE(Dataset):
    def __init__(self,hp,train=True):
        self.train=train
        self.hp = hp
        self.df = pd.read_csv(self.hp.dataset.train_csv_path) if train else pd.read_csv(self.hp.dataset.test_csv_path) 
        self.fs = self.hp.stft.fs
        self.azs = {}
        tmp_az = 90
        for i in range(37): #create a dictionary of all mesurments
            self.azs[str(tmp_az)] = i
            tmp_az -=5
            if tmp_az ==-5:
                tmp_az = 355
        self.sir = None
        
    def __len__(self):
        return len(self.df)
    
     
    def calc_SNR_gain(self,x,n,snr_db):
        x_power = torch.mean(x**2).sum()
        n_power = torch.mean(n**2).sum()
        
        snr_linear = 10**(snr_db/10)
        desierd_noise_power = x_power/snr_linear

        g = torch.sqrt(desierd_noise_power/(n_power+1e-10))
        return g
    
    def preprocess(self,x,fs):
        #cut length
        if fs != self.fs:
            x = F.resample(x, fs, self.fs)
        if x.shape[0]!=1:
            x = x.squeeze()
        if self.hp.dataset.norm_sample:
            x = x/(x.abs().max(dim=-1, keepdim=True).values)
        if x.shape[-1] <= self.hp.dataset.time_len*self.fs:
             #cat zeros
            remain = self.hp.dataset.time_len*self.fs-x.shape[-1]
            x = torch.cat((x,torch.zeros((x.shape[0],remain))),-1)
        else:
            #cut
            x = x[:,:self.hp.dataset.time_len*self.fs] 
        return x
    
    def mix(self,x1,x2,sir_db=0):
        sir_lin = self.calc_SNR_gain(x1,x2,sir_db)
        x2 *=sir_lin
        x = x1+x2
        x = x/(x.abs().max(dim=-1, keepdim=True).values)
        return self.stft_sample(x)
    
    def stft_sample(self,x):
        X = torch.stft(torch.squeeze(x),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length), return_complex=True).to(torch.complex64)
        X[0:2, :] = X[0:2, :] * 0.001
        # mx_x = torch.max(torch.max(torch.abs(torch.real(X)) , torch.max(torch.abs(torch.imag(X)) )))
        # X = X/mx_x
        return X
    
    def hrtf_preprocces(self,hrtf):
        HRTF = torch.fft.fft(hrtf,self.hp.stft.fft_length).to(torch.complex64)
        HRTF = HRTF[:,1:self.hp.stft.fft_length//2 +1]
        return HRTF

    def get_hrtf(self,az,elev):
        h=self.hrtf_obj.get_hrtf(az,elev,fs=self.hp.dataset.fs)
        h = self.hrtf_preprocces(torch.tensor(h.T))
        return h
    
    def conv_h(self,wav_path,h_path):
        wav,fs = sf.read(wav_path)
        h,fs = sf.read(h_path)
        rend_L = signal.fftconvolve(wav,h[:,0])
        rend_R = signal.fftconvolve(wav,h[:,1])
        stereo_audio_h = np.concatenate((rend_L[:,np.newaxis],rend_R[:,np.newaxis]),axis=1)
        return torch.tensor(stereo_audio_h).T,fs
    
    def iSTFT(self,x):
        x_time = torch.istft(torch.squeeze(x).detach().cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length))
        x_time = x_time/(x_time.abs().max(dim=-1, keepdim=True).values)
        return x_time
    
    def __getitem__(self, idx):
        line = self.df.iloc[idx]

        #load irs
        h1_path = line['hrir_rev_1_path']
        hrtf1_path = line['hrir_zero_1_path']
        h2_path = line['hrir_rev_2_path']
        hrtf2_path = line['hrir_zero_2_path']
        
        #conv with audio samples
        x1_h,fs_x1 = self.conv_h(line['speaker_1'],h1_path)
        x2_h,fs_x2 = self.conv_h(line['speaker_2'],h2_path)

        x1_hrtf,fs_x1_hrtf = self.conv_h(line['speaker_1'],hrtf1_path)
        x2_hrtf,fs_x2_hrtf = self.conv_h(line['speaker_2'],hrtf2_path)

        #preprocess: stft
        x1_h,x2_h = self.preprocess(x1_h,fs_x1),self.preprocess(x2_h,fs_x2)
        x1_hrtf,x2_hrtf = self.preprocess(x1_hrtf,fs_x1_hrtf),self.preprocess(x2_hrtf,fs_x2_hrtf)

        # hrtf1,hrtf2 = self.hrtf_preprocces(hrtf1),self.hrtf_preprocces(hrtf2)
        if ('sir' in line) and (self.sir!=0):
            sir = line['sir']
        elif self.sir ==0:
            sir = 0
        else:
            sir=0
        Mix = self.mix(x1_h,x2_h,sir)
        Y1 = self.stft_sample(x1_hrtf)
        Y2 = self.stft_sample(x2_hrtf)
        return Mix,Y1,Y2
    


if __name__ == "__main__":
    from tqdm import tqdm
    
    hp = OmegaConf.load('/home/workspace/yoavellinson/binaural_TSE_Gen/conf/vae.yml')
    ds_db  = PatchDBDataset(hp, train=True)
    ds_mix = ExtractionDatasetRevVAE(hp, train=True)

    joined_ds = JoinedDataset(ds_db, ds_mix)
    loader = DataLoader(
        joined_ds,
        batch_size=2,
        num_workers=1,
        collate_fn=collate_joined
    )
    for step,batch in tqdm(enumerate(loader),total=len(loader)):
        print(batch.keys())   
        #### check the create data script for the rounding of az an elev so it matches in collate fn

    