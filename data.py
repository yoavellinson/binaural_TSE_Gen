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
import sys
sys.path.append("/home/workspace/yoavellinson/binaural_TSE_Gen")
from audio_itd import hrir2itd_fft
DTYPE = torch.complex64
from pathlib import Path
DEBUG_SAMPLES=100


class JoinedDataset(Dataset):
    """
    Combines two datasets sharing the same CSV/row order.
    Left side (db_) is dict from PatchDBDataset.
    Right side (mix_) is tuple (Mix, Y1, Y2) from ExtractionDatasetRevVAE, converted to dict.
    """
    def __init__(self, ds_db: Dataset, ds_mix: Dataset, prefix_db='db_', prefix_mix='mix_',device=None):
        assert len(ds_db) == len(ds_mix), "Datasets must have equal length for index-wise join"
        self.ds_db = ds_db
        self.ds_mix = ds_mix
        self.prefix_db = prefix_db
        self.prefix_mix = prefix_mix
        self.device = device

    def __len__(self):
        return len(self.ds_db)

    def __getitem__(self, i):
        try:
            A = self.ds_db[i]                       # dict: patches/pos/kpm
            B = self.ds_mix[i]                      # tuple: (Mix, Y1, Y2) or dict
        except:
            print(i)            
        if not isinstance(B, dict):
            Mix, Y1, Y2 = B
            B = {"mix": Mix, "y1": Y1, "y2": Y2}
        
        A = {
            self.prefix_db + k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in A.items()
        }
        B = {self.prefix_mix + k: v.to(self.device) for k, v in B.items()}
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

def collate_joined_default(batch):
    # Split into db_ and mix_ sub-dicts
    db_batch  = [{k[3:]: v for k, v in b.items() if k.startswith("db_")}  for b in batch]
    mix_batch = [{k[4:]: v for k, v in b.items() if k.startswith("mix_")} for b in batch]

    db_part  = default_collate(db_batch)                             # padded [B, Nmax, C, T], etc.
    mix_part = default_collate(mix_batch)                           # handles complex tensors too
    
    # Re-namespace keys for clarity
    out = {f"db_{k}": v for k, v in db_part.items()}
    out.update({f"mix_{k}": v for k, v in mix_part.items()})
    return out

class PatchDBDataset(Dataset):
    def __init__(self, hp, train=True,debug=False):
        df = hp.dataset.train_csv_path if train else hp.dataset.test_csv_path
        self.df = pd.read_csv(df)
        if debug:
            self.df = self.df[:DEBUG_SAMPLES]
        self.hp = hp
    def __len__(self):
        return len(self.df)  # <- fix: was len(self.paths)

    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        if self.hp.stft.fft_length == 512:
            p =line['sofa_path']
            p = p.replace('sofas','pts_512').replace('.sofa','.pt')
            d,name = load_db(p)
        else:
            d,name = load_db(line['pt_path'])
        patches = d['patches']    # expect [N, C, T]
        pos     = d['pos']        # [N, 2]
        N, C, T = patches.shape 
        kpm = torch.zeros(N, dtype=torch.bool)

        #get hrtfs from az/elev
        az1 = line['az_1']
        az2 = line['az_2']

        elev1 = line['elev_1']
        elev2 = line['elev_2']

        target1 = torch.tensor([az1, elev1], dtype=pos.dtype, device=pos.device)
        dist1 = torch.norm(pos - target1, dim=1)  # Euclidean distance
        idx1 = torch.argmin(dist1).item()
        target2 = torch.tensor([az2, elev2], dtype=pos.dtype, device=pos.device)
        dist2 = torch.norm(pos - target2, dim=1)  # Euclidean distance
        idx2 = torch.argmin(dist2).item()
        
        hrtf1 = patches[idx1]
        hrtf2 = patches[idx2]
        itd = hrir2itd_fft(patches)
        
        return {'patches': patches, 
                'pos': pos, 
                'kpm': kpm,
                'hrtf1':hrtf1,
                'hrtf2':hrtf2,
                'az1':az1,
                'elev1':elev1,
                'az2':az2,
                'elev2':elev2,
                'itd':itd,
                'name':name}


class HRTFHeadCondDataset(Dataset):
    def __init__(self, hp, train=True,debug=False,device=None):
        df = hp.dataset.train_csv_path if train else hp.dataset.test_csv_path
        self.df = pd.read_csv(df)
        if debug:
            self.df = self.df[:DEBUG_SAMPLES]
        self.hp = hp
        self.device=device
    def __len__(self):
        return len(self.df)  # <- fix: was len(self.paths)

    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        if self.hp.stft.fft_length == 512:
            p =line['sofa_path']
            p = p.replace('sofas','pts_512').replace('.sofa','.pt')
            d,name = load_db(p)
        else:
            d,name = load_db(line['pt_path'])
        p_emb = line['sofa_path'].replace('sofas','ae_res').replace('.sofa','.pt')
        try:
            head_emb = torch.load(p_emb).squeeze()
        except:
            print(p_emb)
        patches = d['patches']    # expect [N, C, T]
        pos     = d['pos']        # [N, 2]
        N, C, T = patches.shape 
        kpm = torch.zeros(N, dtype=torch.bool)

        #get hrtfs from az/elev
        az1 = line['az_1']
        az2 = line['az_2']

        elev1 = line['elev_1']
        elev2 = line['elev_2']

        target1 = torch.tensor([az1, elev1], dtype=pos.dtype, device=pos.device)
        dist1 = torch.norm(pos - target1, dim=1)  # Euclidean distance
        idx1 = torch.argmin(dist1).item()
        target2 = torch.tensor([az2, elev2], dtype=pos.dtype, device=pos.device)
        dist2 = torch.norm(pos - target2, dim=1)  # Euclidean distance
        idx2 = torch.argmin(dist2).item()
        
        hrtf1 = patches[idx1]
        hrtf2 = patches[idx2]
        
        return {'head_emb': head_emb.to(torch.float32), 
                'hrtf1':hrtf1.to(torch.float32),
                'hrtf2':hrtf2.to(torch.float32),
                'az1':torch.tensor(az1),
                'elev1':torch.tensor(elev1),
                'az2':torch.tensor(az2),
                'elev2':torch.tensor(elev2)}

def load_db(path):
    data = torch.load(path)
    p = Path(path)
    name=f'db_{p.parent.stem}_sample_{p.stem}'
    return {'patches': data['patches'], 'pos': data['pos']},name

def collate_unif_sample(batch,Nmax=None):
    """
    batch[i] = {
      "patches": [N_i, C, T],
      "pos":     [N_i, 2],
      "kpm":     [N_i] (bool),
      "hrtf1":   [2*C, T] (real-concat),
      "hrtf2":   [2*C, T] (real-concat),
    }
    Nmax = None, effective batch size
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
    Nmax = max(Ns) if Nmax == None else Nmax

    B    = len(batch)

    dtype_x  = batch[0]["patches"].dtype
    dtype_xy = batch[0]["pos"].dtype
    dtype_itd = batch[0]["itd"].dtype
    device   = batch[0]["patches"].device  # usually CPU inside DataLoader

    # --- allocate padded buffers ---
    patches = torch.zeros(B, Nmax, C, T, dtype=dtype_x, device=device)
    pos     = torch.zeros(B, Nmax, 2,    dtype=dtype_xy, device=device)
    kpm     = torch.ones( B, Nmax,       dtype=torch.bool, device=device)  # True=padded
    itd     = torch.ones( B, Nmax,       dtype=dtype_itd, device=device)  
    # per-sample (complex)
    hrtf1 = torch.empty(B, C//2, T, dtype=DTYPE, device=device)
    hrtf2 = torch.empty(B, C//2, T, dtype=DTYPE, device=device)

    # --- copy and restore ---
    for i, b in enumerate(batch):
        N = min(b["patches"].shape[0],Nmax)
        idx = torch.randperm(b["patches"].shape[0])[:N]
        patches[i, :N] = b["patches"][idx]
        pos[i, :N]     = b["pos"][idx]
        kpm[i, :N]     = False
        itd[i,:N] = b['itd'][idx]
        # restore complex HRTFs
        H1 = b["hrtf1"]; H2 = b["hrtf2"]
        C2 = H1.shape[0] // 2
        Hr1, Hi1 = H1[:C2], H1[C2:]
        Hr2, Hi2 = H2[:C2], H2[C2:]
        hrtf1[i] = torch.complex(Hr1, Hi1)
        hrtf2[i] = torch.complex(Hr2, Hi2)
    
    half = C//2
    patches_real = patches[:,:,:half,:]
    patches_imag = patches[:,:,half:,:]
    patches = torch.complex(patches_real,patches_imag).to(DTYPE).permute(0,2,1,3)
    return {
        "patches": patches,   # [B, 2,Nmax, T]
        "pos":     pos,       # [B, Nmax, 2]
        "kpm":     kpm,       # [B, Nmax] True=padded
        "hrtf1":   hrtf1,     # [B, C, T] complex64
        "hrtf2":   hrtf2,     # [B, C, T] complex64
        'az1':b['az1'],
        'az2':b['az2'],
        'elev1':b['elev1'],
        'elev2':b['elev2'],
        'itd':itd
        
    }
def collate_simple(batch):
    """
    batch[i] = {
      "patches": [N_i, C, T],
      "pos":     [N_i, 2],
      "kpm":     [N_i] (bool),
      "hrtf1":   [2*C, T] (real-concat),
      "hrtf2":   [2*C, T] (real-concat),
    }
    Nmax = None, effective batch size
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
    dtype_itd = batch[0]["itd"].dtype
    device   = batch[0]["patches"].device  # usually CPU inside DataLoader

    # --- allocate padded buffers ---
    patches = torch.zeros(B, Nmax, C, T, dtype=dtype_x, device=device)
    pos     = torch.zeros(B, Nmax, 2,    dtype=dtype_xy, device=device)
    kpm     = torch.ones( B, Nmax,       dtype=torch.bool, device=device)  # True=padded
    itd     = torch.ones( B, Nmax,       dtype=dtype_itd, device=device)
    name = ['' for _ in range(B)]
    # per-sample (complex)
    hrtf1 = torch.empty(B, C//2, T, dtype=DTYPE, device=device)
    hrtf2 = torch.empty(B, C//2, T, dtype=DTYPE, device=device)

    # --- copy and restore ---
    for i, b in enumerate(batch):
        N = min(b["patches"].shape[0],Nmax)
        patches[i, :N] = b["patches"]
        pos[i, :N]     = b["pos"]
        kpm[i, :N]     = False
        itd[i,:N] = b['itd']
        # restore complex HRTFs
        H1 = b["hrtf1"]; H2 = b["hrtf2"]
        C2 = H1.shape[0] // 2
        Hr1, Hi1 = H1[:C2], H1[C2:]
        Hr2, Hi2 = H2[:C2], H2[C2:]
        hrtf1[i] = torch.complex(Hr1, Hi1)
        hrtf2[i] = torch.complex(Hr2, Hi2)
        name[i] = b['name']
    half = C//2
    patches_real = patches[:,:,:half,:]
    patches_imag = patches[:,:,half:,:]
    patches = torch.complex(patches_real,patches_imag).to(DTYPE).permute(0,2,1,3)
    return {
        "patches": patches,   # [B, 2,Nmax, T]
        "pos":     pos,       # [B, Nmax, 2]
        "kpm":     kpm,       # [B, Nmax] True=padded
        "hrtf1":   hrtf1,     # [B, C, T] complex64
        "hrtf2":   hrtf2,     # [B, C, T] complex64
        'az1':b['az1'],
        'az2':b['az2'],
        'elev1':b['elev1'],
        'elev2':b['elev2'],
        'itd':itd,
        'name':name
    }

class ExtractionDatasetRevVAE(Dataset):
    def __init__(self,hp,train=True,debug=False):
        self.train=train
        self.hp = hp
        self.df = pd.read_csv(self.hp.dataset.train_csv_path) if train else pd.read_csv(self.hp.dataset.test_csv_path) 
        if debug:
            self.df = self.df[:DEBUG_SAMPLES]
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
        X = torch.stft(torch.squeeze(x),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length), return_complex=True).to(DTYPE)
        X[0:2, :] = X[0:2, :] * 0.001
        # mx_x = torch.max(torch.max(torch.abs(torch.real(X)) , torch.max(torch.abs(torch.imag(X)) )))
        # X = X/mx_x
        return X
    
    def hrtf_preprocces(self,hrtf):
        HRTF = torch.fft.fft(hrtf,self.hp.stft.fft_length).to(DTYPE)
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
    

import torch

def find_matching_positions(d):
    # First, normalize shapes: remove batch dimension [1, N, 2] -> [N, 2]
    d2 = {k: v.squeeze(0) for k, v in d.items()}  
    
    # Convert each pos tensor to a set of tuples for fast equality checks
    tuple_map = {
        k: {tuple(pair.tolist()) for pair in v}
        for k, v in d2.items()
    }

    # Build result  
    d_res = {}

    for name_i, pos_i in d2.items():
        matches = {}

        for name_j, pos_j in d2.items():
            if name_i == name_j:
                continue

            # find exact matches
            pairs_j = pos_j.tolist()  # list of [az,el]
            idx_list = [
                idx for idx, pair in enumerate(pairs_j)
                if tuple(pair) in tuple_map[name_i]
            ]

            if idx_list:  # only save if there are matches
                mask = (abs(d[name_j][:,idx_list,-1])<30).tolist()[0]
                idx_filtered = [x for x, m in zip(idx_list, mask) if m]
                matches[name_j] = idx_filtered 

        d_res[name_i] = matches

    for name in list(d_res.keys()):
        for sub in list(d_res[name].keys()):
            if len(d_res[name][sub]) == 0:      # empty list
                d_res[name].pop(sub)            # remove it

        # Optionally remove empty sub-dicts
        if len(d_res[name]) == 0:
            d_res.pop(name)

    return d_res
def create_similar():
    new_df = []
    for i in tqdm(df.index):
        line = df.iloc[i].copy()
        p = Path(line['pt_path'])
        name=f'db_{p.parent.stem}_sample_{p.stem}'
        get_new_name = True
        m = 0
        while get_new_name==True and m<10:
            try:
                new_name = random.choice(list(d_res[name].keys()))
            except:
                m+=1
                continue

            if d_res[name][new_name] == []:
                d_res[name].pop(new_name)
                m+=1
                continue
            new_idx1 = random.choice(d_res[name][new_name])
            new_pos1 = d[new_name][:,new_idx1]
            new_idx2 =None
            s = random.sample(d_res[name][new_name],len(d_res[name][new_name]))
            for j in s:
                if abs(d[new_name][:,j] - new_pos1)[:,0] >= 30:
                    new_idx2 = j
                    break
            if new_idx2:
                new_pos2 = d[new_name][:,new_idx2]
                # d_res[name][new_name].remove(new_idx1)
                # d_res[name][new_name].remove(new_idx2)
                line['az_1'] = float(new_pos1[0][0])
                line['elev_1'] = float(new_pos1[0][1])
                line['az_2'] = float(new_pos2[0][0])
                line['elev_2'] = float(new_pos2[0][1])
                db = new_name.split('_sample_')[0].replace('db_','')
                sample = new_name.split('_sample_')[-1]
                pt_path = f'/home/workspace/yoavellinson/binaural_TSE_Gen/pts_512/test_set/{db}/{sample}.pt'
                sofa_path = f'/home/workspace/yoavellinson/binaural_TSE_Gen/sofas/test_set/{db}/{sample}.sofa'
                line.sofa_path = sofa_path
                line.pt_path = pt_path
                new_df.append(line)
                get_new_name = False
                if d_res[name] == {} or len(d_res[name].keys())<2:
                    d_res.pop(name)
            else:
                m+=1
    new_df = pd.DataFrame(new_df) 
    new_df.to_csv('/home/workspace/yoavellinson/binaural_TSE_Gen/csvs/HRTF_test_different_heads_wsj0_1k_mp.csv')
    torch.save(d_res, "positions_match.pt")
    

def find_names_with_pairs(d, az1, elev1, az2, elev2):
    target1 = (float(az1), float(elev1))
    target2 = (float(az2), float(elev2))

    matched_names = []

    for name, pos in d.items():
        p = pos.squeeze(0).tolist()   # convert [1,N,2] → list of (az,el)

        has_az1 = abs(pos[:, :, 0] -az1)<1 
        has_elev1 = abs(pos[:, :, 1] -elev1)<1
        has_az2 = abs(pos[:, :, 0] -az2)<1 
        has_elev2 = abs(pos[:, :, 1] -elev2)<1

        has1= has_az1*has_elev1
        has2= has_az2*has_elev2

        if has1.sum() and has2.sum():
            matched_names.append(name)

    return matched_names

if __name__ == "__main__":
    from tqdm import tqdm
    import random

    hp = OmegaConf.load('/home/workspace/yoavellinson/binaural_TSE_Gen/conf/extraction_nbss_conf_large_large.yml')
    ds_db  = PatchDBDataset(hp, train=False)
    ds_mix = ExtractionDatasetRevVAE(hp, train=False)

    joined_ds = JoinedDataset(ds_db, ds_mix)
    loader = DataLoader(
        joined_ds,
        batch_size=1,
        num_workers=1,
        collate_fn=collate_joined
    )
    d = {}
    for step,batch in tqdm(enumerate(loader),total=len(loader)):
        name = batch['db_name']
        if not name in list(d.keys()):
            pos = batch['db_pos']
            d[name[0]] = pos
        if len(d.keys())==7:
            break
    df = ds_db.df
    new_df = []
    for i in tqdm(df.index):
        line = df.iloc[i].copy()
        p = Path(line['pt_path'])
        name=f'db_{p.parent.stem}_sample_{p.stem}'
        az1,elev1 = line['az_1'],line['elev_1']
        az2,elev2 = line['az_2'],line['elev_2']
        d_tmp = d.copy()
        d_tmp.pop(name)
        names = find_names_with_pairs(d_tmp,az1,elev1,az2,elev2)
        if names == []:
            continue
        else: 
            new_name = random.sample(names,1)[0]

            # new_name = random.choice(list(d_res[name].keys()))
    
            db = new_name.split('_sample_')[0].replace('db_','')
            sample = new_name.split('_sample_')[-1]
            pt_path = f'/home/workspace/yoavellinson/binaural_TSE_Gen/pts_512/test_set/{db}/{sample}.pt'
            sofa_path = f'/home/workspace/yoavellinson/binaural_TSE_Gen/sofas/test_set/{db}/{sample}.sofa'
            line.sofa_path = sofa_path
            line.pt_path = pt_path
            new_df.append(line)
    new_df = pd.DataFrame(new_df) 
    new_df.to_csv('/home/workspace/yoavellinson/binaural_TSE_Gen/csvs/HRTF_test_different_heads_wsj0_1k_mp.csv')

