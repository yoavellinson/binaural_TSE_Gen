from torch.utils.data import Dataset,DataLoader,random_split
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from omegaconf import OmegaConf
from glob import glob
from random import shuffle
import random
from scipy import signal
from pathlib import Path
import rir_generator as rir
import matplotlib.pyplot as plt
import torch.nn.functional as F
from hrtf_convolve import SOFA_HRTF_wrapper

class MonoStereoWhamrDataset(Dataset):
    def __init__(self,hp,train=True,debug=False,mono_input=False):
        self.train=train
        self.hp = hp
        self.df = pd.read_csv(self.hp.dataset.csv_path) if train else pd.read_csv(self.hp.dataset.test_csv_path) 
        if debug:
            self.df = self.df[:10]
        self.fs = self.hp.dataset.fs
        self.sir = None
        self.db_root = hp.dataset.db_root_train if train else hp.dataset.db_root_test
        self.db_root = Path(self.db_root)
        self.wsj0_root = hp.dataset.wsj0_root
        self.wsj0_root = Path(self.wsj0_root)
        self.wsj0_root_gt = self.db_root.parent
        self.wsj0_root_gt = Path(self.wsj0_root_gt)
        d = hp.dataset.array.d #array distance between mics
        n = hp.dataset.array.n #number of mics
        r = hp.dataset.array.r #radius from array
        self.n_ref=n_ref = n//2
        recs = self.generate_array(n,d)
        x_ref,y_ref,z_ref = recs[n_ref]
        s_left = [x_ref-r*np.sin(np.pi/4),1.5,z_ref+r*np.sin(np.pi/4)]
        s_right = [x_ref+r*np.sin(np.pi/4),1.5,z_ref+r*np.sin(np.pi/4)]
        
        self.h_loc_left = self.generate_h(s_left,recs)
        self.h_loc_right = self.generate_h(s_right,recs)
        self.H_loc_left = self.h_preprocces(self.h_loc_left.T)
        self.H_loc_right = self.h_preprocces(self.h_loc_right.T)
        self.mono_input = mono_input
    def generate_h(self,s,recs):
        h = rir.generate(c=343,fs=self.fs,r=recs,s=s,L=[6,2.8,5],reverberation_time=0.0,order=0,nsample=1024) 
        return h


    def generate_array(self,n, d):
        assert n % 2 == 1, "n must be odd"
        mid_index = n // 2
        arr = []

        for i in range(n):
            offset = (i - mid_index) * d
            arr.append((3 + offset,1.5,2.5))
        
        return np.array(arr)

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
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
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
            x = torch.cat((x,torch.zeros((x.shape[0],int(remain)))),-1)
        else:
            #cut
            x = x[:,:int(self.hp.dataset.time_len*self.fs)] 
        return x
    
    def mix(self,x1,x2,sir_db=0):
        # sir_lin = self.calc_SNR_gain(x1,x2,sir_db)
        # x2 *=sir_lin
        x1 = x1/(x1.abs().max(dim=-1, keepdim=True).values)
        x2 = x2/(x2.abs().max(dim=-1, keepdim=True).values)
        x = x1+x2
        x = x/(x.abs().max(dim=-1, keepdim=True).values)
        return self.stft_sample(x)
    
    def stft_sample(self,x):
        X = torch.stft(torch.squeeze(x),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length), return_complex=True).to(torch.complex64)
        # X[0:2, :] = X[0:2, :] * 0.001
        # mx_x = torch.max(torch.max(torch.abs(torch.real(X)) , torch.max(torch.abs(torch.imag(X)) )))
        # X = X/mx_x
        return X
    
    def h_preprocces(self,h):
        h= torch.tensor(h)
        H = torch.fft.fft(h,self.hp.stft.fft_length).to(torch.complex64)
        H = H[:,1:self.hp.stft.fft_length//2 +1]
        return H

    def get_hrtf(self,az,elev):
        h=self.hrtf_obj.get_hrtf(az,elev,fs=self.hp.dataset.fs)
        h = self.hrtf_preprocces(torch.tensor(h.T))
        return h
    

    def conv_h(self,wav_path, h_LM, return_shape="MT"):
        """
        Convolve mono audio with M channel RIRs.
        h_LM: np.ndarray or torch.Tensor with shape (L, M)  [time, mics]
        return_shape: "TM" -> (T_out, M), "MT" -> (M, T_out)
        """
        # Load mono audio
        wav, fs = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav[:, 0]  # take first channel
        
        # To torch
        x = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,T)
        h = torch.as_tensor(h_LM, dtype=torch.float32)                            # (L,M)
        L, M = h.shape

        # PyTorch conv1d is cross-correlation; flip kernel for true convolution
        # Weight shape must be (out_channels, in_channels/groups, kernel_size)
        weight = h.flip(0).T.unsqueeze(1)  # (M,1,L)

        # Duplicate the same input across M channels via broadcast
        x_M = x.expand(1, M, -1)  # (1,M,T) (no extra memory)

        # Depthwise 1D convolution: apply each RIR to the same input
        y = F.conv1d(x_M, weight, bias=None, groups=M, padding=L-1)  # (1,M,T+L-1)
        y = y.squeeze(0)  # (M, T_out)

        if return_shape.upper() == "MT":
            return y.contiguous(), fs   # (T_out, M)
        else:
            return y.T.contiguous(), fs     # (M, T_out)

    
    def iSTFT(self,x):
        x_time = torch.istft(torch.squeeze(x).detach().cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length))
        x_time = x_time/(x_time.abs().max(dim=-1, keepdim=True).values)
        return x_time
    
    def preprocess_mono(self,x,fs,n_channels_out =1):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)        #cut length
        if fs != self.fs:
            x = F.resample(x, fs, self.fs)
        if x.shape[0]!=1:
            x = x.squeeze()
        if self.hp.dataset.norm_sample:
            x = x/(x.abs().max(dim=-1, keepdim=True).values)
        if x.shape[-1] <= self.hp.dataset.time_len*self.fs:
             #cat zeros
            remain = self.hp.dataset.time_len*self.fs-x.shape[-1]
            x = torch.cat((x,torch.zeros((int(remain)))),-1)
        else:
            #cut
            x = x[:int(self.hp.dataset.time_len*self.fs)] 
        # if n_channels_out>1:
        x_repeated = x.repeat(n_channels_out, 1)  # [B, C*N, F, T]
        # if n_channels_out==1:
        #     x_repeated=x_repeated.unsqueeze(0)         
        return x_repeated
      
    def __getitem__(self, idx):
        line = self.df.iloc[idx]

        mix,fs = sf.read(self.db_root/line['output_filename'])
        mix = self.preprocess_mono(mix,fs,n_channels_out=int(self.hp.dataset.array.n) if not self.mono_input else 1)
        Mix = self.stft_sample(mix)
        if self.mono_input:
            Mix = Mix.unsqueeze(0)
        left_speaker = random.randint(0,1)
        right_speaker = abs(1-left_speaker)

        #mix both permutations
        s_left1,fs = self.conv_h(self.wsj0_root/line[f's{left_speaker+1}_path'],self.h_loc_left)
        s_right1,_ = self.conv_h(self.wsj0_root/line[f's{right_speaker+1}_path'],self.h_loc_right)
        s_left1 = self.preprocess(s_left1,fs)
        s_right1 = self.preprocess(s_right1,fs)
        Mix_target_1 = self.mix(s_left1,s_right1,sir_db=0)

        s_left2,_ = self.conv_h(self.wsj0_root/line[f's{right_speaker+1}_path'],self.h_loc_left)
        s_right2,_ = self.conv_h(self.wsj0_root/line[f's{left_speaker+1}_path'],self.h_loc_right)
        s_left2 = self.preprocess(s_left2,fs)
        s_right2 = self.preprocess(s_right2,fs)
        Mix_target_2 = self.mix(s_left2,s_right2,sir_db=0)

        # fixing sisdr shifts:
        s1 =s_left1[self.n_ref,:]
        s2 = s_right1[self.n_ref,:]
        
        # s1 = s1.roll(1)
        # s1[0] = 0
        # s2=s2.roll(-1)
        # s2[-1]=0
        s1,_ = sf.read(self.wsj0_root_gt/f's1_anechoic/{line['output_filename']}')
        s2,_ = sf.read(self.wsj0_root_gt/f's2_anechoic/{line['output_filename']}')
        s1,s2 = self.preprocess_mono(s1,fs,n_channels_out=1),self.preprocess_mono(s2,fs,n_channels_out=1)
        S1,S2 = self.stft_sample(s1),self.stft_sample(s2)
        return Mix,Mix_target_1,Mix_target_2,self.H_loc_left,self.H_loc_right,S1,S2

class ExtractionDatasetRev(Dataset):
    def __init__(self,hp,train=True):
        self.train=train
        self.hp = hp
        self.df = pd.read_csv(self.hp.dataset.csv_path) if train else pd.read_csv(self.hp.dataset.test_csv_path) 
        self.fs = self.hp.dataset.fs
        self.hrtf_obj = SOFA_HRTF_wrapper(path=hp.dataset.hrtf_path)
        self.azs = {}
        tmp_az = 90
        self.h_anechoic_path = Path(hp.dataset.h_anechoic_path)
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
        _,fs_x1_hrtf,hrtf1,real_az1,real_elev1 = self.hrtf_obj.conv_file(line['speaker_1'],az=line['az_1'],elev=-7)#line['elev_1'])
        _,fs_x2_hrtf,hrtf2,real_az2,real_elev2 = self.hrtf_obj.conv_file(line['speaker_2'],az=line['az_2'],elev=-7)#line['elev_2'])

        h_idx_1 = self.azs[str(real_az1)]
        h1_path = line['hrir_path']+f'/h_rt60_receiver_{h_idx_1}.wav'

        hrtf1_path = line['hrir_path']+f'/h_first_receiver_{h_idx_1}.wav'
        h_idx_2 = self.azs[str(real_az2)]
        
        h2_path = line['hrir_path']+f'/h_rt60_receiver_{h_idx_2}.wav'
        hrtf2_path = line['hrir_path']+f'/h_first_receiver_{h_idx_2}.wav'

        x1_h,fs_x1 = self.conv_h(line['speaker_1'],h1_path)
        x2_h,fs_x2 = self.conv_h(line['speaker_2'],h2_path)

        x1_hrtf,fs_x1_hrtf = self.conv_h(line['speaker_1'],hrtf1_path)
        x2_hrtf,fs_x2_hrtf = self.conv_h(line['speaker_2'],hrtf2_path)

        x1_h,x2_h = self.preprocess(x1_h,fs_x1),self.preprocess(x2_h,fs_x2)
        x1_hrtf,x2_hrtf = self.preprocess(x1_hrtf,fs_x1_hrtf),self.preprocess(x2_hrtf,fs_x2_hrtf)

        hrtf1_time,hrtf2_time = hrtf1,hrtf2
        hrtf1,hrtf2 = self.hrtf_preprocces(hrtf1),self.hrtf_preprocces(hrtf2)
        if ('sir' in line) and (self.sir!=0):
            sir = line['sir']
        elif self.sir ==0:
            sir = 0
        else:
            sir=0
        Mix = self.mix(x1_h,x2_h,sir)
        if self.hp.output_channels==1:
            wav1,fs = sf.read(line['speaker_1'])
            x1 = self.preprocess(torch.tensor(wav1).unsqueeze(0),fs)
            wav2,fs = sf.read(line['speaker_2'])
            x2 = self.preprocess(torch.tensor(wav2).unsqueeze(0),fs)
            Y1 = self.stft_sample(x1).unsqueeze(0)
            Y2 = self.stft_sample(x2).unsqueeze(0)
        else: 
            Y1 = self.stft_sample(x1_hrtf)
            Y2 = self.stft_sample(x2_hrtf)
        return Mix,Y1,Y2,hrtf1,hrtf2,hrtf1_time,hrtf2_time,real_az1,real_elev1,real_az2,real_elev2


if __name__ == "__main__":
    hp = OmegaConf.load('/home/workspace/yoavellinson/mono-to-stereo/conf/config_ncsnpp_mono_stereo.yaml')
    hp.dataset.array.d = 0.03
    db = MonoStereoWhamrDataset(hp,train=False)
    loader = DataLoader(db,batch_size=1,shuffle=False)
    for step,batch in enumerate(loader):
        Mix,Mix_target_1,Hl,Hr,S1,S2= batch
        mix = db.iSTFT(Mix)
        mix_target_1 = db.iSTFT(Mix_target_1)
        s1 = db.iSTFT(S1)
        s2 = db.iSTFT(S2)

        sf.write('batch/mix.wav',mix.T.numpy(),16000)
        sf.write(f'batch/mix_target_1_d_{hp.dataset.array.d}.wav',mix_target_1.T.numpy(),16000)
        sf.write('batch/s1_left.wav',s1.T.numpy(),16000)
        sf.write('batch/s2_right.wav',s2.T.numpy(),16000)

        break
