import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
# from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torch_pesq import PesqLoss
from scipy.signal import deconvolve

def kl_divergence(mu, logvar):
    # mean over batch
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

def mae_recon_loss(pred, target, mae_keep, kpm):
    # accept either [B,N,D] or [B,N,C,T]
    if pred.dim() == 4:
        B, N, C, T = pred.shape
        pred   = pred.reshape(B, N, C*T)
        target = target.reshape(B, N, C*T)
    valid  = ~kpm
    masked = (~mae_keep) & valid
    if masked.sum() == 0:
        return pred.new_zeros([])
    return F.l1_loss(pred[masked], target[masked], reduction='mean')

def info_nce(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temperature              # [B,B]
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5*(F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def make_mae_keep(B, N, ratio, device):
    K = int((1.0 - ratio) * N)              # how many to keep
    idx = torch.argsort(torch.rand(B, N, device=device), dim=1)[:, :K]
    keep = torch.zeros(B, N, dtype=torch.bool, device=device)
    keep.scatter_(1, idx, True)
    return keep  # True = visible

class PITPESQLoss(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.pesq = PesqLoss(0.5,
                sample_rate=hp.dataset.fs)
        
    def forward(self, x_hat_time, x_time):
        '''
        h_hat_time: B,1,T
        h_time: B,2,T - 2 it the two possible sources
        '''
        x_hat_time = x_hat_time.repeat(1,2,1)
        pesq = self.pesq.mos(x_time.to(torch.float32).cpu().squeeze(), x_hat_time.to(torch.float32).cpu().squeeze()).max()
        return pesq
    
class SiSDRLoss(nn.Module):
    def __init__(self,hp):
        super(SiSDRLoss, self).__init__()
        self.hp = hp
   
    def si_sdr_calc(self,estimate_source, source):
        """Calculate SI-SNR or SI-SDR (same thing)
        Args:
        source: [B, C, T], B is batch size ,C= channels ,T = frames
        estimate_source: [B, C, T]
        """
        EPS = 1e-08
        ########## reshape to use broadcast ##########
        s_target = torch.unsqueeze(source, dim=1)  # [B, 1, C, T]
        s_estimate = torch.unsqueeze(estimate_source, dim=2)  # [B, C, 1, T]
        ########## s_target = <s', s>s / ||s||^2 ##########
        pair_wise_dot = torch.sum(s_estimate * s_target,
                                dim=3, keepdim=True)  # [B, C, C, 1]
        s_target_energy = torch.sum(
            s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        ########## e_noise = s' - s_target ##########
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        ########## SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)##########
        pair_wise_si_snr = torch.sum(
            pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]
        diag_si_snr = pair_wise_si_snr.diagonal(dim1=1, dim2=2)
        si_sdr_per_sample = diag_si_snr.mean(dim=1)

        return si_sdr_per_sample
  
    def forward(self,x_hat,x,ch=0):
        x_hat_time = torch.stack([torch.istft(torch.squeeze(x_hat[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x_hat.shape[0])])
        x_time = torch.stack([torch.istft(torch.squeeze(x[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x.shape[0])])
        
        sisdr_loss = -self.si_sdr_calc(x_hat_time,x_time)#-self.si_sdr(x_hat_time,x_time)
        if self.hp.dataset.n_channels >1:
            sisdr_loss = sisdr_loss.mean()
        return sisdr_loss
    
    
class SpecMAE(nn.L1Loss):
    def __init__(self, hp, reduction='mean'):
        super(SpecMAE, self).__init__(reduction=reduction)
        self.hp = hp
        self.eps = 1e-8

    def forward(self, x_hat, x):
        # Clamp to avoid NaNs and numerical instability
        x_hat = x_hat.abs().float()
        x = x.abs().float()
        # Compute loss
        loss = super(SpecMAE, self).forward(x_hat, x)
        return loss

    
