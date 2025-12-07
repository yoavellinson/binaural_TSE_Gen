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
    
# class SiSDRLoss(nn.Module):
#     def __init__(self,hp):
#         super(SiSDRLoss, self).__init__()
#         self.hp = hp
   
#     def si_sdr_calc(self,estimate_source, source):
#         """Calculate SI-SNR or SI-SDR (same thing)
#         Args:
#         source: [B, C, T], B is batch size ,C= channels ,T = frames
#         estimate_source: [B, C, T]
#         """
#         EPS = 1e-08
#         ########## reshape to use broadcast ##########
#         s_target = torch.unsqueeze(source, dim=1)  # [B, 1, C, T]
#         s_estimate = torch.unsqueeze(estimate_source, dim=2)  # [B, C, 1, T]
#         ########## s_target = <s', s>s / ||s||^2 ##########
#         pair_wise_dot = torch.sum(s_estimate * s_target,
#                                 dim=3, keepdim=True)  # [B, C, C, 1]
#         s_target_energy = torch.sum(
#             s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
#         pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
#         ########## e_noise = s' - s_target ##########
#         e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
#         ########## SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)##########
#         pair_wise_si_snr = torch.sum(
#             pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
#         pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]
#         diag_si_snr = pair_wise_si_snr.diagonal(dim1=1, dim2=2)
#         si_sdr_per_sample = diag_si_snr.mean(dim=1)

#         return si_sdr_per_sample
    # def forward(self,x_hat,x,ch=0):
    #     x_hat_time = torch.stack([torch.istft(torch.squeeze(x_hat[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x_hat.shape[0])])
    #     x_time = torch.stack([torch.istft(torch.squeeze(x[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x.shape[0])])
        
    #     sisdr_loss = -self.si_sdr_calc(x_hat_time,x_time)#-self.si_sdr(x_hat_time,x_time)
    #     if self.hp.dataset.n_channels >1:
    #         sisdr_loss = sisdr_loss.mean()
    #     return sisdr_loss

class LSD(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, dim=2):
        lsd = torch.sqrt(torch.mean((pred - target).pow(2), dim=dim))
        if self.reduction == "mean":
            lsd = torch.mean(lsd)
        elif self.reduction == "sum":
            lsd = torch.sum(lsd)
        elif self.reduction in ["none", None]:
            pass
        else:
            raise ValueError

        return lsd

def _sisdr_time_safe(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8, return_db: bool = True):
    """
    est, ref: [B, C, L] real tensors (time-domain)
    SI-SDR = 10*log10( ||s_target||^2 / ||e_noise||^2 ), with zero-mean per sample.
    Numerically safe (no div by zero, no log of zero).
    Returns [B, C] (dB if return_db).
    """
    # zero-mean per sample to satisfy SI-SDR definition
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    # projection of est on ref
    ref_energy = (ref**2).sum(dim=-1, keepdim=True).clamp_min(eps)  # [B,C,1]
    scale = (est * ref).sum(dim=-1, keepdim=True) / ref_energy      # [B,C,1]
    s_target = scale * ref                                          # [B,C,L]
    e_noise = est - s_target

    # energies
    num = (s_target**2).sum(dim=-1).clamp_min(eps)  # [B,C]
    den = (e_noise**2).sum(dim=-1).clamp_min(eps)   # [B,C]

    if return_db:
        # 10*log10 safely (no log(0) thanks to clamp)
        return 10.0 * torch.log10(num / den)
    else:
        return num / den  # linear SI-SDR
        

class SiSDRLossFromSTFT(nn.Module):
    """
    SI-SDR loss for batched 2-channel audio given complex STFTs.
    est_stft, ref_stft: [B, 2, F, T], complex dtype
    Returns scalar loss = -mean(SI-SDR_dB over batch & channels).
    """
    def __init__(self, hp, center=True, eps=1e-8, reduction='mean',hop=None):
        super().__init__()
        self.n_fft = hp.stft.fft_length
        self.hop = hop if hop!=None else hp.stft.fft_hop
        self.win = self.n_fft
        self.center = center
        self.eps = eps
        self.reduction = reduction

        # Cache a Hann window (recreated on device/dtype as needed)
        self.register_buffer("_win_buf", torch.tensor([], dtype=torch.float32), persistent=False)

    def _get_window(self, device, dtype):
        if self.hop == self.n_fft:
                self._win_buf = torch.ones(self.win,device=device,dtype=dtype)    
        elif (self._win_buf.numel() == 0) or (self._win_buf.device != device) or (self._win_buf.dtype != dtype):
            self._win_buf = torch.hann_window(self.win, periodic=True, device=device, dtype=dtype)
        
        return self._win_buf

    def forward(self, est_stft: torch.Tensor, ref_stft: torch.Tensor) -> torch.Tensor:
        # Basic checks
        if not est_stft.dtype.is_complex or not ref_stft.dtype.is_complex:
            C = est_stft.shape[2]
            half = C//2
            est_stft_real = est_stft[:,:,:half,:]
            est_stft_imag = est_stft[:,:,half:,:]
            est_stft = torch.complex(est_stft_real,est_stft_imag).permute(0,2,3,1)

            ref_stft_real = ref_stft[:,:,:half,:]
            ref_stft_imag = ref_stft[:,:,half:,:]
            ref_stft = torch.complex(ref_stft_real,ref_stft_imag).permute(0,2,3,1)

        assert est_stft.shape == ref_stft.shape and est_stft.dim() == 4 and est_stft.size(1) == 2, \
            "Expected shape [B, 2, F, T] for both tensors"

        B, C, F, T = est_stft.shape
        device = est_stft.device

        # Use real window dtype = float32 for stability (avoid float16 window)
        win = self._get_window(device=device, dtype=torch.float32)

        # Flatten to [B*C, F, T] for istft vectorization; cast to complex64 to match float32 window
        est_bc = est_stft.to(torch.complex64).reshape(B * C, F, T)
        ref_bc = ref_stft.to(torch.complex64).reshape(B * C, F, T)

        # Reconstruct reference first to get a target length
        ref_time = torch.istft(ref_bc, n_fft=self.n_fft, hop_length=self.hop,
                               win_length=self.win, window=win, center=self.center)

        # If ref is all-zeros for any sample, clamp after the fact
        torch.nan_to_num_(ref_time, nan=0.0, posinf=0.0, neginf=0.0)

        target_len = ref_time.size(-1)

        est_time = torch.istft(est_bc, n_fft=self.n_fft, hop_length=self.hop,
                               win_length=self.win, window=win, center=self.center,
                               length=target_len)

        torch.nan_to_num_(est_time, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape back to [B, C, L] and keep in float32
        ref_time = ref_time.view(B, C, -1).to(torch.float32)
        est_time = est_time.view(B, C, -1).to(torch.float32)

        # Extra guard: if any sample is entirely zero, add a tiny dither (prevents 0/0)
        silent_ref = (ref_time.abs().sum(dim=-1, keepdim=True) == 0)
        if silent_ref.any():
            ref_time = ref_time + silent_ref * (self.eps * torch.randn_like(ref_time))

        sisdr_bc = _sisdr_time_safe(est_time, ref_time, eps=self.eps, return_db=True)  # [B, C]
        loss_per = -sisdr_bc  # maximize SI-SDR

        if self.reduction == 'mean':
            return loss_per.mean()
        elif self.reduction == 'sum':
            return loss_per.sum()
        else:
            return loss_per  # [B, C]

    
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

    
class PESQloss(nn.Module):
    def __init__(self,hp):
        super(PESQloss, self).__init__()
        self.hp = hp
        self.pesq = PesqLoss(factor=hp.loss.pesq_coeff,sample_rate=hp.stft.fs,n_fft=hp.stft.fft_length,hop_length=hp.stft.fft_hop,win_length=hp.stft.fft_length)
    
    def mos(self,x_hat,x):
        x_hat_time = torch.stack([torch.istft(torch.squeeze(x_hat[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x_hat.shape[0])]).flatten(0,1)
        x_time = torch.stack([torch.istft(torch.squeeze(x[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x.shape[0])]).flatten(0,1)
        return self.pesq.mos(x_time,x_hat_time)
        
    def forward(self,x_hat,x):
        x_hat_time = torch.stack([torch.istft(torch.squeeze(x_hat[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x_hat.shape[0])]).flatten(0,1)
        x_time = torch.stack([torch.istft(torch.squeeze(x[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x.shape[0])]).flatten(0,1)
        loss = self.pesq(x_time,x_hat_time).mean()
        return loss