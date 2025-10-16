import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
# from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torch_pesq import PesqLoss
from scipy.signal import deconvolve
    
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
    
class PITSiSDRLoss(nn.Module):
    def __init__(self, hp):
        super().__init__()
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
        if diag_si_snr[0][0] > diag_si_snr[0][1]:
            permutation=0
        else:
            permutation=1
        si_sdr_per_sample = diag_si_snr.max()
        return si_sdr_per_sample,permutation
        
    def forward(self, x_hat_time, x_time):
        '''
        h_hat_time: B,1,T
        h_time: B,2,T - 2 it the two possible sources
        '''
        x_hat_time = x_hat_time.repeat(1,2,1)
        sisdr_loss,per = self.si_sdr_calc(x_hat_time,x_time)
        return -sisdr_loss,per
    
    def forward_stft(self, x_hat, x):
        '''
        h_hat_time: B,1,T
        h_time: B,2,T - 2 it the two possible sources
        '''
        x_hat_time = torch.stack([torch.istft(torch.squeeze(x_hat[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x_hat.shape[0])])
        x_time = torch.stack([torch.istft(torch.squeeze(x[b,:,:,:]).cpu(),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, window=torch.hann_window(self.hp.stft.fft_length)) for b in range(x.shape[0])])
        x_hat_time = x_hat_time.repeat(1,2,1)
        sisdr_loss = self.si_sdr_calc(x_hat_time,x_time)
        return -sisdr_loss
    

# class ECAPA2_loss(nn.Module):
#     def __init__(self, hp,device):
#         super().__init__()
#         self.hp = hp
#         ecapa2_path = hp.loss.ecapa2.path
#         self.ecapa2 = torch.jit.load(ecapa2_path,map_location=device)
#         self.l2 = torch.nn.MSELoss()
#         self.device=device
#     def embedding_normalize(self,embs, use_std=True, eps=1e-10):
#         embs = embs - embs.mean()
#         if use_std:
#             embs = embs / (embs.std() + eps)
#         embs_l2_norm = torch.unsqueeze(torch.norm(embs, p=2, dim=-1), dim=1) #np.expand_dims(np.linalg.norm(embs, ord=2, axis=-1), axis=1)
#         embs = embs / embs_l2_norm
#         return embs

#     def compute_affinity_matrix(self,embeddings):
#         l2_norms = torch.norm(embeddings, dim=1)
#         embeddings_normalized = embeddings / l2_norms[:, None]

#         cosine_similarities = embeddings_normalized@embeddings_normalized.T #np.matmul(embeddings_normalized,np.transpose(embeddings_normalized))

#         affinity = 1- (cosine_similarities + 1.0) / 2.0
#         sigma = torch.sort(affinity**2,1)[0].max()
#         affinity = torch.exp(-(affinity**2)/sigma)
#         return affinity
    
#     def forward(self,x_hat_tuple,s_tuple):
#         #for non batched
#         x_left,x_right = x_hat_tuple
#         s1,s2=s_tuple
#         emb_s1 = self.ecapa2(s1.unsqueeze(0).to(self.device))
#         emb_s2 = self.ecapa2(s2.unsqueeze(0).to(self.device))
#         emb_left = self.ecapa2(x_left.unsqueeze(0))
#         emb_right = self.ecapa2(x_right.unsqueeze(0))
#         emb_s1,emb_s2,emb_left,emb_right = (self.embedding_normalize(emb_s1),
#                                             self.embedding_normalize(emb_s2),
#                                             self.embedding_normalize(emb_left),
#                                             self.embedding_normalize(emb_right))
#         emb = torch.cat((emb_s1,emb_s2,emb_left,emb_right),dim=0)
#         aff = self.compute_affinity_matrix(emb)
        
#         s1s2_s1s2 = aff[:2,:2]
#         lr_lr = aff[-2:,-2:]
#         lr_s1s2 = aff[-2:,:2]

#         #1
#         out = self.l2(lr_lr,s1s2_s1s2)

#         #2
#         trace_main = lr_s1s2[0,0] + lr_s1s2[1,1]
#         trace_sec =  lr_s1s2[0,1] + lr_s1s2[1,0]
#         if trace_sec > trace_main:
#             tmp = lr_s1s2.clone()
#             lr_s1s2[0,:] = tmp[1,:]
#             lr_s1s2[1,:] = tmp[0,:]
#         out+= self.l2(lr_s1s2,s1s2_s1s2)
#         return out
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECAPA2_loss(nn.Module):
    def __init__(self, hp, device):
        super().__init__()
        self.hp = hp
        self.device = device
        self.ecapa2 = torch.jit.load(hp.loss.ecapa2.path, map_location=device)
        self.l2 = nn.MSELoss(reduction='mean')

    @staticmethod
    def embedding_normalize(embs: torch.Tensor, use_std=True, eps=1e-10):
        """
        embs: [B, D]
        - center per sample
        - optional std-norm per sample
        - final L2 normalize per sample
        """
        embs = embs - embs.mean(dim=-1, keepdim=True)
        if use_std:
            embs = embs / (embs.std(dim=-1, keepdim=True) + eps)
        l2 = torch.norm(embs, p=2, dim=-1, keepdim=True).clamp_min(eps)
        return embs / l2

    @staticmethod
    def compute_affinity_matrix(embeddings: torch.Tensor, eps=1e-12):
        """
        embeddings: [B, N, D]  (here N=4: [s1, s2, left, right])
        Returns:
          aff: [B, N, N]
        """
        # L2-normalize over D
        l2 = torch.norm(embeddings, dim=-1, keepdim=True).clamp_min(eps)
        E = embeddings / l2  # [B, N, D]

        # Cosine similarity matrix per batch: E @ E^T
        cos = torch.bmm(E, E.transpose(1, 2))  # [B, N, N], in [-1, 1]

        # Your affinity transform
        aff = 1.0 - (cos + 1.0) / 2.0          # in [0, 1]
        sigma = (aff ** 2).amax(dim=(1, 2), keepdim=True).clamp_min(eps)
        aff = torch.exp(-(aff ** 2) / sigma)   # [B, N, N]
        return aff

    def forward(self, x_hat_tuple, s_tuple):
        """
        x_hat_tuple: (x_left, x_right) each [B, T]
        s_tuple:     (s1, s2)          each [B, T]
        Returns scalar loss.
        """
        x_left, x_right = x_hat_tuple
        s1, s2 = s_tuple

        # Embeddings: [B, D]
        emb_s1   = self.ecapa2(s1.to(self.device))
        emb_s2   = self.ecapa2(s2.to(self.device))
        emb_left = self.ecapa2(x_left.to(self.device))
        emb_right= self.ecapa2(x_right.to(self.device))

        # Normalize per sample
        emb_s1   = self.embedding_normalize(emb_s1)
        emb_s2   = self.embedding_normalize(emb_s2)
        emb_left = self.embedding_normalize(emb_left)
        emb_right= self.embedding_normalize(emb_right)

        # Stack into [B, 4, D]: order = [s1, s2, left, right]
        emb = torch.stack([emb_s1, emb_s2, emb_left, emb_right], dim=1)

        # Affinity per batch: [B, 4, 4]
        aff = self.compute_affinity_matrix(emb)

        # Blocks
        s1s2_s1s2 = aff[:, :2, :2]   # [B,2,2]
        lr_lr     = aff[:, -2:, -2:] # [B,2,2]
        lr_s1s2   = aff[:, -2:, :2]  # [B,2,2]

        # Term 1: L2 between LR-LR and S-S
        loss_1 = self.l2(lr_lr, s1s2_s1s2)

        # Term 2: choose row permutation of lr_s1s2 that maximizes diagonal sum (PIT over {identity, swap})
        # trace_main = lr_s1s2[0,0] + lr_s1s2[1,1], trace_sec = lr_s1s2[0,1] + lr_s1s2[1,0] per batch
        trace_main = lr_s1s2[:, 0, 0] + lr_s1s2[:, 1, 1]  # [B]
        trace_sec  = lr_s1s2[:, 0, 1] + lr_s1s2[:, 1, 0]  # [B]
        swap_mask  = (trace_sec > trace_main)             # [B] boolean

        # Build indices per batch: if swap -> [1,0], else [0,1]
        base_idx   = torch.tensor([0, 1], device=lr_s1s2.device)
        swap_idx   = torch.tensor([1, 0], device=lr_s1s2.device)
        row_idx    = torch.where(swap_mask.unsqueeze(1), swap_idx.unsqueeze(0), base_idx.unsqueeze(0))  # [B,2]

        # Gather rows without in-place ops (keeps autograd happy)
        lr_s1s2_perm = lr_s1s2.gather(
            dim=1,
            index=row_idx.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B,2,2]

        loss_2 = self.l2(lr_s1s2_perm, s1s2_s1s2)

        return loss_1 + loss_2
