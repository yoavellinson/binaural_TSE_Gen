from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

from .NBC2 import NBC2HRTF


def neg_si_sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    si_snr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_snr_val.view(batch_size, -1), dim=1)

def pit_sisdr_stft(pred,target,hp):
    B,C,F,T = target.shape
    device = pred.device
    target=target.to(device)
    window = torch.hann_window(hp.stft.fft_length,device=device)
    pred = torch.istft(pred.reshape(B * C,F,T), n_fft=hp.stft.fft_length, hop_length=hp.stft.fft_hop, window=window, win_length=hp.stft.fft_length )
    pred=pred.reshape(B, C, -1)
    target = torch.istft(target.reshape(B * C,F,T), n_fft=hp.stft.fft_length, hop_length=hp.stft.fft_hop, window=window, win_length=hp.stft.fft_length )
    target=target.reshape(B, C, -1)
    neg_sisdr_val = neg_si_sdr(pred,target)
    loss = neg_sisdr_val.mean()
    return loss

class NBSS(nn.Module):
    """Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training.

    A module version of NBSS which takes time domain signal as input, and outputs time domain signal.

    Arch could be NB-BLSTM or NBC
    """

    def __init__(
            self,
            n_channel: int = 8,
            n_speaker: int = 2,
            n_fft: int = 512,
            n_overlap: int = 256,
            ref_channel: int = 0,
            arch: str = "NB_BLSTM",  # could also be NBC, NBC2
            arch_kwargs: Dict[str, Any] = dict(),
    ):
        super().__init__()
        self.arch = NBC2HRTF(dim_input=n_channel * 2, dim_output=n_speaker * 2, **arch_kwargs)     

        self.register_buffer('window', torch.hann_window(n_fft), False)  # self.window, will be moved to self.device at training time
        self.n_fft = n_fft
        self.n_overlap = n_overlap
        self.ref_channel = ref_channel
        self.n_channel = n_channel
        self.n_speaker = n_speaker

    def forward(self, x: Tensor,hrtf: Tensor) -> Tensor:
        """forward

        Args:
            x: time stft signal, shape [batch, channel, freq,time frame]

        Returns:
            y: the predicted time domain signal, shape [batch, channel, freq,time frame]
        """

        # STFT
        B,C,F,T = x.shape
        X = x.permute(0, 2, 3, 1)  # (batch, freq, time frame, channel)
        hrtf = hrtf.permute(0,2,1).unsqueeze(-2)
        # normalization by using ref_channel
        F, TF = X.shape[1], X.shape[2]
        Xr = X[..., self.ref_channel].clone()  # copy
        XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of X
        X[:, :, :, :] /= (XrMM.reshape(B, F, 1, 1) + 1e-8)

        # to real
        X = torch.view_as_real(X)  # [B, F, T, C, 2]
        X = X.reshape(B, F, TF, C * 2)
        hrtf=torch.view_as_real(hrtf)
        hrtf = hrtf.reshape(B, F, 1, C * 2)
        # network processing
        output = self.arch(X,hrtf)

        # to complex
        output = output.reshape(B, F, TF, self.n_speaker, 2)
        output = torch.view_as_complex(output)  # [B, F, TF, S]
        y_hat = output.permute(0,3,1,2)
        return y_hat


if __name__ == '__main__':
    x = torch.randn(size=(3,2,257,626),dtype=torch.complex64)
    ys = torch.randn(size=(3,2,257,626),dtype=torch.complex64)
    hrtf = torch.randn(size=(3,2,257),dtype=torch.complex64)
    # NBSS_with_NB_BLSTM = NBSS(n_channel=8, n_speaker=2, arch="NB_BLSTM")
    # ys_hat = NBSS_with_NB_BLSTM(x)
    # neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    # print(ys_hat.shape, neg_sisdr_loss.mean())

    # NBSS_with_NBC = NBSS(n_channel=8, n_speaker=2, arch="NBC")
    # ys_hat = NBSS_with_NBC(x)
    # neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    # print(ys_hat.shape, neg_sisdr_loss.mean())

    NBSS_with_NBC_small = NBSS(n_channel=2,
                               n_speaker=2,
                               arch="NBC2",
                               arch_kwargs={
                                   "n_layers": 8, # 12 for large
                                   "dim_hidden": 96, # 192 for large
                                   "dim_ffn": 192, # 384 for large
                                   "block_kwargs": {
                                       'n_heads': 2,
                                       'dropout': 0,
                                       'conv_kernel_size': 3,
                                       'n_conv_groups': 8,
                                       'norms': ("LN", "GBN", "GBN"),
                                       'group_batch_norm_kwargs': {
                                           'group_size': 257, # 129 for 8k Hz
                                           'share_along_sequence_dim': False,
                                       },
                                   }
                               },)
    
    Ys_hat = NBSS_with_NBC_small(x,hrtf)
    n_fft = 512
    n_overlap=128
    window = torch.hann_window(n_fft)
    B,C,F,T = x.shape
    ys_hat = torch.istft(Ys_hat.reshape(B * C,F,T), n_fft=n_fft, hop_length=n_overlap, window=window, win_length=n_fft )
    ys_hat=ys_hat.reshape(B, C, -1)
    ys = torch.istft(x.reshape(B * C,F,T), n_fft=n_fft, hop_length=n_overlap, window=window, win_length=n_fft)
    ys=ys.reshape(B, C, -1)

    neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    print(ys_hat.shape, neg_sisdr_loss.mean())
