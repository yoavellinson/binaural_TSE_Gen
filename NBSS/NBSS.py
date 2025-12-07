from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch import Tensor
from .NBC2 import NBC2HRTF,NBC2HRTFCond,NBC2HRTF_temb
import math


class NBSS(nn.Module):
    """Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training.

    A module version of NBSS which takes time domain signal as input, and outputs time domain signal.

    Arch could be NB-BLSTM or NBC
    """

    def __init__(
            self,
            hp      
    ):
        super().__init__()
        
        self.hp=hp
        self.register_buffer('window', torch.hann_window(hp.stft.fft_length), False)  # self.window, will be moved to self.device at training time
        self.ref_channel = hp.model.ref_channel
        self.n_channel = hp.model.n_channel
        self.n_channel_out = hp.model.output_channels
        arch_kwargs= {  "n_layers": hp.model.n_layers,
                        "dim_hidden": hp.model.dim_hidden,
                        "dim_ffn": hp.model.dim_ffn,
                        "block_kwargs": {
                            'n_heads': hp.model.block.n_heads,
                            'dropout': hp.model.block.dropout,
                            'conv_kernel_size': hp.model.block.conv_kernel_size,
                            'n_conv_groups': hp.model.block.n_conv_groups,
                            'norms': tuple(hp.model.block.norms),
                            'group_batch_norm_kwargs': {
                                'group_size': hp.model.block.group_batch_norm.group_size,
                                'share_along_sequence_dim': hp.model.block.group_batch_norm.share_along_sequence_dim,
                            },
                        }
                    }
        self.arch = NBC2HRTF(dim_input=self.n_channel * 2, dim_output=self.n_channel_out * 2, **arch_kwargs)     

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
        output = output.reshape(B, F, TF, self.n_channel_out, 2)
        output = torch.view_as_complex(output)  # [B, F, TF, S]
        y_hat = output.permute(0,3,1,2)
        return y_hat
    


class NBSS_CFM(nn.Module):
    """Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training.

    A module version of NBSS which takes time domain signal as input, and outputs time domain signal.

    Arch could be NB-BLSTM or NBC
    """

    def __init__(
            self,
            hp      
    ):
        super().__init__()
        
        self.hp=hp
        self.register_buffer('window', torch.hann_window(hp.stft.fft_length), False)  # self.window, will be moved to self.device at training time
        self.ref_channel = hp.model.ref_channel
        self.n_channel = hp.model.n_channel
        self.n_channel_out = hp.model.output_channels
        arch_kwargs= {  "n_layers": hp.model.n_layers,
                        "dim_hidden": hp.model.dim_hidden,
                        "dim_ffn": hp.model.dim_ffn,
                        "block_kwargs": {
                            'n_heads': hp.model.block.n_heads,
                            'dropout': hp.model.block.dropout,
                            'conv_kernel_size': hp.model.block.conv_kernel_size,
                            'n_conv_groups': hp.model.block.n_conv_groups,
                            'norms': tuple(hp.model.block.norms),
                            'group_batch_norm_kwargs': {
                                'group_size': hp.model.block.group_batch_norm.group_size,
                                'share_along_sequence_dim': hp.model.block.group_batch_norm.share_along_sequence_dim,
                            },
                        }
                    }
        self.arch = NBC2HRTF_temb(dim_input=self.n_channel * 2, dim_output=self.n_channel_out * 2, **arch_kwargs)     
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hp.model.time_dim),
            nn.Linear(hp.model.time_dim, hp.model.dim_hidden),
            nn.SiLU(),
            nn.Linear(hp.model.dim_hidden, hp.model.dim_hidden)
        )
    def forward(self, x: Tensor,hrtf: Tensor,s_t:Tensor,t:Tensor) -> Tensor:
        """forward

        Args:
            x: time stft signal, shape [batch, channel, freq,time frame]
            hrtf: spatial clue
            s_t: current state
            t: timestamp (scalar)
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

        S = s_t.permute(0,2,3,1)               # [B,F,T,C_out]
        S = torch.view_as_real(S).reshape(B, F, T, self.n_channel_out * 2)

        #time emb
        t_emb = self.time_mlp(t)               # [B, dim_hidden]
        # network processing
        inp = torch.cat([X, S], dim=-1) 
        V_hat = self.arch(inp, hrtf, t_emb)    # [B,F,T, C_out*2]
        V_hat = V_hat.reshape(B, F, T, self.n_channel_out, 2)
        V_hat = torch.view_as_complex(V_hat).permute(0,3,1,2)
        return V_hat

class SinusoidalPosEmb(nn.Module):
    """
    t -> [sin(freq * t), cos(freq * t), ...]
    Standard timestep embedding used in diffusion / flow models.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] or [B,1]
        returns: [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        # Exponential frequencies: 10^(−4) → 10^4 (same as transformer)
        emb = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Outer product: [B,1] * [half_dim] -> [B,half_dim]
        args = t.unsqueeze(1) * freqs.unsqueeze(0)

        # Concatenate sin + cos
        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        # If dim is odd, pad by one
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0,1))

        return emb


class NBSSCond(nn.Module):
    """Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training.

    A module version of NBSS which takes time domain signal as input, and outputs time domain signal.

    Arch could be NB-BLSTM or NBC
    """

    def __init__(
            self,
            hp      
    ):
        super().__init__()
        
        self.hp=hp
        self.register_buffer('window', torch.hann_window(hp.stft.fft_length), False)  # self.window, will be moved to self.device at training time
        self.ref_channel = hp.model.ref_channel
        self.n_channel = hp.model.n_channel
        self.n_channel_out = hp.model.output_channels
        arch_kwargs= {  "n_layers": hp.model.n_layers,
                        "dim_hidden": hp.model.dim_hidden,
                        "dim_ffn": hp.model.dim_ffn,
                        "block_kwargs": {
                            'n_heads': hp.model.block.n_heads,
                            'dropout': hp.model.block.dropout,
                            'conv_kernel_size': hp.model.block.conv_kernel_size,
                            'n_conv_groups': hp.model.block.n_conv_groups,
                            'norms': tuple(hp.model.block.norms),
                            'group_batch_norm_kwargs': {
                                'group_size': hp.model.block.group_batch_norm.group_size,
                                'share_along_sequence_dim': hp.model.block.group_batch_norm.share_along_sequence_dim,
                            },
                        }
                    }
        self.arch = NBC2HRTFCond(dim_input=self.n_channel * 2, dim_output=self.n_channel_out * 2, **arch_kwargs)     

    def forward(self, x: Tensor,hrtf: Tensor,head_cond: Tensor) -> Tensor:
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
        # hrtf=torch.view_as_real(hrtf)
        # hrtf = hrtf.reshape(B, F, 1, C * 2)

        # head_cond = head_cond.flatten(1)
        # network processing
        output = self.arch(X,hrtf,head_cond)

        # to complex
        output = output.reshape(B, F, TF, self.n_channel_out, 2)
        output = torch.view_as_complex(output)  # [B, F, TF, S]
        y_hat = output.permute(0,3,1,2)
        return y_hat


if __name__ == '__main__':
    x = torch.randn(size=(3,2,257,626),dtype=torch.complex64)
    ys = torch.randn(size=(3,2,257,626),dtype=torch.complex64)
    hrtf = torch.randn(size=(3,2,257),dtype=torch.complex64)
    hp = OmegaConf.load('/home/workspace/yoavellinson/binaural_TSE_Gen/conf/extraction_nbss_conf.yml')
    NBSS_with_NBC_small = NBSS(hp)

    Ys_hat = NBSS_with_NBC_small(x,hrtf)
    n_fft = 512
    n_overlap=128
    window = torch.hann_window(n_fft)
    B,C,F,T = x.shape
    ys_hat = torch.istft(Ys_hat.reshape(B * C,F,T), n_fft=n_fft, hop_length=n_overlap, window=window, win_length=n_fft )
    ys_hat=ys_hat.reshape(B, C, -1)
    ys = torch.istft(x.reshape(B * C,F,T), n_fft=n_fft, hop_length=n_overlap, window=window, win_length=n_fft)
    ys=ys.reshape(B, C, -1)

    # neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    # print(ys_hat.shape, neg_sisdr_loss.mean())
