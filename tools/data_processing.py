import os
import pdb
import torch
from einops import rearrange


def data_processing(
    data: dict,
    preprocess_func: callable, 
    device: str
):
    dataset_name = data["dataset_name"][0]

    if dataset_name == "MUSDB18HQ":

        # Condition: mono mixture
        mono_conditions = data["mixture"].mean(dim=1).unsqueeze(1).to(device)
        # Input: stereo mixture
        mixture = data["mixture"].to(device)
    else:
        raise NotImplementedError(dataset_name)

    # 'Preprocess function' is used when converting waveform to freq domain
    if preprocess_func is not None:
        # Target data
        x = preprocess_func(mixture, device)  # (b, c, t, f)
        # Condition
        cond_tf = preprocess_func(mono_conditions, device)  # (b, c, t, f)
    else:
        x = mixture  # (b, c, t)
        cond_tf = mono_conditions # (b, c, t)
 
    return x, cond_tf

def complex_to_interleaved(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex tensor [..., C, T, F] -> real tensor [..., 2C, T, F]
    Channels alternate real/imag.
    """
    if not x.is_complex():
        raise TypeError(f"Expected complex tensor, got dtype={x.dtype}")

    *batch, C, T, F = x.shape
    out = torch.empty(*batch, 2 * C, T, F, dtype=x.real.dtype, device=x.device)
    out[..., 0::2, :, :] = x.real
    out[..., 1::2, :, :] = x.imag
    out = rearrange(out,'b c f t -> b c t f')
    out = out[:,:,:,1:]
    return out

def interleaved_to_complex(y: torch.Tensor, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    Convert real tensor [..., 2C, T, F] -> complex tensor [..., C, T, F].
    Assumes channels alternate real/imag.
    """
    if y.shape[-3] % 2 != 0:
        raise ValueError(f"Channel dimension must be even; got {y.shape[-3]}")
    if y.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"Expected real float tensor, got dtype={y.dtype}")

    Re = y[..., 0::2, :, :]
    Im = y[..., 1::2, :, :]
    out = torch.complex(Re, Im).to(dtype)
    out = rearrange(out,'b c t f -> b c f t')
    B,C,F,T = out.shape
    if F == 256:
        zero_bin = torch.zeros((B, C, 1, T), dtype=out.dtype, device=out.device)
        out = torch.cat([zero_bin, out], dim=2)
    return out


def iSTFT(x,n_fft=512,hop_length=128):
    B, C, F, T = x.shape
    X_bc = x.reshape(B * C, F, T)
    x_time = torch.istft(torch.squeeze(X_bc).detach().cpu(),n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft))
    x_time = x_time.view(B,C,-1)
    x_time = x_time/(x_time.abs().max(dim=-1, keepdim=True).values)
    return x_time