import torch
from torch import nn
from einops import rearrange

# ---------------------- Helpers ----------------------

def safe_unit_norm(x, dim=None, keepdim=False, min_val=1e-8):
    denom = x.abs().amax(dim=dim, keepdim=keepdim)
    return x / denom.clamp_min(min_val)

# ---------------------- Audio -> Spec ----------------------

class AudioToSpec(nn.Module):
    def __init__(self, n_fft=512, hop=128, win_length=512, drop_dc=True, spec_factor=0.15):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length
        self.drop_dc = drop_dc
        self.spec_factor = spec_factor
        # register window once; dtype/device will be matched at use time
        self.register_buffer("win", torch.hamming_window(win_length), persistent=False)

    def forward(self, audio):  # audio: (B, C, T)
        B, C, T = audio.shape
        # Clamp only if >1 to avoid upscaling quiet signals
        peak = audio.abs().amax(dim=2, keepdim=True)
        audio = audio / peak.clamp_min(1.0)

        # STFT per channel (vectorized): (B*C, T) -> (B*C, F, TT)
        x = audio.reshape(B*C, T)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.win.to(x.device, dtype=x.dtype),
            onesided=True,
            return_complex=True,
        )  # (B*C, F, TT)

        # log-polar compression
        spec = self._log_spec(spec)  # complex (B*C, F, TT)

        # complex -> real-imag at last dim, then pack to (B, 2C, T, F)
        spec_ri = torch.view_as_real(spec)                     # (B*C, F, TT, 2)
        spec_ri = rearrange(spec_ri, '(B C) F TT RI -> B (C RI) TT F', B=B, C=C)

        if self.drop_dc:
            spec_ri = spec_ri[..., 1:]  # drop DC bin

        return spec_ri  # (B, 2C, TT, F or F-1)

    def _log_spec(self, S: torch.Tensor) -> torch.Tensor:
        # S complex, output complex with scaled log magnitude and same phase
        mag = torch.log1p(S.abs())                  # log(1+|S|)
        out = (self.spec_factor * mag) * torch.exp(1j * S.angle())
        return out

# ---------------------- Spec -> Audio ----------------------

class SpecToAudio(nn.Module):
    def __init__(self, n_fft=512, hop=128, win_length=512, had_dc=False, spec_factor=0.15):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length
        self.had_dc = had_dc          # True if your specs include DC; False if you dropped it
        self.spec_factor = spec_factor
        self.register_buffer("win", torch.hamming_window(win_length), persistent=False)

    def forward(self, spec_ri):  # spec_ri: (B, 2C, TT, F or F-1)
        B, twoC, TT, Fsub = spec_ri.shape
        C = twoC // 2

        # restore DC if it was dropped
        if not self.had_dc and Fsub == self.n_fft // 2:
            z = torch.zeros((B, twoC, TT, 1), dtype=spec_ri.dtype, device=spec_ri.device)
            spec_ri = torch.cat([z, spec_ri], dim=-1)  # now F = n_fft//2 + 1

        # -> (B*C, F, TT, 2) then complex
        spec_ri = rearrange(spec_ri, 'B (C RI) TT F -> (B C) F TT RI', C=C, RI=2)
        Z = torch.view_as_complex(spec_ri.contiguous())  # (B*C, F, TT) complex

        # invert log-polar compression (correct math)
        Z = self._log2spec(Z)

        # one istft over the flattened batch
        audio = torch.istft(
            Z,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.win.to(Z.device, dtype=Z.real.dtype),
            onesided=True,
        )  # (B*C, Ttime)

        audio = rearrange(audio, '(B C) T -> B C T', B=B, C=C)
        # safe per-sample normalization (optional)
        audio = audio / audio.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
        return audio

    def _log2spec(self, Z: torch.Tensor) -> torch.Tensor:
        # Z complex where |Z| = spec_factor * log(1+|S|).
        mag_out = Z.abs()
        phase = torch.exp(1j * Z.angle())
        mag_in = torch.exp(mag_out / self.spec_factor) - 1.0  # <- correct inverse
        return mag_in * phase
