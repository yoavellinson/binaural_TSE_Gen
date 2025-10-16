import torch
import torch.nn as nn
from typing import Sequence, Tuple, Dict

class MRSTFTLossMC(nn.Module):
    """
    Multi-Resolution STFT loss for waveforms with channels.
    Accepts [B, C, T] (or [B, T] / [B,1,T]) and averages over batch & channels.

    Args:
        fft_sizes, hop_sizes, win_lengths: per-resolution STFT params
        mag_weight, logmag_weight: weights for |S| and log|S| terms
        center: STFT center flag (must match your pipeline)
        channel_reduce: "mean" or "sum" to combine channel losses
        eps: numerical floor for logs
    """
    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (128, 256, 512),
        win_lengths: Sequence[int] = (512, 1024, 2048),
        mag_weight: float = 1.0,
        logmag_weight: float = 1.0,
        center: bool = True,
        channel_reduce: str = "mean",
        eps: float = 1e-3,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        assert channel_reduce in ("mean", "sum")
        self.fft_sizes = list(fft_sizes)
        self.hops = list(hop_sizes)
        self.wins = list(win_lengths)
        self.mag_w = mag_weight
        self.logmag_w = logmag_weight
        self.center = center
        self.channel_reduce = channel_reduce
        self.eps = eps
        self._win_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _get_window(self, n: int, device, dtype):
        key = (n, device, dtype)
        w = self._win_cache.get(key)
        if w is None:
            w = torch.hann_window(n, device=device, dtype=dtype)
            self._win_cache[key] = w
        return w

    @staticmethod
    def _ensure_bct(x: torch.Tensor) -> torch.Tensor:
        # [B,T] -> [B,1,T]; [B,1,T] stays; [B,C,T] stays
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected [B,T] or [B,C,T], got {tuple(x.shape)}")

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_hat: predicted waveform [B,C,T] / [B,T]
            y:     target   waveform [B,C,T] / [B,T]
        Returns:
            scalar loss
        """
        y_hat = self._ensure_bct(y_hat)
        y     = self._ensure_bct(y)
        B, C, T = y.shape
        device, dtype = y.device, y.dtype

        # Flatten channels into batch: [B*C, T]
        yh = y_hat.reshape(B * C, T)
        yt = y.reshape(B * C, T)

        total = 0.0
        for n_fft, hop, win_len in zip(self.fft_sizes, self.hops, self.wins):
            window = self._get_window(win_len, device, dtype)

            YH = torch.stft(yh, n_fft=n_fft, hop_length=hop, win_length=win_len,
                            window=window, center=self.center, return_complex=True)
            YT = torch.stft(yt, n_fft=n_fft, hop_length=hop, win_length=win_len,
                            window=window, center=self.center, return_complex=True)

            MH, MT = YH.abs(), YT.abs()

            # per-(BC) loss
            mag_l1    = (MH - MT).abs().mean(dim=(1,2))
            logmag_l1 = ((MH + self.eps).log() - (MT + self.eps).log()).abs().mean(dim=(1,2))

            per_bc = self.mag_w * mag_l1 + self.logmag_w * logmag_l1  # [B*C]
            total += per_bc.mean()  # average over (B*C) for this resolution

        return total / len(self.fft_sizes)
