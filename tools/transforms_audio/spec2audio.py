import torch
from torch import nn
from einops import rearrange

class SpecToAudio(nn.Module):
    def __init__(self):
        super().__init__()


    def __call__(self, audio, device):
        # audio: (b, c, t, f)

        # Extract spec feature.
        x = self.spec_to_audio(audio, device)  # (b, c, t)

        # Normalize x 
        x = self.normalize_audio(x)  # (b, c, t)

        return x
    
    # def spec_to_audio(self, x, device):
    #     # x: (b, c*2, t, f)
    #     B,C,T,F = x.shape
    #     if F == 256:
    #         zero_bin = torch.zeros((B, C, T, 1), dtype=x.dtype, device=x.device)
    #         x = torch.cat([zero_bin, x], dim=-1)
    #     # Rearrange for iSTFT
    #     x = rearrange(x, 'b c t f -> b c f t')

    #     # Split two channels
    #     left_channel = x[:, :2, :, :]
    #     right_channel = x[:, 2:, :, :]

    #     left_channel  = rearrange(left_channel, 'b c f t -> b f t c')
    #     right_channel  = rearrange(right_channel, 'b c f t -> b f t c')

    #     # torch.istft requires complex input
    #     left_complex = torch.view_as_complex(left_channel.contiguous()) #[b, f, t]
    #     right_complex = torch.view_as_complex(right_channel.contiguous()) #[b, f, t]

    #     left_complex = self.log2spec(left_complex)
    #     right_complex = self.log2spec(right_complex)

    #     stereo_complex = torch.stack([left_complex, right_complex], dim=1)
        
    #     # List to hold audio for each channel
    #     audio_channels = []
    #     for c in range(stereo_complex.shape[1]):
    #         # Inverse STFT
    #         audio = torch.istft(
    #             input=stereo_complex[:, c],
    #             n_fft=512,
    #             hop_length=128,
    #             win_length=512,
    #             window=torch.hamming_window(512).to(device),
    #             onesided=True,
    #         )
    #         audio_channels.append(audio)
        
    #     # Stack channels back to get (b, c, t) shape
    #     audio = torch.stack(audio_channels, dim=1)

    #     return audio
    def spec_to_audio(self, x, device):
        # x: (b, c*2, t, f)   [c = num_channels]
        B, C, T, F = x.shape
        num_channels = C // 2  # each channel has real+imag

        # Add missing DC bin if needed
        if F == 256:
            zero_bin = torch.zeros((B, C, T, 1), dtype=x.dtype, device=x.device)
            x = torch.cat([zero_bin, x], dim=-1)

        # Rearrange for iSTFT: (b, c*2, f, t)
        x = rearrange(x, 'b c t f -> b c f t')

        # Split into individual channels (each has 2 = real+imag)
        audio_channels = []
        for ch in range(num_channels):
            # Extract the two "subchannels" (real, imag) for this channel
            ch_tensor = x[:, ch*2:(ch+1)*2, :, :]  # (b, 2, f, t)
            ch_tensor = rearrange(ch_tensor, 'b c2 f t -> b f t c2')

            # Convert back to complex
            ch_complex = torch.view_as_complex(ch_tensor.contiguous())  # (b, f, t)
            ch_complex = self.log2spec(ch_complex)

            # Inverse STFT for this channel
            audio = torch.istft(
                input=ch_complex,
                n_fft=512,
                hop_length=128,
                win_length=512,
                window=torch.hamming_window(512).to(device),
                onesided=True,
            )
            audio_channels.append(audio)

        # Stack back to (b, c, t)
        audio = torch.stack(audio_channels, dim=1)
        return audio

    def log2spec(self, spec):
        spec_factor = 0.15
        spec = (torch.exp(spec.abs())-1) * torch.exp(1j * spec.angle())
        spec = spec / spec_factor
        return spec

    def normalize_audio(self, x):
        return x / torch.max(torch.abs(x))
    
 
