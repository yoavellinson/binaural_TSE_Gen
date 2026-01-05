from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn import Module, MultiheadAttention
from torch.nn.parameter import Parameter

class LayerNorm(nn.LayerNorm):

    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transpose = transpose

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose:
            input = input.transpose(-1, -2)  # [B, H, T] -> [B, T, H]
        o = super().forward(input)
        if self.transpose:
            o = o.transpose(-1, -2)
        return o


class BatchNorm1d(nn.Module):

    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__()
        self.transpose = transpose
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose == False:
            input = input.transpose(-1, -2)  # [B, T, H] -> [B, H, T]
        o = self.bn.forward(input)  # accepts [B, H, T]
        if self.transpose == False:
            o = o.transpose(-1, -2)
        return o


class GroupNorm(nn.GroupNorm):

    def __init__(self, transpose: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transpose = transpose

    def forward(self, input: Tensor) -> Tensor:
        if self.transpose == False:
            input = input.transpose(-1, -2)  # [B, T, H] -> [B, H, T]
        o = super().forward(input)  # accepts [B, H, T]
        if self.transpose == False:
            o = o.transpose(-1, -2)
        return o


class GroupBatchNorm(Module):
    """Applies Group Batch Normalization over a group of inputs

    This layer uses statistics computed from input data in both training and
    evaluation modes.
    """

    dim_hidden: int
    group_size: int
    eps: float
    affine: bool
    transpose: bool
    share_along_sequence_dim: bool

    def __init__(
        self,
        dim_hidden: int,
        group_size: int,
        share_along_sequence_dim: bool = False,
        transpose: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        """
        Args:
            dim_hidden (int): hidden dimension
            group_size (int): the size of group
            share_along_sequence_dim (bool): share statistics along the sequence dimension. Defaults to False.
            transpose (bool): whether the shape of input is [B, T, H] or [B, H, T]. Defaults to False, i.e. [B, T, H].
            affine (bool): affine transformation. Defaults to True.
            eps (float): Defaults to 1e-5.
        """
        super(GroupBatchNorm, self).__init__()

        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.transpose = transpose
        self.share_along_sequence_dim = share_along_sequence_dim
        if self.affine:
            if transpose:
                self.weight = Parameter(torch.empty([dim_hidden, 1]))
                self.bias = Parameter(torch.empty([dim_hidden, 1]))
            else:
                self.weight = Parameter(torch.empty([dim_hidden]))
                self.bias = Parameter(torch.empty([dim_hidden]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: shape [B, T, H] if transpose=False, else shape [B, H, T] , where B = num of groups * group size.
        """
        assert (input.shape[0] // self.group_size) * self.group_size, f'batch size {input.shape[0]} is not divisible by group size {self.group_size}'
        if self.transpose == False:
            B, T, H = input.shape
            input = input.reshape(B // self.group_size, self.group_size, T, H)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(input, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(input, dim=(1, 3), unbiased=False, keepdim=True)

            output = (input - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
            output = output.reshape(B, T, H)
        else:
            B, H, T = input.shape
            input = input.reshape(B // self.group_size, self.group_size, H, T)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(input, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(input, dim=(1, 2), unbiased=False, keepdim=True)

            output = (input - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(B, H, T)

        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, {group_size}, share_along_sequence_dim={share_along_sequence_dim}, transpose={transpose}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class NBC2Block(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            n_heads: int,
            dropout: float = 0,
            conv_kernel_size: int = 3,
            n_conv_groups: int = 8,
            norms: Tuple[str, str, str] = ("LN", "GBN", "GBN"),
            group_batch_norm_kwargs: Dict[str, Any] = {
                'group_size': 257,
                'share_along_sequence_dim': False,
            },
    ) -> None:
        super().__init__()
        # self-attention
        self.norm1 = self._new_norm(norms[0], dim_hidden, False, n_conv_groups, **group_batch_norm_kwargs)
        self.self_attn = MultiheadAttention(embed_dim=dim_hidden, num_heads=n_heads, batch_first=True)
        self.cross_attn = MultiheadAttention(embed_dim=dim_hidden, num_heads=n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # Convolutional Feedforward
        self.norm2 = self._new_norm(norms[1], dim_hidden, False, n_conv_groups, **group_batch_norm_kwargs)
        self.linear1 = nn.Linear(dim_hidden, dim_ffn)
        self.conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            self._new_norm(norms[2], dim_ffn, True, n_conv_groups, **group_batch_norm_kwargs),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.linear2 = nn.Linear(dim_ffn, dim_hidden)
        self.dropout2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None,hrtf:Optional[Tensor] = None,gamma:Optional[Tensor] = None, beta:Optional[Tensor] = None) -> Tensor:
        r"""

        Args:
            x: shape [batch, seq, feature]
            att_mask: the mask for attentions. shape [batch, seq, seq]
            hrtf: spatial cond
            gamma: [BF, H]  (from FiLM)
            beta:  [BF, H]

        Shape:
            out: shape [batch, seq, feature]
            attention: shape [batch, head, seq, seq]
        """
        # FiLM at block input
        if gamma is not None and beta is not None:
            x = x * (1 + gamma[:, None, :]) + beta[:, None, :]

        # Self-attention
        x_sa, attn = self._sa_block(self.norm1(x), att_mask)
        x = x + x_sa

        # Cross-attention to HRTF
        if hrtf is not None:
            x_ca, _ = self._ca_block(self.norm1(x), hrtf, attn_mask=None)
            x = x + x_ca

        # FiLM again before FFN 
        if gamma is not None and beta is not None:
            x = x * (1 + gamma[:, None, :]) + beta[:, None, :]

        # FFN
        x_ff = self._ff_block(self.norm2(x))
        x = x + x_ff
        return x, attn

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if isinstance(self.self_attn, MultiheadAttention):
            x, attn = self.self_attn.forward(x, x, x, average_attn_weights=False, attn_mask=attn_mask)
        else:
            x, attn = self.self_attn(x, attn_mask=attn_mask)
        return self.dropout1(x), attn
    
    def _ca_block(self, x: Tensor,hrtf: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        x, attn = self.cross_attn.forward(x, hrtf, hrtf, average_attn_weights=False, attn_mask=attn_mask)

        return self.dropout1(x), attn

    # conv feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.conv(self.linear1(x).transpose(-1, -2)).transpose(-1, -2))
        return self.dropout2(x)

    def _new_norm(self, norm_type: str, dim_hidden: int, transpose: bool, num_conv_groups: int, **freq_norm_kwargs):
        if norm_type == 'LN':
            norm = LayerNorm(normalized_shape=dim_hidden, transpose=transpose)
        elif norm_type == 'GBN':
            norm = GroupBatchNorm(dim_hidden=dim_hidden, transpose=transpose, **freq_norm_kwargs)
        elif norm_type == 'BN':
            norm = BatchNorm1d(num_features=dim_hidden, transpose=transpose)
        elif norm_type == 'GN':
            norm = GroupNorm(num_groups=num_conv_groups, num_channels=dim_hidden, transpose=transpose)
        else:
            raise Exception(norm_type)
        return norm

class NBC2HRTF(nn.Module):

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        n_layers: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        num_freqs: int = 257,
        block_kwargs: Dict[str, Any] = {
            'n_heads': 2,
            'dropout': 0,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'share_along_sequence_dim': False,
            },
        },
    ):
        super().__init__()
        block_kwargs['group_batch_norm_kwargs']['group_size'] = num_freqs

        # encoder
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")
        stride_hrtf = int(dim_hidden / dim_input)
        kernel_size_hrtf = dim_hidden - (dim_input - 1) * stride_hrtf

        self.encoder_hrtf  = nn.ConvTranspose1d(
                                in_channels=1,
                                out_channels=1,
                                kernel_size=kernel_size_hrtf,
                                stride=stride_hrtf
                            )
        # self-attention net
        self.sa_layers = nn.ModuleList()
        for l in range(n_layers):
            self.sa_layers.append(NBC2Block(dim_hidden=dim_hidden, dim_ffn=dim_ffn, **block_kwargs))

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, x: Tensor,hrtf: Tensor) -> Tensor:
        # x: [Batch, NumFreqs, Time, C]
        # hrtf: [Batch,NumFreqs,1,C]
        B, F, T, H = x.shape
        x = x.reshape(B * F, T, H)
        x = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        hrtf = hrtf.reshape(B * F, 1,H)
        hrtf=self.encoder_hrtf(hrtf)
        # attns = []
        for m in self.sa_layers:
            x, attn = m(x,hrtf=hrtf)
            # if i == len(self.sa_layers)//2:
            #     x = x*hrtf
            del attn
            # attns.append(attn)
        y = self.decoder(x)
        y = y.reshape(B, F, T, -1)
        return y.contiguous()  # , attns
    

def get_fbank(sample_rate,nfft,nfilt):
    low_freq_mel = 0
    high_freq_mel = (2595 * torch.log10(torch.tensor(1 + (sample_rate / 2) / 700)))  
    mel_points = torch.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = torch.floor((nfft + 1) * hz_points / sample_rate)
    fbank = torch.zeros((nfilt, int(torch.floor(torch.tensor(nfft / 2 + 1)))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   #left
        f_m = int(bin[m])             #center
        f_m_plus = int(bin[m + 1])    #right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    idxs = {f'bin_{i}': fbank[i].nonzero().flatten() for i in range(nfilt)}
    idxs[f'bin_{nfilt-1}'] = torch.cat((idxs[f'bin_{nfilt-1}'],torch.tensor([nfft//2])))
    idxs[f'bin_{0}'] = torch.cat((torch.tensor([0]),idxs[f'bin_{0}']))

    return fbank,idxs

class mNBC2HRTF2(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        n_layers_mels: int,
        n_layers: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        num_freqs: int = 257,
        num_bins: int = 23,
        sample_rate: int = 16000,
        n_fft: int = 512,
        block_kwargs: Dict[str, Any] = {...},
    ):
        super().__init__()

        fbank,self.bin_indices = get_fbank(sample_rate,n_fft,num_bins)
        self.register_buffer("fbank", fbank, persistent=False)
        # encoder (shared)
        self.encoder = nn.Conv1d(
            dim_input, dim_hidden,
            kernel_size=encoder_kernel_size,
            padding="same"
        )
        self.encoder_nb = nn.Conv1d(
            dim_input, dim_hidden,
            kernel_size=encoder_kernel_size,
            padding="same"
        )

        stride_hrtf = dim_hidden // dim_input
        kernel_size_hrtf = dim_hidden - (dim_input - 1) * stride_hrtf

        self.encoder_hrtf = nn.ConvTranspose1d(
            1, 1, kernel_size=kernel_size_hrtf, stride=stride_hrtf
        )
        self.encoder_hrtf_nb = nn.ConvTranspose1d(
            1, 1, kernel_size=kernel_size_hrtf, stride=stride_hrtf
        )
        block_kwargs["norms"] = ("LN", "LN", "LN")
        block_kwargs["group_batch_norm_kwargs"] = {"share_along_sequence_dim": False, "group_size": 1}

        # NBC2 stacks per bin
        self.bin_blocks = nn.ModuleList()
        for _ in range(num_bins):
            blocks = nn.ModuleList([
                NBC2Block(
                    dim_hidden=dim_hidden,
                    dim_ffn=dim_ffn,
                    **block_kwargs
                )
                for _ in range(n_layers_mels)
            ])
            self.bin_blocks.append(blocks)
        
        self.narrow_band_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.narrow_band_blocks.append(
                NBC2Block(
                    dim_hidden=dim_hidden,
                    dim_ffn=dim_ffn,
                    **block_kwargs
                ))

        # decoder (shared)
        self.decoder = nn.Linear(dim_hidden, dim_output)
        self.decoder_nb = nn.Linear(dim_hidden, dim_output)


    def forward(self, x: Tensor, hrtf: Tensor) -> Tensor:
        # x:    [B, F, T, C]
        # hrtf: [B, F, 1, C]
        B, F, T, C = x.shape
        device = x.device

        y = torch.zeros(
            B, F, T, self.decoder.out_features,
            device=device
        )

        for b, idx in enumerate(self.bin_indices.values()):
            if idx.numel() == 0:
                continue

            xb = x[:, idx]      # [B, Fb, T, C]
            hb = hrtf[:, idx]   # [B, Fb, 1, C]

            Bb, Fb = B, idx.numel()

            # flatten (this is where GPU parallelism happens)
            xb = xb.reshape(Bb * Fb, T, C)
            xb = self.encoder(xb.permute(0, 2, 1)).permute(0, 2, 1)

            hb = hb.reshape(Bb * Fb, 1, C)
            hb = self.encoder_hrtf(hb)

            for block in self.bin_blocks[b]:
                xb, _ = block(xb, hrtf=hb)

            yb = self.decoder(xb)
            y[:, idx] += yb.reshape(Bb, Fb, T, -1)*self.fbank[b][idx][None,:,None,None]

        x_nb = y.reshape(B * F, T, C)
        x_nb = self.encoder_nb(x_nb.permute(0, 2, 1)).permute(0, 2, 1)
        hrtf = hrtf.reshape(B * F, 1,C)
        hrtf=self.encoder_hrtf_nb(hrtf)

        for m in self.narrow_band_blocks:
            x_nb, attn = m(x_nb,hrtf=hrtf)
            del attn
        y = self.decoder(x_nb)
        y = y.reshape(B, F, T, -1)
        return y.contiguous()

        

def build_bin_indices(freq2bin, num_bins):
    return [
        (freq2bin == b).nonzero(as_tuple=True)[0]
        for b in range(num_bins)
    ]

def build_mel_bin_map(
    num_freqs: int,
    sample_rate: int,
    n_fft: int,
    num_bins: int,
):
    freqs = torch.linspace(0, sample_rate / 2, num_freqs)

    def hz_to_mel(f): return 2595 * torch.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10**(m / 2595) - 1)

    mel_edges = torch.linspace(
        hz_to_mel(freqs[0]),
        hz_to_mel(freqs[-1]),
        num_bins + 1
    )
    hz_edges = mel_to_hz(mel_edges)

    bin_ids = torch.bucketize(freqs, hz_edges) - 1
    return bin_ids.clamp(0, num_bins - 1)



class mNBC2HRTF(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        n_layers: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        num_freqs: int = 257,
        num_bins: int = 23,
        sample_rate: int = 16000,
        n_fft: int = 512,
        block_kwargs: Dict[str, Any] = {...},
    ):
        super().__init__()

        self.num_bins = num_bins
        self.num_freqs = num_freqs

        # build bin map (buffer, not parameter)
        bin_map = build_mel_bin_map(
            num_freqs=num_freqs,
            sample_rate=sample_rate,
            n_fft=n_fft,
            num_bins=num_bins,
        )
        self.register_buffer("freq2bin", bin_map, persistent=False)
        self.bin_indices = build_bin_indices(self.freq2bin, num_bins)

        # encoder (shared)
        self.encoder = nn.Conv1d(
            dim_input, dim_hidden,
            kernel_size=encoder_kernel_size,
            padding="same"
        )

        stride_hrtf = dim_hidden // dim_input
        kernel_size_hrtf = dim_hidden - (dim_input - 1) * stride_hrtf

        self.encoder_hrtf = nn.ConvTranspose1d(
            1, 1, kernel_size=kernel_size_hrtf, stride=stride_hrtf
        )
        block_kwargs["norms"] = ("LN", "LN", "LN")
        block_kwargs["group_batch_norm_kwargs"] = {"share_along_sequence_dim": False, "group_size": 1}

        # NBC2 stacks per bin
        self.bin_blocks = nn.ModuleList()
        for _ in range(num_bins):
            blocks = nn.ModuleList([
                NBC2Block(
                    dim_hidden=dim_hidden,
                    dim_ffn=dim_ffn,
                    **block_kwargs
                )
                for _ in range(n_layers)
            ])
            self.bin_blocks.append(blocks)

        # decoder (shared)
        self.decoder = nn.Linear(dim_hidden, dim_output)

    def forward(self, x: Tensor, hrtf: Tensor) -> Tensor:
        # x:    [B, F, T, C]
        # hrtf: [B, F, 1, C]
        B, F, T, C = x.shape
        device = x.device

        y = torch.zeros(
            B, F, T, self.decoder.out_features,
            device=device
        )

        for b, idx in enumerate(self.bin_indices):
            if idx.numel() == 0:
                continue

            xb = x[:, idx]      # [B, Fb, T, C]
            hb = hrtf[:, idx]   # [B, Fb, 1, C]

            Bb, Fb = B, idx.numel()

            # flatten (this is where GPU parallelism happens)
            xb = xb.reshape(Bb * Fb, T, C)
            xb = self.encoder(xb.permute(0, 2, 1)).permute(0, 2, 1)

            hb = hb.reshape(Bb * Fb, 1, C)
            hb = self.encoder_hrtf(hb)

            for block in self.bin_blocks[b]:
                xb, _ = block(xb, hrtf=hb)

            yb = self.decoder(xb)
            y[:, idx] = yb.reshape(Bb, Fb, T, -1)

        return y



class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act  = nn.SiLU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class NBC2HRTF_temb(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        n_layers: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        num_freqs: int = 257,
        block_kwargs: Dict[str, Any] = {
            'n_heads': 2,
            'dropout': 0,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'share_along_sequence_dim': False,
            },
        },
        
    ):
        super().__init__()
        block_kwargs['group_batch_norm_kwargs']['group_size'] = num_freqs


        # self.encoder2d = nn.Sequential(
        #     Conv2dBlock(in_ch=dim_input, out_ch=dim_hidden//2),
        #     Conv2dBlock(in_ch=dim_hidden//2, out_ch=dim_hidden//1),
        #     Conv2dBlock(in_ch=dim_hidden, out_ch=dim_hidden),
        # )
        # self.decoder2d = nn.Sequential(
        #     Conv2dBlock(dim_hidden, dim_hidden),
        #     Conv2dBlock(dim_hidden, dim_hidden),
        #     Conv2dBlock(dim_hidden, dim_hidden//2),
        #     nn.Conv2d(dim_hidden//2, dim_output, kernel_size=1),
        # )
        self.enc1 = Conv2dBlock(in_ch=dim_input, out_ch=dim_hidden//2)
        self.enc2 = Conv2dBlock(in_ch=dim_hidden//2, out_ch=dim_hidden//1)
        self.enc3 = Conv2dBlock(in_ch=dim_hidden, out_ch=dim_hidden)

        self.dec1 = Conv2dBlock(dim_hidden*2, dim_hidden)
        self.dec2 = Conv2dBlock(dim_hidden+(dim_hidden//2), dim_hidden)
        self.dec3 = Conv2dBlock(dim_hidden, dim_hidden//2)
        self.out_proj = nn.Conv2d(dim_hidden//2, dim_output, kernel_size=1)


        self.encoder_hrtf  = nn.Linear(dim_input//2, dim_hidden)

        self.sa_layers = nn.ModuleList()
        for l in range(n_layers):
            self.sa_layers.append(NBC2Block(dim_hidden=dim_hidden, dim_ffn=dim_ffn, **block_kwargs))

        self.film_time = nn.Linear(dim_hidden, 2 * dim_hidden)   # rename old self.film
        self.film_mix  = nn.Linear(dim_hidden, 2 * dim_hidden)

    def forward(self, x: Tensor,hrtf: Tensor,t_emb: Tensor) -> Tensor:
        # x: [Batch, NumFreqs, Time, C]
        # hrtf: [Batch,NumFreqs,1,C]
        B, F, T, H = x.shape

        gamma, beta = self.film_time(t_emb).chunk(2, dim=-1)  # [B, H], [B, H]
        # Repeat per-frequency
        gamma = gamma[:, None, :].repeat(1, F, 1)        # [B, F, H]
        beta  = beta[:, None, :].repeat(1, F, 1)

        # Flatten to [BF, H]
        gamma = gamma.reshape(B * F, -1)
        beta  = beta.reshape(B * F, -1)
        x = x.permute(0, 3, 1, 2)
        # x = self.encoder2d(x)
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        x_latent = self.enc3(f2)
        x = x_latent.permute(0, 2, 3, 1)

        # condition film
        mix_emb = x.mean(dim=(1, 2)) 

        gamma_t, beta_t = self.film_time(t_emb).chunk(2, dim=-1)  # [B, H]
        gamma_x, beta_x = self.film_mix(mix_emb).chunk(2, dim=-1) # [B, H]

        gamma = gamma_t + gamma_x     # [B, H]
        beta  = beta_t  + beta_x      # [B, H]

        gamma = gamma[:, None, :].repeat(1, F, 1)   # [B, F, H]
        beta  = beta[:, None, :].repeat(1, F, 1)

        gamma = gamma.reshape(B * F, -1)            # [BF, H]
        beta  = beta.reshape(B * F, -1)             # [BF, H]

        # ready for NBC
        x = x.reshape(B * F, T, -1)                 # [BF, T, H]

        B, F, _, C_h = hrtf.shape
        # squeeze T=1 and encode per freq
        hrtf = hrtf.squeeze(2)                   # [B, F, C_h]
        hrtf = hrtf.reshape(B * F, C_h)          # [BF, C_h]
        hrtf = self.encoder_hrtf(hrtf)           # [BF, dim_hidden]
        hrtf = hrtf.unsqueeze(1)

        # attns = []
        for m in self.sa_layers:
            x, attn = m(x,hrtf=hrtf,gamma=gamma,beta=beta)
            del attn
        x = x.reshape(B, F, T, -1)
        x = x.permute(0, 3, 1, 2)   # [B, dim_hidden, F, T]

        # y = self.decoder2d(x)
        d1_in = torch.cat([x, f2], dim=1)
        d1 = self.dec1(d1_in)

        d2_in = torch.cat([d1, f1], dim=1)
        d2 = self.dec2(d2_in)

        d3 = self.dec3(d2)
        V_hat = self.out_proj(d3)
        V_hat = V_hat.permute(0, 2, 3, 1)   # [B, F, T, dim_output]

        return V_hat.contiguous()  # , attns


class HRTFEmbEncoder(nn.Module):
    def __init__(self,hidden_dim=96):
        super().__init__()
        # in:  (B, 64, 1029)
        # out: (B, 96, 257)  with kernel_size=4, stride=4
        self.conv = nn.Conv1d(
            in_channels=64,
            out_channels=hidden_dim,
            kernel_size=4,
            stride=4
        )

    def forward(self, x):
        # x: (B, 1029, 64)
        x = x.transpose(1, 2)           # (B, 64, 1029)
        y = self.conv(x)                # (B, 96, 257)
        y = y.transpose(1, 2)           # (B, 257, 96)
        return y

class CrossAttentionFT(nn.Module):
    """
    Cross-attention for tensors shaped:
        X:         [B, F, T, H]
        head_cond: [B, F, 1, H]
    Uses projection to a larger latent space for attention,
    then projects back to original H.
    """

    def __init__(self, H=4, attn_dim=64, num_heads=4, dropout=0.0):
        super().__init__()
        
        self.in_proj = nn.Linear(H, attn_dim)

        # Multihead attention in 64-dim space
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.out_proj = nn.Linear(attn_dim, H)
        self.norm = nn.LayerNorm(H)

    def forward(self, X, head_cond):
        """
        X:         [B, F, T, H]
        head_cond: [B, F, 1, H]
        """
        B, F, T, H = X.shape

        Xp   = self.in_proj(X)          # [B, F, T, attn_dim]
        Cp   = self.in_proj(head_cond)  # [B, F, 1, attn_dim]

        # MultiheadAttention expects [B, seq, dim]
        Q = Xp.reshape(B, F*T, -1)   # [B, F*T, attn_dim]
        K = Cp.reshape(B, F,   -1)   # [B, F,   attn_dim]
        V = K
        out, attn_weights = self.attn(Q, K, V)
        out = out.reshape(B, F, T, -1)
        out = self.out_proj(out)    # [B, F, T, H]
        out = self.norm(out + X)
        return out, attn_weights


class NBC2HRTFCond(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        n_layers: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        num_freqs: int = 257,
        block_kwargs: Dict[str, Any] = {
            'n_heads': 2,
            'dropout': 0,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'share_along_sequence_dim': False,
            },
        },
    ):
        super().__init__()
        block_kwargs['group_batch_norm_kwargs']['group_size'] = num_freqs

        # encoder
        self.dim_hidden=dim_hidden
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")
        stride_hrtf = int(dim_hidden / dim_input)
        kernel_size_hrtf = dim_hidden - (dim_input - 1) * stride_hrtf

        self.encoder_hrtf  = nn.ConvTranspose1d(
                                in_channels=1,
                                out_channels=1,
                                kernel_size=kernel_size_hrtf,
                                stride=stride_hrtf
                            )
        self.head_encoder = HRTFEmbEncoder(hidden_dim=4)
        self.cond_attn = CrossAttentionFT() #MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)
        # self-attention net
        self.sa_layers = nn.ModuleList()
        for l in range(n_layers):
            self.sa_layers.append(NBC2Block(dim_hidden=dim_hidden, dim_ffn=dim_ffn, **block_kwargs))

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, x: Tensor,hrtf: Tensor,head_cond: Tensor) -> Tensor:
        # x: [Batch, NumFreqs, Time, C]
        # hrtf: [Batch,NumFreqs,1,C]
        
        B, F, T, H = x.shape
        head_cond = self.head_encoder(head_cond)
        head_cond = head_cond.reshape(B,F,1,4)

        x, _ = self.cond_attn(x, head_cond)

        x = x.reshape(B * F, T, H)
        x = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        hrtf = hrtf.reshape(B * F, 1,H)
        hrtf=self.encoder_hrtf(hrtf)

        #head conditioning


        # attns = []
        for m in self.sa_layers:
            x, attn = m(x,hrtf=hrtf)
            # if i == len(self.sa_layers)//2:
            #     x = x*hrtf
            del attn
            # attns.append(attn)
        y = self.decoder(x)
        y = y.reshape(B, F, T, -1)
        return y.contiguous()  # , attns

if __name__ == '__main__':
    x = torch.randn((5, 257, 100, 16))
    NBC2_small = NBC2HRTF(
        dim_input=16,
        dim_output=4,
        n_layers=8,
        dim_hidden=96,
        dim_ffn=192,
        block_kwargs={
            'n_heads': 2,
            'dropout': 0,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'group_size': 257,
                'share_along_sequence_dim': False,
            },
        },
    )
    y = NBC2_small(x)
    print(NBC2_small)
    print(y.shape)
    # NBC2_large = NBC2(
    #     dim_input=16,
    #     dim_output=4,
    #     n_layers=12,
    #     dim_hidden=192,
    #     dim_ffn=384,
    #     block_kwargs={
    #         'n_heads': 2,
    #         'dropout': 0,
    #         'conv_kernel_size': 3,
    #         'n_conv_groups': 8,
    #         'norms': ("LN", "GBN", "GBN"),
    #         'group_batch_norm_kwargs': {
    #             'group_size': 257,
    #             'share_along_sequence_dim': False,
    #         },
    #     },
    # )
    # y = NBC2_large(x)
    # print(NBC2_large)
    # print(y.shape)
