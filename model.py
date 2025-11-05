import torch.nn as nn
import torch
import numpy as np
import torch.functional as F
import math
eps = torch.exp(torch.tensor(-6))

DTYPE=torch.complex64
       
class CLeakyReLU(nn.LeakyReLU):
    def forward(self, xr, xi):
        return F.leaky_relu(xr, self.negative_slope, self.inplace),\
                F.leaky_relu(xi, self.negative_slope, self.inplace)

class complex_relu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,z):
        return torch.complex(torch.nn.ReLU()(z.real), torch.nn.ReLU()(z.imag))



class Pos2DRFF(nn.Module):
    """
    Random Fourier Features for 2D positions (RBF kernel).
    xy: [B, N, 2]  ->  out: [B, Cpos or out_channels, N, T]
    
    Similarity ~ exp(-||x - y||^2 / (2*sigma^2)), so close positions are similar.
    All features are inherently relative to (0,0) because the kernel is centered there.
    """
    def __init__(self, num_feats=32, sigma=1.0, include_xy=False, out_channels=None,dtype=torch.float64):
        """
        num_feats   : number of base frequencies (the output is 2*num_feats due to cos/sin)
        sigma       : RBF bandwidth (larger => smoother, farther points still similar)
        include_xy  : optionally append (scaled) raw xy
        out_channels: if set, project to this channel size via 1x1 Conv2d
        """
        super().__init__()
        assert num_feats > 0
        self.num_feats = num_feats
        self.sigma = float(sigma)
        self.include_xy = include_xy
        self.out_channels = out_channels

        # Sample RFF matrix: W ~ N(0, 1/sigma^2 I). Shape [2, num_feats]
        W = torch.randn(2, num_feats) / self.sigma
        self.register_buffer("W", W, persistent=True)

        cpos = 2 * num_feats + (2 if include_xy else 0)  # cos+sin (+ raw xy)
        self.proj = nn.Conv2d(cpos, out_channels, kernel_size=1,dtype=dtype) if out_channels is not None else None

    def forward(self, xy, T, kpm=None, scale=(1.0, 1.0)):
        """
        xy  : [B, N, 2]
        T   : int, replicate along last dim
        kpm : optional mask [B, N] — True => zero out
        scale: (Sx, Sy) to normalize coordinates before encoding
        """
        B, N, two = xy.shape
        assert two == 2
        device, dtype = xy.device, xy.dtype

        # Clean + normalize
        xy = torch.nan_to_num(xy, nan=0.0)
        Sx, Sy = scale
        xy = xy / torch.tensor([Sx, Sy], device=device, dtype=dtype).view(1, 1, 2)

        # RFF mapping: phi(x) = sqrt(2/m) [cos(xW), sin(xW)]
        z = xy @ self.W.to(dtype)                  # [B, N, num_feats]
        phi = math.sqrt(2.0 / self.num_feats) * torch.cat([torch.cos(z), torch.sin(z)], dim=-1)  # [B,N,2*num_feats]

        if self.include_xy:
            phi = torch.cat([phi, xy], dim=-1)     # [B,N,2*num_feats+2]

        pos = phi.permute(0, 2, 1).unsqueeze(-1)   # [B,Cpos,N,1]
        pos = pos.expand(-1, -1, -1, T).contiguous()

        if kpm is not None:
            pos = pos.masked_fill(kpm[:, None, :, None], 0.0)

        if self.proj is not None:
            pos = self.proj(pos)                   # [B,out_channels,N,T]

        return pos

    
class MaskedMean(nn.Module):
    def forward(self, x, kpm=None):           # x: [B,C,N,T]
        x = x.mean(dim=-1)                    # [B,C,N]
        if kpm is None:
            return x.mean(dim=-1)             # [B,C]
        valid = (~kpm).float()                # [B,N]
        denom = valid.sum(-1, keepdim=True).clamp_min(1.0)
        return (x * valid.unsqueeze(1)).sum(-1) / denom


class GlobalToLatent(nn.Module):
    def __init__(self, in_dim=512, latent_dim=128,dtype=DTYPE):
        super().__init__()
        self.mu     = nn.Sequential(GlobLNComplex(in_dim), nn.Linear(in_dim, latent_dim,dtype=dtype))
        self.logvar = nn.Sequential(GlobLNComplex(in_dim), nn.Linear(in_dim, latent_dim,dtype=dtype))
    def forward(self, h):  # h: [B,in_dim]
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class CHRTFtoLatent(nn.Module):
    def __init__(self, hp,d_model=128,latent_dim=128):
        super().__init__()
        self.hp = hp
        kernel_size=(1,4)
        stride=(1,2)
        self.hrtf_encoder = CEncoder(self.hp, ic=self.hp.dataset.n_channels, ngf=16,kernel_size=kernel_size, stride=stride,padding=(0,1))
        self.pool = MaskedMean()
        self.to_latent = GlobalToLatent(in_dim=d_model, latent_dim=latent_dim)
        self.pos_enc = Pos2DRFF(num_feats=32, sigma=0.5, include_xy=False, out_channels=d_model)

    def forward(self,hrtf,pos,kpm):
        conv1, conv2, conv3, conv4, bottleneck = self.hrtf_encoder(hrtf)  # bottleneck:[B,512,N,32]
        T_b = bottleneck.shape[-1]            # e.g., 64 or 32
        pos_enc = self.pos_enc(pos, T=T_b, kpm=kpm, scale=(360,90)) 
        bottleneck += pos_enc
        # Global latent
        h_global = self.pool(bottleneck, kpm)    # [B,512]
        z, mu, logvar = self.to_latent(h_global) # [B,d], [B,d], [B,d]
        return z


class CFiLM2dComplex(nn.Module):
    """
    Fully complex FiLM:
      y = gamma * x + beta,  with gamma, beta complex and per-channel.
    """
    def __init__(self, z_dim: int, C: int, use_bias: bool = True, use_nonlinearity: bool = False):
        super().__init__()
        self.C = C
        self.use_bias = use_bias
        self.fc1 = nn.Linear(z_dim, max(z_dim, 2*C),dtype=DTYPE)
        self.act = complex_relu() if use_nonlinearity else nn.Identity()
        self.fc2 = nn.Linear(max(z_dim, 2*C), (2 if use_bias else 1) * C,dtype=DTYPE)
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] complex
        z: [B, z_dim] complex
        """
        B, C = x.shape[:2]
        h = self.fc1(z)           # [B, Hc] complex
        h = self.act(h)
        params = self.fc2(h)      # [B, P] complex

        if self.use_bias:
            d_gamma, beta = params.split(self.C, dim=-1)
        else:
            d_gamma = params
            beta = None

        # reshape to broadcast
        d_gamma = d_gamma.view(B, C, 1, 1)
        gamma = 1.0 + d_gamma     # start as identity (1+0j)
        y = gamma * x
        if beta is not None:
            beta = beta.view(B, C, 1, 1)
            y = y + beta
        return y
    
class CEncoderFiLM(nn.Module):
    def __init__(self, hp, ic, ngf, kernel_size, stride=2, padding=1, z_dim=128):
        super().__init__()
        self.hp = hp
        self.globln = GlobLNComplex(2)

        self.convlayer1 = CDownBlock(ic,      ngf,     kernel_size, stride=1,     padding=padding, z_dim=z_dim)
        self.convlayer2 = CDownBlock(ngf,     ngf * 2, kernel_size, stride=stride, padding=padding, z_dim=z_dim)
        self.convlayer3 = CDownBlock(ngf * 2, ngf * 4, kernel_size, stride=stride, padding=padding, z_dim=z_dim)
        self.convlayer4 = CDownBlock(ngf * 4, ngf * 8, kernel_size, stride=stride, padding=padding, z_dim=z_dim)
        self.convlayer5 = CDownBlock(ngf * 8, ngf * 8, kernel_size, stride=stride, padding=padding, z_dim=z_dim)

    def forward(self, x, z):  # z: [B,128]
        x = self.globln(x)
        c1 = self.convlayer1(x, z)
        c2 = self.convlayer2(c1, z)
        c3 = self.convlayer3(c2, z)
        c4 = self.convlayer4(c3, z)
        c5 = self.convlayer5(c4, z)
        return c1, c2, c3, c4, c5

class CDownBlock(nn.Module):
    def __init__(self, nc, output_nc, kernel_size, stride, padding, z_dim):
        super().__init__()
        self.conv = nn.Conv2d(nc, output_nc, kernel_size,
                              stride=stride, padding=padding,
                              bias=False, dtype=DTYPE)
        self.norm = GlobLNComplex(output_nc)  # your complex norm
        self.film = CFiLM2dComplex(z_dim=z_dim, C=output_nc, use_bias=True)
        self.act  = complex_relu()            # your complex activation

    def forward(self, x, z):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, z)   # <-- fully complex modulation
        x = self.act(x)
        return x
    
### =================== encoder ==================== ###
class CEncoder(nn.Module):
    def __init__(self, hp, ic, ngf, kernel_size, stride=2,padding=1):
        super().__init__()
        self.hp = hp
        self.globln = GlobLNComplex(2)
        self.convlayer1 = self.unet_downconv(ic, ngf, kernel_size=kernel_size, stride=1,padding=padding)
        self.convlayer2 = self.unet_downconv(ngf, ngf * 2, kernel_size=kernel_size, stride=stride,padding=padding)
        self.convlayer3 = self.unet_downconv(ngf * 2, ngf * 4, kernel_size=kernel_size, stride=stride,padding=padding)
        self.convlayer4 = self.unet_downconv(ngf * 4, ngf * 8, kernel_size=kernel_size, stride=stride,padding=padding)
        self.convlayer5 = self.unet_downconv(ngf * 8, ngf * 8, kernel_size=kernel_size, stride=stride,padding=padding)

    def forward(self, x):
        x = self.globln(x)
        conv1feature = self.convlayer1(x)
        conv2feature = self.convlayer2(conv1feature)
        conv3feature = self.convlayer3(conv2feature)
        conv4feature = self.convlayer4(conv3feature)
        conv5feature = self.convlayer5(conv4feature)
        return conv1feature, conv2feature, conv3feature, conv4feature, conv5feature

    def unet_downconv(self, nc, output_nc, kernel_size, stride,padding=1):
        '''
        output size: [(in_size-kernel_size+2*padding)/stride]+1
        '''
        downconv = nn.Conv2d(nc, output_nc, kernel_size, stride=stride,padding=padding, bias=False,dtype=DTYPE)
        downnorm = GlobLNComplex(output_nc)
        act_fun=complex_relu()
        return nn.Sequential(*[downconv, downnorm, act_fun])

### =================== decoder ==================== ###
class CDecoder(nn.Module):
    def __init__(self, hp, kernel_size,stride=2, dim=1,sa=False,ic=1024):
        super().__init__()
        self.hp = hp
        self.dim = dim
        self.sa=sa
        ngf = hp.model.num_filters
        ch = [ngf*8, ngf*4, ngf*2, ngf, ngf//2]
        # ic = 1024  if (not ( hp.model_def_name=='Two_Stages' or hp.model_def_name=='Three_Stages' or hp.model_def_name=='First_Stage_Separation' ) or hp.use_emb_ref) else 512
        oc = hp.model.output_channels
        self.upconvlayer1 = self.unet_upconv(ic, ch[0], kernel_size,stride)
        self.upconvlayer2 = self.unet_upconv(ch[0]*2, ch[1], kernel_size,stride)
        self.upconvlayer3 = self.unet_upconv(ch[1]*2, ch[2], kernel_size,stride)
        self.upconvlayer4 = self.unet_upconv(ch[2]*2,ch[3], kernel_size,stride)
        self.upconvlayer5 = self.unet_upconv(ch[3]*2,  ch[4], (4,3),stride=1)
        self.postunet = self.last_layer(ch[4],oc)

  
    def forward(self, **inputs):
        bottleneck, conv1feature, conv2feature, conv3feature, conv4feature, = inputs['bottleneck'], inputs[
                'conv1feature'], inputs['conv2feature'], inputs['conv3feature'], inputs['conv4feature']

        upconv1feature = self.upconvlayer1(bottleneck)
        in_layer2 =  torch.cat((upconv1feature, conv4feature), self.dim) 
        upconv2feature = self.upconvlayer2(in_layer2)
 
        in_layer3 =  torch.cat((upconv2feature, conv3feature), self.dim)
        upconv3feature = self.upconvlayer3(in_layer3)
 
        in_layer4 =   torch.cat((upconv3feature, conv2feature), self.dim)
        upconv4feature = self.upconvlayer4(in_layer4)
   
        in_layer5 =   torch.cat((upconv4feature, conv1feature), self.dim)
        output = self.upconvlayer5(in_layer5)
     

        output = self.postunet(output)

        return output



    def unet_upconv(self, nc, output_nc, kernel_size, stride=2, padding=1,norm ='batch'):
        '''
        output_size: (in_size-1)*stride-2*padding+kernel_size
        '''
   
        upconv = nn.ConvTranspose2d(nc, output_nc, kernel_size, stride=stride, padding=padding,dtype=DTYPE)
        upnorm = GlobLNComplex(output_nc)
        act_fun=complex_relu()
        return nn.Sequential(*[upconv, upnorm, act_fun])

    def last_layer(self, nc, output_nc, kernel_size=3):
        postconv1 = nn.Conv2d(
            nc, output_nc, kernel_size=kernel_size, stride=1, padding=1,dtype=DTYPE)

        return nn.Sequential(*[postconv1])
       
class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size, dtype=torch.complex64):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size, dtype=dtype))
        self.beta  = nn.Parameter(torch.zeros(channel_size, dtype=dtype))

    def apply_gain_and_bias(self, normed_x):
        """
        normed_x: [B, C, *]  (channel = dim 1)
        Broadcast gamma/beta over all trailing dims with no transposes.
        """
        B, C = normed_x.shape[:2]
        # [1, C, 1, 1, ...] so it broadcasts over spatial/temporal dims
        shape = (1, C) + (1,) * (normed_x.ndim - 2)

        # Ensure params match input dtype/device (important for complex)
        gamma = self.gamma.to(dtype=normed_x.dtype, device=normed_x.device).view(shape)
        beta  = self.beta.to(dtype=normed_x.dtype,  device=normed_x.device).view(shape)

        # Avoid huge temporaries created by transpose+contiguous
        out = normed_x * gamma
        out = out + beta
        return out

class GlobLNComplex(_LayerNorm):
    """Global Layer Normalization for complex-valued tensors."""

    def forward(self, x):
        """ Applies forward pass.

        Args:
            x (:class:`torch.Tensor`): Complex-valued tensor of shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))

        # Compute magnitude and phase
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        # Compute mean and variance of magnitude
        mean = magnitude.mean(dim=dims, keepdim=True)
        var = torch.pow(magnitude - mean, 2).mean(dim=dims, keepdim=True)

        # Normalize magnitude
        normalized_magnitude = (magnitude - mean) / (var + 1e-8).sqrt()

        # Recombine normalized magnitude with phase
        normalized_x = normalized_magnitude * torch.exp(1j * phase)

        return self.apply_gain_and_bias(normalized_x)

class ComplexExtraction(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp = hp
        d_model=512
        self.encoder = CEncoder(self.hp, ic=self.hp.dataset.n_channels, ngf=64, kernel_size=(4,3), stride=(2,1))
        self.decoder=CDecoder(self.hp,kernel_size=(4,3),stride=(2,1),ic = d_model)
        self.bottelneck = CBottleneck(self.hp)

    def forward(self,x,hrtf): #fft of 2 channels hrtf
        conv1feature, conv2feature, conv3feature, conv4feature, emb  = self.encoder(x)
        emb = self.bottelneck(emb,hrtf)
        output = self.decoder(bottleneck=emb, conv1feature=conv1feature, conv2feature=conv2feature,
                                    conv3feature=conv3feature,conv4feature=conv4feature)
        return output
    
class ComplexBinauralExtraction(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp = hp
        ngf = hp.model.num_filters
        d_model=ngf*8
        self.hrtf_encoder = CHRTFtoLatent(hp,d_model=128)
        self.encoder = CEncoderFiLM(self.hp, ic=self.hp.dataset.n_channels, ngf=ngf, kernel_size=(4,3), stride=(2,1))
        self.decoder=CDecoder(self.hp,kernel_size=(4,3),stride=(2,1),ic = d_model)
        self.bottelneck = CBottleneck(self.hp)

    def forward(self,x,hrtf,hrtf_patches,pos,kpm): #fft of 2 channels hrtf
        hrtf_emb = self.hrtf_encoder(hrtf_patches,pos,kpm)

        conv1feature, conv2feature, conv3feature, conv4feature, emb  = self.encoder(x,hrtf_emb)
        emb = self.bottelneck(emb,hrtf)
        output = self.decoder(bottleneck=emb, conv1feature=conv1feature, conv2feature=conv2feature,
                                    conv3feature=conv3feature,conv4feature=conv4feature)
        return output
    
class CBottleneck(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp=hp
        embed_dim = 512
        hrtf_dim = hp.stft.fft_length//2
        ngf = hp.model.num_filters
        self.fc1= nn.Linear((ngf//4)*embed_dim,embed_dim,dtype=DTYPE)
        t = math.floor(((self.hp.dataset.time_len*self.hp.stft.fs))/self.hp.stft.fft_hop)+1
        self.self_attention_mix = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)

        #bn self attentions *4 with attention map transfer
        self.self_attention_bn_0 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_0 =GlobLNComplex(t)
        self.self_attention_bn_1 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_1 =GlobLNComplex(t)
        self.self_attention_bn_2 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_2 =GlobLNComplex(t)
        self.self_attention_bn_3 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_3 =GlobLNComplex(t)

        self.fc_hrtf = nn.Linear(hrtf_dim*2,512,dtype=DTYPE)
        self.attn_hrtf = CustomMultiheadAttention(embed_dim=hrtf_dim,num_heads=8)
        self.globln_hrtf =GlobLNComplex(2)        
        self.output_proj =nn.Linear(embed_dim,embed_dim*(ngf//4),dtype=DTYPE)
     

    def forward(self,emb_mix,hrtf):
            if hrtf.shape[-1] == (self.hp.stft.fft_length//2 +1):
                hrtf = hrtf[:,:,1:]
            B = emb_mix.shape[0]
            E = emb_mix.shape[1]
            C = emb_mix.shape[2]
            T = emb_mix.shape[3]

            emb_mix = emb_mix.permute(0,3,2,1).flatten(2,3)  #(B, 512, 16,626) ->  (B, 626, 512,16)-> (B,626,512*16)
            emb_mix = self.fc1(emb_mix) #(B,626,512) 
            emb_mix,_ = self.self_attention_mix(emb_mix)
            hrtf,_ = self.attn_hrtf(hrtf)
            hrtf = self.globln_hrtf(hrtf)
            hrtf = hrtf.flatten(1)
            hrtf = self.fc_hrtf(hrtf)
            # hrtf = hrtf.unsqueeze(-1).expand(emb_mix.shape[0], emb_mix.shape[0], emb_mix.shape[0])
            if B>1:
                hrtf=hrtf.unsqueeze(1)
          
            bn = emb_mix * hrtf
            
            #self attn with attn_map transfer
            bn,attn_mask = self.self_attention_bn_0(bn)
            bn = self.globln_0(bn)
            bn,attn_mask = self.self_attention_bn_1(bn,attn_mask=attn_mask)
            bn=self.globln_1(bn)
            bn,attn_mask = self.self_attention_bn_2(bn,attn_mask=attn_mask)
            bn = self.globln_2(bn)
            bn,_ = self.self_attention_bn_3(bn,attn_mask=attn_mask)
            bn = self.globln_3(bn)
            
            output_proj = self.output_proj(bn)
            output_proj = output_proj.reshape(B,T,E,C).permute(0,2,3,1)
            return output_proj


class complex_relu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,z):
        return torch.complex(torch.nn.ReLU()(z.real), torch.nn.ReLU()(z.imag))
    
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0,dtype=DTYPE):
        """
        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            activation_fn (callable, optional): Custom activation function (e.g., ReLU, Sigmoid, etc.).
            dropout (float): Dropout probability for attention weights.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5  # Scaling factor for QK^T

        # Linear layers for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim,dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim,dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim,dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim,dtype=dtype)

        # Custom activation function
        self.activation_fn = complex_relu()

    def forward(self, query, key=None, value=None, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query (Tensor): Query embeddings of shape (B, T, C).
            key (Tensor): Key embeddings of shape (B, T, C).
            value (Tensor): Value embeddings of shape (B, T, C).
            attn_mask (Tensor, optional): Attention mask of shape (T, T).
            key_padding_mask (Tensor, optional): Padding mask of shape (B, T).

        Returns:
            Tensor: Attention output of shape (B, T, C).
        """
        if key == None and value == None: #self attn
            key = query.clone()
            value = query.clone()
        batch_size, seq_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"

        # Linear projections for Q, K, V
        Q = self.q_proj(query)  # (B, T, C)
        K = self.k_proj(key)    # (B, T, C)
        V = self.v_proj(value)  # (B, T, C)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling  # (B, H, T, T)

        # Apply attention mask, if provided
        if attn_mask is not None:
            attn_weights += attn_mask

        # Apply key padding mask, if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Apply custom activation function to attention weights
        attn_weights = self.activation_fn(attn_weights)  # Custom activation

        # Dropout on attention weights
        # attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, D)

        # Reshape back to (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, H, D)
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)  # (B, T, C)

        # Final output projection
        attn_output = self.out_proj(attn_output)  # (B, T, C)

        return attn_output, attn_weights
    
    

