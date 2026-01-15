import math, torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import math, torch
import torch.nn as nn
import torch.nn.functional as F



### =================== encoder ==================== ###
class Encoder(nn.Module):
    def __init__(self, hp, ic, ngf, kernel_size, stride=2):
        super().__init__()

        self.hp = hp
        self.convlayer1 = self.unet_downconv(ic, ngf, kernel_size=kernel_size, stride=1,norm=hp.model.norm)
        self.convlayer2 = self.unet_downconv(ngf, ngf * 2, kernel_size=kernel_size, stride=stride,norm=hp.model.norm)
        self.convlayer3 = self.unet_downconv(ngf * 2, ngf * 4, kernel_size=kernel_size, stride=stride,norm=hp.model.norm)
        self.convlayer4 = self.unet_downconv(ngf * 4, ngf * 8, kernel_size=kernel_size, stride=stride,norm=hp.model.norm)
        self.convlayer5 = self.unet_downconv(ngf * 8, ngf * 8, kernel_size=kernel_size, stride=stride,norm=hp.model.norm)

    def forward(self, x):
      
        conv1feature = self.convlayer1(x)
        conv2feature = self.convlayer2(conv1feature)
        conv3feature = self.convlayer3(conv2feature)
        conv4feature = self.convlayer4(conv3feature)
        conv5feature = self.convlayer5(conv4feature)  
        return conv1feature, conv2feature, conv3feature, conv4feature, conv5feature

    def unet_downconv(self, nc, output_nc, kernel_size, stride,padding=(0,1),norm='layer',transformer_norm_first=False):
        '''
        output size: [(in_size-kernel_size+2*padding)/stride]+1
        '''
        downconv = nn.Conv2d(nc, output_nc, kernel_size, stride=stride,padding=padding, bias=False)
        if not transformer_norm_first: # if True apply layernorm before self attention
            if norm =='batch':
                downnorm = nn.BatchNorm2d(output_nc)
            elif norm =='layer':
                downnorm = GlobLN(output_nc)
        if self.hp.model.act_fun=='relu':
            act_fun = nn.ReLU() 
        elif self.hp.model.act_fun=='prelu':
            act_fun = nn.PReLU()
        return nn.Sequential(*[downconv, downnorm, act_fun])

class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class MaskedMean(nn.Module):
    def forward(self, x, kpm=None):           # x: [B,C,N,T]
        x = x.mean(dim=-1)                    # [B,C,N]
        if kpm is None:
            return x.mean(dim=-1)             # [B,C]
        valid = (~kpm).float()                # [B,N]
        denom = valid.sum(-1, keepdim=True).clamp_min(1.0)
        return (x * valid.unsqueeze(1)).sum(-1) / denom

class Pos2DChannels(nn.Module):
    def __init__(self, num_bands=8, include_xy=True, base_freq=1.0):
        super().__init__()
        self.num_bands, self.include_xy, self.base = num_bands, include_xy, base_freq
    def forward(self, xy, T, kpm=None, scale=(1.0,1.0)):  # xy:[B,N,2] -> [B,Cpos,N,T]
        B,N,_ = xy.shape; device = xy.device
        xy = torch.nan_to_num(xy, nan=0.0)
        Sx,Sy = scale
        xy = xy / torch.tensor([Sx,Sy], device=device).view(1,1,2)
        Freq = (self.base * (2.0 ** torch.arange(self.num_bands, device=device))) * math.pi
        x,y = xy[...,0:1], xy[...,1:2]
        xw,yw = x*Freq, y*Freq
        feats = [torch.sin(xw), torch.cos(xw), torch.sin(yw), torch.cos(yw)]
        if self.include_xy: feats += [x,y]
        pos = torch.cat(feats, dim=-1)                  # [B,N,Cpos]
        pos = pos.permute(0,2,1).unsqueeze(-1).expand(-1,-1,-1,T).contiguous()  # [B,Cpos,N,T]
        if kpm is not None:
            pos = pos.masked_fill(kpm[:,None,:,None], 0.0)
        return pos

class FiLMFromZXY(nn.Module):
    def __init__(self, ch, d_latent, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_latent + 2, hidden), nn.GELU(),
            nn.Linear(hidden, 2*ch)   # -> gamma, beta
        )
    def forward(self, feat, z, xy, kpm=None):  # feat:[B,C,N,T], z:[B,d], xy:[B,N,2]
        B,C,N,T = feat.shape
        zt = z.unsqueeze(1).expand(-1,N,-1)          # [B,N,d]
        gb = self.mlp(torch.cat([zt, xy], dim=-1))   # [B,N,2C]
        gamma, beta = gb.split(C, dim=-1)            # [B,N,C] each
        gamma = gamma.permute(0,2,1).unsqueeze(-1)   # [B,C,N,1]
        beta  =  beta.permute(0,2,1).unsqueeze(-1)
        if kpm is not None:
            gamma = gamma.masked_fill(kpm[:,None,:,None], 0.0)
            beta  =  beta.masked_fill(kpm[:,None,:,None], 0.0)
        return feat * (1.0 + gamma) + beta
    
class GlobalToLatent(nn.Module):
    def __init__(self, in_dim=512, latent_dim=128):
        super().__init__()
        self.mu     = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, latent_dim))
        self.logvar = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, latent_dim))
    def forward(self, h):  # h: [B,in_dim]
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-20.0, max=20.0)  # keep numerically stable
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    
class CondUNetDecoderFiLM(nn.Module):
    def __init__(self, hp, d_latent=128, cpos=34, kernel_size=(1,4), stride=(1,2), padding=(0,1)):
        super().__init__()
        # Example: 5 up stages mirroring your U-Net; adapt channels as you want
        ngf = hp.model.num_filters
        ch = [ngf*8, ngf*4, ngf*2, ngf, ngf//2]  # decoder feature widths
        self.in_conv = nn.Conv2d(cpos, ch[0], kernel_size=1)  # map pos->features

        # Up blocks (ConvTranspose2d or upsample+conv)
        self.up1 = nn.ConvTranspose2d(ch[0], ch[1], kernel_size, stride, padding)
        self.up2 = nn.ConvTranspose2d(ch[1], ch[2], kernel_size, stride, padding)
        self.up3 = nn.ConvTranspose2d(ch[2], ch[3], kernel_size, stride, padding)
        self.up4 = self.up4 = nn.ConvTranspose2d(ch[3], ch[4],
                              kernel_size=(1,4), stride=(1,2), padding=(0,1)) 

        # FiLM at each stage
        self.film1 = FiLMFromZXY(ch[0], d_latent)
        self.film2 = FiLMFromZXY(ch[1], d_latent)
        self.film3 = FiLMFromZXY(ch[2], d_latent)
        self.film4 = FiLMFromZXY(ch[3], d_latent)

        self.out = nn.Conv2d(ch[4], 4, kernel_size=1)  # -> [B,4,N,513]

        self.act = nn.GELU()
        self.norms = nn.ModuleList([nn.GroupNorm(8, c) for c in ch])

    def forward(self, pos_ch, z, xy, kpm=None):  # pos_ch:[B,Cpos,N,T], z:[B,d]
        pos_ch = pos_ch.to(dtype=torch.float32)
        z      = z.to(torch.float32)
        xy     = xy.to(torch.float32)
        y = self.in_conv(pos_ch)                          # [B,ch0,N,Tb]
        y = self.film1(self.norms[0](y), z, xy, kpm); y = self.act(y)
        y = self.up1(y)
        y = self.film2(self.norms[1](y), z, xy, kpm); y = self.act(y)
        y = self.up2(y)
        y = self.film3(self.norms[2](y), z, xy, kpm); y = self.act(y)
        y = self.up3(y)
        y = self.film4(self.norms[3](y), z, xy, kpm); y = self.act(y)
        y = self.up4(y)
        y = F.pad(y, (0, 1)) 
        return self.out(y)

class PatchDBVAE(nn.Module):
    def __init__(self, hp, latent_dim=256, num_bands=8, include_xy=True):
        super().__init__()
        self.hp = hp
        kernel_size=(1,4)
        stride=(1,2)
        self.ic = 4 #2ch -> Real,Imag
        self.oc = 4
        ngf = hp.model.num_filters

        d_model=ngf*8

        self.encoder = Encoder(hp, self.ic, ngf,kernel_size,stride=stride)
        self.pool = MaskedMean()
        self.to_latent = GlobalToLatent(in_dim=d_model, latent_dim=latent_dim)

        self.pos_enc = Pos2DChannels(num_bands=num_bands, include_xy=include_xy, base_freq=1.0)
        cpos = 4*num_bands + (2 if include_xy else 0)
        self.decoder = CondUNetDecoderFiLM(hp, d_latent=latent_dim, cpos=cpos)

    def forward(self, x_in, xy, kpm=None, scale=(360,90)):
        """
        x_in: [B, 2, N, 513]  (patch tensor permuted to [B,C,N,T])
        xy:   [B, N, 2]
        kpm:  [B, N] True=PAD
        """
        # Encoder
        x_in = torch.view_as_real(x_in).permute(0,1,4,2,3)
        B,C,RI,N,F = x_in.shape
        x_in = x_in.reshape(B,C*RI,N,F).contiguous() #-> [B, C=4, H=N, W=257]
        # x_in = x_in.permute(0, 2, 1, 3).contiguous()  # -> [B, C=4, H=N, W=513]
        # hrtf=torch.view_as_real(hrtf)
        # hrtf = hrtf.reshape(B, F, 1, C * 2)
        conv1, conv2, conv3, conv4, bottleneck = self.encoder(x_in)  # bottleneck:[B,256,N,32]

        # Global latent
        h_global = self.pool(bottleneck, kpm)    # [B,512]
        z, mu, logvar = self.to_latent(h_global) # [B,d], [B,d], [B,d]

        T_b = bottleneck.shape[-1]            # e.g., 64 or 32
        pos_ch = self.pos_enc(xy, T=T_b, kpm=kpm, scale=scale) 

        # Decode (conditioned on z & xy only; no encoder skips)
        x_hat = self.decoder(pos_ch, z, xy, kpm)  # [B,2,N,513]
        x_hat = x_hat.permute(0,2,1,3)
        return x_hat, z, mu, logvar
    

    

# ===== Collate: pad variable-N DBs into a batch =====
def collate_db(db_list, N_target=None, ignore_value=float('nan')):
    """
    Pads a list of patch DBs (each with variable N) to the same length.

    Each item in db_list is a dict:
        {
          'patches': [N_i, 4, 513],
          'pos': [N_i, 4]
        }

    Returns:
        patches : [B, N_max, 4, 513]
        pos     : [B, N_max, 4]
        kpm     : [B, N_max]   True = PAD
    """
    B = len(db_list)
    Ns = [d['patches'].shape[0] for d in db_list]
    Nmax = max(Ns) if N_target is None else N_target

    patches = torch.zeros(B, Nmax, 4, 513, dtype=torch.float32)
    pos     = torch.full((B, Nmax, 2), ignore_value, dtype=torch.float32)
    kpm     = torch.ones(B, Nmax, dtype=torch.bool)  # True = PAD/ignore

    for b, d in enumerate(db_list):
        P = torch.as_tensor(d['patches'], dtype=torch.float32)  # [N_i, 2, 513]
        XY = torch.as_tensor(d['pos'], dtype=torch.float32)      # [N_i, 2]
        N = P.shape[0]

        if N_target is not None and N > N_target:
            idx = torch.randperm(N)[:N_target]
            P, XY = P[idx], XY[idx]
            N = N_target

        patches[b, :N] = P
        pos[b, :N] = XY
        kpm[b, :N] = False  # valid
    return patches, pos, kpm



if __name__=="__main__":
    hp = OmegaConf.load('/home/workspace/yoavellinson/binaural_TSE_Gen/conf/vaef.yml')

    db1 = torch.load('/home/workspace/yoavellinson/binaural_TSE_Gen/pts/3d3a/Subject1_HRIRs.pt')
    db2 = torch.load('/home/workspace/yoavellinson/binaural_TSE_Gen/pts/ari_atl_and_full/hrtf b_nh5.pt')
    db3 = torch.load('/home/workspace/yoavellinson/binaural_TSE_Gen/pts/sadie/H5_48K_24bit_256tap_FIR_SOFA.pt')
    P, XY, KPM = collate_db([db1, db2], N_target=None) 
    model = PatchDBVAE(hp)
    x_hat, z, mu, logvar = model(P,XY,KPM)
    print(x_hat.shape)