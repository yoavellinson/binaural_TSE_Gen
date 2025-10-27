import torch.nn as nn
import torch
import numpy as np
import torch.functional as F
eps = torch.exp(torch.tensor(-6))

       
class CLeakyReLU(nn.LeakyReLU):
    def forward(self, xr, xi):
        return F.leaky_relu(xr, self.negative_slope, self.inplace),\
                F.leaky_relu(xi, self.negative_slope, self.inplace)

class complex_relu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,z):
        return torch.complex(torch.nn.ReLU()(z.real), torch.nn.ReLU()(z.imag))
      
### =================== encoder ==================== ###
class CEncoder(nn.Module):
    def __init__(self, hp, ic, ngf, kernel_size, stride=2):
        super().__init__()
        self.hp = hp
        self.globln = GlobLNComplex(2)
        self.convlayer1 = self.unet_downconv(ic, ngf, kernel_size=kernel_size, stride=1)
        self.convlayer2 = self.unet_downconv(ngf, ngf * 2, kernel_size=kernel_size, stride=stride)
        self.convlayer3 = self.unet_downconv(ngf * 2, ngf * 4, kernel_size=kernel_size, stride=stride)
        self.convlayer4 = self.unet_downconv(ngf * 4, ngf * 8, kernel_size=kernel_size, stride=stride)
        self.convlayer5 = self.unet_downconv(ngf * 8, ngf * 8, kernel_size=kernel_size, stride=stride)

        
    def forward(self, x):
        x = self.globln(x)
        conv1feature = self.convlayer1(x)
        conv2feature = self.convlayer2(conv1feature)
        conv3feature = self.convlayer3(conv2feature)
        conv4feature = self.convlayer4(conv3feature)
        conv5feature = self.convlayer5(conv4feature)
        return conv1feature, conv2feature, conv3feature, conv4feature, conv5feature

    def unet_downconv(self, nc, output_nc, kernel_size, stride,padding=1,norm='layer',transformer_norm_first=False):
        '''
        output size: [(in_size-kernel_size+2*padding)/stride]+1
        '''
        downconv = nn.Conv2d(nc, output_nc, kernel_size, stride=stride,padding=padding, bias=False,dtype=torch.complex64)
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
        
        # ic = 1024  if (not ( hp.model_def_name=='Two_Stages' or hp.model_def_name=='Three_Stages' or hp.model_def_name=='First_Stage_Separation' ) or hp.use_emb_ref) else 512
        oc = hp.output_channels
        self.upconvlayer1 = self.unet_upconv(ic, 512, kernel_size,stride)
        self.upconvlayer2 = self.unet_upconv(512*2, 256, kernel_size,stride)
        self.upconvlayer3 = self.unet_upconv(512, 128, kernel_size,stride)
        self.upconvlayer4 = self.unet_upconv(256, 64, kernel_size,stride)
        self.upconvlayer5 = self.unet_upconv(128,  16, (4,3),stride=1)
        self.postunet = self.last_layer(16,oc)

  
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
   
        upconv = nn.ConvTranspose2d(nc, output_nc, kernel_size, stride=stride, padding=padding,dtype=torch.complex64)
        upnorm = GlobLNComplex(output_nc)
        act_fun=complex_relu()
        return nn.Sequential(*[upconv, upnorm, act_fun])

    def last_layer(self, nc, output_nc, kernel_size=3):
        postconv1 = nn.Conv2d(
            nc, output_nc, kernel_size=kernel_size, stride=1, padding=1,dtype=torch.complex64)

        return nn.Sequential(*[postconv1])
       
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
    
class CBottleneck(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp=hp
        embed_dim = 512
        hrtf_dim = hp.stft.fft_length//2
        self.fc1= nn.Linear(16*embed_dim,embed_dim,dtype=torch.complex64)

        self.self_attention_mix = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)

        #bn self attentions *4 with attention map transfer
        self.self_attention_bn_0 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_0 =GlobLNComplex(626)
        self.self_attention_bn_1 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_1 =GlobLNComplex(626)
        self.self_attention_bn_2 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_2 =GlobLNComplex(626)
        self.self_attention_bn_3 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_3 =GlobLNComplex(626)
        #

        self.fc_hrtf = nn.Linear(hrtf_dim*2,512,dtype=torch.complex64)
        self.attn_hrtf = CustomMultiheadAttention(embed_dim=hrtf_dim,num_heads=8)
        self.globln_hrtf =GlobLNComplex(2)        
        self.output_proj =nn.Linear(embed_dim,embed_dim*16,dtype=torch.complex64)
        if hp.bottleneck.process == 'cross_attn':
            self.cross_attn = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
            self.globln_bn = GlobLNComplex(626)

    def forward(self,emb_mix,hrtf):
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
            if self.hp.bottleneck.process == 'cross_attn':
                KV = hrtf.repeat(1,T,1)
                bn,_ = self.cross_attn(emb_mix,KV,KV)
                bn = self.globln_bn(bn)
            else:
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
    def __init__(self, embed_dim, num_heads, dropout=0.0,dtype=torch.complex64):
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
    


class ComplexExtractionDOA(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp = hp
        d_model=512
        self.encoder = CEncoder(self.hp, ic=self.hp.dataset.n_channels, ngf=64, kernel_size=(4,3), stride=(2,1))
        self.decoder=CDecoder(self.hp,kernel_size=(4,3),stride=(2,1),ic = d_model)
        self.bottelneck = CBottleneckDOA(self.hp)

    def forward(self,x,doa): #doa : B,2 : batch,az,elev
        conv1feature, conv2feature, conv3feature, conv4feature, emb  = self.encoder(x)
        emb = self.bottelneck(emb,doa)
        output = self.decoder(bottleneck=emb, conv1feature=conv1feature, conv2feature=conv2feature,
                                    conv3feature=conv3feature,conv4feature=conv4feature)
        return output
    
class CBottleneckDOA(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp=hp
        embed_dim = 512
        doa_dim = 37
        self.fc1= nn.Linear(16*embed_dim,embed_dim,dtype=torch.complex64)

        self.self_attention_mix = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)

        #bn self attentions *4 with attention map transfer
        self.self_attention_bn_0 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_0 =GlobLNComplex(626)
        self.self_attention_bn_1 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_1 =GlobLNComplex(626)
        self.self_attention_bn_2 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_2 =GlobLNComplex(626)
        self.self_attention_bn_3 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_3 =GlobLNComplex(626) #

        self.fc_proj_doa = nn.Linear(doa_dim,256,dtype=torch.complex64)
        self.fc_doa = nn.Linear(embed_dim,embed_dim,dtype=torch.complex64)
        self.attn_doa = CustomMultiheadAttention(embed_dim=256,num_heads=8)
        self.globln_doa =GlobLNComplex(2)        
        self.output_proj =nn.Linear(embed_dim,embed_dim*16,dtype=torch.complex64)
        if hp.bottleneck.process == 'cross_attn':
            self.cross_attn = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
            self.globln_bn = GlobLNComplex(626)

    def forward(self,emb_mix,doa):
            B = emb_mix.shape[0]
            E = emb_mix.shape[1]
            C = emb_mix.shape[2]
            T = emb_mix.shape[3]

            emb_mix = emb_mix.permute(0,3,2,1).flatten(2,3)  #(B, 512, 16,626) ->  (B, 626, 512,16)-> (B,626,512*16)
            emb_mix = self.fc1(emb_mix) #(B,626,512)
            emb_mix,_ = self.self_attention_mix(emb_mix)
            doa =doa+ 1j
            doa = self.fc_proj_doa(doa) # 37->256
            doa,_ = self.attn_doa(doa)
            doa = self.globln_doa(doa)
            doa = doa.flatten(1)
            doa = self.fc_doa(doa)
            # doa = doa.unsqueeze(-1).expand(emb_mix.shape[0], emb_mix.shape[0], emb_mix.shape[0])
            if B>1:
                doa=doa.unsqueeze(1)
            if self.hp.bottleneck.process == 'cross_attn':
                KV = doa.repeat(1,T,1)
                bn,_ = self.cross_attn(emb_mix,KV,KV)
                bn = self.globln_bn(bn)
            else:
                bn = emb_mix * doa
            
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
    

class ComplexDOA(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp = hp
        d_model=512
        self.encoder = CEncoder(self.hp, ic=self.hp.dataset.n_channels, ngf=64, kernel_size=(4,3), stride=(2,1))
        self.decoder=CDecoder(self.hp,kernel_size=(4,3),stride=(2,1),ic = d_model)
        self.bottelneck = CBottleneckNoClue(self.hp)

    def forward(self,x): #fft of 2 channels hrtf
        conv1feature, conv2feature, conv3feature, conv4feature, emb  = self.encoder(x)
        emb = self.bottelneck(emb)
        output = self.decoder(bottleneck=emb, conv1feature=conv1feature, conv2feature=conv2feature,
                                    conv3feature=conv3feature,conv4feature=conv4feature)
        return output
    

class CBottleneckNoClue(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.hp=hp
        embed_dim = 512
        self.fc1= nn.Linear(16*embed_dim,embed_dim,dtype=torch.complex64)

        self.self_attention_mix = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)

        #bn self attentions *4 with attention map transfer
        self.self_attention_bn_0 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_0 =GlobLNComplex(626)
        self.self_attention_bn_1 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_1 =GlobLNComplex(626)
        self.self_attention_bn_2 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_2 =GlobLNComplex(626)
        self.self_attention_bn_3 = CustomMultiheadAttention(embed_dim=embed_dim,num_heads=8)
        self.globln_3 =GlobLNComplex(626)
    
        self.output_proj =nn.Linear(embed_dim,embed_dim*16,dtype=torch.complex64)

        self.fc_out = nn.Linear(512 * 2 * 626, 360)

    def forward(self,emb_mix):
            B = emb_mix.shape[0]
            E = emb_mix.shape[1]
            C = emb_mix.shape[2]
            T = emb_mix.shape[3]

            emb_mix = emb_mix.permute(0,3,2,1).flatten(2,3)  #(B, 512, 16,626) ->  (B, 626, 512,16)-> (B,626,512*16)
            emb_mix = self.fc1(emb_mix) #(B,626,512)
            emb_mix,_ = self.self_attention_mix(emb_mix)
        
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
            flattened = output_proj.reshape(B, -1)
            return self.fc_out(flattened)