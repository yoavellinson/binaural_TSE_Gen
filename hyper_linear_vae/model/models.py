import torch as th
import torch.nn as nn

from .modules import HyperLinearBlock, FourierFeatureMapping

class AttrDict(dict):
    def __getattr__(self, name):
        value = None
        if name in self.keys():
            value = self[name]
        if isinstance(value, dict):
            value = AttrDict(value)
        return value
    
class FreqSrcPosCondAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = AttrDict(config)
        self.config = config

        # Stats for standardization
        self.stats = {
            "none": {
                "hrtf_mag": {"mean": th.tensor(0.0), "std": th.tensor(1.0)},
                "itd": {"mean": th.tensor(0.0), "std": th.tensor(1.0)}
            }
        }

        # Fourier Feature Mapping
        self.ffm_srcpos = FourierFeatureMapping(
            num_features=config.fourier_feature_mapping.num_features.source_position,
            input_dim=2,
            trainable=config.fourier_feature_mapping.trainable)
        self.ffm_freq = FourierFeatureMapping(
            num_features=config.fourier_feature_mapping.num_features.frequency,
            input_dim=1,
            trainable=config.fourier_feature_mapping.trainable)
        self.radius_norm = config.radius_norm
        self.freq_norm = config.freq_norm
        self.num_mes_norm = config.num_mes_norm

        # Encoder
        dim_cond_vec_enc = config.fourier_feature_mapping.num_features.source_position * 2 + 2
        if config.encoder.use_freq in [None, True]:
            self.encoder_use_freq = True
            dim_cond_vec_enc += config.fourier_feature_mapping.num_features.frequency * 2
        else:
            self.encoder_use_freq = False
        modules = []
        for l_e in range(config.encoder.num_layers):
            in_dim = 1 if l_e == 0 else config.encoder.mid_dim
            out_dim = config.encoder.out_dim if l_e == config.encoder.num_layers - 1 else config.encoder.mid_dim
            if config.encoder.nonlinear in [None, True]:
                post_prcs = l_e != (config.encoder.num_layers - 1)
            else:
                post_prcs = False

            modules.extend([
                HyperLinearBlock(in_dim=in_dim,
                                 out_dim=out_dim,
                                 hidden_dim=config.weight_bias_generator.mid_dim,
                                 num_hidden=config.weight_bias_generator.num_layers,
                                 cond_dim=dim_cond_vec_enc, post_prcs=post_prcs),
            ])
        self.encoder = nn.Sequential(*modules)

        # Decoder
        dim_cond_vec_dec = config.fourier_feature_mapping.num_features.source_position * 2 + 1
        if config.decoder.use_freq in [None, True]:
            self.decoder_use_freq = True
            dim_cond_vec_dec += config.fourier_feature_mapping.num_features.frequency * 2
        else:
            self.decoder_use_freq = False
        modules = []
        for l_d in range(config.decoder.num_layers):
            in_dim = config.decoder.in_dim if l_d == 0 else config.decoder.mid_dim
            out_dim = 1 if l_d == config.decoder.num_layers - 1 else config.decoder.mid_dim
            if config.decoder.nonlinear in [None, True]:
                post_prcs = l_d != (config.encoder.num_layers - 1)
            else:
                post_prcs = False

            modules.extend([
                HyperLinearBlock(in_dim=in_dim,
                                 out_dim=out_dim,
                                 hidden_dim=config.weight_bias_generator.mid_dim,
                                 num_hidden=config.weight_bias_generator.num_layers,
                                 cond_dim=dim_cond_vec_dec, post_prcs=post_prcs),
            ])
        self.decoder = nn.Sequential(*modules)

    def set_stats(self, mean, std, dataset_name, data_type="hrtf_mag"):
        if dataset_name not in self.stats:
            self.stats[dataset_name] = {}
        self.stats[dataset_name][data_type] = {"mean": mean, "std": std}

    def standardize(self, input, dataset_name, data_type="hrtf_mag", reverse=False, device="cuda"):
        if not reverse:
            output = (input - self.stats[dataset_name][data_type]["mean"].to(device)) / self.stats[dataset_name][data_type]["std"].to(device)
        else:
            output = input * self.stats[dataset_name][data_type]["std"].to(device) + self.stats[dataset_name][data_type]["mean"].to(device)
        return output

    def switch_device(self, inputs, device="cuda"):
        outputs = []
        for input in inputs:
            outputs.append(input.to(device))
        return outputs

    def get_conditioning_vector(self, pos_cart, freq=None, use_freq=True, use_num_pos=False, device="cuda"):
        S, B, _ = pos_cart.shape
        L = freq.shape[1]
        conditioning_vector = []

        pos_cart = pos_cart / self.radius_norm  # (S, B, 2)
        pos_cart_lr_flip = pos_cart * th.tensor([1, -1, 1], device=device)[None, None, :]  # (S, B, 3)

        pos_cart = self.ffm_srcpos(pos_cart)  # (S, B, 32)
        pos_cart = pos_cart.unsqueeze(2).tile(1, 1, L, 1)  # (S, B, L, 32)

        pos_cart_lr_flip = self.ffm_srcpos(pos_cart_lr_flip)  # (S, B, 32)
        pos_cart_lr_flip = pos_cart_lr_flip.unsqueeze(2).tile(1, 1, L, 1)  # (S, B, L, 32)

        pos_cart_all = th.cat((pos_cart, pos_cart_lr_flip, pos_cart[:, :, 0:1, :]), dim=2)  # (S, B, 2L+1, 32)
        conditioning_vector.append(pos_cart_all)

        if use_freq:
            freq = freq / self.freq_norm
            freq = self.ffm_freq(freq.unsqueeze(-1))  # (1, L, 16)
            freq = freq.reshape(1, 1, L, -1).tile(S, B, 2, 1)  # (S, B, 2L, 16)
            freq = th.cat((freq, th.zeros(S, B, 1, freq.shape[-1], device=device, dtype=th.float32)), dim=2)  # (S, B, 2L+1, 16)
            conditioning_vector.append(freq)

        if use_num_pos:
            num_pos = B / self.num_mes_norm * th.ones(S, B, 2 * L + 1, 1, device=device, dtype=th.float32)  # (S, B, 2L+1, 1)
            conditioning_vector.append(num_pos)

        delta = th.cat((th.zeros(2 * L, device=device, dtype=th.float32),
                        th.ones(1, device=device, dtype=th.float32)), dim=0)  # (2L + 1)
        delta = delta.reshape(1, 1, 2 * L + 1, 1).tile(S, B, 1, 1)  # (S, B, 2L+1, 1)
        conditioning_vector.append(delta)

        conditioning_vector = th.cat(conditioning_vector, dim=-1)  # (S, B, 2L+1, 50 or 49 or 34 or 33)
        return conditioning_vector.to(device)
    

    def get_conditioning_vector_sph(self, pos_sph, freq=None, use_freq=True, use_num_pos=False, device="cuda"):
        S, B, _ = pos_sph.shape          # pos_sph: (S, B, 2)
        L = freq.shape[0]                # L = 257
        CH = 4                           # 4 channels

        # ======= positional encoding =======
        pos_sph = pos_sph / th.tensor(self.radius_norm, device=device)

        # Encode
        pos_encoded = self.ffm_srcpos(pos_sph)       # (S, B, 32)

        # Expand for each channel
        pos_encoded = pos_encoded.unsqueeze(2).tile(1, 1, CH*L, 1)  # (S, B, 4L, 32)

        # Add the extra +1 bin to reach 4L+1
        last = pos_encoded[:, :, :1, :]

        pos_all = th.cat([pos_encoded, last], dim=2)  # (S, B, 4L+1, 32)
        conditioning_vector = [pos_all]

        # ======= frequency encoding =======
        if use_freq:
            freq = freq / self.freq_norm
            freq_enc = self.ffm_freq(freq.unsqueeze(-1))  # (1, L, 16)

            # Repeat freq encoding for 4 channels
            freq_enc = freq_enc.reshape(1, 1, L, -1).tile(S, B, CH, 1)  # (S, B, 4L, 16)

            # Add the extra +1 element
            extra = th.zeros(S, B, 1, freq_enc.shape[-1], device=device)
            freq_enc = th.cat([freq_enc, extra], dim=2)  # (S, B, 4L+1, 16)

            conditioning_vector.append(freq_enc)

        # ======= num_pos =======
        if use_num_pos:
            num_pos = B / self.num_mes_norm * th.ones(S, B, CH*L+1, 1, device=device)
            conditioning_vector.append(num_pos)

        # ======= delta =======
        delta = th.cat([
            th.zeros(CH * L, device=device),
            th.ones(1, device=device)
        ], dim=0)                          # (4L+1)
        delta = delta.reshape(1, 1, CH*L+1, 1).tile(S, B, 1, 1)
        conditioning_vector.append(delta)

        # ======= final output =======
        conditioning_vector = th.cat(conditioning_vector, dim=-1)  # (S, B, 4L+1, K)

        return conditioning_vector


    def forward(self, hrtf, itd, freq, mes_pos_cart, tar_pos_cart, device="cuda"):
        '''
        Args:
            hrtf:     (S ,B_m, 4, L)
            itd:          (S, B_m)
            freq:         (S, L)
            mes_pos_cart: (S, B_m, 2) 2: az,elev
            tar_pos_cart: (S, B_t, 2)
            device: str

        Returns:
            hrtf_mag_pred: (S, B_t, 4, L)
            itd_pred:      (S, B_t)
        '''
 

        _, B_m, _, L = hrtf.shape
        assert hrtf.shape[1] == itd.shape[1] == B_m
        B_t = tar_pos_cart.shape[1]

        hrtf, itd, freq, mes_pos_cart, tar_pos_cart = self.switch_device([hrtf, itd, freq, mes_pos_cart, tar_pos_cart], device=device)


        hrtf = th.cat((hrtf[:, :, 0, :], hrtf[:, :, 1, :],hrtf[:, :, 2, :],hrtf[:, :, 3, :]), dim=-1)  # (S, B_m, 4L)
        encoder_input = th.cat((hrtf, itd[:,:,None]), dim=-1).unsqueeze(-1)  # (S, B_m, 4L+1, 1)
        encoder_cond = self.get_conditioning_vector_sph(mes_pos_cart, freq, use_freq=self.encoder_use_freq, use_num_pos=True, device=device)  # (S, B_m, 2L+1, 50 or 34)

        latent = self.encoder((encoder_input, encoder_cond))[0]  # (S, B_m, 4L+1, D)
        prototype = th.mean(latent, dim=1, keepdim=True)  # (S, 1, 4L+1, D)

        decoder_input = prototype.tile(1, B_t, 1, 1)  # (S, B_t, 2L+1, D)
        decoder_cond = self.get_conditioning_vector_sph(tar_pos_cart, freq, use_freq=self.decoder_use_freq, use_num_pos=False, device=device)  # (S, B_t, 2L+1, 49 or 33)
        decoder_output = self.decoder((decoder_input, decoder_cond))[0]  # (S, B_t, 2L+1, 1)

        hrtf_pred = th.cat((decoder_output[:, :, None, :L, 0], decoder_output[:, :, None, L:2 * L, 0],decoder_output[:, :, None, 2*L:3*L, 0],decoder_output[:, :, None, 3*L:4*L, 0]), dim=2)  # (S, B_t, 4, L)
        itd_pred = decoder_output[:, :, -1, 0]  # (S, B_t)

        return hrtf_pred, itd_pred

