from pathlib import Path
import matplotlib.pyplot as plt
import torchaudio
import torch
from tqdm import tqdm


MAX_SHIFT = 0.1 #Seconds

def roll_tnzr(x:torch.Tensor):
    x = x.roll(-1)
    x[:,-1] = 0
    return x

def plot_sisdr(mix,y_ckpt,target,save_dir_i,max_shift=MAX_SHIFT):
    sisdri_res = []
    for i in tqdm(range(int(max_shift*8000))):
        sisdri = si_sdr_calc(y_ckpt,target).mean()
        sisdri_res.append(float(sisdri))
        y_ckpt = roll_tnzr(y_ckpt)

    plt.figure()
    idx_sisdri = range(len(sisdri_res))
    plt.plot(idx_sisdri,sisdri_res,color='blue', label='SI-SDRi')
    best_sisdri = max(sisdri_res)
    print(best_sisdri)
    idx_best = sisdri_res.index(best_sisdri)
    plt.scatter(idx_best,best_sisdri,color='cyan',s=100, edgecolor='black')
    plt.text(idx_best+1000,best_sisdri + 3, f'{idx_best} samples, SI-SDRi = {round(best_sisdri,5)}', color='black', fontsize=10, ha='center')
    plt.ylim(top = 15)
    plt.xlabel('shift in samples')
    plt.ylabel('SiSdri (dB)')
    plt.savefig(save_dir_i/'sisdr_plot.png')
    return(idx_best)

def si_sdr_calc(estimate_source, source):
    """Calculate SI-SNR or SI-SDR (same thing)
    Args:
    source: [B, C, T], B is batch size ,C= channels ,T = frames
    estimate_source: [B, C, T]
    """

    EPS = 1e-08
    source = torch.unsqueeze(source, dim=1)
    estimate_source = torch.unsqueeze(estimate_source, dim=1)
    ########## reshape to use broadcast ##########
    s_target = torch.unsqueeze(source, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(estimate_source, dim=2)  # [B, C, 1, T]
    ########## s_target = <s', s>s / ||s||^2 ##########
    pair_wise_dot = torch.sum(s_estimate * s_target,
                              dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(
        s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    ########## e_noise = s' - s_target ##########
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    ########## SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)##########
    pair_wise_si_snr = torch.sum(
        pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]
    pair_wise_si_snr = torch.transpose(pair_wise_si_snr, 1, 2)
    # si_snr = torch.sum(torch.squeeze(pair_wise_si_snr))

    return pair_wise_si_snr

def pad_time_domain_torchaudio(x,l):
    if x.shape[-1] > l:
        return x[:,:l]
    else:
        return torch.cat((x,torch.zeros(1,l-x.shape[-1])),-1)

save_dir_i = Path('/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/')

mix,sr = torchaudio.load('/dsi/scratch/users/yoavellinson/outputs/blcmv_anechoic_wsj0/mix_0.wav')
mix = torchaudio.functional.resample(mix,sr,16000)
max_len = mix.shape[-1]
y_ckpt,sr = torchaudio.load('/dsi/scratch/users/yoavellinson/outputs/blcmv_anechoic_wsj0/y_hat_1_0_az_270_elev_0_sisdr_-2.3984197816137676.wav')
y_ckpt = torchaudio.functional.resample(y_ckpt,sr,16000)
max_len = max(max_len,y_ckpt.shape[-1])
target,sr = torchaudio.load('/dsi/scratch/users/yoavellinson/outputs/blcmv_anechoic_wsj0/y1_0_az_270_elev_0.wav')
target = torchaudio.functional.resample(target,sr,16000)

# max_len = max(max_len,target.shape[-1])
# mix = pad_time_domain_torchaudio(mix,max_len)
# target = pad_time_domain_torchaudio(target,max_len)
# y_ckpt = pad_time_domain_torchaudio(y_ckpt,max_len)



print(plot_sisdr(mix,target,y_ckpt,save_dir_i))
