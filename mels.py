import torch
import matplotlib.pyplot as plt
# sample_rate = 16000
# nfilt = 8
# NFFT =512



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

    idxs = {f'bin_{i}':fbank[i].nonzero().flatten().tolist() for i in range(nfilt)}
    idxs[f'bin_{nfilt-1}'].append(nfft//2)
    return fbank,idxs
    

fbank,idxs=get_fbank(16000,512,9)
fbank = fbank.detach().cpu()  # in case it's on GPU / requires grad

plt.figure()
for i in range(fbank.shape[0]):
    plt.plot(fbank[i], label=f'Line {i}')

plt.xlabel("Index (0–256)")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig('mels.png')


# filter_banks = torch.dot(pow_frames, fbank.T)
# filter_banks = torch.where(filter_banks == 0, torch.finfo(float).eps, filter_banks)  # Numerical Stability
# filter_banks = 20 * torch.log10(filter_banks)  # dB