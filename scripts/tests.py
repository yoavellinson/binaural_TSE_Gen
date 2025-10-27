import torch
from numpy import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from data import ExtractionDataset
from torch.utils.data import DataLoader
from losses import SiSDRLoss,HRTFdeConvSiSDRLoss
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from pathlib import Path
from model import ComplexExtraction
import torchaudio
import numpy as np

# # Load the noisy speech signal
# hp = OmegaConf.load('/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/conf/config.yaml')
# file_path = "/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/mixs_ys/y_hat_2_39_az_tensor([275])_elev_tensor([-10])_sisdr_-11.481229782104492.wav"
# num_components = 250 


# def stft_sample(x):
#     X = torch.stft(torch.squeeze(x),n_fft=hp.stft.fft_length, hop_length=hp.stft.fft_hop, window=torch.hann_window(hp.stft.fft_length), return_complex=True).to(torch.complex64)
#     X[0:2, :] = X[0:2, :] * 0.001
#     mx_x = torch.max(torch.max(torch.abs(torch.real(X)) , torch.max(torch.abs(torch.imag(X)) )))
#     X = X/mx_x
#     return X

# def iSTFT(x):
#     x_time = torch.istft(torch.squeeze(x).detach().cpu(),n_fft=hp.stft.fft_length, hop_length=hp.stft.fft_hop, window=torch.hann_window(hp.stft.fft_length))
#     x_time = x_time/(x_time.abs().max(dim=-1, keepdim=True).values)
#     return x_time

# def apply_wiener_filter(Y, noise_estimate, alpha=0.9):
#     """
#     Apply Wiener filtering to reduce musical noise.
#     Y: Complex STFT of noisy signal
#     noise_estimate: Estimated noise STFT
#     alpha: Smoothing factor (0.9 - 0.99)
#     """
#     noise_power = ((noise_estimate.T@noise_estimate)**2).abs() #torch.abs(noise_estimate) ** 2
#     signal_power = torch.sum(torch.abs(Y) ** 2, dim=0, keepdim=True)
#     SNR = torch.clamp(signal_power / (noise_power + 1e-8), min=0)  # Avoid division by zero
#     Wiener_gain = SNR / (SNR + 1)  # Wiener filter gain

#     # Apply smoothing to reduce artifacts
#     Wiener_gain = alpha * Wiener_gain + (1 - alpha) * Wiener_gain.mean(dim=-1, keepdim=True)

#     return Y * Wiener_gain  # Apply the filter

# # Example: Apply Wiener filter after PCA denoising


# y, sr = torchaudio.load(file_path)
# Y = stft_sample(y)
# Y_hat = torch.empty(Y.shape,dtype=torch.complex64)
# frame_energy = torch.sum(torch.abs(Y) ** 2, dim=(0, 1))  # Sum over frequencies and channels

# # Get indices of 10 lowest-energy frames
# lowest_energy_indices = torch.argsort(frame_energy)[:100]

# # Compute noise estimate (average of 10 lowest-energy frames)
# for i in range(Y.shape[0]):
#     Y_i = Y[i,:,:]
#     noise_estimate = Y_i[:,:100].mean(dim=1, keepdim=True)
#     Y_hat[i,:,:] = apply_wiener_filter(Y_i,noise_estimate)

# y_hat = iSTFT(Y_hat).detach().cpu()
# torchaudio.save('/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/hat.wav',y_hat,sr)


# path = '/home/bari/workspace/svd/agg_nosiy.wav'
# wav,sr = torchaudio.load(path)[:16000]
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

# s = torch.stft(wav,512).squeeze().to(device)
# # s = torch.view_as_complex(s)
# s_mag = s[:,:,0]
# s_phase =s[:,:,1]
# s_hat_m = torch.empty(s_mag.shape).to(device)
# s_hat_p = torch.empty(s_mag.shape).to(device)
# cxx = torch.matmul(wav.T,wav)
# U, S, Vh = torch.linalg.svd(cxx, full_matrices=True)
# v1 = Vh[0].to(device)
# wav_hat = torch.matmul(wav.T,v1)*v1
# torch.save('time_pac.wav',wav_hat,sr)

# for i in range(int(512/2) +1):
#     Cxx = torch.matmul(s_mag[i].unsqueeze(1),s_mag[i].unsqueeze(1).T).to(device) # -mean?
#     U, S, Vh = torch.linalg.svd(Cxx, full_matrices=True)
#     v1 = Vh[0].to(device)
#     s_hat_m[i] = torch.matmul(s_mag[i].unsqueeze(1).T,v1)*v1
#     print(i)
# for i in range(int(512/2) +1):
#     Cxx = torch.matmul(s_phase[i].unsqueeze(1),s_phase[i].unsqueeze(1).T).to(device) # -mean?
#     U, S, Vh = torch.linalg.svd(Cxx, full_matrices=True)
#     v1 = Vh[0].to(device)
#     s_hat_p[i] = torch.matmul(s_phase[i].unsqueeze(1).T,v1)*v1
#     print(i)
# print('done')
# s_hat = torch.stack((s_hat_m,s_hat_p),dim=2)
# w_hat = torch.istft(s_hat.cpu(),512)
# torchaudio.save('pca2.wav',w_hat.unsqueeze(0),sr)
# f, axarr = plt.subplots(2)
# axarr[0,0] = plt.imshow(s.abs())
# axarr[0,1] = plt.imshow(s_hat.abs())

# U, S, V = torch.svd(Y)
# top_components = V[:, :num_components]
# reduced_frames = torch.matmul(frames_centered, top_components)

# # Reconstruct the frames
# reconstructed_frames = torch.matmul(reduced_frames, top_components.T) + mean

# # Overlap-add to reconstruct signal
# reconstructed_signal = torch.zeros(y.shape)
# count = torch.zeros(y.shape)

# for i in range(reconstructed_frames.shape[0]):
#     start = i * hop_size
#     reconstructed_signal[start : start + frame_size] += reconstructed_frames[i]
#     count[start : start + frame_size] += 1

# # Avoid division by zero
# count[count == 0] = 1
# reconstructed_signal /= count

# # Save the denoised speech
# sf.write("denoised_speech.wav", reconstructed_signal.numpy(), sr)

# print("Denoising complete! Saved as 'denoised_speech.wav'.")


if __name__=="__main__":
    device_idx = 1
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device_idx)  
    out_dir = Path('/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/mixs_ys')
    hp = OmegaConf.load('/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/conf/config.yaml')
    db = ExtractionDataset(hp,train=False)
    test_loader = DataLoader(db, batch_size=6, shuffle=False)
    criterion = HRTFdeConvSiSDRLoss(hp)
    with torch.no_grad():
        model =ComplexExtraction(hp).to(device)

        checkpoint_path = "/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/checkpoints/vibrant-thunder-175_ComplexExtraction_lr_1e-05_bs_6_loss_sisdr_L1_pretrain_from_150/model_epoch_199.pth"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model.eval()
        for batch in test_loader:
            Mix,Y1,Y2,hrtf1,hrtf2,hrtf1_time,hrtf2_time,az1,elev1,az2,elev2 = batch
            Mix,Y1,Y2,hrtf1,hrtf2 = Mix.to(device),Y1.to(device),Y2.to(device),hrtf1.to(device),hrtf2.to(device)
            outputs1 = model(Mix,hrtf1)
            loss = criterion(outputs1, Y1,hrtf1_time)*hp.loss.sisdr_coeff
            
            outputs2 = model(Mix,hrtf2)
            loss += criterion(outputs2,Y2,hrtf2_time)*hp.loss.sisdr_coeff
            sisdr_loss = loss.item()
            # l = criterion(Y1,Mix,hrtf1_time)
            # print(l)
            print(loss)
            break

