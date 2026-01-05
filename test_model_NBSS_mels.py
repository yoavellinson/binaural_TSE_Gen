from torch.utils.data import Dataset,DataLoader
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as F
import numpy as np
from losses import PESQloss,SiSDRLossFromSTFT
from data import ExtractionDatasetRevVAE,PatchDBDataset,JoinedDataset,collate_joined
from NBSS.NBSS import NBSS
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def roll_tnzr(x:torch.Tensor):
    x = x.roll(-1)
    x[:,:,-1] = 0
    return x

def get_best_sisdr(crit,y_hat,y,max_shift=0.1):
    sisdr_res = []
    y_hat = y_hat[:,None,:]
    y = y[:,None,:]
    for i in tqdm(range(int(max_shift*16000))):
        sisdr = crit.channel_sisdr(y_hat,y)
        sisdr_res.append(float(sisdr))
        y= roll_tnzr(y)
    best_sisdr = min(sisdr_res)
    idx_best = sisdr_res.index(best_sisdr)
    print(idx_best)

    return best_sisdr


def load_checkpoint(model, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return 

# check the HRTF downsampling
if __name__ == "__main__":
    one_speaker=False
    device_idx = 1
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device_idx)  
    out_dir = Path('/home/workspace/yoavellinson/binaural_TSE_Gen/outputs/mixs_ys_rev_NBSS_mels2')
    hp = OmegaConf.load('/home/workspace/yoavellinson/binaural_TSE_Gen/conf/extraction_nbss_conf_large_mel.yml')
    ds_db  = PatchDBDataset(hp, train=False,debug=True)
    ds_mix = ExtractionDatasetRevVAE(hp, train=False,debug=True)
    joined_ds = JoinedDataset(ds_db, ds_mix)

    # db.sir=0
    test_loader = DataLoader(joined_ds, batch_size=1, shuffle=False,collate_fn=collate_joined)
    
    criterion_sisdr = SiSDRLossFromSTFT(hp)
    criterion_pesq = PESQloss(hp)

    sisdr_out = []
    sisdr_in = []
    pesq_out = []
    pesq_in = []
    dnsmos_ovrl = []
    dnsmos_sig = []
    dnsmos_bak = []
    sisdri=[]

    with torch.no_grad():
        model = NBSS(hp)
        model = model.to(device)
        checkpoint_path = "/home/workspace/yoavellinson/binaural_TSE_Gen/checkpoints/binaural_NBSS_large_mel2/None_NBSS_lr_0.001_bs_2_loss_sisdr_L1_rev/model_epoch_best.pth"
        load_checkpoint(model,path=checkpoint_path,device=device)
        model.eval()
        i=0
        #dns compute score
        for step,batch in tqdm(enumerate(test_loader),total=len(test_loader)):
            Mix = batch['mix_mix']
            Y1,Y2 = batch['mix_y1'],batch['mix_y2']
            hrtf1,hrtf2 = batch['db_hrtf1'],batch['db_hrtf2']
            hrtf_patches = batch['db_patches']
            Mix,Y1,Y2,hrtf1,hrtf2 = Mix.to(device),Y1.to(device),Y2.to(device),hrtf1.to(device),hrtf2.to(device)
            outputs1 = model(Mix,hrtf1)
            az1,elev1,az2,elev2 =batch['db_az1'],batch['db_elev1'],batch['db_az2'],batch['db_elev2']

            #sisdr
            sisdr_1 = criterion_sisdr(outputs1,Y1)
            sisdr_out.append(-sisdr_1.cpu())
            sisdr_in_1 = criterion_sisdr(Mix,Y1)
            sisdr_in.append(-sisdr_in_1.cpu())
            
            sisdri.append((-sisdr_1+sisdr_in_1).cpu())

            # pesq
            pesq_out_1 = criterion_pesq.mos(outputs1,Y1).max()
            pesq_out.append(pesq_out_1)
            pesq_in_1 = criterion_pesq.mos(Mix,Y1).max()
            pesq_in.append(pesq_in_1)

            mix = ds_mix.iSTFT(Mix).detach().cpu()
            sf.write(out_dir/f'mix_{step}.wav',mix.T,ds_mix.fs)
            y1 = ds_mix.iSTFT(Y1).detach().cpu()
            sf.write(out_dir/f'y1_{step}_az_{int(az1)}_elev_{int(elev1)}.wav',y1.T,ds_mix.fs)

            y_hat_1 = ds_mix.iSTFT(outputs1).detach().cpu()
            sf.write(out_dir/f'y_hat_1_{step}_az_{int(az1)}_elev_{int(elev1)}_sisdr_{sisdr_1:.3f}.wav',y_hat_1.T,ds_mix.fs)

            if not one_speaker:
                outputs2 = model(Mix,hrtf2)
                sisdr_2 = criterion_sisdr(outputs2,Y2)
                sisdr_out.append(-sisdr_2.cpu())
                sisdr_in_2 = criterion_sisdr(Mix,Y2)
                sisdr_in.append(-sisdr_in_2.cpu())
                sisdri.append((-sisdr_2+sisdr_in_2).cpu())
                pesq_out_2= criterion_pesq.mos(outputs2,Y2).mean()
                pesq_out.append(pesq_out_2)
                pesq_in_2= criterion_pesq.mos(Mix,Y2).mean()
                pesq_in.append(pesq_in_2)
                y2 = ds_mix.iSTFT(Y2).detach().cpu()
                sf.write(out_dir/f'y2_{step}_az_{int(az2)}_elev_{int(elev2)}.wav',y2.T,ds_mix.fs)
                y_hat_2 = ds_mix.iSTFT(outputs2).detach().cpu()
                sf.write(out_dir/f'y_hat_2_{step}_az_{int(az2)}_elev_{int(elev2)}_sisdr_{sisdr_2:.3f}.wav',y_hat_2.T,ds_mix.fs)
        # Ensure 1D arrays
        sisdr_in = reject_outliers(np.ravel(sisdr_in))
        sisdr_out = reject_outliers(np.ravel(sisdr_out))
        sisdri = reject_outliers(np.ravel(sisdri))

        pesq_in = reject_outliers(np.ravel(pesq_in))
        pesq_out = reject_outliers(np.ravel(pesq_out))
        # dnsmos_ovrl = reject_outliers(np.ravel(dnsmos_ovrl))
        # dnsmos_sig = reject_outliers(np.ravel(dnsmos_sig))
        # dnsmos_bak = reject_outliers(np.ravel(dnsmos_bak))
        sisdri_mean = np.mean(sisdri)


        # Calculate means
        mean_sisdr_out = np.mean(sisdr_out)
        mean_sisdr_in = np.mean(sisdr_in)
        mean_pesq_out = np.mean(pesq_out)
        mean_pesq_in = np.mean(pesq_in)
        mean_dnsmos = {'OVRL':np.mean(0),'SIG':np.mean(0),'BAK':np.mean(0)}

        # Create two side-by-side histograms
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        # fig.suptitle(f'SI-SDRi={sisdri_mean:.3f}\nDNSMOS OVRL/SIG/BAK:{mean_dnsmos['OVRL']:.3f}/{mean_dnsmos['SIG']:.3f}/{mean_dnsmos['BAK']:.3f}')
        fig.suptitle(f'SI-SDRi={sisdri_mean:.3f}')
        # SI-SDR Histogram
        axes[0].hist(sisdr_in, bins=30, alpha=0.5, color='blue', edgecolor='black', label="SI-SDR In")
        axes[0].hist(sisdr_out, bins=30, alpha=0.5, color='red', edgecolor='black', label="SI-SDR Out")
        axes[0].set_title(f"SI-SDR Histogram\nMean In: {mean_sisdr_in:.2f}, Mean Out: {mean_sisdr_out:.2f}")
        axes[0].set_xlabel("SI-SDR Score")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        # PESQ Histogram
        axes[1].hist(pesq_in, bins=30, alpha=0.5, color='green', edgecolor='black', label="PESQ In")
        axes[1].hist(pesq_out, bins=30, alpha=0.5, color='purple', edgecolor='black', label="PESQ Out")
        axes[1].set_title(f"PESQ Histogram\nMean In: {mean_pesq_in:.2f}, Mean Out: {mean_pesq_out:.2f}")
        axes[1].set_xlabel("PESQ Score")
        axes[1].legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(out_dir / 'sisdr_pesq_histograms.png')
