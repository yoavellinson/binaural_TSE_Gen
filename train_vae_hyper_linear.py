import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data import PatchDBDataset,collate_simple,collate_unif_sample
from losses import SiSDRLossFromSTFT
import os, time
from tqdm import tqdm
from hyper_linear_vae.model import FreqSrcPosCondAutoEncoder
from functools import partial
import wandb
from wandb_key import WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

def train_step(model, batch, optimizer,sisdr_loss,l1_loss,hp,device):
    hrtf, itd, pos = batch["patches"].to(device), batch["itd"].to(device), batch["pos"].to(device)
    S ,C,B_m,L=hrtf.shape
    hrtf = hrtf.permute(0,2,3,1).unsqueeze(-2)
    hrtf=torch.view_as_real(hrtf)
    hrtf = hrtf.reshape(S, B_m, L, C * 2).permute(0,1,3,2)
    
    B_m = hp.training.b_m
    B_t = hp.training.num_samples-B_m
    idx = torch.randperm(hrtf.shape[1]) #unif sample train and target idxs
    idx_train = idx[:B_m]
    idx_target = idx[-B_t:]
    hrtf_m,itd_m,pos_m = hrtf[:,idx_train],itd[:,idx_train],pos[:,idx_train]
    hrtf_t,itd_t,pos_t = hrtf[:,idx_target],itd[:,idx_target],pos[:,idx_target]

    #sample
    optimizer.zero_grad()
    freq = torch.arange(0, hp.stft.fft_length//2 + 1) * ((hp.stft.fs//2) / (hp.stft.fft_length//2 + 1))
    hrtf_pred, itd_pred,prototype = model(hrtf_m,itd_m,freq,pos_m,pos_t,device)

    sisdr = sisdr_loss(hrtf_pred,hrtf_t)
    l1_itd = l1_loss(itd_pred,itd_t)

    loss = sisdr + l1_itd
    
    loss.backward()
    optimizer.step()
    stats = {'loss':loss}
    return stats

def main(device=None,debug=False):
    cfg_path = '/home/workspace/yoavellinson/binaural_TSE_Gen/conf/extraction_nbss_conf_large.yml'
    hp = OmegaConf.load(cfg_path)
    # W&B
    wandb.init(
        project=str(hp.project),
        config={
            "num_epochs": hp.training.num_epochs,
            "learning_rate": hp.training.lr,
            "batch_size": hp.training.batch_size,
        },mode= "online" if not debug else "offline")
    # wandb.init(mode='offline')
    # Data
    ds = PatchDBDataset(hp,train=True if not debug else False)  
    collate_fn = partial(collate_unif_sample,Nmax=hp.training.num_samples)
    dl = DataLoader(
        ds,
        batch_size=int(hp.training.batch_size),
        shuffle=True,
        num_workers=int(hp.training.num_workers),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model_vae = FreqSrcPosCondAutoEncoder(OmegaConf.to_container(hp.vae.architecture,resolve=True)).to(device)
    optimizer = torch.optim.AdamW(
        model_vae.parameters(),
        lr=float(hp.training.lr),
        weight_decay=float(hp.training.weight_decay),
    )
    #losses

    criterion_sisdr = SiSDRLossFromSTFT(hp,hop=hp.stft.fft_length)
    criterion_l1_itd = torch.nn.L1Loss()

    os.makedirs(str(hp.checkpoint_path), exist_ok=True)

    global_step = 0
    for epoch in range(int(hp.training.num_epochs)):
        model_vae.train()

        for batch in tqdm(dl,total=len(ds)//hp.training.batch_size):
            stats = train_step(
                model_vae, batch, optimizer,criterion_sisdr,criterion_l1_itd, hp,device=device )
            wandb.log({
                "epoch": epoch,
                "step": global_step,
                "loss/total": stats["loss"],                
            }, step=global_step)

            # Periodic checkpoint
            if global_step and global_step % 500 == 0:
                ckpt_path = os.path.join(str(hp.checkpoint_path), f"vae_step{global_step}.pth")
                torch.save({
                    "model": model_vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "config": OmegaConf.to_container(hp, resolve=True),
                }, ckpt_path)
                wandb.save(ckpt_path)

            global_step += 1


    # Final save
    final_ckpt = os.path.join(str(hp.checkpoint_path), "vae_final.pth")
    torch.save({
        "model": model_vae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": global_step,
        "config": OmegaConf.to_container(hp, resolve=True),
    }, final_ckpt)
    wandb.save(final_ckpt)
    wandb.finish()

if __name__=="__main__":
    device_idx = 1
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    main(device,debug=False)



