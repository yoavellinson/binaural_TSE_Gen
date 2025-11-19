import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data import PatchDBDataset,collate_simple
import os, time
from tqdm import tqdm
from hyper_linear_vae.model import FreqSrcPosCondAutoEncoder

import wandb
from wandb_key import WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

def train_step(model, batch, optimizer,hp,device):
    hrtf, itd, pos = batch["patches"].to(device), batch["itd"].to(device), batch["pos"].to(device)
    
    #sample
    optimizer.zero_grad()
    freq = torch.arange(1, hp.stft.fft_length//2 + 1) * ((hp.stft.fs//2) / (hp.stft.fft_length//2 + 1))

    hrtf_pred, itd_pred = model(hrtf,itd,freq,pos,pos,device)
    print('')
    

def main(device=None,debug=False):

    cfg_path = '/home/workspace/yoavellinson/binaural_TSE_Gen/conf/extraction_nbss_conf_large.yml'
    hp = OmegaConf.load(cfg_path)

    # W&B
    # wandb.init(
    #     project=str(hp.project),
    #     config={
    #         "num_epochs": hp.training.num_epochs,
    #         "learning_rate": hp.training.lr,
    #         "batch_size": hp.training.batch_size,
    #     },mode= "online" if not debug else "offline")
    wandb.init(mode='offline')
    # Data
    ds = PatchDBDataset(hp,train=True if not debug else False)  
    dl = DataLoader(
        ds,
        batch_size=int(hp.training.batch_size),
        shuffle=True,
        num_workers=int(hp.training.num_workers),
        pin_memory=True,
        collate_fn=collate_simple,
    )

    model_vae = FreqSrcPosCondAutoEncoder(OmegaConf.to_container(hp.vae.architecture,resolve=True)).to(device)
    optimizer = torch.optim.AdamW(
        model_vae.parameters(),
        lr=float(hp.training.lr),
        weight_decay=float(hp.training.weight_decay),
    )

    os.makedirs(str(hp.checkpoint_path), exist_ok=True)

    global_step = 0
    for epoch in range(int(hp.training.num_epochs)):
        model_vae.train()
        t0 = time.time()

        for batch in tqdm(dl,total=len(ds)//hp.training.batch_size):
            stats = train_step(
                model_vae, batch, optimizer, hp,device=device )
            wandb.log({
                "epoch": epoch,
                "step": global_step,
                "loss/total": stats["loss"],
                "loss/recon": stats["rec"],
                "loss/kl": stats["kl"],
                "loss/contrastive": stats["con"],
                "lr": optimizer.param_groups[0]["lr"],
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

        wandb.log({"epoch_time_s": time.time() - t0}, step=global_step)
        print(f"Epoch {epoch+1}/{int(hp.training.num_epochs)} in {time.time()-t0:.1f}s")

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



