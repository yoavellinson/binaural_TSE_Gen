from model_enc import PatchDBVAE
import torch
from omegaconf import OmegaConf
from losses import mae_recon_loss,make_mae_keep,kl_divergence,info_nce
from torch.utils.data import DataLoader
from data import PatchDBDataset,collate_simple
import os, time, random, numpy as np
from tqdm import tqdm

import wandb
from wandb_key import WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

def beta_schedule(step, warmup_steps=10000):
    return min(1.0, step / warmup_steps)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train_step(model_vae, batch, optimizer, beta, mask_ratio=0.65, pos_jitter=0.01,
               scale_xy=(1.0, 1.0), temp=0.1, amp=False, scaler=None, device="cuda"):
    x_in = batch['patches'].to(device, non_blocking=True)   # [B, C, N, T]
    xy   = batch['pos'].to(device, non_blocking=True)       # [B, N, 2]
    kpm  = batch['kpm'].to(device, non_blocking=True)       # [B, N] True=PAD

    B, N,C, T = x_in.shape
    mae_keep_A = make_mae_keep(B, N, mask_ratio, device)
    mae_keep_B = make_mae_keep(B, N, mask_ratio, device)

    valid = (~kpm).unsqueeze(-1)
    xyA = xy + pos_jitter * torch.randn_like(xy) * valid
    xyB = xy + pos_jitter * torch.randn_like(xy) * valid

    optimizer.zero_grad(set_to_none=True)

    if amp:
        assert scaler is not None
        with torch.amp.autocast('cuda',dtype=torch.float16):
            x_hat_A, zA, muA, logvarA = model_vae(x_in, xyA, kpm, scale=scale_xy)
            x_hat_B, zB, muB, logvarB = model_vae(x_in, xyB, kpm, scale=scale_xy)
            L_rec = mae_recon_loss(x_hat_A, x_in, mae_keep_A, kpm) + mae_recon_loss(x_hat_B, x_in, mae_keep_B, kpm)
            L_kl  = kl_divergence(muA, logvarA) + kl_divergence(muB, logvarB)
            L_con = info_nce(zA, zB, temperature=temp)
            loss  = L_rec + beta * L_kl + L_con
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model_vae.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
    else:
        x_hat_A, zA, muA, logvarA = model_vae(x_in, xyA, kpm, scale=scale_xy)
        x_hat_B, zB, muB, logvarB = model_vae(x_in, xyB, kpm, scale=scale_xy)
        L_rec = mae_recon_loss(x_hat_A, x_in, mae_keep_A, kpm) + mae_recon_loss(x_hat_B, x_in, mae_keep_B, kpm)
        L_kl  = kl_divergence(muA, logvarA) + kl_divergence(muB, logvarB)
        L_con = info_nce(zA, zB, temperature=temp)
        loss  = L_rec + beta * L_kl + L_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_vae.parameters(), 1.0)
        optimizer.step()

    return {"loss": float(loss.detach()), "rec": float(L_rec.detach()),
            "kl": float(L_kl.detach()), "con": float(L_con.detach())}



def main(device=None,debug=False):

    cfg_path = '/home/workspace/yoavellinson/binaural_TSE_Gen/conf/vae.yml'
    hp = OmegaConf.load(cfg_path)
    set_seed(int(hp.training.seed))

    # W&B
    wandb.init(
        project=str(hp.project),
        config={
            "num_epochs": hp.training.num_epochs,
            "learning_rate": hp.training.lr,
            "batch_size": hp.training.batch_size,
        },mode= "online" if not debug else "offline")

    # Data
    ds = PatchDBDataset(hp,train=True if not debug else False)  # your dataset reads from cfg/hp as before
    dl = DataLoader(
        ds,
        batch_size=int(hp.training.batch_size),
        shuffle=True,
        num_workers=int(hp.training.num_workers),
        pin_memory=True,
        collate_fn=collate_simple,
    )

    # Model / Opt
    model_vae = PatchDBVAE(hp).to(device)
    optimizer = torch.optim.AdamW(
        model_vae.parameters(),
        lr=float(hp.training.lr),
        weight_decay=float(hp.training.weight_decay),
    )
    scaler = torch.amp.GradScaler('cuda',enabled=bool(hp.training.amp))

    os.makedirs(str(hp.checkpoint_path), exist_ok=True)

    global_step = 0
    for epoch in range(int(hp.training.num_epochs)):
        model_vae.train()
        t0 = time.time()

        for batch in tqdm(dl,total=len(ds)//hp.training.batch_size):
            beta = beta_schedule(global_step, warmup_steps=int(hp.training.warmup_steps))
            stats = train_step(
                model_vae, batch, optimizer,
                beta=beta,
                mask_ratio=float(hp.training.mask_ratio),
                pos_jitter=float(hp.training.pos_jitter),
                scale_xy=(1.0, 1.0),
                temp=float(hp.training.temp),
                amp=bool(hp.training.amp),
                scaler=scaler,
                device=device,
            )
            wandb.log({
                "epoch": epoch,
                "step": global_step,
                "loss/total": stats["loss"],
                "loss/recon": stats["rec"],
                "loss/kl": stats["kl"],
                "loss/contrastive": stats["con"],
                "beta": beta,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

            # Periodic checkpoint
            if global_step and global_step % 500 == 0:
                ckpt_path = os.path.join(str(hp.checkpoint_path), f"vae_step{global_step}.pth")
                torch.save({
                    "model": model_vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if bool(hp.training.amp) else None,
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
        "scaler": scaler.state_dict() if bool(hp.training.amp) else None,
        "step": global_step,
        "config": OmegaConf.to_container(hp, resolve=True),
    }, final_ckpt)
    wandb.save(final_ckpt)
    wandb.finish()

if __name__=="__main__":
    device_idx = 0
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    main(device,debug=False)



