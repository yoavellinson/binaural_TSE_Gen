from model_enc import PatchDBVAE
import torch
from omegaconf import OmegaConf
from losses import mae_recon_loss,make_mae_keep,kl_divergence,info_nce
from torch.utils.data import DataLoader
from data import PatchDBDataset,collate_simple
import os, time, random, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def test_step(model_vae, batch, beta=1.0, mask_ratio=0.65, pos_jitter=0.01,
              scale_xy=(1.0, 1.0), temp=0.1, amp=False, device="cuda"):
    """
    Evaluate the VAE on a held-out batch (no gradient updates).

    Args:
        model_vae: your PatchDBVAE model.
        batch: dict with
            'patches': [B, C, N, T]
            'pos': [B, N, 2]
            'kpm': [B, N] (True=PAD)
        beta: KL weight.
        mask_ratio: same as in train_step.
        pos_jitter: random noise added to positions.
        scale_xy: normalization scale for xy coords.
        temp: contrastive temperature for info_nce.
        amp: use mixed precision if True.
        device: CUDA or CPU.
    """
    model_vae.eval()
    with torch.no_grad():
        x_in = batch['patches'].to(device, non_blocking=True)
        xy   = batch['pos'].to(device, non_blocking=True)
        kpm  = batch['kpm'].to(device, non_blocking=True)
        B, C, N, T = x_in.shape

        mae_keep_A = make_mae_keep(B, N, mask_ratio, device)
        xyA = xy

        if amp:
            with torch.amp.autocast('cuda',dtype=torch.float16):
                x_hat_A, zA, muA, logvarA = model_vae(x_in, xyA, kpm, scale=scale_xy)
        else:
            x_hat_A, zA, muA, logvarA = model_vae(x_in, xyA, kpm, scale=scale_xy)

        return zA


def main(device=None,path=''):

    ckpt = torch.load(path,weights_only=False)
    cfg = ckpt['config']
    hp = OmegaConf.create(cfg)
    set_seed(int(hp.training.seed))

    ds = PatchDBDataset(hp,train=False)  # your dataset reads from cfg/hp as before
    dl = DataLoader(
        ds,
        batch_size=int(hp.training.batch_size),
        shuffle=True,
        num_workers=int(hp.training.num_workers),
        pin_memory=True,
        collate_fn=collate_simple,
    )

    # # Model / Opt
    model_vae = PatchDBVAE(hp).to(device)
    model_vae.load_state_dict(ckpt['model'])
    model_vae.eval()
    zs = []
    for batch in tqdm(dl,total=len(ds)//hp.training.batch_size):
        z = test_step(
            model_vae, batch,
            beta=1,
            mask_ratio=float(hp.training.mask_ratio),
            pos_jitter=float(hp.training.pos_jitter),
            scale_xy=(1.0, 1.0),
            temp=float(hp.training.temp),
            amp=bool(hp.training.amp),
            device=device,
        )
        zs.append(z)
    zs = torch.cat(zs)
    # print(zs.shape)
    X = zs.detach().cpu().float().numpy()               # (80, 256)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Xc = U S V^T
    X2 = Xc @ Vt[:2].T                                 # (80, 2)

    # explained variance
    var = (S**2) / (X.shape[0] - 1)
    evr = var[:2] / var.sum()
    print(f"Explained variance: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}")

    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2"); plt.title("Latent z PCA (80×256 → 2D)")
    plt.tight_layout()
    plt.savefig('pca.png')
    plt.close()

if __name__=="__main__":
    device_idx = 0
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    main(device,path='/home/workspace/yoavellinson/binaural_TSE_Gen/checkpoints/vae_step14000.pt')



