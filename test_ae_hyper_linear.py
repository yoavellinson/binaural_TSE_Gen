import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data import PatchDBDataset, collate_simple, collate_unif_sample
from tqdm import tqdm
from hyper_linear_vae.model import FreqSrcPosCondAutoEncoder
from pathlib import Path
import gc

@torch.no_grad()
def get_emb(model, batch, hp, device, freq):
    hrtf, itd, pos, name = (
        batch["patches"].to(device, non_blocking=True),
        batch["itd"].to(device, non_blocking=True),
        batch["pos"].to(device, non_blocking=True),
        batch["name"],
    )

    S, C, B_m, L = hrtf.shape
    hrtf = hrtf.permute(0, 2, 3, 1).unsqueeze(-2)
    hrtf = torch.view_as_real(hrtf)
    hrtf = hrtf.reshape(S, B_m, L, C * 2).permute(0, 1, 3, 2)

    # model forward
    _, _, prototype = model(hrtf, itd, freq, pos, pos, device)

    # cleanup local tensors
    del hrtf, itd, pos, batch
    return prototype, name

@torch.no_grad()
def main(device=None, debug=False):
    ckpt_path = '/home/workspace/yoavellinson/binaural_TSE_Gen/checkpoints/VAE/vae_step714000.pth'
    cfg_path = '/home/workspace/yoavellinson/binaural_TSE_Gen/conf/extraction_nbss_conf_large.yml'
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    hp = OmegaConf.load(cfg_path)
    ds = PatchDBDataset(hp, train=True, debug=debug)

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(hp.training.num_workers),
        pin_memory=True,
        collate_fn=collate_simple,
    )

    model_vae = FreqSrcPosCondAutoEncoder(OmegaConf.to_container(hp.vae.architecture, resolve=True))
    model_vae.load_state_dict(ckpt['model'])
    model_vae = model_vae.to(device)
    model_vae.eval()

    # precompute freq once, on the right device
    freq = torch.arange(0, hp.stft.fft_length // 2 + 1, device=device) * (
        (hp.stft.fs // 2) / (hp.stft.fft_length // 2 + 1)
    )

    out_dir = Path('/home/workspace/yoavellinson/binaural_TSE_Gen/ae_res/train_set')

    with torch.no_grad():
        for batch in tqdm(dl, total=len(ds)):
            name = batch['name']
            db_name = name[0].split('_sample_')[0].replace('db_', '')
            db_dir = out_dir / db_name
            db_dir.mkdir(exist_ok=True)
            sample_name = name[0].split('_sample_')[-1] + '.pt'
            path = db_dir / sample_name

            if path.exists():#in ['db_fhk_sample_HRIR_FULL2DEG','db_sadie_sample_D2_48K_24bit_256tap_FIR_SOFA','db_sadie_sample_D1_48K_24bit_256tap_FIR_SOFA']:
                continue

            try:
                latent, _ = get_emb(
                    model_vae, batch, hp, device=device, freq=freq
                )
                torch.save(latent.cpu(), path)  # save on CPU to be safe
                print('saved', path)

            except torch.cuda.OutOfMemoryError:
                print(f"[WARN] CUDA OOM on sample {name[0]}, skipping.")
                # clean up as much as possible
                if 'latent' in locals():
                    del latent
                del batch
                gc.collect()
                torch.cuda.empty_cache()
                # print(f'{name} not saved')
                continue

            except RuntimeError as e:
                # extra safety for older PyTorch where OOM is a RuntimeError
                if "out of memory" in str(e).lower():
                    print(f"[WARN] CUDA OOM (RuntimeError) on sample {name[0]}, skipping.")
                    if 'latent' in locals():
                        del latent
                    del batch
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                else:
                    # real bug: re-raise
                    raise

            # optional light cleanup each iteration
            del latent
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    device_idx = 0
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    main(device, debug=False)
