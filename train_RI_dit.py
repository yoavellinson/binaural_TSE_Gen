import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusion import create_diffusion
from models_dit import DiT
# from datasets import get_dataset
from tools.data_processing import complex_to_interleaved
import torch.optim as optim
from data import ExtractionDatasetRev
from omegaconf import OmegaConf
from wandb_key import WANDB_API_KEY
import wandb
wandb.login(key=WANDB_API_KEY)
import os, signal, threading 
from pathlib import Path
import numpy as np
import copy
import math
import torch.nn.utils as nn_utils

class EMA:
    def __init__(self, model, decay=0.9999, device=None):
        # deep copy so buffers/shapes match exactly
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        self.ema_model.requires_grad_(False)
        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model):
        # supports DDP-wrapped model
        src = model.module if hasattr(model, "module") else model
        for (name, ema_v) in self.ema_model.state_dict().items():
            if name not in src.state_dict():
                continue
            cur_v = src.state_dict()[name]
            if not cur_v.dtype.is_floating_point:
                ema_v.copy_(cur_v)  # keep buffers (e.g., ints) in sync
            else:
                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * cur_v)

    def to(self, device):
        self.ema_model.to(device)
        return self


def train(hp,resume_checkpoint_path=''):
    # Dataset & DataLoader
    train_dataset = ExtractionDatasetRev(hp,train=True,mono_input=True,debug=False)
    dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
    wandb.init(project='mono-to-stereo-DiT', config={
            "num_epochs": hp.num_epochs,
            "learning_rate": hp.lr,
            "batch_size": hp.batch_size,
        },mode= "online")
    # wandb.init(mode='offline')
    # Model and optimizers
    device = hp.device if torch.cuda.is_available() else "cpu"

    # 1) weights
    model = DiT(
        input_size=tuple(hp.input_size),
        patch_size=hp.patch_size,
        in_channels=hp.in_channels, 
        hidden_size=hp.hidden_size,
        depth=hp.depth,
        num_heads=hp.num_heads,
    )

    epoch=0
    ema = EMA(model, decay=0.9999, device=device)

    if resume_checkpoint_path!='':
        ckpt = torch.load(resume_checkpoint_path, map_location=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        try:
            ema.ema_model.load_state_dict(ckpt["ema"])
        except KeyError:
            # first run without EMA saved—initialize EMA from current model
            ema.ema_model.load_state_dict((model.module if hasattr(model,"module") else model).state_dict())
        ema.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = ckpt.get("epoch", 0)
        print(f'###### RESUMING: from -{resume_checkpoint_path}')
    else:
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
    
    
    # Create diffusion model
    diffusion = create_diffusion(timestep_respacing="",diffusion_steps=30,predict_v=True,predict_xstart=False)

    # Create directory to save model checkpoints
    os.makedirs(hp.checkpoint_dir, exist_ok=True)

    # Training Loop
    num_epochs=hp.num_epochs
    sample_interval=hp.sample_interval
    
    model.train()


    for epoch in range(epoch,num_epochs):
        total_loss = 0.0
        for step, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):

            cond_tf,x1,x2,_,_,_,_=data
            B = cond_tf.shape[0]
            speaker_idx = torch.randint(0, 2, (1,))
            x = [x1,x2][speaker_idx]
            #complex to RI channels
            x = complex_to_interleaved(x).to(device)
            cond_tf = complex_to_interleaved(cond_tf).to(device)
            # Sample t
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            t = torch.from_numpy(t).to(device).long()

            model_kwargs = dict(y=cond_tf)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Grad norm
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

            total_loss += loss.item()
            wandb.log({"step_loss": loss.item(), "step": step + 1 + epoch * len(dataloader)})
            if shutdown_event.is_set():
                print(f"[INFO] - signal seto saving checkpoint")
                checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema':ema.ema_model.state_dict() 
                }
                checkpoint_path = f"{hp.checkpoint_dir}/model_ckpt_slurm_shutdown.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Model checkpoint saved on slurm shutdown")
                return
            
        avg_loss = total_loss / len(dataloader)
        wandb.log({"train_loss": avg_loss,"epoch":epoch + 1})
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % sample_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema':ema.ema_model.state_dict()
            }
            checkpoint_path = f"{hp.checkpoint_dir}/model_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1}")
         
        
def handle_sig(signum, frame):
    # mark for graceful shutdown
        shutdown_event.set()

if __name__ == "__main__":
    hp = OmegaConf.load('/home/workspace/yoavellinson/mono-to-stereo/conf/train.yaml')
    resume = False
    runs = sorted(Path(hp.checkpoint_dir).glob("*.pt"), key=os.path.getmtime)
    if runs:
        latest_checkpoint_pth = runs[-1]
        resume = True
    if resume:
        print(f"[INFO] Resuming from checkpoint: {latest_checkpoint_pth}")
    else:
        print("[INFO] Starting fresh training run.")    
        latest_checkpoint_pth=''
    shutdown_event = threading.Event()
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, handle_sig)

    train(hp,resume_checkpoint_path=latest_checkpoint_pth)
    # writer.close()

