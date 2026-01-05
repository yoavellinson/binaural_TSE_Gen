import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf
from data import JoinedDataset,ExtractionDatasetRevVAE,PatchDBDataset,collate_joined
from torch.utils.data import random_split, DataLoader
from NBSS.NBSS import NBSS,NBSS_CFM
from losses import SiSDRLossFromSTFT,SpecMAE
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, signal, threading 
from test_model_NBSS_gen import sample_flow

from tqdm import tqdm
from pathlib import Path

from wandb_key import WANDB_API_KEY
import wandb
wandb.login(key=WANDB_API_KEY)
DEBUG=False


def get_scheduler(optimizer, total_steps: int, warmup_steps: int, lr_min_ratio: float = 0.01):
    """
    Creates a learning rate scheduler with a linear warmup followed by 
    cosine annealing decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer being used (e.g., AdamW).
        total_steps (int): Total number of training steps (iterations).
        warmup_steps (int): Number of steps for the linear warmup phase.
        lr_min_ratio (float): The final LR will be (initial_LR * lr_min_ratio).
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: The combined scheduler.
    """
    
    # --- 1. Warmup Function (Linear Increase) ---
    def warmup_func(step):
        if step < warmup_steps:
            # Linear increase from 0 to 1
            return float(step) / float(max(1, warmup_steps))
        
        # --- 2. Cosine Decay Function ---
        # After warmup, the effective step starts from 0 for the remaining steps.
        cosine_steps = total_steps - warmup_steps
        current_cosine_step = step - warmup_steps
        
        # Cosine Annealing (from 1 down to lr_min_ratio)
        if current_cosine_step < cosine_steps:
            # Cosine factor goes from 1.0 down to 0.0
            cosine_factor = 0.5 * (1. + torch.cos(
                torch.tensor(current_cosine_step / cosine_steps) * torch.pi
            ))
            # Scale the factor to land at lr_min_ratio instead of 0
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_factor
        
        # If training extends beyond total_steps, LR stays at the minimum
        return lr_min_ratio 

    return LambdaLR(optimizer, lr_lambda=warmup_func)


def train_with_wandb(model,optimizer,scheduler,epoch_start,step_start, hp, train_loader, val_loader, num_epochs, device, learning_rate=0.0001, project_name="binaural_complex_extraction_many_heads"):
    """
    Trains a PyTorch model on multiple GPUs and logs metrics to Weights & Biases (wandb).

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        hp: Hyperparameter object with checkpoint directory information.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
        learning_rate (float): Learning rate for the optimizer.
        project_name (str): Name of the wandb project.
    """
    if not DEBUG:
        wandb.init(project=project_name, config={
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_loader.batch_size,
            "architecture": model.__class__.__name__,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "sisdr_coeff":hp.loss.sisdr_coeff,
        })
    else:
        wandb.init(mode='offline')

    # Send model to device
    device = torch.device(device)
    model = model.to(device)

    criterion_sisdr = SiSDRLossFromSTFT(hp) #pit_sisdr_stft
    val_loss = 90000

    for epoch in tqdm(range(epoch_start,num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        step_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Steps",
            leave=False,
            total=len(train_loader)
        )
        for step,batch in enumerate(step_bar):
            optimizer.zero_grad()
            
            Mix = batch['mix_mix']
            Y1,Y2 = batch['mix_y1'],batch['mix_y2']
            hrtf1,hrtf2 = batch['db_hrtf1'],batch['db_hrtf2']
            Mix,Y1,Y2,hrtf1,hrtf2 = Mix.to(device),Y1.to(device),Y2.to(device),hrtf1.to(device),hrtf2.to(device)
            hrtfs = (hrtf1,hrtf2)
            Ys = (Y1,Y2)
            i = torch.randint(0, 2, (1,)).item()

            #sample ti
            B = Mix.shape[0]
            t = torch.rand(B, device=Ys[i].device)
            z = torch.randn_like(Ys[i])
            # reshape t to broadcast across [C_out, F, T]
            t_ = t.view(B, 1, 1, 1)
            s_t = (1 - t_) * z + t_ * Ys[i]

            # Forward pass SPEAKERi
            v_pred = model(Mix,hrtfs[i],s_t,t)
            v_true = Ys[i] - z

            diff = v_pred - v_true
            loss = (diff.real**2 + diff.imag**2).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Metrics
            train_loss += loss.item() * Mix.size(0)
            global_step = step_start + step + 1 + epoch * len(train_loader)

            wandb.log({"step_loss": loss.item(), "step": global_step,'current_lr':scheduler.get_last_lr()[0]})
            step_bar.set_postfix({"Train Loss": loss.item()})
        train_loss /= len(train_loader.dataset)

        # Validation loop
        if epoch%10==0 or not DEBUG:
            model.eval()
            val_loss = 0.0
        
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation Steps", leave=False):
                    Mix = val_batch['mix_mix']
                    Y1,Y2 = val_batch['mix_y1'],val_batch['mix_y2']
                    hrtf1,hrtf2 = val_batch['db_hrtf1'],val_batch['db_hrtf2']
                    Mix,Y1,Y2,hrtf1,hrtf2 = Mix.to(device),Y1.to(device),Y2.to(device),hrtf1.to(device),hrtf2.to(device)
                    # Mix: [B, C, F, T]
                    Mix_big = torch.cat([Mix, Mix], dim=0)     # [2B, C, F, T]
                    S_ref_big = torch.cat([Y1, Y2], dim=0)     # [2B, Cout, F, T]
                    H_big = torch.cat([hrtf1, hrtf2], dim=0)   # [2B, C, F]

                    s_gen =  sample_flow(model, Mix_big, H_big)
                    val_loss_i = criterion_sisdr(s_gen, S_ref_big)*hp.loss.sisdr_coeff
                    val_loss += val_loss_i.item() * Mix.size(0)

            val_loss /= len(val_loader.dataset)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

            
        # Save the model checkpoint
        if epoch == 0 or epoch==epoch_start:
            train_loss_best = train_loss
            val_loss_best = val_loss
            checkpoint_dir_run = Path(hp.checkpoint_path)/f'{str(wandb.run.name)}_{model.__class__.__name__}_lr_{learning_rate}_bs_{train_loader.batch_size}_loss_sisdr_L1_rev'
            try:
                checkpoint_dir_run.mkdir(exist_ok=True)
            except:
                exit(0)

        if shutdown_event.is_set():
            print(f"[INFO] - signal seto saving checkpoint")
            checkpoint_path = f"model_ckpt_slurm_shutdown.pth"
            save_checkpoint(model,epoch,step,optimizer,scheduler,checkpoint_dir_run / checkpoint_path)
            print(f"Model checkpoint saved on slurm shutdown")
            return
        
        # checkpoint_path = f"model_epoch_{epoch + 1}.pth"
        if train_loss <= train_loss_best or val_loss <= val_loss_best:
            checkpoint_path = f"model_epoch_best.pth"
            train_loss_best = train_loss
            val_loss_best = val_loss
            save_checkpoint(model,epoch,step,optimizer,scheduler,checkpoint_dir_run / checkpoint_path)
            print(f'Model saved: Epoch-{epoch+1} Train loss:{train_loss_best} Val loss:{val_loss_best}')
        else:
            checkpoint_path = f"model_last_epoch.pth"
            save_checkpoint(model,epoch,step,optimizer,scheduler,checkpoint_dir_run / checkpoint_path)


        # Print metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    wandb.finish()

def save_checkpoint(model,epoch,step,optimizer,scheduler,path): # <--- ADD scheduler here
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), # <--- CRITICAL ADDITION
        'step':step
     }
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved at epoch {epoch + 1}")

def load_checkpoint(model, optimizer, scheduler, path, device='cpu'): # <--- ADD scheduler here
    if not os.path.isfile(path):
        print(f"No checkpoint found at {path}")
        return 0, 0  # start fresh

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # <--- CRITICAL ADDITION

    start_epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)

    print(f"Loaded checkpoint from epoch {start_epoch}, step {step}")
    return start_epoch, step

def handle_sig(signum, frame):
    # mark for graceful shutdown
    shutdown_event.set()

if __name__=="__main__":
    device_idx=1
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device_idx)  
    hp = OmegaConf.load('/home/workspace/yoavellinson/binaural_TSE_Gen/conf/extraction_nbss_conf_large_large_gen.yml')

    ds_db  = PatchDBDataset(hp, train=True,debug=True if DEBUG else False)
    ds_mix = ExtractionDatasetRevVAE(hp, train=True,debug=True if DEBUG else False)

    joined_ds = JoinedDataset(ds_db, ds_mix)
    train_size = len(joined_ds)-hp.training.batch_size #int(0.95 * len(joined_ds))  # 80% for training
    test_size = len(joined_ds) - train_size  # Remaining 20% for testing
    train_dataset, test_dataset = random_split(joined_ds, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.training.batch_size,
        num_workers=hp.training.num_workers,
        collate_fn=collate_joined,
        shuffle=True
    )    
    val_loader = DataLoader(
        test_dataset,
        batch_size=hp.training.batch_size,
        num_workers=hp.training.num_workers,
        collate_fn=collate_joined,
        shuffle=False
    )    


    model = NBSS_CFM(hp)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.training.lr,weight_decay=hp.training.weight_decay)

    scheduler = get_scheduler(
    optimizer, 
    total_steps=len(train_loader)*hp.training.num_epochs, 
    warmup_steps=10000, 
    lr_min_ratio=0.05
    )
    
    runs = sorted(Path(hp.checkpoint_path).glob("**/*.pth"), key=os.path.getmtime)
    resume=False
    if runs:
        latest_checkpoint_pth = runs[-1]
        resume = True
    if resume:
        print(f"[INFO] Resuming from checkpoint: {latest_checkpoint_pth}")
        # checkpoint = torch.load(latest_checkpoint_pth,map_location=device,weights_only=False)
        epoch,step = load_checkpoint(model,optimizer,scheduler,latest_checkpoint_pth,device)
    else:
        print("[INFO] Starting fresh training run.")    
        latest_checkpoint_pth=''
        epoch,step=0,0
    shutdown_event = threading.Event()

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, handle_sig)
    

    train_with_wandb(model,optimizer,scheduler,epoch,step,hp,train_loader,val_loader,learning_rate=hp.training.lr,num_epochs=hp.training.num_epochs,device=device,project_name=hp.project)
