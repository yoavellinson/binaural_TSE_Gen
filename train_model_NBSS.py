import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf
from data import JoinedDataset,ExtractionDatasetRevVAE,PatchDBDataset,collate_joined
from torch.utils.data import random_split, DataLoader
from NBSS.NBSS import NBSS
from losses import SiSDRLossFromSTFT,SpecMAE
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, signal, threading 


from tqdm import tqdm
from pathlib import Path

from wandb_key import WANDB_API_KEY
import wandb
wandb.login(key=WANDB_API_KEY)
DEBUG=False

def train_with_wandb(model,optimizer,epoch_start,step_start, hp, train_loader, val_loader, num_epochs, device, learning_rate=0.0001, project_name="binaural_complex_extraction_many_heads"):
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
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Send model to device
    device = torch.device(device)
    model = model.to(device)

    criterion_sisdr = SiSDRLossFromSTFT(hp) #pit_sisdr_stft
    criterion_mae = SpecMAE(hp)

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
            i = torch.randint(0,2,(1,))
            # Forward pass SPEAKERi
            outputs1 = model(Mix,hrtfs[i])
            loss = criterion_sisdr(outputs1, Ys[i])#,hp)
            sisdr_loss = loss.item()
            loss = loss.to(device)
            # mae
            mae_loss = criterion_mae(outputs1,Ys[i])
            loss += (mae_loss*hp.loss.mae_coeff)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item() * Mix.size(0)
            global_step = step_start + step + 1 + epoch * len(train_loader)

            wandb.log({"step_loss": loss.item(), "step": global_step,
                       "step_sisdr_loss":sisdr_loss,
                        "step_mae_loss":mae_loss})

            step_bar.set_postfix({"Train Loss": loss.item()})
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation Steps", leave=False):
                Mix = val_batch['mix_mix']
                Y1,Y2 = val_batch['mix_y1'],val_batch['mix_y2']
                hrtf1,hrtf2 = val_batch['db_hrtf1'],val_batch['db_hrtf2']
                Mix,Y1,Y2,hrtf1,hrtf2 = Mix.to(device),Y1.to(device),Y2.to(device),hrtf1.to(device),hrtf2.to(device)
                # Forward pass SPEAKER1
                val_outputs1 = model(Mix,hrtf1)
                val_loss_i = criterion_sisdr(val_outputs1, Y1)*hp.loss.sisdr_coeff
                val_outputs2 = model(Mix,hrtf2)
                val_loss_i += criterion_sisdr(val_outputs2, Y2)*hp.loss.sisdr_coeff
                val_loss += val_loss_i.item() * Mix.size(0)

                val_mae_loss = criterion_mae(val_outputs1,Y1)
                val_mae_loss += criterion_mae(val_outputs2,Y2)
                loss += (val_mae_loss*hp.loss.mae_coeff)

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
            checkpoint_dir_run = Path(hp.training.checkpoint_dir)/f'{str(wandb.run.name)}_{model.__class__.__name__}_lr_{learning_rate}_bs_{train_loader.batch_size}_loss_sisdr_L1_rev'
            try:
                checkpoint_dir_run.mkdir(exist_ok=True)
            except:
                exit(0)

        if shutdown_event.is_set():
            print(f"[INFO] - signal seto saving checkpoint")
            checkpoint_path = f"model_ckpt_slurm_shutdown.pth"
            save_checkpoint(model,epoch,step,optimizer,checkpoint_dir_run / checkpoint_path)
            print(f"Model checkpoint saved on slurm shutdown")
            return
        
        # checkpoint_path = f"model_epoch_{epoch + 1}.pth"
        if train_loss <= train_loss_best and val_loss <= val_loss_best:
            checkpoint_path = f"model_epoch_best.pth"
            train_loss_best = train_loss
            val_loss_best = val_loss
            save_checkpoint(model,epoch,step,optimizer,checkpoint_dir_run / checkpoint_path)
            print(f'Model saved: Epoch-{epoch+1} Train loss:{train_loss_best} Val loss:{val_loss_best}')
        else:
            checkpoint_path = f"model_last_epoch.pth"
            save_checkpoint(model,epoch,step,optimizer,checkpoint_dir_run / checkpoint_path)


        # Print metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    wandb.finish()

def save_checkpoint(model,epoch,step,optimizer,path):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step':step
     }
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved at epoch {epoch + 1}")

def load_checkpoint(model, optimizer, path, device='cpu'):
    if not os.path.isfile(path):
        print(f"No checkpoint found at {path}")
        return 0, 0  # start fresh

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)

    print(f"Loaded checkpoint from epoch {start_epoch}, step {step}")
    return start_epoch, step

def handle_sig(signum, frame):
    # mark for graceful shutdown
    shutdown_event.set()

if __name__=="__main__":
    device_idx=0
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device_idx)  
    hp = OmegaConf.load('/home/workspace/yoavellinson/binaural_TSE_Gen/conf/extraction_nbss_conf_large.yml')

    ds_db  = PatchDBDataset(hp, train=True,debug=True if DEBUG else False)
    ds_mix = ExtractionDatasetRevVAE(hp, train=True,debug=True if DEBUG else False)

    joined_ds = JoinedDataset(ds_db, ds_mix)
    train_size = int(0.8 * len(joined_ds))  # 80% for training
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


    model = NBSS(hp)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.training.lr,weight_decay=hp.training.weight_decay)
    runs = sorted(Path(hp.training.checkpoint_dir).glob("**/*.pth"), key=os.path.getmtime)
    resume=False
    if runs:
        latest_checkpoint_pth = runs[-1]
        resume = True
    if resume:
        print(f"[INFO] Resuming from checkpoint: {latest_checkpoint_pth}")
        # checkpoint = torch.load(latest_checkpoint_pth,map_location=device,weights_only=False)
        epoch,step = load_checkpoint(model,optimizer,latest_checkpoint_pth,device)
    else:
        print("[INFO] Starting fresh training run.")    
        latest_checkpoint_pth=''
        epoch,step=0,0
    shutdown_event = threading.Event()

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, handle_sig)
        
    train_with_wandb(model,optimizer,epoch,step,hp,train_loader,val_loader,learning_rate=hp.training.lr,num_epochs=hp.training.num_epochs,device=device,project_name=hp.project)
