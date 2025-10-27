import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf
from data import ExtractionDataset
from torch.utils.data import random_split, DataLoader
from model import ComplexExtraction
from losses import SiSDRLoss,SpecMAE
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


from tqdm import tqdm
from pathlib import Path

def train_with_wandb(model, hp, train_loader, val_loader, num_epochs, device, learning_rate=0.0001, project_name="complex_extraction"):
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
    wandb.init(project=project_name, config={
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": train_loader.batch_size,
        "architecture": model.__class__.__name__,
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "sisdr_coeff":hp.loss.sisdr_coeff,
        "pesq_coeff":hp.loss.pesq_coeff
    })
    # wandb.init(mode='offline')
    # Send model to device
    device = torch.device(device)
    model = model.to(device)

    criterion_sisdr = SiSDRLoss(hp)
    criterion_mae = SpecMAE(hp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=hp.weight_decay)
    if hp.lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0

        step_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Steps", leave=False)
        for step,batch in enumerate(step_bar):
            optimizer.zero_grad()
            

            Mix,Y1,Y2,hrtf1,hrtf2,_,_,_,_,_,_ = batch
            Mix,Y1,Y2,hrtf1,hrtf2 = Mix.to(device),Y1.to(device),Y2.to(device),hrtf1.to(device),hrtf2.to(device)
            # Forward pass SPEAKER1
            outputs1 = model(Mix,hrtf1)
            loss = criterion_sisdr(outputs1, Y1)*hp.loss.sisdr_coeff
            # Forward pass SPEAKER2
            outputs2 = model(Mix,hrtf2)
            loss += criterion_sisdr(outputs2,Y2)*hp.loss.sisdr_coeff
            sisdr_loss = loss.item()
            loss = loss.to(device)
            # mae
            mae_loss = criterion_mae(outputs1,Y1)
            mae_loss += criterion_mae(outputs2,Y2)
            loss += (mae_loss*hp.loss.mae_coeff)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item() * Mix.size(0)
            wandb.log({"step_loss": loss.item(), "step": step + 1 + epoch * len(train_loader),
                       "step_sisdr_loss":sisdr_loss,
                    #    "step_pesq_loss":pesq_loss,
                        # "step_sisdr_mono": (mono_sisdr_loss_1+mono_sisdr_loss_2)/2,
                        "step_mae_loss":mae_loss})

            # Update progress bar
            step_bar.set_postfix({"Train Loss": loss.item()})
        if hp.lr_scheduler:
            scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation Steps", leave=False):
                Mix,Y1,Y2,hrtf1,hrtf2,_,_,_,_,_,_=val_batch
                Mix,Y1,Y2,hrtf1,hrtf2 = Mix.to(device),Y1.to(device),Y2.to(device),hrtf1.to(device),hrtf2.to(device)

                val_outputs1 = model(Mix,hrtf1)
                val_loss_i = criterion_sisdr(val_outputs1, Y1)*hp.loss.sisdr_coeff
                val_outputs2 = model(Mix,hrtf2)
                val_loss_i += criterion_sisdr(val_outputs2, Y2)*hp.loss.sisdr_coeff
                val_loss += val_loss_i.item() * Mix.size(0)

                # mix_sisdr = criterion_mix(outputs1,outputs2,Mix)
                # loss +=mix_sisdr

                # val_pesq_loss = criterion_pesq(val_outputs1,Y1) + criterion_pesq(val_outputs2,Y2)
                # val_loss += (val_pesq_loss*hp.loss.pesq_coeff)
                # val_loss = loss.to(device)
                val_mae_loss = criterion_mae(val_outputs1,Y1)
                val_mae_loss += criterion_mae(val_outputs2,Y2)
                loss += (val_mae_loss*hp.loss.mae_coeff)
                #mse
                # val_mse_loss = criterion_mse(torch.abs(val_outputs1).float(),torch.abs(Y1).float())
                # val_mse_loss += criterion_mse(torch.abs(val_outputs2).float(),torch.abs(Y2).float())
                # val_loss += (val_mse_loss*hp.loss.mse_coeff)                

        val_loss /= len(val_loader.dataset)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # Save the model checkpoint
        if epoch == 0:
            train_loss_best = train_loss
            val_loss_best = val_loss
            checkpoint_dir_run = Path(hp.checkpoint_dir)/f'{str(wandb.run.name)}_{model.__class__.__name__}_lr_{learning_rate}_bs_{train_loader.batch_size}_loss_sisdr_L1_rev'
            try:
                checkpoint_dir_run.mkdir(exist_ok=False)
            except:
                exit(0)
        # checkpoint_path = f"model_epoch_{epoch + 1}.pth"
        if train_loss <= train_loss_best and val_loss <= val_loss_best:
            checkpoint_path = f"model_epoch_best.pth"
            train_loss_best = train_loss
            val_loss_best = val_loss
            torch.save(model.state_dict(),checkpoint_dir_run / checkpoint_path)
            print(f'Model saved: Epoch-{epoch+1} Train loss:{train_loss_best} Val loss:{val_loss_best}')
        else:
            checkpoint_path = f"model_last_epoch.pth"
            torch.save(model.state_dict(),checkpoint_dir_run / checkpoint_path)

        # Print metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    wandb.finish()

if __name__=="__main__":
    device_idx = 0
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device_idx)  
    hp = OmegaConf.load('/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/conf/conf_rv.yaml')
    dm = ExtractionDataset(hp,train=True)
    train_size = int(0.8 * len(dm))  # 80% for training
    test_size = len(dm) - train_size  # Remaining 20% for testing

    train_dataset, test_dataset = random_split(dm, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=hp.dataset.batch_size_train, shuffle=True,num_workers=32,pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=hp.dataset.batch_size_val, shuffle=False,num_workers=32,pin_memory=False)

    model =ComplexExtraction(hp).to(device)
    resume = True
    if resume:
        checkpoint_path = "/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/checkpoints/best/breezy-glade-189_ComplexExtraction_lr_1e-05_bs_6_loss_sisdr_L1_pretrain_from_188/model_epoch_best.pth"
        checkpoint = torch.load(checkpoint_path,map_location=device)
        model.load_state_dict(checkpoint)
        
    train_with_wandb(model,hp,train_loader,test_loader,learning_rate=hp.lr,num_epochs=hp.num_epochs,device=device)
