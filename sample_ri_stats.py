import torch
import soundfile as sf
from models_dit import DiT
from diffusion import create_diffusion
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from tools.data_processing import complex_to_interleaved,interleaved_to_complex
from data import MonoStereoWhamrDataset
from omegaconf import OmegaConf
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
from train_RI_dit import EMA
from lcmv.torch_lcmv_class import LCMV_torch
from losses import PITPESQLoss,PITSiSDRLoss
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def tf_probe(diffusion, model, x0, num_points=10, model_kwargs=None):
    device = x0.device
    B = x0.shape[0]
    timesteps = torch.linspace(
        diffusion.num_timesteps - 1, 0, num_points, dtype=torch.long, device=device
    )
    results = []
    for t_val in timesteps:
        t = t_val.repeat(B)
        eps = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise=eps)
        out = diffusion.p_mean_variance(model, x_t, t, clip_denoised=False, model_kwargs=model_kwargs)
        x0_hat = out["pred_xstart"]
        mse = ((x0_hat - x0) ** 2).mean().item()
        results.append((int(t_val.item()), mse))
    return results

def load_trained_model(checkpoint_path,hp):
    model = DiT(
        input_size=tuple(hp.input_size),
        patch_size=hp.patch_size,
        in_channels=hp.in_channels, 
        hidden_size=hp.hidden_size,
        depth=hp.depth,
        num_heads=hp.num_heads,
    )
    model.to(device)
    checkpoint = torch.load(checkpoint_path)
    ema = EMA(model, decay=0.9999, device=device)

    try:
        ema.ema_model.load_state_dict(checkpoint["ema"])
    except KeyError:
    # first run without EMA saved—initialize EMA from current model
        ema.ema_model.load_state_dict((model.module if hasattr(model,"module") else model).state_dict())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # model = ema.ema_model
    # model.eval()
    return model


def infer_and_generate_audio(model, diffusion, spec_cond,hp):
    latent_size = tuple(hp.input_size)
    in_channels= hp.in_channels
    z = torch.randn(1, in_channels, latent_size[0], latent_size[1], device=device) # (b, c, t, f)
    model_kwargs = dict(y=spec_cond)

    with torch.no_grad():
        samples = diffusion.ddim_sample_loop(
            model=model.forward,shape= z.shape,noise=None, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
        )
    return samples

def save_audio(audio, output_path, it, sr=16000):
    audio = audio.cpu().numpy()
    output_path = Path(output_path, f"sample{it}.wav")
    if audio.shape[0] == 1:
        audio = audio.squeeze(0)  
        sf.write(file=output_path, data=audio, samplerate=sr)
    else:
        sf.write(file=output_path, data=audio.T, samplerate=sr)
    
    print(f"Write out to {output_path}")

def main():
    epoch=2460
    NUM_SAMPLES=2
    import argparse
    parser = argparse.ArgumentParser(description='Generate audio using DiT')
    parser.add_argument('--checkpoint', type=str,default=f'/home/workspace/yoavellinson/mono-to-stereo/checkpoints_RI_v/model_epoch_{epoch}.pt', help='Path to model checkpoint')
    parser.add_argument('--hp', type=str, default='/home/workspace/yoavellinson/mono-to-stereo/conf/train.yaml', help='Path to conf')
    parser.add_argument('--output_gt_dir', type=str, default='/home/workspace/yoavellinson/mono-to-stereo/diff_outputs/gt_dir', help='Directory to save ground truth audio')
    parser.add_argument('--output_gt_mono_dir', type=str,default='/home/workspace/yoavellinson/mono-to-stereo/diff_outputs/gt_mono_dir', help='Directory to save ground truth mono audio')
    parser.add_argument('--output_gen_dir', type=str, default='/home/workspace/yoavellinson/mono-to-stereo/diff_outputs/gen_dir', help='Directory to save generated audio')
    parser.add_argument('--output_gen_dir_lcmv', type=str, default='/home/workspace/yoavellinson/mono-to-stereo/diff_outputs/gen_dir_lcmv', help='Directory to save generated audio')
    parser.add_argument('--output_gt_dir_mono_lcmv', type=str, default='/home/workspace/yoavellinson/mono-to-stereo/diff_outputs/gt_dir_mono_lcmv', help='Directory to save generated audio')
    
    args = parser.parse_args()
    hp = OmegaConf.load(args.hp)
    lcmv = LCMV_torch()
    pit_sisdr = PITSiSDRLoss(hp)
    pit_pesq = PITPESQLoss(hp)
    model = load_trained_model(args.checkpoint,hp)
    diffusion = create_diffusion(timestep_respacing="",diffusion_steps=30,predict_v=True,predict_xstart=False)

    test_dataset = MonoStereoWhamrDataset(hp,train=False,mono_input=True,debug=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    sisdr_out = []
    sisdr_in = []

    pesq_out = []
    pesq_in = []
    for i, batch in enumerate(tqdm(test_loader)):
        mono_cond,x1,x2,_,_,S1,S2=batch
        cond_spec = complex_to_interleaved(mono_cond).to(device)
        generated_spec = infer_and_generate_audio(model, diffusion, cond_spec,hp)  #(b, c, t, f)
        output_audio = interleaved_to_complex(generated_spec).squeeze(0) 
        output_audio_time = test_dataset.iSTFT(output_audio) #(c, t)
        x1 = interleaved_to_complex(complex_to_interleaved(x1))
        x1_time = test_dataset.iSTFT(x1.to(device)).squeeze(0)  
        x2_time = test_dataset.iSTFT(x2.to(device)).squeeze(0)  
        mono_cond_time = test_dataset.iSTFT(mono_cond.to(device)).squeeze(0).to(device)


        s1 = test_dataset.iSTFT(S1.to(device)).squeeze(0)  
        s2 = test_dataset.iSTFT(S2.to(device)).squeeze(0)  
        s12=torch.stack((s1,s2)).unsqueeze(0).to(device)

        x_hat_left,x_hat_right = lcmv.time_doamin_bf(output_audio_time)
        x_hat_left=x_hat_left.to(device)
        x_hat_right=x_hat_right.to(device)
        sisdr_1,per = pit_sisdr(x_hat_left.unsqueeze(0),s12)
        sisdr_2,_ = pit_sisdr(x_hat_right.unsqueeze(0),s12)
        sisdr_in_i,_ = pit_sisdr(mono_cond_time.unsqueeze(0).unsqueeze(0),s12)

        sisdr_out.append(sisdr_1.detach().cpu().numpy().item())
        sisdr_out.append(sisdr_2.detach().cpu().numpy().item())
        sisdr_in.append(sisdr_in_i.detach().cpu().numpy().item())
        ####
        pesq_1 = pit_pesq(x_hat_left.unsqueeze(0),s12)
        pesq_2 = pit_pesq(x_hat_right.unsqueeze(0),s12)
        pesq_in_i = pit_pesq(mono_cond_time.unsqueeze(0).unsqueeze(0),s12)

        pesq_out.append(pesq_1.detach().cpu().numpy().item())
        pesq_out.append(pesq_2.detach().cpu().numpy().item())
        pesq_in.append(pesq_in_i.detach().cpu().numpy().item())

        s1_per = s1 if not per else s2
        s2_per = s2 if not per else s1
        save_audio(audio=output_audio_time, output_path=args.output_gen_dir, it=f'{i}_{epoch}_stereo', sr=hp.dataset.fs)
        save_audio(audio=x_hat_left, output_path=args.output_gen_dir_lcmv, it=f's1_{i}_{epoch}', sr=hp.dataset.fs)
        save_audio(audio=x_hat_right, output_path=args.output_gen_dir_lcmv, it=f's2_{i}_{epoch}', sr=hp.dataset.fs)
        save_audio(audio=s1_per, output_path=args.output_gt_mono_dir, it=f's1_{i}_{epoch}', sr=hp.dataset.fs)
        save_audio(audio=s2_per, output_path=args.output_gt_mono_dir, it=f's2_{i}_{epoch}', sr=hp.dataset.fs)

        if i >= (NUM_SAMPLES-1):
            break
    sisdr_out=-np.array(sisdr_out)
    sisdr_in = -np.array(sisdr_in)
    pesq_out=np.array(pesq_out)
    pesq_in = np.array(pesq_in)
    print(f'Output sisdr mean : {sisdr_out.mean()}, Best SI-SDR :{sisdr_out.max()}')
    print(f'Input sisdr mean : {sisdr_in.mean()}')
    print(f'Output pesq mean : {pesq_out.mean()}, Best PESQ :{pesq_out.max()}')
    print(f'Input pesq mean : {pesq_in.mean()}')

    mean_sisdr_out = np.mean(sisdr_out)
    mean_sisdr_in = np.mean(sisdr_in)
    mean_pesq_out = np.mean(pesq_out)
    mean_pesq_in = np.mean(pesq_in)

    # Create two side-by-side histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

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
    plt.savefig(Path(args.output_gen_dir).parent/'sisdr_pesq_histograms.png')



if __name__ == "__main__":
    main()

