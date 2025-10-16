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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from train_RI_dit import EMA

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
    import argparse
    parser = argparse.ArgumentParser(description='Generate audio using DiT')
    parser.add_argument('--checkpoint', type=str,default=f'/home/workspace/yoavellinson/mono-to-stereo/checkpoints_RI_v/model_epoch_{epoch}.pt', help='Path to model checkpoint')
    parser.add_argument('--hp', type=str, default='/home/workspace/yoavellinson/mono-to-stereo/conf/train.yaml', help='Path to conf')
    parser.add_argument('--output_gt_dir', type=str, default='/home/workspace/yoavellinson/mono-to-stereo/diff_outputs/gt_dir', help='Directory to save ground truth audio')
    parser.add_argument('--output_gt_mono_dir', type=str,default='/home/workspace/yoavellinson/mono-to-stereo/diff_outputs/gt_mono_dir', help='Directory to save ground truth mono audio')
    parser.add_argument('--output_gen_dir', type=str, default='/home/workspace/yoavellinson/mono-to-stereo/diff_outputs/gen_dir', help='Directory to save generated audio')
    args = parser.parse_args()
    hp = OmegaConf.load(args.hp)
    model = load_trained_model(args.checkpoint,hp)
    diffusion = create_diffusion(timestep_respacing="",diffusion_steps=30,predict_v=True,predict_xstart=False)

    test_dataset = MonoStereoWhamrDataset(hp,train=False,mono_input=True,debug=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for i, batch in enumerate(tqdm(test_loader)):
        mono_cond,x1,x2,_,_,_,_=batch
        cond_spec = complex_to_interleaved(mono_cond).to(device)
        generated_spec = infer_and_generate_audio(model, diffusion, cond_spec,hp)  # (b, c, t, f)
        output_audio = interleaved_to_complex(generated_spec).squeeze(0) 
        output_audio_time = test_dataset.iSTFT(output_audio) # (c, t)
        x1 = interleaved_to_complex(complex_to_interleaved(x1))
        x1_time = test_dataset.iSTFT(x1.to(device)).squeeze(0)  
        x2_time = test_dataset.iSTFT(x2.to(device)).squeeze(0)  
        mono_cond_time = test_dataset.iSTFT(mono_cond.to(device)).squeeze(0)

        save_audio(audio=output_audio_time, output_path=args.output_gen_dir, it=f'{i}_{epoch}', sr=hp.dataset.fs)
        save_audio(audio=x1_time.squeeze(0), output_path=args.output_gt_dir, it=f'{i}_1', sr=hp.dataset.fs)
        save_audio(audio=x2_time.squeeze(0), output_path=args.output_gt_dir, it=f'{i}_2', sr=hp.dataset.fs)
        save_audio(audio=mono_cond_time.squeeze(0), output_path=args.output_gt_mono_dir, it=i, sr=hp.dataset.fs)
        
if __name__ == "__main__":
    main()

