import torch

def hrir2itd_fft(hrir, thrsh_ms=1000.0, fs=16000.0,nfft=512):
    """
    FFT-domain HRIR -> ITD estimator.

    Args:
        hrir: (S, 4, L) tensor S=Samples, 4: Rl,Rr,Il,Ir, L: length (nfft//2 +1)
        thrsh_ms: threshold [ms]. ITD is forced into [-thrsh_ms, +thrsh_ms].
    Returns:
        itd: (S) tensor, ITD in seconds
    """

    S, C, L = hrir.shape
    C2 = C//2
    Hr1, Hi1 = hrir[:,:C2], hrir[:,C2:]
    hrir = torch.complex(Hr1, Hi1)
    # Separate channels: (S, B, L)
    hrir_l = hrir[:, 0, :]
    hrir_r = hrir[:, 1, :]
    crs_cor = torch.fft.irfft(hrir_l*(hrir_r.conj()),n=nfft).real
    crs_cor = torch.fft.fftshift(crs_cor,dim=-1)

    thrsh_idx = round(fs / thrsh_ms)
    center = nfft // 2

    idx_beg = center - thrsh_idx
    idx_end = center + thrsh_idx + 1

    window = crs_cor[ :, idx_beg:idx_end]
    idx_max = torch.argmax(window, axis=-1) - thrsh_idx
    itd = idx_max / fs
    return itd



if __name__=="__main__":
    from data import load_db

    # d = load_db('/home/workspace/yoavellinson/binaural_TSE_Gen/pts_512/test_set/ari_atl_and_full/hrtf_las_nh919.pt')
    # hrir = d['patches'] 
    # pos =  d['pos'] 
    S = 440
    C = 2
    L = 1500

    hrir = torch.zeros(S, C, L)
    hrir[:, 0, 10] = 1.0   # first channel → 1 at index 10
    hrir[:, 1, 10+16] = 1.0  

    HRIR = torch.fft.rfft(hrir,n=512,dim=-1)
    HRIR = torch.concat((HRIR.real,HRIR.imag),dim=1).real
    itd = hrir2itd_fft(HRIR)  
    print(itd)