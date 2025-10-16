import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hrtf_convolve import SOFA_HRTF_db
import numpy as np
# ---------- Angle encoding ----------
class SinCosAngleEnc(nn.Module):
    def __init__(self, L=4):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(L)) * math.pi)
    @property
    def out_dim(self): return 2 * 2 * self.freqs.numel()  # (az,el) × (sin,cos) × L
    def forward(self, azel):  # azel: [N, 2] radians
        a = azel.unsqueeze(-1) * self.freqs  # [N,2,L]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1).flatten(1)  # [N,4L]

# ---------- Simple IR featureizer (1D CNN over waveform) ----------
class IRFeatureizer(nn.Module):
    def __init__(self, in_ch=1, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 9, 2, 4), nn.GELU(),
            nn.Conv1d(32, 64, 9, 2, 4),    nn.GELU(),
            nn.Conv1d(64, 128, 9, 2, 4),   nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.out = nn.Linear(128, feat_dim)
    @property
    def out_dim(self): return self.out.out_features
    def forward(self, x):                 # x: [N,C,T]
        h = self.net(x).squeeze(-1)      # [N,128]
        return self.out(h)               # [N,F]

# ---------- Set Transformer blocks (SAB + PMA) ----------
class MAB(nn.Module):
    def __init__(self, dim_q, dim_kv, dim, heads=4, dropout=0.0, return_weights=False):
        super().__init__()
        self.q = nn.Linear(dim_q, dim)
        self.k = nn.Linear(dim_kv, dim)
        self.v = nn.Linear(dim_kv, dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff  = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim), nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(dim)
        self.return_weights = return_weights
    def forward(self, Q, K, key_padding_mask=None):
        Qh, Kh, Vh = self.q(Q), self.k(K), self.v(K)
        H, w = self.attn(Qh, Kh, Vh, key_padding_mask=key_padding_mask, need_weights=self.return_weights)
        H = self.ln1(Qh + H)
        O = self.ln2(H + self.ff(H))
        return (O, w) if self.return_weights else O

class SAB(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.mab = MAB(dim, dim, dim, heads, dropout)
    def forward(self, X, key_padding_mask=None):
        return self.mab(X, X, key_padding_mask=key_padding_mask)

class PMA(nn.Module):
    def __init__(self, dim, k=1, heads=4, dropout=0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, dim))
        self.mab = MAB(dim, dim, dim, heads, dropout)
    def forward(self, X, key_padding_mask=None):
        B = X.size(0); S = self.S.expand(B, -1, -1)
        return self.mab(S, X, key_padding_mask=key_padding_mask)  # [B,k,D]

# ---------- Focus pooling: seed comes from the highlighted token ----------
class FocusPMA(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0, return_weights=False):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.mab = MAB(dim, dim, dim, heads, dropout, return_weights=return_weights)
    def forward(self, seed, X, key_padding_mask=None):  # seed: [B,D], X: [B,N,D]
        S = self.q_proj(seed).unsqueeze(1)              # [B,1,D]
        out = self.mab(S, X, key_padding_mask=key_padding_mask)
        if isinstance(out, tuple):  # returning weights
            O, w = out
            return O.squeeze(1), w  # O: [B,1,D] -> [B,D]; w: [B,1,N]
        return out.squeeze(1), None

# ---------- The operation: all IRs -> z_db and highlighted z_focus ----------
class DatasetEmbedOp(nn.Module):
    """
    One op that:
      - builds tokens for ALL IRs of a database,
      - returns z_db (global embedding) and z_focus (embedding conditioned on a highlighted IR).
    Also supports batching multiple databases (variable N) in one call.
    """
    def __init__(self, in_ch=1, ir_feat_dim=128, d_model=256, depth=4, heads=4, dropout=0.1, return_focus_attn=False):
        super().__init__()
        self.ir_enc  = IRFeatureizer(in_ch=in_ch, feat_dim=ir_feat_dim)
        self.ang_enc = SinCosAngleEnc(L=4)
        token_dim = ir_feat_dim + self.ang_enc.out_dim + 1  # +1 for "is_missing"
        self.input_proj = nn.Linear(token_dim, d_model)
        self.sabs = nn.ModuleList([SAB(d_model, heads, dropout) for _ in range(depth)])
        self.pma_global = PMA(d_model, k=1, heads=heads, dropout=dropout)
        self.pma_focus  = FocusPMA(d_model, heads=heads, dropout=dropout, return_weights=return_focus_attn)
        self.norm = nn.LayerNorm(d_model)
        self.return_focus_attn = return_focus_attn

    # ---- Public API: SINGLE database ----
    def forward_single(self, irs, azel, valid_mask, highlight_idx=None):
        """
        irs:         [N, C, T]  float
        azel:        [N, 2]     radians
        valid_mask:  [N]        bool  (True = valid IR, False = missing)
        highlight_idx: int or None
        Returns:
          z_db:    [D]
          z_focus: [D]
          focus_attn (optional): [N] attention weights over IRs for the highlight
        """
        tokens = self._make_tokens(irs, azel, valid_mask)       # [N, token_dim]
        z_db, z_focus, focus_w = self._encode_and_pool([tokens], [highlight_idx])
        z_db, z_focus = z_db[0], z_focus[0]
        if self.return_focus_attn:
            # focus_w: [B,1,Nmax], but only first sample matters; slice to N valid
            N = tokens.size(0)
            return z_db, z_focus, focus_w[0, 0, :N]
        return z_db, z_focus

    # ---- Public API: BATCH of databases (list inputs with varying N) ----
    def forward_batch(self, batch_irs, batch_azel, batch_valid, batch_highlight_idx):
        """
        batch_*: lists of tensors per DB
        Returns:
          z_db:    [B, D]
          z_focus: [B, D]
          focus_attn (optional): [B, Nmax] (padded; mask is returned too)
        """
        tokens_list = [self._make_tokens(irs, az, vm) for irs, az, vm in zip(batch_irs, batch_azel, batch_valid)]
        z_db, z_focus, focus_w = self._encode_and_pool(tokens_list, batch_highlight_idx)
        if self.return_focus_attn:
            padmask = self._make_padmask(tokens_list)
            return z_db, z_focus, focus_w, padmask  # (so you can ignore padded positions)
        return z_db, z_focus

    # ---- internals ----
    def _make_tokens(self, irs, azel, valid_mask):
        # Featureize IRs + encode angles + append is_missing flag (1 if missing)
        ir_feat  = self.ir_enc(irs)                        # [N,F]
        ang_feat = self.ang_enc(azel)                      # [N,A]
        miss     = (~valid_mask).float().unsqueeze(-1)     # [N,1]
        return torch.cat([ir_feat, ang_feat, miss], dim=-1)  # [N, F+A+1]

    def _make_padmask(self, tokens_list):
        B = len(tokens_list); Nmax = max(t.size(0) for t in tokens_list)
        padmask = torch.ones(B, Nmax, dtype=torch.bool, device=tokens_list[0].device)
        for b, t in enumerate(tokens_list):
            padmask[b, :t.size(0)] = False
        return padmask  # True = PAD

    def _encode_and_pool(self, tokens_list, highlight_indices):
        # pad
        B = len(tokens_list); D_in = tokens_list[0].size(-1)
        Nmax = max(t.size(0) for t in tokens_list)
        device = tokens_list[0].device
        X = torch.zeros(B, Nmax, D_in, device=device)
        padmask = self._make_padmask(tokens_list)
        for b, t in enumerate(tokens_list):
            n = t.size(0)
            X[b, :n] = t

        # Input projection + SAB stack
        H = self.input_proj(X)                  # [B,N,D]
        for sab in self.sabs:
            H = sab(H, key_padding_mask=padmask)

        # Global pooling
        z_db = self.pma_global(H, key_padding_mask=padmask).squeeze(1)   # [B,D]
        z_db = self.norm(z_db)

        # Focus seeds
        seeds = []
        for b in range(B):
            hi = highlight_indices[b]
            if hi is None:
                seeds.append(z_db[b])           # fallback: use global as seed
            else:
                seeds.append(H[b, hi])          # highlighted token (post-SAB)
        seeds = torch.stack(seeds, 0)           # [B,D]

        # Focus pooling (with optional attention weights)
        z_focus, w = self.pma_focus(seeds, H, key_padding_mask=padmask)  # z_focus: [B,D], w: [B,1,N]
        z_focus = self.norm(z_focus)
        return (z_db, z_focus, w if self.return_focus_attn else None)


if __name__=="__main__":
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    db = SOFA_HRTF_db('/home/workspace/yoavellinson/binaural_TSE_Gen/sofas/riec_full/RIEC_hrir_subject_001.sofa')
    azimuths_deg = np.arange(0, 356, 1)   
    elevations_deg = np.arange(-30, 90,5)
    az_grid, el_grid = np.meshgrid(azimuths_deg, elevations_deg, indexing="xy")
    desired_azel_deg = np.stack([az_grid.ravel(), el_grid.ravel()], axis=1)  # [N, 2]

    irs, azel, valid_mask = db.build_db_tensors_deg(desired_azel_deg=desired_azel_deg,device=device)
    op = DatasetEmbedOp(in_ch=2, d_model=256, return_focus_attn=True).to(device)
    z_db, z_focus, attn = op.forward_single(irs, azel, valid_mask, highlight_idx=17)