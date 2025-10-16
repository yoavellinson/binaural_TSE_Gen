import torch
from torch import nn

class ConditionalFlowMatching:
    """
    Conditional Flow Matching (Rectified Flow) utility.

    Path: x_t = (1 - gamma(t)) * x0 + gamma(t) * z,   z ~ N(0, I),  t in [0,1]
    Velocity target: u*(x_t, t) = gamma'(t) * (z - x0)

    Model contract:
      u_pred = model(x_t, t, **model_kwargs)  # predicts velocity field (same shape as x_t)
    """

    def __init__(
        self,
        rho: float = 1.5,            # time warp gamma(t) = t^rho
        t_eps: float = 1e-3,         # avoid singularities exactly at {0,1}
        guidance_fade_p: float = 1.5,# CFG fade power: s_t = 1 + (s-1)*t^p
        null_cond_fn=None,           # optional: fn(cond) -> uncond (for CFG + CF training)
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.rho = float(rho)
        self.t_eps = float(t_eps)
        self.guidance_fade_p = float(guidance_fade_p)
        self.null_cond_fn = null_cond_fn
        self.device = device
        self.dtype = dtype

    # ---------- time utilities ----------
    def _gamma(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0,1], gamma(t) = t^rho
        return torch.clamp(t, self.t_eps, 1.0).pow(self.rho)

    def _dgamma(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, self.t_eps, 1.0)
        return self.rho * t.pow(self.rho - 1.0)

    def _expand_t(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # expand scalar/batchwise t -> [B,1,1,...] to broadcast with 'like'
        while t.ndim < like.ndim:
            t = t.view(*t.shape, 1)
        return t

    # ---------- training: loss ----------
    @torch.no_grad()
    def _make_xt_and_target(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Build x_t and u_star for given x0 and t.
        Returns: x_t, u_star, gamma, dgamma
        """
        B = x0.shape[0]
        t = t.to(device=x0.device, dtype=x0.dtype).view(B, 1, 1, *([1] * (x0.ndim - 3)))
        g = self._gamma(t)
        dg = self._dgamma(t)

        z = torch.randn_like(x0)
        x_t = (1.0 - g) * x0 + g * z
        u_star = dg * (z - x0)
        return x_t, u_star, g, dg

    def loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,                 # [B, C, T...] normalized like training audio
        model_kwargs: dict | None = None, # conditioning dict passed to model
        cond_drop_prob: float = 0.0,      # classifier-free training drop prob (if null_cond_fn provided)
        weight_endpoints: float = 0.3,    # small emphasis near t≈0 or 1
    ):
        """
        Compute CFM loss for a batch.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x0.shape[0]
        device = x0.device
        t = torch.rand(B, device=device, dtype=x0.dtype)
        # keep t away from exact endpoints
        t = t * (1.0 - 2 * self.t_eps) + self.t_eps

        # classifier-free drop (optional)
        if self.null_cond_fn is not None and cond_drop_prob > 0:
            if torch.rand(()) < cond_drop_prob:
                model_kwargs = self.null_cond_fn(model_kwargs)

        x_t, u_star, g, dg = self._make_xt_and_target(x0, t)

        # predict velocity
        u_pred = model(x_t, self._expand_t(t, x_t), **model_kwargs)
        if isinstance(u_pred, tuple):  # allow (pred, extra)
            u_pred, _ = u_pred

        # mild endpoint weighting
        w_t = ((t * (1 - t)) + 1e-3).pow(-weight_endpoints)
        w_t = self._expand_t(w_t, x_t)

        l2 = (w_t * (u_pred - u_star) ** 2).mean()

        return l2, {
            "l2": l2.detach(),
            "t_mean": t.mean().detach(),
        }

    # ---------- CFG for velocity ----------
    def _cfg_velocity(
        self,
        model, x: torch.Tensor, t: torch.Tensor,
        cond: dict | None, uncond: dict | None, guidance_scale: float | None
    ):
        if guidance_scale is None or uncond is None:
            u = model(x, t, **(cond or {}))
            if isinstance(u, tuple): u = u[0]
            return u

        # fade guidance as t->0 to avoid HF hiss amplification
        # t in [0,1], early steps t~1 ==> s_t≈s; late t~0 ==> s_t≈1
        s = float(guidance_scale)
        s_t = 1.0 + (s - 1.0) * torch.clamp(t, 0.0, 1.0).pow(self.guidance_fade_p)

        u_u = model(x, t, **uncond)
        u_c = model(x, t, **(cond or {}))
        if isinstance(u_u, tuple): u_u = u_u[0]
        if isinstance(u_c, tuple): u_c = u_c[0]
        return u_u + self._expand_t(s_t, x) * (u_c - u_u)

    # ---------- sampling (ODE; Heun predictor-corrector) ----------
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cond: dict | None,
        steps: int = 40,
        shape: tuple | None = None,     # required if x_init is None
        x_init: torch.Tensor | None = None,
        t0: float | None = None,        # start time in [0,1]; default 1.0 (pure noise) or based on img2img init
        guidance_scale: float | None = None,
        uncond: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Integrate the flow ODE from t0 -> 0 with Heun's method.
        - If x_init is None: start from z ~ N(0, I) at t0=1.
        - If x_init is provided: you must also pass t0 in (0,1]; see img2img_init().
        """
        device = device or self.device or (x_init.device if x_init is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = dtype or self.dtype

        if x_init is None:
            assert shape is not None, "Provide `shape` when x_init is None."
            x = torch.randn(*shape, device=device, dtype=dtype)
            t0 = 1.0 if t0 is None else float(t0)
        else:
            x = x_init.to(device=device, dtype=dtype)
            assert t0 is not None, "Provide t0 for a custom x_init (e.g., from img2img_init)."
            t0 = float(t0)

        # time grid
        ts = torch.linspace(t0, 0.0, steps + 1, device=device, dtype=dtype)

        B = x.shape[0]
        for k in range(steps, 0, -1):
            t_k   = ts[k].expand(B)          # scalar -> [B]
            t_km1 = ts[k - 1].expand(B)
            dt = (t_km1 - t_k)               # positive

            t_k_exp   = self._expand_t(t_k, x)
            t_km1_exp = self._expand_t(t_km1, x)

            # predictor (Euler)
            u1 = self._cfg_velocity(model, x, t_k_exp, cond, uncond, guidance_scale)
            x_pred = x - dt.view(-1, *([1]*(x.ndim-1))) * u1

            # corrector (Heun)
            u2 = self._cfg_velocity(model, x_pred, t_km1_exp, cond, uncond, guidance_scale)
            x = x - dt.view(-1, *([1]*(x.ndim-1))) * (u1 + u2) * 0.5

        return x  # x at t=0

    # ---------- img2img-style init (sample “around” a condition) ----------
    @torch.no_grad()
    def img2img_init(
        self,
        x_cond: torch.Tensor,    # reference audio, same preprocessing as x0
        strength: float,         # in [0,1]; 0→near x_cond, 1→pure noise
    ):
        """
        Build x_init at t0 corresponding to the given strength.
        """
        x_cond = x_cond.to(device=self.device or x_cond.device, dtype=self.dtype or x_cond.dtype)
        B = x_cond.shape[0]
        t0 = float(torch.clamp(torch.tensor(strength), 0.0, 1.0).item())
        t = torch.full((B,), t0, device=x_cond.device, dtype=x_cond.dtype)

        g = self._gamma(t).view(B, *([1]*(x_cond.ndim-1)))
        z = torch.randn_like(x_cond)
        x_init = (1.0 - g) * x_cond + g * z
        return x_init, t0
