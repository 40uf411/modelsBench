"""
Gaussian diffusion: training loss, ancestral DDPM sampling and DDIM sampling.

Both samplers use the same beta schedule as training (stored in every
checkpoint), run their timesteps from noisy to clean, and clip the
predicted x0 to [-1, 1] at every step.

  DDPM (Ho et al., 2020): x_{t-1} ~ q(x_{t-1} | x_t, x0_hat), all T steps.
  DDIM (Song et al., 2021): deterministic for eta = 0; any number of steps
  taken from the training timesteps. With steps = T and eta = 1 it reduces
  to DDPM with the posterior ("fixedsmall") variance; test_ddpm3d.py checks this.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch


def make_betas(schedule: str, T: int) -> np.ndarray:
    if schedule == "linear":
        scale = 1000.0 / T  # keeps the Ho et al. endpoints meaningful for T != 1000
        betas = np.linspace(scale * 1e-4, scale * 0.02, T, dtype=np.float64)
        return np.minimum(betas, 0.999)  # small T would otherwise reach beta = 1 (alpha_bar = 0)
    if schedule == "cosine":  # Nichol & Dhariwal (2021)
        s = 0.008
        f = lambda t: math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        return np.array([min(1 - f(i + 1) / f(i), 0.999) for i in range(T)], dtype=np.float64)
    raise ValueError(f"unknown schedule {schedule}")


def _gather(a: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    return a.to(t.device)[t].float().view(-1, *([1] * (ndim - 1)))


class GaussianDiffusion:
    def __init__(self, betas: np.ndarray, var_type: str = "fixedsmall", clip_x0: bool = True):
        if var_type not in ("fixedsmall", "fixedlarge"):
            raise ValueError(var_type)
        b = torch.as_tensor(betas, dtype=torch.float64)
        self.T = len(b)
        self.var_type = var_type
        self.clip_x0 = clip_x0
        a = 1.0 - b
        ab = torch.cumprod(a, 0)
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
        self.betas = b
        self.alphas_cumprod = ab
        self.alphas_cumprod_prev = ab_prev
        self.sqrt_ab = ab.sqrt()
        self.sqrt_1m_ab = (1 - ab).sqrt()
        self.sqrt_recip_ab = (1 / ab).sqrt()
        self.sqrt_recipm1_ab = (1 / ab - 1).sqrt()
        self.post_var = b * (1 - ab_prev) / (1 - ab)
        self.post_coef_x0 = b * ab_prev.sqrt() / (1 - ab)
        self.post_coef_xt = (1 - ab_prev) * a.sqrt() / (1 - ab)
        # "fixedlarge" (Ho et al.): sigma_t^2 = beta_t, with the posterior value at t = 0
        self.large_var = torch.cat([self.post_var[1:2], b[1:]])

    # ---- training --------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        n = x0.ndim
        return _gather(self.sqrt_ab, t, n) * x0 + _gather(self.sqrt_1m_ab, t, n) * noise

    def training_loss(self, model: Callable, x0: torch.Tensor,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """L_simple: mean squared error of the predicted noise (mean over all elements)."""
        b = x0.shape[0]
        t = torch.randint(self.T, (b,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        eps = model(self.q_sample(x0, t, noise), t)
        return torch.mean((eps.float() - noise) ** 2)

    # ---- shared helpers ---------------------------------------------------------
    def _x0_from_eps(self, x_t, t, eps):
        n = x_t.ndim
        x0 = _gather(self.sqrt_recip_ab, t, n) * x_t - _gather(self.sqrt_recipm1_ab, t, n) * eps
        return x0.clamp(-1.0, 1.0) if self.clip_x0 else x0

    def _eps_from_x0(self, x_t, t, x0):
        n = x_t.ndim
        return (_gather(self.sqrt_recip_ab, t, n) * x_t - x0) / _gather(self.sqrt_recipm1_ab, t, n)

    # ---- samplers ---------------------------------------------------------------
    @torch.no_grad()
    def ddpm_sample(self, model: Callable, shape, device, generator: Optional[torch.Generator] = None,
                    progress: bool = False) -> torch.Tensor:
        x = torch.randn(shape, device=device, generator=generator)
        var = self.post_var if self.var_type == "fixedsmall" else self.large_var
        steps = range(self.T - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc="DDPM", leave=False)
        for i in steps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t).float()
            x0 = self._x0_from_eps(x, t, eps)
            n = x.ndim
            mean = _gather(self.post_coef_x0, t, n) * x0 + _gather(self.post_coef_xt, t, n) * x
            if i > 0:
                noise = torch.randn(shape, device=device, generator=generator)
                x = mean + _gather(var, t, n).sqrt() * noise
            else:
                x = mean
        return x

    def ddim_timesteps(self, steps: int) -> np.ndarray:
        """Evenly spaced training timesteps, ascending, always including T-1."""
        if not 1 <= steps <= self.T:
            raise ValueError(f"steps must be in [1, {self.T}]")
        ts = np.round(np.linspace(0, self.T - 1, steps)).astype(np.int64)
        return np.unique(ts)

    @torch.no_grad()
    def ddim_sample(self, model: Callable, shape, device, steps: int = 50, eta: float = 0.0,
                    generator: Optional[torch.Generator] = None, progress: bool = False) -> torch.Tensor:
        ts = self.ddim_timesteps(steps)
        x = torch.randn(shape, device=device, generator=generator)
        seq = list(range(len(ts) - 1, -1, -1))  # noisy -> clean
        if progress:
            from tqdm import tqdm
            seq = tqdm(seq, desc=f"DDIM{steps}", leave=False)
        n = x.ndim
        for j in seq:
            i = int(ts[j])
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            ab = float(self.alphas_cumprod[i])
            ab_prev = float(self.alphas_cumprod[ts[j - 1]]) if j > 0 else 1.0
            eps = model(x, t).float()
            x0 = self._x0_from_eps(x, t, eps)
            eps = self._eps_from_x0(x, t, x0)  # consistent with the clipped x0
            sigma = eta * math.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev))
            x = math.sqrt(ab_prev) * x0 + math.sqrt(max(1 - ab_prev - sigma ** 2, 0.0)) * eps
            if sigma > 0:
                x = x + sigma * torch.randn(shape, device=device, generator=generator)
        return x
