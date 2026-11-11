"""
Discrete-time Gaussian diffusion: schedule, training loss, and three samplers.

Choices (defaults) and why:
  * Noise schedule: shifted cosine (Hoogeboom et al., 2023, "simple diffusion"),
    logSNR(t) = -2 log tan(pi t / 2) + log(snr_scale). A 64^3 volume has as many
    elements as a 512^2 image; averaging over many voxels removes noise, so the
    standard schedule leaves coarse structure almost noise-free for most of the
    chain. snr_scale = 1/8 lowers the SNR so that the 8^3 bottleneck sees the
    same SNR as the 8^2 bottleneck of a 64^2 image (3D pooling by 8 divides
    noise variance by 512 instead of 64).
  * Zero terminal SNR (Lin et al., 2024): alpha_bar(T) = 0, so training sees
    pure noise at the last step, as sampling does.
  * v-prediction (Salimans & Ho, 2022): v = sqrt(ab) eps - sqrt(1-ab) x0.
    Well-defined at zero SNR (eps-prediction is not) and stable at high noise.
  * Self-conditioning (Chen, Zhang & Hinton, 2023, "Analog Bits"): with
    probability 0.5 the network also receives its own x0 estimate. Designed for
    discrete data diffused as +-1 values, which is what segmented volumes are.
  * x0 is clipped to [-1, 1] at every sampling step.

Samplers (all use the training schedule, run noisy -> clean):
  ddpm  : ancestral, all T steps, posterior ("fixedsmall") or beta variance
  ddim  : Song et al. (2021), any number of steps, eta = 0 deterministic
  dpmpp : DPM-Solver++(2M) (Lu et al., 2022), second-order multistep in the
          data-prediction form; first-order first and last steps
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------------
# schedules
# ----------------------------------------------------------------------------
def make_alphas_cumprod(schedule: str, T: int, snr_scale: float = 1.0,
                        zero_terminal_snr: bool = True) -> np.ndarray:
    """alpha_bar for t = 0..T-1 (index t = 0 is the least noisy step)."""
    if schedule == "cosine":
        t = np.arange(1, T + 1, dtype=np.float64) / T
        with np.errstate(divide="ignore"):
            logsnr = -2.0 * np.log(np.tan(np.pi * t / 2)) + np.log(snr_scale)
        ab = 1.0 / (1.0 + np.exp(-logsnr))
    elif schedule == "linear":
        scale = 1000.0 / T
        betas = np.minimum(np.linspace(scale * 1e-4, scale * 0.02, T, dtype=np.float64), 0.999)
        ab = np.cumprod(1.0 - betas)
        if snr_scale != 1.0:
            logsnr = np.log(ab) - np.log1p(-ab) + np.log(snr_scale)
            ab = 1.0 / (1.0 + np.exp(-logsnr))
    else:
        raise ValueError(f"unknown schedule {schedule}")
    if zero_terminal_snr:
        # Lin et al. (2024): rescale sqrt(ab) so the last step is pure noise
        s = np.sqrt(ab)
        s0, sT = s[0], s[-1]
        s = (s - sT) * s0 / (s0 - sT)
        ab = s ** 2
    return ab


# kept for backwards compatibility with the first version of this module
def make_betas(schedule: str, T: int) -> np.ndarray:
    ab = make_alphas_cumprod(schedule, T, 1.0, zero_terminal_snr=False)
    return 1.0 - ab / np.concatenate([[1.0], ab[:-1]])


def _col(a: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    return a.to(t.device)[t].float().view(-1, *([1] * (ndim - 1)))


class GaussianDiffusion:
    def __init__(self, alphas_cumprod: np.ndarray, prediction: str = "v",
                 var_type: str = "fixedsmall", clip_x0: bool = True, self_condition: bool = False):
        if prediction not in ("v", "eps"):
            raise ValueError(prediction)
        if var_type not in ("fixedsmall", "fixedlarge"):
            raise ValueError(var_type)
        ab = torch.as_tensor(alphas_cumprod, dtype=torch.float64)
        if prediction == "eps" and ab[-1] <= 0:
            raise ValueError("eps-prediction is undefined at zero terminal SNR; use prediction='v'")
        self.T = len(ab)
        self.prediction = prediction
        self.var_type = var_type
        self.clip_x0 = clip_x0
        self.self_condition = self_condition
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
        b = 1.0 - ab / ab_prev
        self.alphas_cumprod = ab
        self.betas = b
        self.sqrt_ab = ab.sqrt()
        self.sqrt_1m_ab = (1 - ab).sqrt()
        self.post_var = b * (1 - ab_prev) / (1 - ab)
        self.post_coef_x0 = b * ab_prev.sqrt() / (1 - ab)
        self.post_coef_xt = (1 - ab_prev) * (1 - b).sqrt() / (1 - ab)
        self.large_var = torch.cat([self.post_var[1:2], b[1:]])

    # ---- conversions --------------------------------------------------------------
    def q_sample(self, x0, t, noise):
        n = x0.ndim
        return _col(self.sqrt_ab, t, n) * x0 + _col(self.sqrt_1m_ab, t, n) * noise

    def target(self, x0, t, noise):
        if self.prediction == "eps":
            return noise
        n = x0.ndim
        return _col(self.sqrt_ab, t, n) * noise - _col(self.sqrt_1m_ab, t, n) * x0

    def to_x0_eps(self, out, x_t, t):
        """Network output -> (x0, eps); x0 clipped, eps made consistent with it."""
        n = x_t.ndim
        a, s = _col(self.sqrt_ab, t, n), _col(self.sqrt_1m_ab, t, n)
        out = out.float()
        if self.prediction == "v":
            x0 = a * x_t - s * out
        else:
            x0 = (x_t - s * out) / a
        if self.clip_x0:
            x0 = x0.clamp(-1.0, 1.0)
        eps = (x_t - a * x0) / s
        return x0, eps

    def _denoise(self, model, x_t, t, x_sc):
        out = model(x_t, t, x_sc) if self.self_condition else model(x_t, t)
        return self.to_x0_eps(out, x_t, t)

    # ---- training -------------------------------------------------------------------
    def training_loss(self, model: Callable, x0: torch.Tensor,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Mean squared error on the prediction target (v by default)."""
        b = x0.shape[0]
        t = torch.randint(self.T, (b,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        x_t = self.q_sample(x0, t, noise)
        if self.self_condition:
            x_sc = torch.zeros_like(x0)
            if torch.rand((), device=x0.device, generator=generator).item() < 0.5:
                with torch.no_grad():
                    x_sc, _ = self.to_x0_eps(model(x_t, t, x_sc), x_t, t)
            out = model(x_t, t, x_sc.detach())
        else:
            out = model(x_t, t)
        return torch.mean((out.float() - self.target(x0, t, noise)) ** 2)

    # ---- samplers ---------------------------------------------------------------------
    def sample_timesteps(self, steps: int) -> np.ndarray:
        """Evenly spaced training timesteps, ascending, always including 0 and T-1."""
        if not 1 <= steps <= self.T:
            raise ValueError(f"steps must be in [1, {self.T}]")
        return np.unique(np.round(np.linspace(0, self.T - 1, steps)).astype(np.int64))

    ddim_timesteps = sample_timesteps  # name used by sample.py and the tests

    @staticmethod
    def _progress(seq, desc, progress):
        if not progress:
            return seq
        from tqdm import tqdm
        return tqdm(seq, desc=desc, leave=False)

    @torch.no_grad()
    def ddpm_sample(self, model, shape, device, generator=None, progress=False):
        x = torch.randn(shape, device=device, generator=generator)
        var = self.post_var if self.var_type == "fixedsmall" else self.large_var
        x_sc = torch.zeros(shape, device=device)
        n = x.ndim
        for i in self._progress(range(self.T - 1, -1, -1), "DDPM", progress):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x0, _ = self._denoise(model, x, t, x_sc)
            x_sc = x0
            mean = _col(self.post_coef_x0, t, n) * x0 + _col(self.post_coef_xt, t, n) * x
            if i > 0:
                x = mean + _col(var, t, n).sqrt() * torch.randn(shape, device=device, generator=generator)
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, steps=50, eta=0.0, generator=None, progress=False):
        ts = self.sample_timesteps(steps)
        x = torch.randn(shape, device=device, generator=generator)
        x_sc = torch.zeros(shape, device=device)
        for j in self._progress(range(len(ts) - 1, -1, -1), f"DDIM{steps}", progress):
            i = int(ts[j])
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            ab = float(self.alphas_cumprod[i])
            ab_prev = float(self.alphas_cumprod[ts[j - 1]]) if j > 0 else 1.0
            x0, eps = self._denoise(model, x, t, x_sc)
            x_sc = x0
            sigma = eta * math.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)) if ab > 0 else eta * math.sqrt(1 - ab_prev)
            x = math.sqrt(ab_prev) * x0 + math.sqrt(max(1 - ab_prev - sigma ** 2, 0.0)) * eps
            if sigma > 0:
                x = x + sigma * torch.randn(shape, device=device, generator=generator)
        return x

    @torch.no_grad()
    def dpmpp_sample(self, model, shape, device, steps=20, generator=None, progress=False):
        """DPM-Solver++(2M), data prediction, first-order first and last steps."""
        ts = self.sample_timesteps(steps)[::-1]  # noisy -> clean
        a = lambda i: float(self.sqrt_ab[i]) if i >= 0 else 1.0
        s = lambda i: float(self.sqrt_1m_ab[i]) if i >= 0 else 0.0
        lam = lambda i: math.log(a(i)) - math.log(s(i)) if 0 < a(i) and s(i) > 0 else None
        x = torch.randn(shape, device=device, generator=generator)
        x_sc = torch.zeros(shape, device=device)
        x0_prev, lam_prev = None, None
        seq = list(ts) + [-1]  # -1 = clean data (alpha = 1, sigma = 0)
        for k in self._progress(range(len(ts)), f"DPM++{steps}", progress):
            i, nxt = int(seq[k]), int(seq[k + 1])
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x0, _ = self._denoise(model, x, t, x_sc)
            x_sc = x0
            lam_i, lam_n = lam(i), lam(nxt)
            d = x0
            if x0_prev is not None and lam_prev is not None and lam_i is not None and lam_n is not None:
                r = (lam_i - lam_prev) / (lam_n - lam_i)
                d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * x0_prev
            # x_next = (s_n / s_i) x + (a_n - a_i s_n / s_i) D   (exact first-order step)
            x = (s(nxt) / s(i)) * x + (a(nxt) - a(i) * s(nxt) / s(i)) * d
            x0_prev, lam_prev = x0, lam_i
        return x
