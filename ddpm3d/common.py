"""Shared helpers: EMA, model/diffusion construction, checkpoint I/O."""
from __future__ import annotations

import copy
import os
from typing import Dict

import torch

from .data import PhaseCodec
from .diffusion import GaussianDiffusion, make_betas
from .unet import UNet3D


class EMA:
    """Exponential moving average of model weights (fp32)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().float()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        # warm-up so early averages are not dominated by the random init
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        ema_p = [p for p in self.model.parameters()]
        src_p = [p.detach().float() for p in model.parameters()]
        torch._foreach_mul_(ema_p, d)
        torch._foreach_add_(ema_p, src_p, alpha=1 - d)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "model": self.model.state_dict()}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.updates = sd["updates"]
        self.model.load_state_dict(sd["model"])


def build_model(cfg: Dict) -> UNet3D:
    return UNet3D(**cfg)


def build_diffusion(dcfg: Dict) -> GaussianDiffusion:
    return GaussianDiffusion(make_betas(dcfg["schedule"], dcfg["T"]),
                             var_type=dcfg.get("var_type", "fixedsmall"),
                             clip_x0=dcfg.get("clip_x0", True))


def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_for_sampling(path: str, device):
    """Load a snapshot (EMA weights + configs). Returns model, diffusion, codec, meta."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model_config"])
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    model.to(device).eval()
    diffusion = build_diffusion(ck["diffusion_config"])
    codec = PhaseCodec(len(ck["data"]["values"]), ck["data"]["encoding"])
    return model, diffusion, codec, ck


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=bool(enabled) and device.type == "cuda")
