import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', cosine_beta_schedule(T))
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        DDPM_LOSS =  F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return DDPM_LOSS


class ImprovedDDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, training_T, sampling_T, eta=0.0):
        super().__init__()
        self.model = model
        self.training_T = training_T
        self.sampling_T = sampling_T
        self.eta = eta
        
        # Use cosine beta schedule
        self.register_buffer('betas', cosine_beta_schedule(training_T))
        
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # Timestep mapping
        self.register_buffer('timestep_map', self.get_sampling_timesteps(training_T, sampling_T))

    def get_sampling_timesteps(self, training_T, sampling_T):
        return torch.linspace(0, training_T - 1, sampling_T).long()

    def ddim_sample(self, x_t, t, prev_t):
        at = extract(self.alphas_cumprod, t, x_t.shape)
        at_next = extract(self.alphas_cumprod, prev_t, x_t.shape)
        
        et = self.model(x_t, t)
        
        x0_t = (x_t - et * torch.sqrt(1 - at)) / torch.sqrt(at)
        c1 = torch.sqrt(at_next)
        c2 = torch.sqrt(1 - at_next - self.eta ** 2 * (1 - at))
        
        xt_next = c1 * x0_t + c2 * et
        
        if self.eta > 0:
            noise = torch.randn_like(x_t)
            sigma = self.eta * torch.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
            xt_next += sigma * noise
        
        return xt_next

    def forward(self, x_T):
        x_t = x_T
        batch_size = x_T.shape[0]
        device = x_T.device

        for i in range(self.sampling_T - 1):
            t = self.timestep_map[i].repeat(batch_size).to(device)
            prev_t = self.timestep_map[i+1].repeat(batch_size).to(device)
            x_t = self.ddim_sample(x_t, t, prev_t)
        
        # Handle the last step separately
        t = self.timestep_map[-1].repeat(batch_size).to(device)
        prev_t = torch.zeros_like(t)  # For the last step, prev_t is 0
        x_t = self.ddim_sample(x_t, t, prev_t)
        
        return x_t
