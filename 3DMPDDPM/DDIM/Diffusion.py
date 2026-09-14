import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract(v, t, x_shape):
    """Extract the appropriate t index for a batch of indices."""
    out = torch.gather(v, index=t, dim=0).float().to(t.device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two Gaussians.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2)**2 * torch.exp(-logvar2))

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs

class ImprovedDDIMTrainer(nn.Module):
    def __init__(self, model, num_timesteps, use_ddim=True, use_ddpm=True, use_vlb=True, loss_balance_lambda=0.5, loss_balance_policy='static', vlb_lambda=0.001):
        """
        Args:
            model (nn.Module): Model to train
            num_timesteps (int): Number of timesteps to train for
            loss_balance_lambda (float): balance the DDIM and DDPM loss. Value between 0 and 1, where 0 is DDIM only and 1 is DDPM only
            loss_balance_policy (str): Policy for updating loss_balance_lambda. One of 'static', 'linear_decrease', 'linear_increase', 'cosine', 'step'
            vlb_lambda (float): Weight for VLB loss
            use_ddim (bool): Whether to use DDIM loss
            use_ddpm (bool): Whether to use DDPM loss
            use_vlb (bool): Whether to use VLB loss
        """

        super().__init__()

        self.model = model
        self.num_timesteps = num_timesteps
        self.lambda_ratio = loss_balance_lambda
        self.loss_balance_policy = loss_balance_policy
        self.vlb_lambda = vlb_lambda
        self.current_step = 0

        # Ensure at least one of DDIM or DDPM is used
        if not use_ddim and not use_ddpm:
            raise ValueError("At least one of use_ddim or use_ddpm must be True")
        self.use_ddim = use_ddim
        self.use_ddpm = use_ddpm
        self.use_vlb = use_vlb

        # Use cosine schedule for betas
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.register_buffer('posterior_variance', self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', torch.log(self.posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', torch.sqrt(alphas_cumprod_prev) * self.betas / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        noise_pred = self.model(x, t)
        x_recon = self.predict_start_from_noise(x, t, noise_pred)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance, noise_pred

    def _update_lambda(self):
        if self.loss_balance_policy == 'static':
            pass
        elif self.loss_balance_policy == 'linear_decrease':
            self.lambda_ratio = max(0, self.lambda_ratio - 1e-5)
        elif self.loss_balance_policy == 'linear_increase':
            self.lambda_ratio = min(1, self.lambda_ratio + 1e-5)
        elif self.loss_balance_policy == 'cosine':
            self.lambda_ratio = 0.5 * (1 + torch.cos(torch.tensor(self.current_step * np.pi / self.num_timesteps)).item())
        elif self.loss_balance_policy == 'step':
            if self.current_step == self.num_timesteps // 2:
                self.lambda_ratio = 1 - self.lambda_ratio
        self.current_step += 1

    def set_lambda_policy(self, policy, loss_balance_lambda=None):
        """
        Set the lambda policy and optionally reset the lambda ratio.
        Args:
        policy (str): One of 'static', 'linear_decrease', 'linear_increase', 'cosine', 'step'
        loss_balance_lambda (float, optional): If provided, reset lambda_ratio to this value
        """
        assert policy in ['static', 'linear_decrease', 'linear_increase', 'cosine', 'step'], "Invalid policy"
        self.loss_balance_policy = policy
        if loss_balance_lambda is not None:
            self.lambda_ratio = loss_balance_lambda
        self.current_step = 0

    def forward(self, x_0, noise=None):
        b = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x_0.device).long()
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        model_mean, model_log_variance, noise_pred = self.p_mean_variance(x_t, t, clip_denoised=False)
        
        loss = 0
        loss_components = {}
        
        if self.use_ddim:
            ddim_loss = F.mse_loss(model_mean, x_0, reduction='none').mean()
            loss_components['ddim'] = ddim_loss
        
        if self.use_ddpm:
            ddpm_loss = F.mse_loss(noise_pred, noise, reduction='none').mean()
            loss_components['ddpm'] = ddpm_loss
        
        if self.use_vlb:
            true_mean, true_log_variance_clipped = self.q_posterior(x_0, x_t, t)
            kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
            kl = mean_flat(kl) / np.log(2.0)
            
            decoder_nll = -discretized_gaussian_log_likelihood(x_0, means=model_mean, log_scales=0.5 * model_log_variance)
            decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
            
            vlb = torch.where(t == 0, decoder_nll, kl).mean()
            loss_components['vlb'] = vlb * self.vlb_lambda

        # Combine losses based on active components
        if self.use_ddim and self.use_ddpm:
            loss += self.lambda_ratio * loss_components['ddim'] + (1 - self.lambda_ratio) * loss_components['ddpm']
        elif self.use_ddim:
            loss += loss_components['ddim']
        elif self.use_ddpm:
            loss += loss_components['ddpm']

        if self.use_vlb:
            loss += loss_components['vlb']

        self._update_lambda()
        return loss, loss_components

class ImprovedDDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, training_T, sampling_T, eta=0.0):
        super().__init__()
        self.model = model
        self.training_T = training_T
        self.sampling_T = sampling_T
        self.eta = eta
        # Use cosine beta schedule
        self.register_buffer('betas', self.cosine_beta_schedule(training_T))
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # Timestep mapping
        self.register_buffer('timestep_map', self.get_sampling_timesteps(training_T, sampling_T))

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

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
