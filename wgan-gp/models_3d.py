# models_3d.py
import torch
import torch.nn as nn


class Generator3D(nn.Module):
    """
    3D DCGAN-style generator for (D,H,W) volumes.
    Output: [B, C=1, D, H, W]
    """
    def __init__(self, vol_size, latent_dim: int, dim: int, out_channels: int = 1):
        super().__init__()

        if isinstance(vol_size, int):
            D = H = W = vol_size
        else:
            D, H, W = vol_size

        assert D % 16 == 0 and H % 16 == 0 and W % 16 == 0, "vol_size must be divisible by 16"

        self.dim = dim
        self.latent_dim = latent_dim
        self.vol_size = (D, H, W)
        self.feature_sizes = (D // 16, H // 16, W // 16)  # start spatial size

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1] * self.feature_sizes[2]),
            nn.ReLU(inplace=True),
        )

        self.features_to_volume = nn.Sequential(
            nn.ConvTranspose3d(8 * dim, 4 * dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(4 * dim),

            nn.ConvTranspose3d(4 * dim, 2 * dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(2 * dim),

            nn.ConvTranspose3d(2 * dim, dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(dim),

            nn.ConvTranspose3d(dim, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.latent_to_features(z)
        x = x.view(
            -1,
            8 * self.dim,
            self.feature_sizes[0],
            self.feature_sizes[1],
            self.feature_sizes[2],
        )
        return self.features_to_volume(x)

    def sample_latent(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.latent_dim), device=next(self.parameters()).device)


class Discriminator3D(nn.Module):
    """
    3D critic for WGAN-GP. Input: [B, C=1, D, H, W], output: [B, 1]
    """
    def __init__(self, vol_size, dim: int, in_channels: int = 1):
        super().__init__()

        if isinstance(vol_size, int):
            D = H = W = vol_size
        else:
            D, H, W = vol_size

        self.vol_size = (D, H, W)

        self.volume_to_features = nn.Sequential(
            nn.Conv3d(in_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(dim, 2 * dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(2 * dim, 4 * dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(4 * dim, 8 * dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        out_size = 8 * dim * (D // 16) * (H // 16) * (W // 16)
        self.features_to_score = nn.Linear(out_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        f = self.volume_to_features(x)
        f = f.view(b, -1)
        return self.features_to_score(f)
