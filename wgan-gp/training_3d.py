# training_3d.py
from __future__ import annotations

import os
from typing import Literal

import imageio
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def _save_slices_grid_png(slices: torch.Tensor, out_path: str, nrow: int = 8) -> None:
    """
    Minimal replacement for torchvision.utils.make_grid/save_image.

    slices: [N, 1, H, W] in [0,1]
    Saves a PNG grid using imageio.
    """
    if slices.dim() != 4 or slices.size(1) != 1:
        raise ValueError(f"Expected slices [N,1,H,W], got {tuple(slices.shape)}")

    slices = slices.detach().cpu().clamp(0.0, 1.0)

    N, _, H, W = slices.shape
    nrow = max(1, int(nrow))
    ncol = int(np.ceil(N / nrow))

    grid = torch.zeros((ncol * H, nrow * W), dtype=torch.float32)
    for i in range(N):
        r = i // nrow
        c = i % nrow
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = slices[i, 0]

    img = (grid.numpy() * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.imwrite(out_path, img)


def _slices_grid_frame_uint8(slices: torch.Tensor, nrow: int = 8) -> np.ndarray:
    """
    slices: [N,1,H,W] in [0,1]
    Returns a uint8 HxW frame suitable for GIF.
    """
    if slices.dim() != 4 or slices.size(1) != 1:
        raise ValueError(f"Expected slices [N,1,H,W], got {tuple(slices.shape)}")

    slices = slices.detach().cpu().clamp(0.0, 1.0)

    N, _, H, W = slices.shape
    nrow = max(1, int(nrow))
    ncol = int(np.ceil(N / nrow))

    grid = torch.zeros((ncol * H, nrow * W), dtype=torch.float32)
    for i in range(N):
        r = i // nrow
        c = i % nrow
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = slices[i, 0]

    frame = (grid.numpy() * 255.0).astype(np.uint8)
    return frame


class Trainer3D:
    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        dis_optimizer,
        gp_weight: float = 10.0,
        critic_iterations: int = 5,
        print_every: int = 500,
        use_cuda: bool = False,
        checkpoint_every: int = 100,
        checkpoint_dir: str = "./checkpoints_3d",
        save_png_every: int = 10,
        png_grid_size: int = 32,
        png_nrow: int = 8,
        slice_axis: Literal["z", "y", "x"] = "z",
        gif_frame_duration: float = 0.15,
    ):
        self.G = generator
        self.D = discriminator
        self.G_opt = gen_optimizer
        self.D_opt = dis_optimizer

        self.losses = {"G": [], "D": [], "GP": [], "gradient_norm": []}
        self.num_steps = 0
        self.use_cuda = use_cuda

        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.save_png_every = save_png_every
        self.png_grid_size = png_grid_size
        self.png_nrow = png_nrow
        self.slice_axis = slice_axis
        self.gif_frame_duration = gif_frame_duration

        with torch.no_grad():
            self.fixed_latents = self.G.sample_latent(self.png_grid_size)
            if self.use_cuda:
                self.fixed_latents = self.fixed_latents.cuda()

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    # -------------------------
    # Checkpointing
    # -------------------------
    def _save_checkpoint(self, epoch: int):
        ckpt = {
            "epoch": epoch,
            "num_steps": self.num_steps,
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
            "G_opt_state_dict": self.G_opt.state_dict(),
            "D_opt_state_dict": self.D_opt.state_dict(),
            "losses": self.losses,
            "gp_weight": self.gp_weight,
            "critic_iterations": self.critic_iterations,
            # store model config if present (helps sample_3d.py)
            "vol_size": getattr(self.G, "vol_size", None),
            "latent_dim": getattr(self.G, "latent_dim", None),
            "dim": getattr(self.G, "dim", None),
        }
        path = os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch:04d}.pt")
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _to_01(x: torch.Tensor) -> torch.Tensor:
        # If generator uses tanh: [-1,1] -> [0,1]
        if float(x.min().item()) < 0.0:
            x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    def _extract_central_slice(self, vols: torch.Tensor) -> torch.Tensor:
        """
        vols: [B,1,D,H,W] -> slices: [B,1,H,W] (z axis) or [B,1,D,W]/[B,1,D,H] for y/x
        Returned tensor is always 4D [B,1,*,*] suitable for grid saving.
        """
        if vols.dim() != 5 or vols.size(1) != 1:
            raise ValueError(f"Expected vols [B,1,D,H,W], got {tuple(vols.shape)}")

        B, _, D, H, W = vols.shape

        if self.slice_axis == "z":
            k = D // 2
            sl = vols[:, :, k, :, :]  # [B,1,H,W]
        elif self.slice_axis == "y":
            k = H // 2
            sl = vols[:, :, :, k, :]  # [B,1,D,W]
        elif self.slice_axis == "x":
            k = W // 2
            sl = vols[:, :, :, :, k]  # [B,1,D,H]
        else:
            raise ValueError(f"Unknown slice_axis={self.slice_axis}")

        if sl.dim() != 4:
            raise ValueError(f"Slice must be 4D, got {sl.dim()}D with shape {tuple(sl.shape)}")
        return sl

    @torch.no_grad()
    def _save_epoch_png(self, epoch: int):
        self.G.eval()
        z = self.fixed_latents

        vols = self.G(z).detach().cpu()  # [N,1,D,H,W]
        vols = self._to_01(vols)

        slices = self._extract_central_slice(vols)  # [N,1,*,*]
        out_path = os.path.join(self.checkpoint_dir, f"samples_epoch_{epoch:04d}_{self.slice_axis}.png")
        _save_slices_grid_png(slices, out_path, nrow=self.png_nrow)
        print(f"Saved PNG samples: {out_path}")

        self.G.train()

    # -------------------------
    # Training steps
    # -------------------------
    def _critic_train_iteration(self, data: torch.Tensor):
        """
        One critic step. Expects real volumes as:
          - [B,1,D,H,W] (preferred) or [B,D,H,W] (we'll fix)
        """
        if self.use_cuda:
            data = data.cuda(non_blocking=True)

        if data.dim() == 4:
            data = data.unsqueeze(1)  # [B,1,D,H,W]

        if data.dim() != 5:
            raise ValueError(f"Expected real data 5D [B,1,D,H,W], got {tuple(data.shape)}")

        batch_size = data.size(0)

        generated_data = self.sample_generator(batch_size)
        if self.use_cuda:
            generated_data = generated_data.cuda(non_blocking=True)

        if generated_data.dim() == 4:
            generated_data = generated_data.unsqueeze(1)

        if generated_data.dim() != 5:
            raise ValueError(f"Expected fake data 5D [B,1,D,H,W], got {tuple(generated_data.shape)}")

        if data.shape != generated_data.shape:
            raise ValueError(f"Shape mismatch: real {tuple(data.shape)} vs fake {tuple(generated_data.shape)}")

        d_real = self.D(data)
        d_generated = self.D(generated_data)

        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses["GP"].append(gradient_penalty.item())

        self.D_opt.zero_grad(set_to_none=True)
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        self.losses["D"].append(d_loss.item())

    def _generator_train_iteration(self, data: torch.Tensor):
        """One generator step."""
        self.G_opt.zero_grad(set_to_none=True)
        batch_size = data.size(0)

        generated_data = self.sample_generator(batch_size)
        if self.use_cuda:
            generated_data = generated_data.cuda(non_blocking=True)

        if generated_data.dim() == 4:
            generated_data = generated_data.unsqueeze(1)
        if generated_data.dim() != 5:
            raise ValueError(f"Expected fake data 5D [B,1,D,H,W], got {tuple(generated_data.shape)}")

        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()

        g_loss.backward()
        self.G_opt.step()

        self.losses["G"].append(g_loss.item())

    def _gradient_penalty(self, real_data: torch.Tensor, generated_data: torch.Tensor) -> torch.Tensor:
        """
        WGAN-GP:
          GP = λ * E[(||∇_x D(x̂)||_2 - 1)^2]
        """
        if real_data.dim() == 4:
            real_data = real_data.unsqueeze(1)
            generated_data = generated_data.unsqueeze(1)

        if real_data.dim() != 5:
            raise ValueError(f"Expected 5D [B,1,D,H,W], got {tuple(real_data.shape)}")
        if real_data.shape != generated_data.shape:
            raise ValueError(f"Shape mismatch: real {tuple(real_data.shape)} vs fake {tuple(generated_data.shape)}")

        device = real_data.device
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device, dtype=real_data.dtype)
        interpolated = real_data + alpha * (generated_data - real_data)
        interpolated.requires_grad_(True)

        d_interpolated = self.D(interpolated)
        grad_outputs = torch.ones_like(d_interpolated, device=device)

        gradients = torch_grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]  # [B,1,D,H,W]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        self.losses["gradient_norm"].append(grad_norm.mean().item())

        gp = ((grad_norm - 1.0) ** 2).mean()
        return self.gp_weight * gp

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1

            # Support both:
            #  - dataset returns tensor
            #  - dataset returns (tensor, label)
            if isinstance(data, (list, tuple)):
                data = data[0]

            self._critic_train_iteration(data)

            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

            if i % self.print_every == 0:
                print(f"Iteration {i + 1}")
                print(f"D: {self.losses['D'][-1]}")
                print(f"GP: {self.losses['GP'][-1]}")
                print(f"Gradient norm: {self.losses['gradient_norm'][-1]}")
                if len(self.losses["G"]) > 0:
                    print(f"G: {self.losses['G'][-1]}")

    # -------------------------
    # Public API
    # -------------------------
    def train(self, data_loader, epochs: int, save_training_gif: bool = True):
        frames = []

        # fixed latents for GIF
        if save_training_gif:
            fixed_latents = Variable(self.G.sample_latent(self.png_grid_size))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}")
            self._train_epoch(data_loader)

            if (epoch + 1) % self.save_png_every == 0:
                self._save_epoch_png(epoch + 1)

            if (epoch + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1)

            if save_training_gif:
                with torch.no_grad():
                    self.G.eval()
                    vols = self.G(fixed_latents).detach().cpu()  # [N,1,D,H,W]
                    vols = self._to_01(vols)
                    slices = self._extract_central_slice(vols)  # [N,1,*,*]
                    frame = _slices_grid_frame_uint8(slices, nrow=self.png_nrow)  # HxW uint8
                    frames.append(frame)
                    self.G.train()

        if save_training_gif and frames:
            out_gif = os.path.join(self.checkpoint_dir, f"training_{epochs}_epochs_{self.slice_axis}.gif")
            imageio.mimsave(out_gif, frames, duration=float(self.gif_frame_duration))
            print(f"Saved training GIF: {out_gif}")

    def sample_generator(self, num_samples: int) -> torch.Tensor:
        latent = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent = latent.cuda()
        return self.G(latent)

    def sample(self, num_samples: int) -> np.ndarray:
        vols = self.sample_generator(num_samples).detach().cpu().numpy().astype(np.float32)  # [N,1,D,H,W]
        return vols

