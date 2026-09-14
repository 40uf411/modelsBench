import os
import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=False, checkpoint_every=100, checkpoint_dir="./checkpoints",
                 save_png_every=1, png_grid_size=64, png_nrow=8):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
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
        # Fixed latents for consistent visualization across epochs
        with torch.no_grad():
            self.fixed_latents = self.G.sample_latent(self.png_grid_size)
            if self.use_cuda:
                self.fixed_latents = self.fixed_latents.cuda()


        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

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
        }
        path = f"{self.checkpoint_dir}/ckpt_epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    @torch.no_grad()
    def _save_epoch_png(self, epoch: int):
        self.G.eval()
        z = self.fixed_latents

        samples = self.G(z).detach().cpu()  # [N, C, H, W]

        # If generator is tanh -> [-1,1], map to [0,1] for saving
        if samples.min().item() < 0:
            samples = (samples + 1.0) / 2.0

        samples = samples.clamp(0.0, 1.0)

        out_path = os.path.join(self.checkpoint_dir, f"samples_epoch_{epoch:04d}.png")
        save_image(samples, out_path, nrow=self.png_nrow)
        print(f"Saved PNG samples: {out_path}")

        self.G.train()


    def _critic_train_iteration(self, data: torch.Tensor):
        """
        One critic (discriminator) training step for WGAN-GP.
        Expects `data` to be a batch of images. Preferred shape: [B, C, H, W].
        """

        # ---- Move real data to correct device and fix shape ----
        if self.use_cuda:
            data = data.cuda(non_blocking=True)

        # If real data arrives as [B, H, W], recover grayscale channel -> [B, 1, H, W]
        # (This is only correct for single-channel datasets. If you train RGB, fix upstream transforms.)
        if data.dim() == 3:
            data = data.unsqueeze(0)

        if data.dim() != 4:
            raise ValueError(f"Expected real data 4D [B,C,H,W], got {tuple(data.shape)}")

        batch_size = data.size(0)

        # ---- Generate fake data (make sure it is on same device) ----
        generated_data = self.sample_generator(batch_size)

        if self.use_cuda:
            generated_data = generated_data.cuda(non_blocking=True)

        # If generator returns [B,H,W] for grayscale, fix it too
        if generated_data.dim() == 3:
            generated_data = generated_data.unsqueeze(1)

        if generated_data.dim() != 4:
            raise ValueError(f"Expected generated data 4D [B,C,H,W], got {tuple(generated_data.shape)}")

        # Strict shape/channel match (prevents subtle bugs)
        if data.shape != generated_data.shape:
            raise ValueError(f"Shape mismatch: real {tuple(data.shape)} vs fake {tuple(generated_data.shape)}")

        # ---- Critic scores ----
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # ---- Gradient penalty ----
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses.setdefault('GP', []).append(gradient_penalty.item())

        # ---- Optimize critic ----
        self.D_opt.zero_grad(set_to_none=True)
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        # ---- Record loss ----
        self.losses.setdefault('D', []).append(d_loss.item())

    def _generator_train_iteration(self, data):
        """One generator training step for WGAN-GP."""

        # Zero generator grads
        self.G_opt.zero_grad(set_to_none=True)

        # Batch size from real batch (only used to sample noise)
        batch_size = data.size(0)

        # Sample fake images
        generated_data = self.sample_generator(batch_size)

        # Ensure generated data is on the same device as the critic
        if self.use_cuda:
            generated_data = generated_data.cuda(non_blocking=True)

        # If generator returns [B,H,W] for grayscale, make it [B,1,H,W]
        if generated_data.dim() == 3:
            generated_data = generated_data.unsqueeze(1)

        if generated_data.dim() != 4:
            raise ValueError(f"Expected generated data 4D [B,C,H,W], got {tuple(generated_data.shape)}")

        # Generator loss: maximize critic score on fake -> minimize negative mean
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()

        # Backprop + step
        g_loss.backward()
        self.G_opt.step()

        # Record scalar loss properly
        self.losses.setdefault('G', []).append(g_loss.item())

    def _gradient_penalty(self, real_data: torch.Tensor, generated_data: torch.Tensor) -> torch.Tensor:
        """
        WGAN-GP gradient penalty:
            GP = λ * E[(||∇_x D(x̂)||_2 - 1)^2]
        where x̂ is an interpolation between real and generated samples.

        Assumptions:
        - real_data and generated_data should be same shape.
        - Expected image shape: [B, C, H, W] for conv critics.
        """

        # --- Basic validation / shape hygiene ---
        if real_data.dim() != generated_data.dim():
            raise ValueError(f"real_data dim {real_data.dim()} != generated_data dim {generated_data.dim()}")

        # If something upstream dropped the channel dim (common with .squeeze()), recover it.
        # Only safe if your data is actually single-channel.
        if real_data.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
            real_data = real_data.unsqueeze(1)
            generated_data = generated_data.unsqueeze(1)

        if real_data.dim() != 4:
            raise ValueError(f"Expected real_data to be 4D [B,C,H,W], got shape {tuple(real_data.shape)}")

        if real_data.shape != generated_data.shape:
            raise ValueError(
                f"Shape mismatch: real_data {tuple(real_data.shape)} vs generated_data {tuple(generated_data.shape)}"
            )

        device = real_data.device
        batch_size = real_data.size(0)

        # --- Interpolation ---
        # alpha needs to broadcast across C,H,W
        alpha = torch.rand(batch_size, 1, 1, 1, device=device, dtype=real_data.dtype)
        interpolated = real_data + alpha * (generated_data - real_data)
        interpolated.requires_grad_(True)

        # --- Critic output on interpolated samples ---
        d_interpolated = self.D(interpolated)

        # Ensure shape is [B, ...] and build matching grad_outputs
        grad_outputs = torch.ones_like(d_interpolated, device=device)

        # --- Compute gradients of critic outputs w.r.t. interpolated inputs ---
        gradients = torch_grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # same shape as interpolated: [B, C, H, W]

        # --- Per-sample gradient norm ---
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)  # [B]

        # Optional logging (safe)
        if hasattr(self, "losses") and isinstance(self.losses, dict):
            self.losses.setdefault("gradient_norm", []).append(grad_norm.mean().item())

        # --- Penalty ---
        gp = ((grad_norm - 1.0) ** 2).mean()
        return self.gp_weight * gp

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data[0])
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data[0])

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(64))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)
                if (epoch + 1) % self.save_png_every == 0:
                    self._save_epoch_png(epoch + 1)
                # Save checkpoint every N epochs
                if (epoch + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(epoch + 1)
        if save_training_gif:
            imageio.mimsave('./training_{}_epochs.gif'.format(epochs),
                            training_progress_images)

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
