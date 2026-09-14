#!/usr/bin/env python3
"""
sample.py

Two utilities for a WGAN-GP training run that saves checkpoints (recommended format):
  ckpt = {
    "epoch": int,
    "num_steps": int,
    "G_state_dict": ...,
    "D_state_dict": ...,
    "G_opt_state_dict": ...,
    "D_opt_state_dict": ...,
    # optionally: "img_size", "latent_dim", "dim" (highly recommended)
  }

JOB 1) make-gif:
  - Load checkpoints in order (by epoch number)
  - For each checkpoint, generate an image grid from fixed latents
  - Save an animated GIF

JOB 2) sample-npy:
  - Load a specific checkpoint epoch
  - Generate N images
  - Save them into one .npy file (shape: [N, C, H, W], float32)

Notes:
  - This script reconstructs the Generator from your `models.py`.
  - If your checkpoints do NOT store generator config (img_size/latent_dim/dim),
    you must pass them via CLI flags.
"""

import argparse
import os
import re
from typing import List, Tuple, Optional

import imageio
import numpy as np
import torch
from torchvision.utils import make_grid

from models import Generator  # assumes your Generator class is in models.py


CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pt$")


def _device_from_args(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _list_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Return sorted list of (epoch, path) for ckpt_epoch_XXXX.pt files."""
    out = []
    for fn in os.listdir(checkpoint_dir):
        m = CKPT_RE.match(fn)
        if m:
            epoch = int(m.group(1))
            out.append((epoch, os.path.join(checkpoint_dir, fn)))
    out.sort(key=lambda x: x[0])
    return out


def _load_ckpt(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def _parse_img_size(img_size_str: str) -> Tuple[int, int, int]:
    """
    Accepts "256,256,1" or "256 256 1" or "(256,256,1)".
    Returns (H, W, C).
    """
    s = img_size_str.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = re.split(r"[,\s]+", s)
    parts = [p for p in parts if p]
    if len(parts) != 3:
        raise ValueError(f"img_size must have 3 ints (H,W,C). Got: {img_size_str}")
    h, w, c = map(int, parts)
    return (h, w, c)


def _build_generator_from_ckpt_or_args(
    ckpt: dict,
    img_size: Optional[Tuple[int, int, int]],
    latent_dim: Optional[int],
    dim: Optional[int],
    device: torch.device,
) -> Generator:
    # Prefer config inside checkpoint if available
    ckpt_img_size = ckpt.get("img_size", None)
    ckpt_latent_dim = ckpt.get("latent_dim", None)
    ckpt_dim = ckpt.get("dim", None)

    final_img_size = ckpt_img_size if ckpt_img_size is not None else img_size
    final_latent_dim = ckpt_latent_dim if ckpt_latent_dim is not None else latent_dim
    final_dim = ckpt_dim if ckpt_dim is not None else dim

    missing = []
    if final_img_size is None:
        missing.append("img_size")
    if final_latent_dim is None:
        missing.append("latent_dim")
    if final_dim is None:
        missing.append("dim")

    if missing:
        raise ValueError(
            "Cannot reconstruct Generator; missing config: "
            + ", ".join(missing)
            + ". Provide them via CLI flags or store them in checkpoints."
        )

    G = Generator(img_size=final_img_size, latent_dim=final_latent_dim, dim=final_dim).to(device)
    if "G_state_dict" in ckpt:
        G.load_state_dict(ckpt["G_state_dict"], strict=True)
    elif "state_dict" in ckpt:
        # fallback if you saved only model weights
        G.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise KeyError(f"Checkpoint {ckpt.keys()} does not contain G_state_dict or state_dict.")

    G.eval()
    return G


@torch.no_grad()
def _generate_grid_frame(
    G: Generator,
    device: torch.device,
    fixed_latents: torch.Tensor,
    grid_nrow: int,
) -> np.ndarray:
    """
    Returns an HxWxC uint8 frame for GIF.
    Assumes G outputs in range [-1, 1] or [0, 1]; we clamp and convert.
    """
    samples = G(fixed_latents.to(device)).detach().cpu()  # [N, C, H, W]

    # Make grid tensor [C, H, W]
    grid = make_grid(samples, nrow=grid_nrow)

    # Convert to [H, W, C]
    grid = grid.permute(1, 2, 0).contiguous()

    # Heuristic scaling: if looks like [-1,1], map to [0,1]
    gmin = float(grid.min().item())
    gmax = float(grid.max().item())
    if gmin < 0.0:
        grid = (grid + 1.0) / 2.0

    grid = grid.clamp(0.0, 1.0)
    frame = (grid.numpy() * 255.0).astype(np.uint8)
    return frame


def cmd_make_gif(args: argparse.Namespace) -> None:
    device = _device_from_args(args.cuda)
    ckpts = _list_checkpoints(args.checkpoint_dir)

    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints matching ckpt_epoch_XXXX.pt found in: {args.checkpoint_dir}"
        )

    # Optional filtering
    if args.start_epoch is not None:
        ckpts = [(e, p) for (e, p) in ckpts if e >= args.start_epoch]
    if args.end_epoch is not None:
        ckpts = [(e, p) for (e, p) in ckpts if e <= args.end_epoch]

    if args.every is not None and args.every > 1:
        ckpts = [(e, p) for (e, p) in ckpts if (e % args.every) == 0]

    if not ckpts:
        raise FileNotFoundError("After filtering, no checkpoints remain to build a GIF.")

    # Fixed latents: keep constant across checkpoints
    fixed_latents = torch.randn(args.grid_size, args.latent_dim if args.latent_dim else 100)

    frames: List[np.ndarray] = []
    for epoch, path in ckpts:
        ckpt = _load_ckpt(path, device=device)

        # Rebuild generator and load weights
        G = _build_generator_from_ckpt_or_args(
            ckpt=ckpt,
            img_size=_parse_img_size(args.img_size) if args.img_size else None,
            latent_dim=args.latent_dim,
            dim=args.dim,
            device=device,
        )

        # Ensure fixed_latents matches ckpt latent_dim if stored (optional)
        ckpt_latent_dim = ckpt.get("latent_dim", None)
        if ckpt_latent_dim is not None and fixed_latents.shape[1] != int(ckpt_latent_dim):
            fixed_latents = torch.randn(args.grid_size, int(ckpt_latent_dim))

        frame = _generate_grid_frame(G, device, fixed_latents, grid_nrow=args.grid_nrow)
        frames.append(frame)
        print(f"[make-gif] added frame for epoch {epoch}: {os.path.basename(path)}")

    out_path = args.out_gif
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # duration: seconds per frame
    imageio.mimsave(out_path, frames, duration=args.frame_duration)
    print(f"[make-gif] saved GIF with {len(frames)} frames to: {out_path}")


@torch.no_grad()
def cmd_sample_npy(args: argparse.Namespace) -> None:
    device = _device_from_args(args.cuda)

    ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch_{args.epoch:04d}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = _load_ckpt(ckpt_path, device=device)

    G = _build_generator_from_ckpt_or_args(
        ckpt=ckpt,
        img_size=_parse_img_size(args.img_size) if args.img_size else None,
        latent_dim=args.latent_dim,
        dim=args.dim,
        device=device,
    )

    # Use ckpt latent_dim if stored; otherwise use CLI latent_dim or 100 fallback
    ld = ckpt.get("latent_dim", None)
    if ld is None:
        ld = args.latent_dim if args.latent_dim is not None else 100
    ld = int(ld)

    latents = torch.randn(args.num_samples, ld, device=device)
    samples = G(latents).detach().cpu().numpy().astype(np.float32)  # [N, C, H, W]

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, samples)
    print(f"[sample-npy] saved {samples.shape} samples to: {args.out_npy}")


def main():
    p = argparse.ArgumentParser(description="Checkpoint sampling utilities (GIF + NPY)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- make-gif ----
    pg = sub.add_parser("make-gif", help="Build a GIF by sampling one grid per checkpoint")
    pg.add_argument("--checkpoint-dir", required=True, help="Directory containing ckpt_epoch_XXXX.pt files")
    pg.add_argument("--out-gif", required=True, help="Output GIF path, e.g. ./progress.gif")

    # Model config if not stored in checkpoints
    pg.add_argument("--img-size", default=None, help='H,W,C e.g. "256,256,1" (only needed if ckpt lacks config)')
    pg.add_argument("--latent-dim", type=int, default=None, help="Latent dim (only needed if ckpt lacks config)")
    pg.add_argument("--dim", type=int, default=None, help="Base dim used to construct Generator (if ckpt lacks config)")

    # Sampling controls
    pg.add_argument("--grid-size", type=int, default=64, help="Number of samples per frame (grid)")
    pg.add_argument("--grid-nrow", type=int, default=8, help="Grid columns (nrow) for make_grid")
    pg.add_argument("--frame-duration", type=float, default=0.15, help="Seconds per GIF frame")

    # Checkpoint filtering
    pg.add_argument("--start-epoch", type=int, default=None, help="Only epochs >= this")
    pg.add_argument("--end-epoch", type=int, default=None, help="Only epochs <= this")
    pg.add_argument("--every", type=int, default=None, help="Use only epochs divisible by N (e.g. 10)")

    pg.add_argument("--cuda", action="store_true", help="Use CUDA if available")

    # ---- sample-npy ----
    pn = sub.add_parser("sample-npy", help="Sample N images from a specific checkpoint and save to .npy")
    pn.add_argument("--checkpoint-dir", required=True, help="Directory containing ckpt_epoch_XXXX.pt files")
    pn.add_argument("--epoch", type=int, required=True, help="Epoch number to load (matches ckpt_epoch_XXXX.pt)")
    pn.add_argument("--num-samples", type=int, required=True, help="Number of images to generate")
    pn.add_argument("--out-npy", required=True, help="Output .npy path, e.g. ./samples_epoch_0100.npy")

    # Model config if not stored in checkpoints
    pn.add_argument("--img-size", default=None, help='H,W,C e.g. "256,256,1" (only needed if ckpt lacks config)')
    pn.add_argument("--latent-dim", type=int, default=None, help="Latent dim (only needed if ckpt lacks config)")
    pn.add_argument("--dim", type=int, default=None, help="Base dim used to construct Generator (if ckpt lacks config)")

    pn.add_argument("--cuda", action="store_true", help="Use CUDA if available")

    args = p.parse_args()

    if args.cmd == "make-gif":
        cmd_make_gif(args)
    elif args.cmd == "sample-npy":
        cmd_sample_npy(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

