# sample_3d.py
import argparse
import os
import re
from typing import List, Tuple, Optional, Literal

import imageio
import numpy as np
import torch
from torchvision.utils import make_grid

from models_3d import Generator3D

CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pt$")


def _device_from_args(use_cuda: bool) -> torch.device:
    return torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")


def _list_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
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


def _parse_vol_size(vol_size_str: str) -> Tuple[int, int, int]:
    s = vol_size_str.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = re.split(r"[,\s]+", s)
    parts = [p for p in parts if p]
    if len(parts) != 3:
        raise ValueError(f"vol_size must have 3 ints (D,H,W). Got: {vol_size_str}")
    d, h, w = map(int, parts)
    return (d, h, w)


def _build_generator_from_ckpt_or_args(
    ckpt: dict,
    vol_size: Optional[Tuple[int, int, int]],
    latent_dim: Optional[int],
    dim: Optional[int],
    device: torch.device,
) -> Generator3D:
    ckpt_vol_size = ckpt.get("vol_size", None)
    ckpt_latent_dim = ckpt.get("latent_dim", None)
    ckpt_dim = ckpt.get("dim", None)

    final_vol_size = ckpt_vol_size if ckpt_vol_size is not None else vol_size
    final_latent_dim = ckpt_latent_dim if ckpt_latent_dim is not None else latent_dim
    final_dim = ckpt_dim if ckpt_dim is not None else dim

    missing = []
    if final_vol_size is None:
        missing.append("vol_size")
    if final_latent_dim is None:
        missing.append("latent_dim")
    if final_dim is None:
        missing.append("dim")
    if missing:
        raise ValueError("Missing config: " + ", ".join(missing) + " (provide via CLI or store in ckpt).")

    G = Generator3D(vol_size=tuple(final_vol_size), latent_dim=int(final_latent_dim), dim=int(final_dim)).to(device)

    if "G_state_dict" in ckpt:
        G.load_state_dict(ckpt["G_state_dict"], strict=True)
    elif "state_dict" in ckpt:
        G.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise KeyError("Checkpoint missing G_state_dict/state_dict")

    G.eval()
    return G


def _to_01(x: torch.Tensor) -> torch.Tensor:
    if float(x.min().item()) < 0.0:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def _central_slice(vols: torch.Tensor, axis: Literal["z", "y", "x"]) -> torch.Tensor:
    # vols: [N,1,D,H,W] -> [N,1,*,*]
    assert vols.dim() == 5 and vols.size(1) == 1
    N, _, D, H, W = vols.shape
    if axis == "z":
        return vols[:, :, D // 2, :, :]  # [N,1,H,W]
    if axis == "y":
        return vols[:, :, :, H // 2, :]  # [N,1,D,W]
    if axis == "x":
        return vols[:, :, :, :, W // 2]  # [N,1,D,H]
    raise ValueError(axis)


@torch.no_grad()
def _generate_grid_frame(G: Generator3D, device: torch.device, fixed_latents: torch.Tensor, grid_nrow: int, axis: str):
    vols = G(fixed_latents.to(device)).detach().cpu()  # [N,1,D,H,W]
    vols = _to_01(vols)
    sl = _central_slice(vols, axis=axis)  # [N,1,*,*]
    grid = make_grid(sl, nrow=grid_nrow)  # [C,H,W]
    grid = grid.permute(1, 2, 0).contiguous()  # [H,W,C]
    return (grid.numpy() * 255.0).astype(np.uint8)


def cmd_make_gif(args: argparse.Namespace) -> None:
    device = _device_from_args(args.cuda)
    ckpts = _list_checkpoints(args.checkpoint_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")

    if args.start_epoch is not None:
        ckpts = [(e, p) for (e, p) in ckpts if e >= args.start_epoch]
    if args.end_epoch is not None:
        ckpts = [(e, p) for (e, p) in ckpts if e <= args.end_epoch]
    if args.every is not None and args.every > 1:
        ckpts = [(e, p) for (e, p) in ckpts if (e % args.every) == 0]
    if not ckpts:
        raise FileNotFoundError("After filtering, no checkpoints remain.")

    fixed_latents = torch.randn(args.grid_size, args.latent_dim if args.latent_dim else 100)

    frames = []
    for epoch, path in ckpts:
        ckpt = _load_ckpt(path, device=device)
        G = _build_generator_from_ckpt_or_args(
            ckpt=ckpt,
            vol_size=_parse_vol_size(args.vol_size) if args.vol_size else None,
            latent_dim=args.latent_dim,
            dim=args.dim,
            device=device,
        )

        ckpt_latent_dim = ckpt.get("latent_dim", None)
        if ckpt_latent_dim is not None and fixed_latents.shape[1] != int(ckpt_latent_dim):
            fixed_latents = torch.randn(args.grid_size, int(ckpt_latent_dim))

        frame = _generate_grid_frame(G, device, fixed_latents, grid_nrow=args.grid_nrow, axis=args.axis)
        frames.append(frame)
        print(f"[make-gif] added frame for epoch {epoch}: {os.path.basename(path)}")

    os.makedirs(os.path.dirname(args.out_gif) or ".", exist_ok=True)
    imageio.mimsave(args.out_gif, frames, duration=args.frame_duration)
    print(f"[make-gif] saved GIF ({len(frames)} frames) to: {args.out_gif}")


@torch.no_grad()
def cmd_sample_npy(args: argparse.Namespace) -> None:
    device = _device_from_args(args.cuda)

    ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch_{args.epoch:04d}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = _load_ckpt(ckpt_path, device=device)
    G = _build_generator_from_ckpt_or_args(
        ckpt=ckpt,
        vol_size=_parse_vol_size(args.vol_size) if args.vol_size else None,
        latent_dim=args.latent_dim,
        dim=args.dim,
        device=device,
    )

    ld = ckpt.get("latent_dim", None)
    if ld is None:
        ld = args.latent_dim if args.latent_dim is not None else 100
    ld = int(ld)

    latents = torch.randn(args.num_samples, ld, device=device)
    vols = G(latents).detach().cpu().numpy().astype(np.float32)  # [N,1,D,H,W]

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, vols)
    print(f"[sample-npy] saved {vols.shape} volumes to: {args.out_npy}")


def main():
    p = argparse.ArgumentParser(description="3D checkpoint sampling utilities (GIF + NPY)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("make-gif")
    pg.add_argument("--checkpoint-dir", required=True)
    pg.add_argument("--out-gif", required=True)

    pg.add_argument("--vol-size", default=None, help='D,H,W e.g. "64,64,64" (if ckpt lacks config)')
    pg.add_argument("--latent-dim", type=int, default=None)
    pg.add_argument("--dim", type=int, default=None)

    pg.add_argument("--grid-size", type=int, default=32)
    pg.add_argument("--grid-nrow", type=int, default=8)
    pg.add_argument("--frame-duration", type=float, default=0.15)

    pg.add_argument("--axis", type=str, default="z", choices=["z", "y", "x"], help="Central slice axis")
    pg.add_argument("--start-epoch", type=int, default=None)
    pg.add_argument("--end-epoch", type=int, default=None)
    pg.add_argument("--every", type=int, default=None)

    pg.add_argument("--cuda", action="store_true")

    pn = sub.add_parser("sample-npy")
    pn.add_argument("--checkpoint-dir", required=True)
    pn.add_argument("--epoch", type=int, required=True)
    pn.add_argument("--num-samples", type=int, required=True)
    pn.add_argument("--out-npy", required=True)

    pn.add_argument("--vol-size", default=None, help='D,H,W e.g. "64,64,64" (if ckpt lacks config)')
    pn.add_argument("--latent-dim", type=int, default=None)
    pn.add_argument("--dim", type=int, default=None)

    pn.add_argument("--cuda", action="store_true")

    args = p.parse_args()
    if args.cmd == "make-gif":
        cmd_make_gif(args)
    elif args.cmd == "sample-npy":
        cmd_sample_npy(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
