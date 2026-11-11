"""
Generate evaluation samples from a DDPM snapshot.

    python -m ddpm3d.sample --ckpt RUN/best.pt --out_dir RUN/samples/ddpm --sampler ddpm
    python -m ddpm3d.sample --ckpt RUN/best.pt --out_dir RUN/samples/ddim50 --sampler ddim --steps 50
    python -m ddpm3d.sample --ckpt RUN/best.pt --out_dir RUN/samples/dpmpp20 --sampler dpmpp --steps 20

Writes to out_dir/
  labels/sample_XXX.npy   one volume per file, in the original data values
                          (e.g. {0,1}), dtype of the smallest fitting integer
  raw/samples_raw.npy     network output before discretisation, float16,
                          (N, C, D, H, W)
  timing.json             sampler, steps, network evaluations, wall time
                          (generation only), GPU, batch size
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from .common import autocast, load_for_sampling


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim", "dpmpp"])
    p.add_argument("--steps", type=int, default=50, help="DDIM / DPM-Solver++ steps (ignored for ddpm)")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--bf16", type=int, default=1)
    return p.parse_args(argv)


def label_dtype(values):
    vals = np.asarray(values)
    if np.all(vals == np.round(vals)):
        for dt in (np.uint8, np.int8, np.int16, np.int32):
            if vals.min() >= np.iinfo(dt).min and vals.max() <= np.iinfo(dt).max:
                return dt
    return np.float32


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model, diffusion, codec, ck = load_for_sampling(args.ckpt, device)
    shape = tuple(ck["data"]["shape"])
    values = np.asarray(ck["data"]["values"])
    dt = label_dtype(values)

    os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)
    g = torch.Generator(device=device).manual_seed(args.seed)

    raws, gen_seconds, idx = [], 0.0, 0
    for s in range(0, args.n, args.batch):
        b = min(args.batch, args.n - s)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with autocast(device, bool(args.bf16)):
            if args.sampler == "ddpm":
                x = diffusion.ddpm_sample(model, (b, codec.channels, *shape), device, generator=g, progress=True)
            elif args.sampler == "ddim":
                x = diffusion.ddim_sample(model, (b, codec.channels, *shape), device, steps=args.steps,
                                          eta=args.eta, generator=g, progress=True)
            else:
                x = diffusion.dpmpp_sample(model, (b, codec.channels, *shape), device, steps=args.steps,
                                           generator=g, progress=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_seconds += time.time() - t0
        x = x.float()
        lab = codec.decode(x).cpu().numpy()
        for v in values[lab].astype(dt):
            np.save(os.path.join(args.out_dir, "labels", f"sample_{idx:03d}.npy"), v)
            idx += 1
        raws.append(x.cpu().numpy().astype(np.float16))
        print(f"{idx}/{args.n} samples ({gen_seconds:.1f} s)", flush=True)

    np.save(os.path.join(args.out_dir, "raw", "samples_raw.npy"), np.concatenate(raws))
    nfe = diffusion.T if args.sampler == "ddpm" else len(diffusion.ddim_timesteps(args.steps))
    with open(os.path.join(args.out_dir, "timing.json"), "w") as fh:
        json.dump({
            "checkpoint": os.path.abspath(args.ckpt),
            "snapshot_step": ck.get("step"),
            "sampler": args.sampler,
            "steps": nfe,
            "eta": args.eta if args.sampler == "ddim" else None,
            "snapshot_ema_decay": ck.get("ema_decay"),
            "network_evaluations_per_sample": nfe,
            "n_samples": args.n,
            "batch": args.batch,
            "bf16": bool(args.bf16),
            "generation_seconds": round(gen_seconds, 2),
            "seconds_per_100": round(100 * gen_seconds / args.n, 2),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        }, fh, indent=2)
    print(f"Done: {args.n} samples in {gen_seconds:.1f} s -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
