"""
Train a 3D DDPM on a folder of segmented .npy volumes.

    python -m ddpm3d.train --data_dir DATA --out_dir RUN [options]

Writes to RUN/
  last.pt                     model, EMA, optimiser, step, RNG state (resume)
  snapshots/ema_XXXXXXX.pt    EMA weights + configs every --snapshot_every steps
  previews/step_XXXXXXX.png   central slices of 4 DDIM-50 samples (EMA)
  train_log.csv               step, loss, lr, grad norm, throughput
  config.json
  TRAINING_DONE               when --iters is reached

Resumes automatically from RUN/last.pt. On SIGUSR1 or SIGTERM it saves
last.pt and exits with code 99 (the Slurm script requeues the job).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from .common import EMA, atomic_save, autocast, build_diffusion, build_model
from .data import VolumeBank

EXIT_REQUEUE = 99
_STOP = False


def _on_signal(signum, frame):
    global _STOP
    print(f"signal {signum}: saving and stopping at the next step", flush=True)
    _STOP = True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    # data
    p.add_argument("--encoding", default="auto", choices=["auto", "scalar", "onehot"])
    p.add_argument("--flips", type=int, default=1, help="random flips along x, y, z")
    p.add_argument("--permute_axes", type=int, default=0,
                   help="random axis permutations; only for isotropic materials")
    p.add_argument("--max_files", type=int, default=None, help="debug: use only the first N volumes")
    # model
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--channel_mult", default="1,2,2,4")
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--attention_resolutions", default="16,8")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head_channels", type=int, default=64)
    # diffusion
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    p.add_argument("--var_type", default="fixedsmall", choices=["fixedsmall", "fixedlarge"])
    # optimisation
    p.add_argument("--iters", type=int, default=150_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    # bookkeeping
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--snapshot_every", type=int, default=10_000)
    p.add_argument("--preview", type=int, default=1)
    return p.parse_args(argv)


def _ints(s):
    return [int(v) for v in str(s).split(",") if v.strip()]


@torch.no_grad()
def save_preview(ema_model, diffusion, codec, shape, device, path, bf16):
    import imageio
    g = torch.Generator(device=device).manual_seed(0)
    with autocast(device, bf16):
        x = diffusion.ddim_sample(ema_model, (4, codec.channels, *shape), device,
                                  steps=min(50, diffusion.T), generator=g)
    lab = codec.decode(x).cpu().numpy()
    k = codec.num_phases - 1
    sl = [(lab[i, shape[0] // 2] * (255 // k)).astype(np.uint8) for i in range(lab.shape[0])]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, np.concatenate(sl, axis=1))


def main(argv=None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    os.makedirs(args.out_dir, exist_ok=True)
    done = os.path.join(args.out_dir, "TRAINING_DONE")
    if os.path.exists(done):
        print("TRAINING_DONE exists: nothing to do.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    bank = VolumeBank(args.data_dir, encoding=args.encoding, device=device, max_files=args.max_files)
    codec = bank.codec
    print(f"{len(bank)} volumes {bank.shape}, values {bank.values}, encoding {codec.encoding}, "
          f"phase fractions {np.round(bank.phase_fractions(), 4).tolist()}", flush=True)

    model_config = dict(in_channels=codec.channels, out_channels=codec.channels, image_size=bank.shape[0],
                        base_channels=args.base_channels, channel_mult=_ints(args.channel_mult),
                        num_res_blocks=args.num_res_blocks,
                        attention_resolutions=_ints(args.attention_resolutions),
                        dropout=args.dropout, head_channels=args.head_channels)
    diffusion_config = dict(T=args.T, schedule=args.schedule, var_type=args.var_type, clip_x0=True)
    data_meta = dict(values=bank.values, encoding=codec.encoding, shape=list(bank.shape),
                     data_dir=os.path.abspath(args.data_dir), n_volumes=len(bank))

    model = build_model(model_config).to(device)
    diffusion = build_diffusion(diffusion_config)
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params / 1e6:.2f} M", flush=True)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    step = 0
    last_path = os.path.join(args.out_dir, "last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        ema.model.to(device)
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        gen.set_state(ck["data_rng"])
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        print(f"Resumed from {last_path} at step {step}", flush=True)
    else:
        with open(os.path.join(args.out_dir, "config.json"), "w") as fh:
            json.dump({"args": vars(args), "model_config": model_config, "diffusion_config": diffusion_config,
                       "data": data_meta, "n_params": n_params}, fh, indent=2)

    fwd = torch.compile(model) if args.compile else model

    def state():
        return {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "data_rng": gen.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "model_config": model_config, "diffusion_config": diffusion_config, "data": data_meta,
                "args": vars(args)}

    def snapshot():
        path = os.path.join(args.out_dir, "snapshots", f"ema_{step:07d}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_save({"ema": ema.model.state_dict(), "step": step, "model_config": model_config,
                     "diffusion_config": diffusion_config, "data": data_meta}, path)
        if args.preview:
            save_preview(ema.model, diffusion, codec, bank.shape, device,
                         os.path.join(args.out_dir, "previews", f"step_{step:07d}.png"), args.bf16)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    new_log = not os.path.exists(log_path)
    log_fh = open(log_path, "a", newline="")
    log = csv.writer(log_fh)
    if new_log:
        log.writerow(["step", "loss", "lr", "grad_norm", "it_per_s", "gpu_mem_gb"])

    model.train()
    t0, loss_acc, gn_acc, n_acc = time.time(), 0.0, 0.0, 0
    while step < args.iters:
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        x0 = bank.sample_batch(args.batch_size, gen, flips=bool(args.flips), permute_axes=bool(args.permute_axes))
        with autocast(device, bool(args.bf16)):
            loss = diffusion.training_loss(fwd, x0, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model)
        step += 1

        loss_acc += loss.item()
        gn_acc += float(gn)
        n_acc += 1
        if step % args.log_every == 0 or step == args.iters:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            row = [step, loss_acc / n_acc, lr, gn_acc / n_acc, n_acc / dt, round(mem, 2)]
            log.writerow(row)
            log_fh.flush()
            print(f"step {step:7d}  loss {row[1]:.5f}  lr {lr:.2e}  |g| {row[3]:.3f}  {row[4]:.2f} it/s  {mem:.1f} GB",
                  flush=True)
            t0, loss_acc, gn_acc, n_acc = time.time(), 0.0, 0.0, 0
        if step % args.snapshot_every == 0 or step == args.iters:
            snapshot()
            model.train()
        if step % args.save_every == 0 or step == args.iters:
            atomic_save(state(), last_path)
        if _STOP:
            atomic_save(state(), last_path)
            print(f"Saved {last_path} at step {step}; exiting for requeue.", flush=True)
            log_fh.close()
            return EXIT_REQUEUE

    log_fh.close()
    open(done, "w").close()
    print("Training finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
