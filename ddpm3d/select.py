"""
Pick the best EMA snapshot with a stated criterion.

    python -m ddpm3d.select --run_dir RUN [--sampler ddim --steps 50]

Each RUN/snapshots/ema<decay>_<step>.pt (all EMA decays) generates --n_eval volumes (fixed seed)
and is scored against --n_ref training volumes with metrics.score_population
(phase-fraction Wasserstein-1 + error of the mean two-point function along
x, y, z); the same criterion is used for WGAN-GP. Selection samples with
DDIM-50 by default to keep it cheap; --sampler ddpm uses the full chain.

Writes RUN/selection/scores.csv, RUN/selection/best.json and copies the
winner to RUN/best.pt.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import torch

from .common import autocast, load_for_sampling
from .data import list_volume_files, load_labels
from .metrics import reference_from_labels, score_population


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default=None, help="reference data (default: the training data)")
    p.add_argument("--n_ref", type=int, default=256)
    p.add_argument("--n_eval", type=int, default=32)
    p.add_argument("--min_step", type=int, default=0)
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max_lag", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--bf16", type=int, default=1)
    return p.parse_args(argv)


@torch.no_grad()
def generate_labels(model, diffusion, codec, shape, n, batch, device, sampler, steps, seed, bf16):
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for s in range(0, n, batch):
        b = min(batch, n - s)
        with autocast(device, bf16):
            if sampler == "ddim":
                x = diffusion.ddim_sample(model, (b, codec.channels, *shape), device, steps=steps, generator=g)
            else:
                x = diffusion.ddpm_sample(model, (b, codec.channels, *shape), device, generator=g)
        out.append(codec.decode(x.float()))
    return torch.cat(out)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snaps = sorted(glob.glob(os.path.join(args.run_dir, "snapshots", "ema*.pt")))
    step_of = lambda p: int(re.search(r"_(\d+)\.pt$", os.path.basename(p)).group(1))
    snaps = [s for s in snaps if step_of(s) >= args.min_step]
    if not snaps:
        print("no snapshots found", file=sys.stderr)
        return 1

    with open(os.path.join(args.run_dir, "config.json")) as fh:
        cfg = json.load(fh)
    data_dir = args.data_dir or cfg["data"]["data_dir"]
    values = cfg["data"]["values"]
    files = list_volume_files(data_dir)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), size=min(args.n_ref, len(files)), replace=False)
    ref_labels, _ = load_labels([files[i] for i in pick], values=values)
    k = len(values)
    ref = reference_from_labels(torch.from_numpy(ref_labels).to(device), k, args.max_lag)
    shape = tuple(cfg["data"]["shape"])

    rows = []
    for path in snaps:
        model, diffusion, codec, ck = load_for_sampling(path, device)
        lab = generate_labels(model, diffusion, codec, shape, args.n_eval, args.batch, device,
                              args.sampler, min(args.steps, diffusion.T), args.seed, bool(args.bf16))
        sc = score_population(lab, ref)
        sc.update({"step": ck["step"], "ema_decay": ck.get("ema_decay"), "snapshot": os.path.basename(path)})
        rows.append(sc)
        print(f"step {ck['step']:7d}  ema {ck.get('ema_decay')}  score {sc['score']:.4f}  frac {sc['frac_w1_norm']:.4f}  "
              f"s2 {sc['s2_rel_l2']:.4f}", flush=True)

    out = os.path.join(args.run_dir, "selection")
    os.makedirs(out, exist_ok=True)
    keys = ["step", "ema_decay", "snapshot", "score", "frac_w1_norm", "s2_rel_l2"] + \
        sorted(x for x in rows[0] if x.startswith("frac") and x != "frac_w1_norm")
    with open(os.path.join(out, "scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    best = min(rows, key=lambda r: r["score"])
    with open(os.path.join(out, "best.json"), "w") as fh:
        json.dump({**best, "criterion": "min(frac_w1_norm + s2_rel_l2)", "sampler": args.sampler,
                   "steps": args.steps, "n_eval": args.n_eval, "n_ref": int(len(pick))}, fh, indent=2)
    shutil.copyfile(os.path.join(args.run_dir, "snapshots", best["snapshot"]),
                    os.path.join(args.run_dir, "best.pt"))
    print(f"Best snapshot: {best['snapshot']} (score {best['score']:.4f}) -> best.pt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
