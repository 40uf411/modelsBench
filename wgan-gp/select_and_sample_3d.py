"""
select_and_sample_3d.py - pick the best WGAN-GP checkpoint with a stated
criterion, then generate the evaluation samples from it.

Selection replaces visual inspection. For every checkpoint
<run_dir>/checkpoints/ckpt_epoch_XXXX.pt (or every k-th one) it generates
--n_eval volumes, thresholds them at 0 and scores them against --n_ref
training volumes with microstats.score_population (phase-fraction
Wasserstein-1 + error of the mean two-point function along x, y and z).
The lowest score wins. All scores go to <run_dir>/selection/scores.csv.

The winner then generates --n_samples volumes (fixed seed):
  samples/raw/samples_raw.npy      float16, (N, D, H, W), tanh output
  samples/bin/sample_XXX.npy       uint8 {0,1}, same convention as the data
  samples/bin_median{k}/...        optional median-filtered copies (--median 3)
  samples/timing.json              wall time of generation only, GPU name

Example
  python select_and_sample_3d.py --run_dir RUN --data_root DATA --n_samples 150 --median 3
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microstats import load_label_volumes, reference_from_labels, score_population  # noqa: E402
from models_3d import Generator3D  # noqa: E402

CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pt$")


def list_checkpoints(ckpt_dir):
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.pt")):
        m = CKPT_RE.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def build_generator(ck, device):
    G = Generator3D(vol_size=ck["vol_size"], latent_dim=ck["latent_dim"], dim=ck["dim"], out_channels=1)
    G.load_state_dict(ck["G_state_dict"])
    return G.to(device).eval()


@torch.no_grad()
def generate(G, n, batch, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    outs = []
    for s in range(0, n, batch):
        b = min(batch, n - s)
        z = torch.randn(b, G.latent_dim, device=device, generator=gen)
        outs.append(G(z)[:, 0].float())
    return torch.cat(outs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_root", required=True, help="training volumes, used as the reference")
    p.add_argument("--n_ref", type=int, default=256)
    p.add_argument("--n_eval", type=int, default=64)
    p.add_argument("--every", type=int, default=1, help="score every k-th checkpoint")
    p.add_argument("--min_epoch", type=int, default=0, help="ignore checkpoints before this epoch")
    p.add_argument("--n_samples", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--median", type=int, nargs="*", default=[3],
                   help="also save median-filtered copies with these kernel sizes ([] for none)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_lag", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = list_checkpoints(os.path.join(args.run_dir, "checkpoints"))
    ckpts = [c for c in ckpts if c[0] >= args.min_epoch][:: max(1, args.every)]
    if not ckpts:
        raise SystemExit("no checkpoints found")

    files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.npy"), recursive=True))
    rng = np.random.default_rng(0)
    ref_files = [files[i] for i in rng.choice(len(files), size=min(args.n_ref, len(files)), replace=False)]
    ref_labels, values = load_label_volumes(ref_files)
    if values != [0, 1]:
        print(f"WARNING: reference values are {values}; generated class 1 is mapped to value {values[-1]}")
    ref = reference_from_labels(torch.from_numpy(ref_labels).to(device), 2, args.max_lag)

    sel_dir = os.path.join(args.run_dir, "selection")
    os.makedirs(sel_dir, exist_ok=True)
    rows = []
    for epoch, path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        G = build_generator(ck, device)
        vols = generate(G, args.n_eval, args.batch, device, seed=args.seed)
        labels = (vols > 0).to(torch.uint8)
        sc = score_population(labels, ref)
        sc.update({"epoch": epoch, "checkpoint": os.path.basename(path)})
        rows.append(sc)
        print(f"epoch {epoch:5d}  score {sc['score']:.4f}  frac {sc['frac_w1_norm']:.4f}  s2 {sc['s2_rel_l2']:.4f}", flush=True)

    keys = ["epoch", "checkpoint", "score", "frac_w1_norm", "s2_rel_l2"] + \
           sorted(k for k in rows[0] if k.startswith("frac") and k != "frac_w1_norm")
    with open(os.path.join(sel_dir, "scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    best = min(rows, key=lambda r: r["score"])
    with open(os.path.join(sel_dir, "best.json"), "w") as fh:
        json.dump({**best, "criterion": "min(frac_w1_norm + s2_rel_l2)", "n_eval": args.n_eval,
                   "n_ref": len(ref_files)}, fh, indent=2)
    print(f"Best checkpoint: {best['checkpoint']} (score {best['score']:.4f})", flush=True)

    # ---- final samples from the best checkpoint ----------------------------
    ck = torch.load(os.path.join(args.run_dir, "checkpoints", best["checkpoint"]),
                    map_location="cpu", weights_only=False)
    G = build_generator(ck, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    vols = generate(G, args.n_samples, args.batch, device, seed=args.seed + 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gen_seconds = time.time() - t0

    out = os.path.join(args.run_dir, "samples")
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(out, "bin"), exist_ok=True)
    raw = vols.cpu().numpy()
    np.save(os.path.join(out, "raw", "samples_raw.npy"), raw.astype(np.float16))
    lab = (raw > 0).astype(np.uint8)
    lab_values = np.asarray(values, dtype=np.uint8)[lab] if len(values) == 2 else lab
    for i, v in enumerate(lab_values):
        np.save(os.path.join(out, "bin", f"sample_{i:03d}.npy"), v)

    if args.median:
        from scipy.ndimage import median_filter
        for k in args.median:
            d = os.path.join(out, f"bin_median{k}")
            os.makedirs(d, exist_ok=True)
            for i, v in enumerate(lab_values):
                np.save(os.path.join(d, f"sample_{i:03d}.npy"), median_filter(v, size=k))

    with open(os.path.join(out, "timing.json"), "w") as fh:
        json.dump({
            "checkpoint": best["checkpoint"],
            "n_samples": args.n_samples,
            "batch": args.batch,
            "generation_seconds": round(gen_seconds, 3),
            "seconds_per_100": round(100 * gen_seconds / args.n_samples, 3),
            "network_evaluations_per_sample": 1,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        }, fh, indent=2)
    print(f"Saved {args.n_samples} samples to {out} ({gen_seconds:.1f} s generation)", flush=True)


if __name__ == "__main__":
    main()
