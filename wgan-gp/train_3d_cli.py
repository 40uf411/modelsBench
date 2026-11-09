"""
train_3d_cli.py - command-line, resumable entry point for 3D WGAN-GP training.

Uses the unchanged models_3d.py, dataloaders_3d.py and training_3d.py.
The hyperparameter defaults are the ones hard-coded in main_3d.py, so

    python train_3d_cli.py --data_root <folder of .npy volumes> --out_dir <dir>

trains exactly the main_3d.py configuration, without editing the file
for every dataset.

Additions over main_3d.py (none change the model, loss or optimiser):
  * all paths and hyperparameters are arguments;
  * resume: if <out_dir>/checkpoints holds ckpt_epoch_XXXX.pt files, training
    continues from the latest one (models, optimisers, losses, step count);
  * on SIGUSR1/SIGTERM (Slurm time limit) the current epoch finishes, a
    checkpoint is written and the process exits with code 99, so the job
    script can requeue itself;
  * a DONE marker is written when all epochs are finished.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import sys
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloaders_3d import get_npy_volume_dataloader  # noqa: E402
from models_3d import Discriminator3D, Generator3D  # noqa: E402
from training_3d import Trainer3D  # noqa: E402

EXIT_REQUEUE = 99
CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.pt$")


class ResumableTrainer3D(Trainer3D):
    """Trainer3D with a start epoch and a graceful stop flag."""

    stop_requested = False

    def train_from(self, data_loader, start_epoch: int, epochs: int, log_path: str):
        last_saved = start_epoch
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            print(f"\nEpoch {epoch + 1}/{epochs}", flush=True)
            self._train_epoch(data_loader)

            if (epoch + 1) % self.save_png_every == 0:
                self._save_epoch_png(epoch + 1)
            if (epoch + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1)
                last_saved = epoch + 1

            with open(log_path, "a") as fh:
                fh.write(json.dumps({
                    "epoch": epoch + 1,
                    "steps": self.num_steps,
                    "D": self.losses["D"][-1] if self.losses["D"] else None,
                    "G": self.losses["G"][-1] if self.losses["G"] else None,
                    "GP": self.losses["GP"][-1] if self.losses["GP"] else None,
                    "grad_norm": self.losses["gradient_norm"][-1] if self.losses["gradient_norm"] else None,
                    "epoch_seconds": round(time.time() - t0, 2),
                }) + "\n")

            if ResumableTrainer3D.stop_requested:
                if last_saved != epoch + 1:
                    self._save_checkpoint(epoch + 1)
                print(f"Stop requested: checkpoint at epoch {epoch + 1}, exiting for requeue.", flush=True)
                return False
        return True


def _request_stop(signum, frame):
    print(f"Received signal {signum}: will stop after the current epoch.", flush=True)
    ResumableTrainer3D.stop_requested = True


def latest_checkpoint(ckpt_dir: str):
    found = []
    for path in glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.pt")):
        m = CKPT_RE.search(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return max(found) if found else None


def parse_args():
    p = argparse.ArgumentParser(description="Resumable 3D WGAN-GP training (main_3d.py defaults)")
    p.add_argument("--data_root", required=True, help="folder with one .npy volume per file, values {0,1}")
    p.add_argument("--out_dir", required=True, help="run folder; checkpoints go to <out_dir>/checkpoints")
    p.add_argument("--vol_size", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--gp_weight", type=float, default=10.0)
    p.add_argument("--critic_iterations", type=int, default=5)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--save_png_every", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    done_marker = os.path.join(args.out_dir, "TRAINING_DONE")
    if os.path.exists(done_marker):
        print(f"{done_marker} exists: nothing to do.")
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    with open(os.path.join(args.out_dir, "train_args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    loader = get_npy_volume_dataloader(args.data_root, batch_size=args.batch_size,
                                       num_workers=args.num_workers)
    print(f"Dataset: {len(loader.dataset)} volumes from {args.data_root}", flush=True)

    vol = (args.vol_size,) * 3
    G = Generator3D(vol_size=vol, latent_dim=args.latent_dim, dim=args.dim, out_channels=1)
    D = Discriminator3D(vol_size=vol, dim=args.dim, in_channels=1)
    G_opt = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    D_opt = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    start_epoch = 0
    resume = latest_checkpoint(ckpt_dir)
    if resume is not None:
        epoch, path = resume
        ck = torch.load(path, map_location="cpu", weights_only=False)
        G.load_state_dict(ck["G_state_dict"])
        D.load_state_dict(ck["D_state_dict"])
        G_opt.load_state_dict(ck["G_opt_state_dict"])
        D_opt.load_state_dict(ck["D_opt_state_dict"])
        start_epoch = int(ck["epoch"])
        print(f"Resuming from {path} (epoch {start_epoch})", flush=True)

    trainer = ResumableTrainer3D(
        G, D, G_opt, D_opt,
        gp_weight=args.gp_weight,
        critic_iterations=args.critic_iterations,
        use_cuda=use_cuda,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        save_png_every=args.save_png_every,
        png_grid_size=32,
        png_nrow=8,
        slice_axis="z",
    )
    if resume is not None:
        trainer.losses = ck.get("losses", trainer.losses)
        trainer.num_steps = int(ck.get("num_steps", 0))
        if use_cuda:  # optimiser state was loaded on CPU before G/D moved to GPU
            for opt in (G_opt, D_opt):
                for state in opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    finished = trainer.train_from(loader, start_epoch, args.epochs,
                                  log_path=os.path.join(args.out_dir, "train_log.jsonl"))
    if not finished:
        return EXIT_REQUEUE

    torch.save(trainer.G.state_dict(), os.path.join(args.out_dir, "generator_final.pt"))
    open(done_marker, "w").close()
    print("Training finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
