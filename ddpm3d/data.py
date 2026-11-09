"""
Volume data for 3D DDPM training.

All volumes of a dataset are loaded once, converted to class indices
(uint8) and kept on the training device, so batches are drawn without a
DataLoader. 2,500 volumes of 64^3 take about 0.65 GB.

Encodings
  scalar : one channel; class k of K maps to the level -1 + 2k/(K-1)
           (binary: -1/+1). Decoded by nearest level.
  onehot : K channels in {-1, +1}. Decoded by argmax. Avoids imposing an
           order on the phases (e.g. pore < clay < mineral), so it is the
           default for K > 2.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

import numpy as np
import torch


def list_volume_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .npy files under {data_dir}")
    return files


def load_labels(files: Sequence[str], values: Optional[Sequence[float]] = None):
    """Load volumes (one per file) and map values to class indices."""
    vols = []
    for f in files:
        a = np.squeeze(np.load(f, allow_pickle=False))
        if a.ndim != 3:
            raise ValueError(f"{f}: expected one 3D volume, got shape {a.shape}")
        vols.append(a)
    shapes = {v.shape for v in vols}
    if len(shapes) != 1:
        raise ValueError(f"volumes have different shapes: {sorted(shapes)[:5]}")
    arr = np.stack(vols)
    if values is None:
        values = np.unique(arr).tolist()
    if len(values) > 16:
        raise ValueError(f"{len(values)} distinct values: data look continuous, expected segmented labels")
    labels = np.full(arr.shape, 255, dtype=np.uint8)
    for i, v in enumerate(values):
        labels[arr == v] = i
    if (labels == 255).any():
        raise ValueError("data contain values outside the label set")
    return labels, [float(v) for v in values]


class PhaseCodec:
    """Maps class indices <-> network tensors."""

    def __init__(self, num_phases: int, encoding: str = "auto"):
        if num_phases < 2:
            raise ValueError("need at least two phases")
        if encoding == "auto":
            encoding = "scalar" if num_phases == 2 else "onehot"
        if encoding not in ("scalar", "onehot"):
            raise ValueError(f"unknown encoding {encoding}")
        self.num_phases = num_phases
        self.encoding = encoding

    @property
    def channels(self) -> int:
        return 1 if self.encoding == "scalar" else self.num_phases

    def levels(self, device=None) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, self.num_phases, device=device)

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        """(B, D, H, W) integer -> (B, C, D, H, W) float32 in [-1, 1]."""
        labels = labels.long()
        if self.encoding == "scalar":
            return self.levels(labels.device)[labels].unsqueeze(1)
        oh = torch.nn.functional.one_hot(labels, self.num_phases)  # (B, D, H, W, K)
        return (oh.permute(0, 4, 1, 2, 3).float() * 2.0 - 1.0)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) float -> (B, D, H, W) uint8 class indices."""
        if self.encoding == "scalar":
            lv = self.levels(x.device).view(1, -1, 1, 1, 1)
            return (x[:, :1] - lv).abs().argmin(dim=1).to(torch.uint8)
        return x.argmax(dim=1).to(torch.uint8)


class VolumeBank:
    """All training volumes on one device, with random batches and flips."""

    def __init__(self, data_dir: str, encoding: str = "auto", device="cpu",
                 max_files: Optional[int] = None):
        files = list_volume_files(data_dir)
        if max_files:
            files = files[:max_files]
        labels, values = load_labels(files)
        self.files = files
        self.values = values
        self.codec = PhaseCodec(len(values), encoding)
        self.labels = torch.from_numpy(labels).to(device)
        self.shape = tuple(labels.shape[1:])
        self.device = torch.device(device)

    def __len__(self):
        return self.labels.shape[0]

    def phase_fractions(self) -> np.ndarray:
        lab = self.labels
        return np.array([(lab == k).float().mean().item() for k in range(self.codec.num_phases)])

    def sample_labels(self, batch_size: int, generator: torch.Generator,
                      flips: bool = True, permute_axes: bool = False) -> torch.Tensor:
        idx = torch.randint(len(self), (batch_size,), device=self.device, generator=generator)
        lab = self.labels[idx]
        if flips:
            for dim in (1, 2, 3):
                m = torch.rand(batch_size, device=self.device, generator=generator) < 0.5
                if m.any():
                    lab[m] = lab[m].flip(dims=(dim,))
        if permute_axes:
            if len(set(self.shape)) != 1:
                raise ValueError("axis permutations need cubic volumes")
            perm = torch.randperm(3, generator=generator, device=self.device).tolist()
            lab = lab.permute(0, *(p + 1 for p in perm)).contiguous()
        return lab

    def sample_batch(self, batch_size: int, generator: torch.Generator,
                     flips: bool = True, permute_axes: bool = False) -> torch.Tensor:
        return self.codec.encode(self.sample_labels(batch_size, generator, flips, permute_axes))
