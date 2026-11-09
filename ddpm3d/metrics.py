"""
metrics.py - cheap population statistics for checkpoint selection
(same as wgan-gp/microstats.py, so both models are selected by one criterion).

Used to pick a checkpoint with a stated, automatic criterion instead of
visual inspection. For every phase k the score compares a generated
population with a reference population (training crops) on

  * the phase-fraction distribution: Wasserstein-1 distance divided by the
    reference standard deviation, and
  * the mean two-point probability function S2_k(r) along each axis,
    r = 1..max_lag: relative L2 error of the mean curve.

score = mean_k [ W1_k / std_ref_k ] + mean_k,axis [ relL2(S2_k,axis) ]

Lower is better. Everything runs in torch on the given device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch


# ----------------------------------------------------------------------------
# basic statistics
# ----------------------------------------------------------------------------
def phase_fractions(labels: torch.Tensor, num_phases: int) -> torch.Tensor:
    """labels: (N, D, H, W) integer class indices -> (N, K) fractions."""
    n = labels.shape[0]
    flat = labels.reshape(n, -1)
    out = torch.stack([(flat == k).float().mean(dim=1) for k in range(num_phases)], dim=1)
    return out


def s2_curves(labels: torch.Tensor, num_phases: int, max_lag: int = 32,
              chunk: int = 16) -> torch.Tensor:
    """
    Two-point probability S2_k(r) per sample, phase and axis (non-periodic
    estimator). labels: (N, D, H, W). Returns (N, K, 3, max_lag + 1).
    """
    n = labels.shape[0]
    res = torch.zeros(n, num_phases, 3, max_lag + 1, device=labels.device)
    for s in range(0, n, chunk):
        lab = labels[s:s + chunk]
        for k in range(num_phases):
            f = (lab == k).float()
            for ax in range(3):
                dim = ax + 1
                length = f.shape[dim]
                lags = min(max_lag, length - 1)
                for r in range(lags + 1):
                    a = f.narrow(dim, 0, length - r)
                    b = f.narrow(dim, r, length - r)
                    res[s:s + chunk, k, ax, r] = (a * b).reshape(a.shape[0], -1).mean(dim=1)
    return res


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein-1 distance between two empirical samples."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    qs = np.linspace(0.0, 1.0, 1001)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


# ----------------------------------------------------------------------------
# reference + scoring
# ----------------------------------------------------------------------------
@dataclass
class ReferenceStats:
    fractions: np.ndarray        # (N_ref, K)
    s2_mean: torch.Tensor        # (K, 3, R+1)
    num_phases: int
    max_lag: int


def reference_from_labels(labels: torch.Tensor, num_phases: int,
                          max_lag: int = 32) -> ReferenceStats:
    fr = phase_fractions(labels, num_phases).cpu().numpy()
    s2 = s2_curves(labels, num_phases, max_lag).mean(dim=0)
    return ReferenceStats(fr, s2, num_phases, max_lag)


def score_population(labels: torch.Tensor, ref: ReferenceStats) -> Dict[str, float]:
    """Score a generated population (N, D, H, W) of class indices."""
    k_ = ref.num_phases
    fr = phase_fractions(labels, k_).cpu().numpy()
    s2 = s2_curves(labels, k_, ref.max_lag).mean(dim=0).to(ref.s2_mean.device)

    frac_terms = []
    for k in range(k_):
        sd = max(float(ref.fractions[:, k].std()), 1e-2)  # floor: 1% phase fraction
        frac_terms.append(wasserstein1(fr[:, k], ref.fractions[:, k]) / sd)

    s2_terms = []
    for k in range(k_):
        for ax in range(3):
            g = s2[k, ax, 1:]
            r = ref.s2_mean[k, ax, 1:]
            s2_terms.append(float(torch.linalg.norm(g - r) / torch.linalg.norm(r).clamp_min(1e-8)))

    out = {
        "frac_w1_norm": float(np.mean(frac_terms)),
        "s2_rel_l2": float(np.mean(s2_terms)),
    }
    for k in range(k_):
        out[f"frac{k}_mean"] = float(fr[:, k].mean())
        out[f"frac{k}_ref_mean"] = float(ref.fractions[:, k].mean())
    out["score"] = out["frac_w1_norm"] + out["s2_rel_l2"]
    return out


# ----------------------------------------------------------------------------
# loading helpers
# ----------------------------------------------------------------------------
def load_label_volumes(files: Sequence[str], values: Optional[Sequence[float]] = None):
    """
    Load .npy volumes and map their values to class indices 0..K-1.
    Returns (labels uint8 array (N, D, H, W), sorted value list).
    """
    vols = []
    for f in files:
        a = np.load(f)
        a = np.squeeze(a)
        if a.ndim != 3:
            raise ValueError(f"{f}: expected one 3D volume, got shape {a.shape}")
        vols.append(a)
    arr = np.stack(vols)
    if values is None:
        values = np.unique(arr).tolist()
    if len(values) > 16:
        raise ValueError(f"data looks continuous ({len(values)} distinct values); expected labels")
    lut = {v: i for i, v in enumerate(values)}
    labels = np.zeros(arr.shape, dtype=np.uint8)
    for v, i in lut.items():
        labels[arr == v] = i
    return labels, list(values)
