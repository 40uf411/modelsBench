# dataloaders_3d.py
from __future__ import annotations

import os
import glob
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NPYVolumeFolderDataset(Dataset):
    """
    Minimal 3D NPY loader.

    Assumptions:
      - Each file is a single sample
      - Data is already binary {0,1}
      - Shape is already either (D,H,W) or (1,D,H,W)
      - No transforms, no resizing, no cropping, no transposing
    Returns:
      - x: torch.float32 of shape (1, D, H, W)
      - dummy label: 0 (to match existing training loop that uses data[0])
    """
    def __init__(self, root: str, recursive: bool = True, map_to_tanh: bool = True):
        pattern = "**/*.npy" if recursive else "*.npy"
        self.files: List[str] = sorted(glob.glob(os.path.join(root, pattern), recursive=recursive))
        if not self.files:
            raise RuntimeError(f"No .npy files found in: {root}")
        self.map_to_tanh = map_to_tanh

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path)  # expects already 0/1
        x = torch.from_numpy(arr).to(torch.float32)

        # Ensure exact shape (1, D, H, W)
        if x.ndim == 4:
            # accept (1, D, H, W) only
            if x.shape[0] != 1:
                raise ValueError(f"{path}: expected shape (1,D,H,W) or (D,H,W), got {tuple(x.shape)}")
        elif x.ndim == 3:
            x = x.unsqueeze(0)  # (1, D, H, W)
        else:
            raise ValueError(f"{path}: expected 3D or 4D array, got {x.ndim}D with shape {tuple(x.shape)}")

        # Optional: match Generator tanh output range [-1,1]
        if self.map_to_tanh:
            x = x * 2.0 - 1.0

        return x, 0


def get_npy_volume_dataloader(
    root_dir: str,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    recursive: bool = True,
    map_to_tanh: bool = True,
) -> DataLoader:
    ds = NPYVolumeFolderDataset(root=root_dir, recursive=recursive, map_to_tanh=map_to_tanh)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=True
    )

