"""PyTorch Dataset and DataLoader factory for pointwise DCN-v2 training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PointwiseDataset(Dataset):
    """Row-level dataset for pointwise BCE / focal-loss training.

    Each ``__getitem__`` returns ``(numeric, cat_indices, target)`` and
    optionally ``pretrained_emb`` when available.
    Target: ``target_is_native_order`` (1D float32 array).
    """

    def __init__(
        self,
        numeric: np.ndarray,
        cat_indices: np.ndarray,
        targets: np.ndarray,
        pretrained_emb: np.ndarray | None = None,
    ) -> None:
        self.numeric = torch.as_tensor(numeric, dtype=torch.float32)
        self.cat_indices = torch.as_tensor(cat_indices, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.pretrained_emb = (
            torch.as_tensor(pretrained_emb, dtype=torch.float32)
            if pretrained_emb is not None
            else None
        )

    def __len__(self) -> int:
        return self.numeric.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        items = [self.numeric[idx], self.cat_indices[idx], self.targets[idx]]
        if self.pretrained_emb is not None:
            items.append(self.pretrained_emb[idx])
        return tuple(items)


def make_dataloader(
    numeric: np.ndarray,
    cat_indices: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 16,
    use_cuda: bool = True,
    pretrained_emb: np.ndarray | None = None,
) -> DataLoader:
    """Build a DataLoader for pointwise (item-level) training."""
    ds = PointwiseDataset(numeric, cat_indices, targets, pretrained_emb)
    nw = num_workers if use_cuda else 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=nw > 0,
        drop_last=False,
    )
