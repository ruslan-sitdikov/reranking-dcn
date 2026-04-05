"""Quantile binning utilities shared by training and serving."""

from __future__ import annotations

import numpy as np

_QUANTILE_BROADCAST_THRESHOLD = 32


def quantile_encode_np(numeric: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Vectorized CPU quantile binning shared by training and serving.

    For small batches (N <= 32), uses a fully-vectorized broadcasting
    comparison that eliminates the Python loop over features (18x faster
    at B=1). For larger batches, falls back to per-feature searchsorted
    to avoid the O(N*F*Q) intermediate allocation.

    Args:
        numeric: (N, F) float32 raw feature values.
        boundaries: (F, Q-1) float32 quantile bin edges.

    Returns:
        (N, F) int64 bin indices in [1, Q]. Index 0 is reserved for padding.
    """
    if numeric.shape[0] <= _QUANTILE_BROADCAST_THRESHOLD:
        return (
            (numeric[:, :, np.newaxis] > boundaries[np.newaxis, :, :]).sum(axis=2).astype(np.int64)
        ) + 1

    n_features = boundaries.shape[0]
    n_rows = numeric.shape[0]
    bins = np.empty((n_rows, n_features), dtype=np.int64)
    for i in range(n_features):
        bins[:, i] = np.searchsorted(boundaries[i], numeric[:, i]) + 1
    return bins
