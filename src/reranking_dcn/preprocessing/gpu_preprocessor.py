"""GPU-accelerated feature preprocessor for training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

import numpy as np
import torch
import torch.nn as nn

from ..features import CAT_FEATURES, HASH_SEEDS, NUMERIC_FEATURES, PRETRAINED_EMB_DIM

log = logging.getLogger(__name__)


class GPUFeaturePreprocessor(nn.Module):
    """Feature preprocessor with GPU-accelerated quantile binning.

    Replaces z-score normalization with distribution-agnostic quantile
    discretization.  Each numeric feature is mapped to one of Q bins
    whose boundaries are computed from training-set quantiles.  Bin
    indices are later converted to dense vectors via a learned
    nn.Embedding inside DCNv2Ranker.

    Categorical encoding (string -> int) remains on CPU.
    """

    def __init__(
        self,
        id_hash_config: dict[str, tuple[int, int]] | None = None,
        cat_emb_dim_overrides: dict[str, int] | None = None,
        n_quantile_bins: int = 64,
    ) -> None:
        super().__init__()
        self.cat_encoders: dict[str, dict[str, int]] = {}
        self.vocab_sizes: dict[str, int] = {}
        self.embedding_dims: dict[str, int] = {}
        self._id_hash_config: dict[str, tuple[int, int]] = id_hash_config or {}
        self._cat_emb_dim_overrides: dict[str, int] = cat_emb_dim_overrides or {}
        self._n_quantile_bins = n_quantile_bins

    def fit(self, df: "pl.DataFrame") -> GPUFeaturePreprocessor:
        """Fit quantile boundaries and categorical encoders from training data."""
        import polars as pl

        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

        quantile_fracs = np.linspace(0, 1, self._n_quantile_bins + 1)[1:-1]
        boundaries_list: list[np.ndarray] = []
        for c in num_cols:
            vals = df[c].fill_null(0.0).fill_nan(0.0).cast(pl.Float32).to_numpy()
            b = np.quantile(vals, quantile_fracs).astype(np.float32)
            boundaries_list.append(b)

        boundaries = np.stack(boundaries_list)
        self.register_buffer("_quantile_boundaries", torch.from_numpy(boundaries))

        for col in CAT_FEATURES:
            if col not in df.columns:
                continue
            unique_vals = sorted(v for v in df[col].unique().to_list() if v is not None)
            self.cat_encoders[col] = {v: i + 1 for i, v in enumerate(unique_vals)}
            vocab_size = len(unique_vals) + 1
            self.vocab_sizes[col] = vocab_size
            if col in self._cat_emb_dim_overrides:
                self.embedding_dims[col] = self._cat_emb_dim_overrides[col]
            else:
                self.embedding_dims[col] = min(50, max(1, vocab_size // 2))

        for col, (n_buckets, emb_dim) in self._id_hash_config.items():
            self.vocab_sizes[col] = n_buckets + 1
            self.embedding_dims[col] = emb_dim
            log.info("  ID hash: %-25s -> %d buckets, dim %d", col, n_buckets, emb_dim)

        return self

    def quantile_encode_on_device(self, numeric: torch.Tensor) -> torch.Tensor:
        """Map raw numeric features to quantile bin indices on GPU.

        Uses torch.searchsorted with per-feature boundaries for a
        fully vectorized O(B x F x log Q) binning pass.

        Args:
            numeric: (B, F) float32 raw feature values.

        Returns:
            (B, F) int64 bin indices in [1, Q]. Index 0 is reserved for padding.
        """
        vals_t = numeric.T.contiguous()
        bins_t = torch.searchsorted(self._quantile_boundaries, vals_t)
        return (bins_t.T.contiguous() + 1).long()

    def transform(
        self,
        df: "pl.DataFrame",
        *,
        skip_targets: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Extract numeric + encoded cats + pretrained embeddings + optional target.

        Returns:
            numeric:        (N, num_features) float32 -- raw values (binning deferred to GPU).
            cat_indices:    (N, num_cat) int32 -- encoded categoricals + hashed IDs.
            pretrained_emb: (N, 2*768) float32 or None if columns absent.
            targets:        (N,) float32 (target_is_native_order) or None if skip_targets=True.
        """
        import polars as pl

        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        numeric = df.select(
            [pl.col(c).fill_null(0.0).fill_nan(0.0).cast(pl.Float32) for c in num_cols]
        ).to_numpy(allow_copy=True)

        all_cat_order = sorted(self.vocab_sizes.keys())
        cat_exprs: list[pl.Expr] = []
        for col in all_cat_order:
            if col in self.cat_encoders:
                cat_exprs.append(
                    pl.col(col).replace_strict(
                        self.cat_encoders[col], default=0, return_dtype=pl.Int32
                    )
                )
            elif col in self._id_hash_config and col in df.columns:
                n_buckets = self._id_hash_config[col][0]
                cat_exprs.append(
                    (pl.col(col).hash(**HASH_SEEDS) % n_buckets + 1).fill_null(0).cast(pl.Int32)
                )
            else:
                cat_exprs.append(pl.lit(0, dtype=pl.Int32).alias(col))

        cat_indices = (
            df.select(cat_exprs).to_numpy(allow_copy=True).astype(np.int32, copy=False)
            if cat_exprs
            else np.empty((df.shape[0], 0), dtype=np.int32)
        )

        product_emb_cols = tuple(f"product_emb_{i}" for i in range(PRETRAINED_EMB_DIM))
        user_emb_cols = tuple(f"user_emb_{i}" for i in range(PRETRAINED_EMB_DIM))
        has_product = all(c in df.columns for c in product_emb_cols)
        has_user = all(c in df.columns for c in user_emb_cols)
        if has_product and has_user:
            all_emb_cols = product_emb_cols + user_emb_cols
            pretrained_emb = df.select([pl.col(c).cast(pl.Float32) for c in all_emb_cols]).to_numpy(
                allow_copy=True
            )
        else:
            pretrained_emb = None

        if skip_targets:
            return numeric, cat_indices, pretrained_emb, None

        targets = df["target_is_native_order"].fill_null(0).cast(pl.Float32).to_numpy()
        return numeric, cat_indices, pretrained_emb, targets

    @property
    def num_continuous(self) -> int:
        return self._quantile_boundaries.shape[0]

    @property
    def n_quantile_bins(self) -> int:
        return self._n_quantile_bins

    # -- Safe serialization (replaces pickle-based checkpoint) ---------------

    def save_state(self) -> dict:
        """Serialize preprocessor state as a plain dict (no pickle).

        All values are tensors or basic Python types, safe for
        ``torch.save`` / ``torch.load(weights_only=True)``.
        """
        return {
            "quantile_boundaries": self._quantile_boundaries.cpu(),
            "cat_encoders": self.cat_encoders,
            "vocab_sizes": self.vocab_sizes,
            "embedding_dims": self.embedding_dims,
            "id_hash_config": {k: list(v) for k, v in self._id_hash_config.items()},
            "cat_emb_dim_overrides": dict(self._cat_emb_dim_overrides),
            "n_quantile_bins": self._n_quantile_bins,
        }

    @classmethod
    def from_state(cls, state: dict) -> GPUFeaturePreprocessor:
        """Reconstruct from a state dict produced by ``save_state()``.

        Avoids pickling the entire nn.Module, enabling ``weights_only=True``
        on checkpoint load and version-independent deserialization.
        """
        preprocessor = cls(
            id_hash_config={k: tuple(v) for k, v in state["id_hash_config"].items()},
            cat_emb_dim_overrides=state.get("cat_emb_dim_overrides", {}),
            n_quantile_bins=state["n_quantile_bins"],
        )
        preprocessor.register_buffer("_quantile_boundaries", state["quantile_boundaries"])
        preprocessor.cat_encoders = state["cat_encoders"]
        preprocessor.vocab_sizes = state["vocab_sizes"]
        preprocessor.embedding_dims = state["embedding_dims"]
        return preprocessor
