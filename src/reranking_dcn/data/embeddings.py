"""Vectorised embedding lookup for Nomic product and user-average embeddings.

Uses sorted-array + ``np.searchsorted`` for O(N log M) lookup per chunk,
avoiding per-chunk Polars LEFT JOINs that would rebuild a hash table each time.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from ..features import PRETRAINED_EMB_DIM
from .sql import USER_DATE_KEY_SCALE

log = logging.getLogger(__name__)


class EmbeddingLookup:
    """Fast vectorised embedding lookup via sorted binary search."""

    def __init__(
        self,
        ids: np.ndarray,
        embeddings: np.ndarray,
        *,
        presorted: bool = False,
    ) -> None:
        if presorted:
            self._sorted_ids = ids
            self._sorted_emb = (
                embeddings if embeddings.dtype == np.float16 else embeddings.astype(np.float16)
            )
        else:
            order = np.argsort(ids)
            self._sorted_ids = ids[order]
            self._sorted_emb = embeddings[order].astype(np.float16, copy=False)
        self.emb_dim = embeddings.shape[1]

    def lookup(self, query_ids: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Return ``(len(query_ids), emb_dim)`` float16 array; zeros for missing IDs."""
        n = len(query_ids)
        if out is not None:
            assert out.shape == (n, self.emb_dim)
            out[:] = 0.0
            result = out
        else:
            result = np.zeros((n, self.emb_dim), dtype=np.float16)

        sort_idx = np.argsort(query_ids)
        sorted_queries = query_ids[sort_idx]

        positions = np.searchsorted(self._sorted_ids, sorted_queries)
        positions = np.clip(positions, 0, len(self._sorted_ids) - 1)
        valid_mask = self._sorted_ids[positions] == sorted_queries

        gathered = self._sorted_emb[positions[valid_mask]]
        result[sort_idx[valid_mask]] = gathered
        return result

    @classmethod
    def from_parquet(
        cls,
        local_dir: Path,
        split_name: str,
        id_col: str,
        emb_cols: tuple[str, ...],
    ) -> EmbeddingLookup:
        from .exporter import Exporter

        t0 = time.monotonic()
        df = Exporter.load(local_dir, split_name).sort(id_col)
        ids = df[id_col].to_numpy()
        emb = df.select(list(emb_cols)).to_numpy().astype(np.float16, copy=False)
        log.info(
            "  EmbeddingLookup(%s): %s IDs, dim=%d (sort+convert %.1fs)",
            id_col,
            f"{len(ids):,}",
            emb.shape[1],
            time.monotonic() - t0,
        )
        del df
        return cls(ids, emb, presorted=True)


def user_date_key(df: pl.DataFrame) -> np.ndarray:
    """Composite key: ``MOD(ABS(FARM_FINGERPRINT(unified_buyer_id)), 9e12) * 100_000 + UNIX_DATE(day)``.

    The ``ubi_hash`` column is pre-computed by BigQuery using
    ``MOD(ABS(FARM_FINGERPRINT(unified_buyer_id)), _UBI_HASH_MOD)`` during
    training data export to avoid INT64 overflow when multiplied by
    USER_DATE_KEY_SCALE.
    """
    hashes = df["ubi_hash"].fill_null(0).cast(pl.Int64).to_numpy()
    days = (
        df["event_timestamp_day"]
        .cast(pl.Date)
        .cast(pl.Int32)
        .fill_null(0)
        .to_numpy()
        .astype(np.int64)
    )
    return hashes * USER_DATE_KEY_SCALE + days


def load_embedding_lookups(
    local_dir: Path,
) -> tuple[EmbeddingLookup | None, EmbeddingLookup | None]:
    """Load product and user embedding lookups from exported Parquet."""
    product_cols = tuple(f"product_emb_{i}" for i in range(PRETRAINED_EMB_DIM))
    user_cols = tuple(f"user_emb_{i}" for i in range(PRETRAINED_EMB_DIM))

    product_lookup: EmbeddingLookup | None = None
    user_lookup: EmbeddingLookup | None = None

    try:
        product_lookup = EmbeddingLookup.from_parquet(
            local_dir, "product_embeddings", "product_id", product_cols
        )
    except FileNotFoundError:
        log.warning("Product embeddings not found — skipping")

    try:
        user_lookup = EmbeddingLookup.from_parquet(
            local_dir, "user_embeddings", "user_date_key", user_cols
        )
    except FileNotFoundError:
        log.warning("User embeddings not found — skipping")

    return product_lookup, user_lookup


def lookup_pretrained_embeddings(
    df: pl.DataFrame,
    product_lookup: EmbeddingLookup | None,
    user_lookup: EmbeddingLookup | None,
) -> np.ndarray | None:
    """Build combined ``(N, product_dim + user_dim)`` float16 embedding array."""
    if product_lookup is None and user_lookup is None:
        return None

    p_dim = product_lookup.emb_dim if product_lookup else 0
    u_dim = user_lookup.emb_dim if user_lookup else 0
    total_dim = p_dim + u_dim
    if total_dim == 0:
        return None

    n = df.shape[0]
    t0 = time.monotonic()
    result = np.zeros((n, total_dim), dtype=np.float16)

    def _do_product() -> None:
        pids = df["product_id"].fill_null(0).to_numpy()
        product_lookup.lookup(pids, out=result[:, :p_dim])  # type: ignore[union-attr]

    def _do_user() -> None:
        keys = user_date_key(df)
        user_lookup.lookup(keys, out=result[:, p_dim : p_dim + u_dim])  # type: ignore[union-attr]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        if product_lookup is not None:
            futures.append(pool.submit(_do_product))
        if user_lookup is not None:
            futures.append(pool.submit(_do_user))
        for f in futures:
            f.result()

    log.info(
        "  Pretrained embeddings: %s rows x %d dim (%.1fs)",
        f"{n:,}",
        total_dim,
        time.monotonic() - t0,
    )
    return result
