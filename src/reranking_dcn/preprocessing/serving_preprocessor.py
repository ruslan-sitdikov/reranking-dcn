"""CPU-only serving-time preprocessor that replicates training transforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..features import HASH_SEEDS
from .quantile import quantile_encode_np

log = logging.getLogger(__name__)


class ServingPreprocessor:
    """CPU-only preprocessing that replicates GPUFeaturePreprocessor.transform().

    Loads serialized artifacts and applies:
    1. Quantile binning (np.searchsorted on boundaries)
    2. Categorical encoding (dict lookup)
    3. ID hashing (polars hash % n_buckets + 1)
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

        boundaries_path = artifacts_dir / "quantile_boundaries.npy"
        if boundaries_path.exists():
            self._quantile_boundaries = np.load(boundaries_path)
            log.info("  Loaded quantile_boundaries: %s", self._quantile_boundaries.shape)
        else:
            self._quantile_boundaries = None
            log.warning("  No quantile_boundaries found")

        with open(artifacts_dir / "cat_encoders.json") as f:
            self.cat_encoders = json.load(f)

        with open(artifacts_dir / "id_hash_config.json") as f:
            raw = json.load(f)
            self.id_hash_config = {col: tuple(v) for col, v in raw.items()}

        with open(artifacts_dir / "model_config.json") as f:
            self.model_config = json.load(f)

        self._cat_feature_order = self.model_config["cat_feature_order"]

    def quantile_encode(self, numeric: np.ndarray) -> np.ndarray:
        """(N, 185) float32 -> (N, 185) int64 bin indices (1-based, 0=padding)."""
        if self._quantile_boundaries is None:
            raise RuntimeError("Quantile boundaries not loaded")
        return quantile_encode_np(numeric, self._quantile_boundaries)

    def encode_categoricals(self, cat_values: dict[str, list]) -> np.ndarray:
        """Encode categorical string values to integer indices.

        Uses Polars hash (with the same seeds as training) for ID columns
        to ensure parity with GPUFeaturePreprocessor.transform().

        Args:
            cat_values: {feature_name: [values for each row]}

        Returns:
            (N, N_cat) int64 array of encoded indices.
        """
        import polars as pl

        n_rows = len(next(iter(cat_values.values())))
        n_cat = len(self._cat_feature_order)
        encoded = np.zeros((n_rows, n_cat), dtype=np.int64)

        for i, col in enumerate(self._cat_feature_order):
            if col in self.id_hash_config:
                n_buckets, _ = self.id_hash_config[col]
                values = cat_values.get(col, [None] * n_rows)
                series = pl.Series(col, values)
                hashed = series.hash(**HASH_SEEDS)
                encoded[:, i] = (hashed % n_buckets + 1).fill_null(0).cast(pl.Int64).to_numpy()
            else:
                encoder = self.cat_encoders.get(col, {})
                vals = cat_values.get(col, ["__MISSING__"] * n_rows)
                encoded[:, i] = np.fromiter(
                    (encoder.get(str(v), 0) for v in vals), dtype=np.int64, count=n_rows
                )

        return encoded
