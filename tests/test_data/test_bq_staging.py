"""Tests for BigQuery staging manager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from reranking_dcn.data.bq_staging import BQStagingManager
from reranking_dcn.features import CAT_FEATURES, NUMERIC_FEATURES


def _write_minimal_parquet(local_dir: Path, split: str, n_rows: int = 20) -> None:
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "product_id": rng.integers(1, 1000, size=n_rows),
            "unified_buyer_id": [f"ubid_{i}" for i in range(n_rows)],
            "event_timestamp_day": pl.Series(["2026-03-24"] * n_rows).str.to_date(),
        }
    )
    split_dir = local_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(split_dir / "shard_000.parquet")


class TestBQStagingExtractIds:
    def test_extracts_unique_product_ids(self):
        from unittest.mock import MagicMock

        manager = BQStagingManager(MagicMock(), "test-project")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            _write_minimal_parquet(local_dir, "train", n_rows=50)

            product_ids, buyer_days = manager.extract_ids_from_parquet(local_dir, "train")

            assert len(product_ids) > 0
            assert len(product_ids) == len(set(product_ids))
            assert isinstance(buyer_days, pl.DataFrame)
            assert "unified_buyer_id" in buyer_days.columns
            assert "event_timestamp_day" in buyer_days.columns

    def test_raises_on_missing_split(self):
        from unittest.mock import MagicMock

        manager = BQStagingManager(MagicMock(), "test-project")

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                manager.extract_ids_from_parquet(Path(tmpdir), "nonexistent")
