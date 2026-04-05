"""Tests for ServingPreprocessor hash parity with training."""

from __future__ import annotations

import numpy as np

from conftest import make_dummy_preprocessor
from reranking_dcn import Config, HASH_SEEDS, NUMERIC_FEATURES, ServingPreprocessor
from reranking_dcn.serving.artifacts import serialize_preprocessing_artifacts


class TestServingPreprocessorHashParity:
    """Verify ServingPreprocessor uses the same hash as training."""

    def test_id_hash_matches_training(self, tmp_path):
        import polars as pl

        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        serialize_preprocessing_artifacts(prep, cfg, tmp_path)
        serving_prep = ServingPreprocessor(tmp_path)

        test_ids = [12345, 67890, 99999, None]
        series = pl.Series("broker_id", test_ids)
        n_buckets = cfg.id_hash_config["broker_id"][0]

        training_hashed = (
            (series.hash(**HASH_SEEDS) % n_buckets + 1).fill_null(0).cast(pl.Int64).to_numpy()
        )

        serving_encoded = serving_prep.encode_categoricals({"broker_id": test_ids})
        broker_col_idx = serving_prep._cat_feature_order.index("broker_id")
        serving_hashed = serving_encoded[:, broker_col_idx]

        np.testing.assert_array_equal(
            training_hashed,
            serving_hashed,
            err_msg="Serving hash must match training Polars hash",
        )

    def test_null_id_hashed_consistently(self, tmp_path):
        """Null IDs get hashed to a consistent bucket (Polars hashes nulls to non-null)."""
        import polars as pl

        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        serialize_preprocessing_artifacts(prep, cfg, tmp_path)
        serving_prep = ServingPreprocessor(tmp_path)
        n_buckets = cfg.id_hash_config["broker_id"][0]

        training_ref = (
            (pl.Series("broker_id", [None]).hash(**HASH_SEEDS) % n_buckets + 1)
            .fill_null(0)
            .cast(pl.Int64)
            .to_numpy()
        )

        encoded = serving_prep.encode_categoricals({"broker_id": [None]})
        broker_col_idx = serving_prep._cat_feature_order.index("broker_id")
        assert encoded[0, broker_col_idx] == training_ref[0]

    def test_roundtrip_quantile_encode(self, tmp_path):
        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        serialize_preprocessing_artifacts(prep, cfg, tmp_path)

        serving_prep = ServingPreprocessor(tmp_path)
        data = np.random.randn(8, len(NUMERIC_FEATURES)).astype(np.float32)
        bins = serving_prep.quantile_encode(data)

        assert bins.shape == (8, len(NUMERIC_FEATURES))
        assert bins.dtype == np.int64
        assert (bins >= 1).all()
