"""Tests for GPU feature preprocessor."""

from __future__ import annotations

import numpy as np
import torch

from reranking_dcn import (
    CAT_FEATURES,
    Config,
    GPUFeaturePreprocessor,
    NUMERIC_FEATURES,
)


class TestGPUFeaturePreprocessor:
    def test_quantile_encode_on_device(self):
        prep = GPUFeaturePreprocessor(n_quantile_bins=4)
        boundaries = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        prep.register_buffer("_quantile_boundaries", boundaries)

        data = torch.tensor([[0.5, 5.0], [1.5, 15.0]])
        bins = prep.quantile_encode_on_device(data)

        assert bins.shape == (2, 2)
        assert bins.dtype == torch.int64
        assert bins[0, 0].item() == 1
        assert bins[0, 1].item() == 1

    def test_num_continuous_property(self):
        prep = GPUFeaturePreprocessor(n_quantile_bins=4)
        boundaries = torch.randn(10, 3)
        prep.register_buffer("_quantile_boundaries", boundaries)
        assert prep.num_continuous == 10

    def test_n_quantile_bins_property(self):
        prep = GPUFeaturePreprocessor(n_quantile_bins=64)
        assert prep.n_quantile_bins == 64

    def test_id_hash_config_default(self):
        cfg = Config()
        prep = GPUFeaturePreprocessor(
            id_hash_config=cfg.id_hash_config,
            cat_emb_dim_overrides=cfg.cat_emb_dim_overrides,
        )
        assert "shop_id" in prep._id_hash_config
        assert "product_id" not in prep._id_hash_config

    def test_transform_handles_nan_null(self):
        """Verify transform() imputes NaN/null to 0 (matching fit() behavior)."""
        import polars as pl

        prep = GPUFeaturePreprocessor(n_quantile_bins=4)
        boundaries = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        prep.register_buffer("_quantile_boundaries", torch.from_numpy(boundaries))

        feat_name = NUMERIC_FEATURES[0]
        df = pl.DataFrame({feat_name: [1.5, None, float("nan"), 2.5]})
        numeric, cat_indices, _, _ = prep.transform(df, skip_targets=True)

        assert numeric.shape == (4, 1)
        assert np.isfinite(numeric).all(), "NaN/null should be imputed to 0"

    def test_fit_handles_null_categoricals(self):
        """Verify fit() doesn't crash when categorical columns contain nulls."""
        import polars as pl

        prep = GPUFeaturePreprocessor(n_quantile_bins=4)
        feat_name = NUMERIC_FEATURES[0]
        cat_name = CAT_FEATURES[0]
        df = pl.DataFrame(
            {
                feat_name: [1.0, 2.0, 3.0],
                cat_name: ["a", None, "b"],
            }
        )
        prep.fit(df)
        assert None not in prep.cat_encoders[cat_name]
        assert "a" in prep.cat_encoders[cat_name]
        assert "b" in prep.cat_encoders[cat_name]
