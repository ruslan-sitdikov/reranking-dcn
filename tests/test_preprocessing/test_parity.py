"""Parity tests between GPUFeaturePreprocessor (training) and ServingPreprocessor (serving).

Ensures that the two independent preprocessing implementations produce
identical outputs for the same input data, preventing silent train-serve skew.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from reranking_dcn import Config
from reranking_dcn.features import CAT_FEATURES, NUMERIC_FEATURES
from reranking_dcn.preprocessing.gpu_preprocessor import GPUFeaturePreprocessor
from reranking_dcn.preprocessing.serving_preprocessor import ServingPreprocessor
from reranking_dcn.serving.artifacts import serialize_preprocessing_artifacts


@pytest.fixture()
def parity_env(tmp_path: Path):
    """Build a fitted GPU preprocessor, serialize artifacts, and create a ServingPreprocessor."""
    rng = np.random.default_rng(42)
    n_rows = 500

    data: dict[str, list | np.ndarray] = {}
    for col in NUMERIC_FEATURES:
        data[col] = rng.standard_normal(n_rows).astype(np.float32).tolist()
    for col in CAT_FEATURES:
        data[col] = [f"val_{rng.integers(0, 8)}" for _ in range(n_rows)]

    data["shop_id"] = rng.integers(1, 10_000, size=n_rows).tolist()

    df = pl.DataFrame(data)
    cfg = Config(device="cpu")

    gpu_pp = GPUFeaturePreprocessor(
        id_hash_config=cfg.id_hash_config,
        cat_emb_dim_overrides=cfg.cat_emb_dim_overrides,
        n_quantile_bins=cfg.num_quantile_bins,
    ).fit(df)

    serialize_preprocessing_artifacts(gpu_pp, cfg, tmp_path)
    serving_pp = ServingPreprocessor(tmp_path)

    return gpu_pp, serving_pp, cfg, df


class TestQuantileBinningParity:
    def test_bins_match_exactly(self, parity_env):
        gpu_pp, serving_pp, cfg, df = parity_env

        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        numeric = df.select(
            [pl.col(c).fill_null(0.0).fill_nan(0.0).cast(pl.Float32) for c in num_cols]
        ).to_numpy()

        gpu_bins = gpu_pp.quantile_encode_on_device(torch.from_numpy(numeric)).numpy()
        cpu_bins = serving_pp.quantile_encode(numeric)

        np.testing.assert_array_equal(
            gpu_bins,
            cpu_bins,
            err_msg="Quantile bin indices differ between GPU and CPU paths",
        )

    def test_boundary_values(self, parity_env):
        """Verify parity on values that sit exactly on quantile boundaries."""
        gpu_pp, serving_pp, _, _ = parity_env

        boundaries = gpu_pp._quantile_boundaries.numpy()
        n_features = boundaries.shape[0]

        exact_boundary_vals = np.zeros((1, n_features), dtype=np.float32)
        for i in range(n_features):
            exact_boundary_vals[0, i] = boundaries[i, boundaries.shape[1] // 2]

        gpu_bins = gpu_pp.quantile_encode_on_device(torch.from_numpy(exact_boundary_vals)).numpy()
        cpu_bins = serving_pp.quantile_encode(exact_boundary_vals)

        np.testing.assert_array_equal(gpu_bins, cpu_bins)

    def test_extreme_values(self, parity_env):
        """Verify parity on values far below/above all boundaries."""
        gpu_pp, serving_pp, _, _ = parity_env
        n_features = gpu_pp._quantile_boundaries.shape[0]

        extremes = np.zeros((2, n_features), dtype=np.float32)
        extremes[0, :] = -1e9
        extremes[1, :] = 1e9

        gpu_bins = gpu_pp.quantile_encode_on_device(torch.from_numpy(extremes)).numpy()
        cpu_bins = serving_pp.quantile_encode(extremes)

        np.testing.assert_array_equal(gpu_bins, cpu_bins)


class TestCategoricalEncodingParity:
    def test_categorical_indices_match(self, parity_env):
        gpu_pp, serving_pp, _, df = parity_env

        _, cat_indices_gpu, _, _ = gpu_pp.transform(df, skip_targets=True)

        all_cat_order = sorted(gpu_pp.vocab_sizes.keys())
        cat_values: dict[str, list] = {}
        for col in all_cat_order:
            if col in gpu_pp.cat_encoders:
                cat_values[col] = (
                    df[col].cast(pl.Utf8).fill_null("__MISSING__").to_list()
                    if col in df.columns
                    else ["__MISSING__"] * df.shape[0]
                )
            elif col in df.columns:
                cat_values[col] = df[col].to_list()
            else:
                cat_values[col] = [None] * df.shape[0]

        cat_indices_cpu = serving_pp.encode_categoricals(cat_values)

        np.testing.assert_array_equal(
            cat_indices_gpu.astype(np.int64),
            cat_indices_cpu,
            err_msg="Categorical encoding differs between GPU and CPU paths",
        )
