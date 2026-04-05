"""Shared test fixtures for reranking_dcn."""

from __future__ import annotations

import numpy as np
import torch

from reranking_dcn import CAT_FEATURES, NUMERIC_FEATURES, Config, GPUFeaturePreprocessor


def make_dummy_preprocessor(cfg: Config | None = None) -> GPUFeaturePreprocessor:
    """Build a preprocessor with random quantile boundaries for testing."""
    if cfg is None:
        cfg = Config(device="cpu")
    prep = GPUFeaturePreprocessor(
        id_hash_config=cfg.id_hash_config,
        cat_emb_dim_overrides=cfg.cat_emb_dim_overrides,
        n_quantile_bins=cfg.num_quantile_bins,
    )
    n_num = len(NUMERIC_FEATURES)
    boundaries = np.sort(
        np.random.randn(n_num, cfg.num_quantile_bins - 1).astype(np.float32), axis=1
    )
    prep.register_buffer("_quantile_boundaries", torch.from_numpy(boundaries))
    for col in CAT_FEATURES:
        prep.cat_encoders[col] = {f"val_{i}": i + 1 for i in range(10)}
        prep.vocab_sizes[col] = 11
        prep.embedding_dims[col] = cfg.cat_emb_dim_overrides.get(col, min(50, 11 // 2))
    for col, (n_buckets, emb_dim) in cfg.id_hash_config.items():
        prep.vocab_sizes[col] = n_buckets + 1
        prep.embedding_dims[col] = emb_dim
    return prep
