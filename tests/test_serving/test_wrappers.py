"""Tests for ONNX-compatible serving wrappers."""

from __future__ import annotations

import pytest
import torch

from conftest import make_dummy_preprocessor
from reranking_dcn import (
    Config,
    DCNv2Serving,
    DCNv2ServingNoEmb,
    NUMERIC_FEATURES,
)
from reranking_dcn.export.checkpoint import build_model_from_config
from reranking_dcn.serving.wrappers import _DCNv2ServingBase


class TestDCNv2ServingWrappers:
    @pytest.fixture()
    def model_with_emb(self):
        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        model.eval()
        return model, prep, cfg

    @pytest.fixture()
    def model_no_emb(self):
        cfg = Config(device="cpu", use_pretrained_embeddings=False)
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        model.eval()
        return model, prep, cfg

    def test_serving_with_emb_forward(self, model_with_emb):
        model, prep, cfg = model_with_emb
        serving = DCNv2Serving(model)
        serving.eval()

        B = 4
        n_num = len(NUMERIC_FEATURES)
        n_cat = len(prep.vocab_sizes)
        bins = torch.randint(1, cfg.num_quantile_bins, (B, n_num), dtype=torch.long)
        cats = torch.randint(0, 5, (B, n_cat), dtype=torch.long)
        proj_prod = torch.randn(B, cfg.projected_emb_dim)
        proj_user = torch.randn(B, cfg.projected_emb_dim)
        has_prod = torch.ones(B, 1)
        has_user = torch.ones(B, 1)

        with torch.no_grad():
            scores = serving(bins, cats, proj_prod, proj_user, has_prod, has_user)

        assert scores.shape == (B,)
        assert torch.isfinite(scores).all()

    def test_serving_with_emb_rejects_no_emb_model(self, model_no_emb):
        model, _, _ = model_no_emb
        with pytest.raises(ValueError, match="DCNv2Serving requires"):
            DCNv2Serving(model)

    def test_serving_no_emb_forward(self, model_no_emb):
        model, prep, cfg = model_no_emb
        serving = DCNv2ServingNoEmb(model)
        serving.eval()

        B = 4
        n_num = len(NUMERIC_FEATURES)
        n_cat = len(prep.vocab_sizes)
        bins = torch.randint(1, cfg.num_quantile_bins, (B, n_num), dtype=torch.long)
        cats = torch.randint(0, 5, (B, n_cat), dtype=torch.long)

        with torch.no_grad():
            scores = serving(bins, cats)

        assert scores.shape == (B,)
        assert torch.isfinite(scores).all()


class TestServingWrapperBase:
    def test_both_wrappers_share_base(self):
        assert issubclass(DCNv2Serving, _DCNv2ServingBase)
        assert issubclass(DCNv2ServingNoEmb, _DCNv2ServingBase)

    def test_serving_batch_size_one(self):
        cfg = Config(device="cpu", use_pretrained_embeddings=False)
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        model.eval()
        serving = DCNv2ServingNoEmb(model)
        serving.eval()

        n_num = len(NUMERIC_FEATURES)
        n_cat = len(prep.vocab_sizes)
        bins = torch.randint(1, cfg.num_quantile_bins, (1, n_num), dtype=torch.long)
        cats = torch.randint(0, 5, (1, n_cat), dtype=torch.long)

        with torch.no_grad():
            scores = serving(bins, cats)
        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()
