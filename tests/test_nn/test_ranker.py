"""Tests for DCNv2Ranker model."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import make_dummy_preprocessor
from reranking_dcn import (
    Config,
    DCNv2Ranker,
    PRETRAINED_EMB_DIM,
    SwiGLUBlock,
)
from reranking_dcn.export.checkpoint import build_model_from_config


def _make_small_model(
    with_emb: bool = True,
    n_num: int = 10,
    n_quantile_bins: int = 8,
    num_emb_dim: int = 4,
) -> tuple[DCNv2Ranker, dict[str, int], dict[str, int]]:
    """Build a small DCN-v2 for testing."""
    vocab_sizes = {
        "page_type": 5,
        "shop_id": 101,
    }
    embedding_dims = {
        "page_type": 3,
        "shop_id": 16,
    }
    pretrained_emb_dim = PRETRAINED_EMB_DIM * 2 if with_emb else 0

    model = DCNv2Ranker(
        num_continuous=n_num,
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        n_quantile_bins=n_quantile_bins,
        num_emb_dim=num_emb_dim,
        pretrained_emb_dim=pretrained_emb_dim,
        projected_emb_dim=64,
        num_cross_layers=2,
        num_experts=2,
        expert_rank=8,
        deep_dims=(32, 16),
        head_dims=(16, 8),
        dropout=0.0,
    )
    return model, vocab_sizes, embedding_dims


class TestDCNv2Ranker:
    @pytest.mark.parametrize("with_emb", [False, True], ids=["no_emb", "with_emb"])
    def test_forward_output(self, with_emb):
        model, vocab_sizes, _ = _make_small_model(with_emb=with_emb)
        model.eval()
        B, n_num = 4, 10
        bins = torch.randint(1, 8, (B, n_num), dtype=torch.long)
        cats = torch.randint(0, 3, (B, len(vocab_sizes)), dtype=torch.long)
        emb = torch.randn(B, PRETRAINED_EMB_DIM * 2) if with_emb else None
        with torch.no_grad():
            logits = model(bins, cats, emb) if with_emb else model(bins, cats)
        assert logits.shape == (B,)
        assert torch.isfinite(logits).all()

    def test_forward_with_none_emb_uses_zeros(self):
        model, vocab_sizes, _ = _make_small_model(with_emb=True)
        model.eval()
        B, n_num = 4, 10
        bins = torch.randint(1, 8, (B, n_num), dtype=torch.long)
        cats = torch.randint(0, 3, (B, len(vocab_sizes)), dtype=torch.long)
        with torch.no_grad():
            logits = model(bins, cats, pretrained_emb=None)
        assert logits.shape == (B,)
        assert torch.isfinite(logits).all()

    def test_projection_is_sequential(self):
        model, _, _ = _make_small_model(with_emb=True)
        assert isinstance(model.product_proj, nn.Sequential)
        assert isinstance(model.user_proj, nn.Sequential)
        assert isinstance(model.product_proj[0], nn.Linear)
        assert isinstance(model.product_proj[1], nn.LayerNorm)
        assert isinstance(model.product_proj[2], nn.GELU)

    @pytest.mark.parametrize(
        "attr",
        ["deep_net", "head_hidden"],
        ids=["deep_net", "head_hidden"],
    )
    def test_uses_swiglu_blocks(self, attr):
        model, _, _ = _make_small_model(with_emb=False)
        module = getattr(model, attr)
        assert any(isinstance(m, SwiGLUBlock) for m in module)

    def test_head_logit_is_linear(self):
        model, _, _ = _make_small_model(with_emb=False)
        assert isinstance(model.head_logit, nn.Linear)
        assert model.head_logit.out_features == 1

    def test_cat_embeddings_have_max_norm(self):
        model, _, _ = _make_small_model(with_emb=False)
        for col in model._cat_feature_order:
            emb = model.embeddings[col]
            assert emb.max_norm == 1.0

    def test_num_bin_offsets(self):
        n_num, n_bins = 10, 8
        model, _, _ = _make_small_model(with_emb=False, n_num=n_num, n_quantile_bins=n_bins)
        expected = torch.arange(n_num, dtype=torch.long) * (n_bins + 1)
        torch.testing.assert_close(model._num_bin_offsets, expected)

    def test_numeric_emb_table_size(self):
        n_num, n_bins, emb_dim = 10, 8, 4
        model, _, _ = _make_small_model(
            with_emb=False,
            n_num=n_num,
            n_quantile_bins=n_bins,
            num_emb_dim=emb_dim,
        )
        expected_num_embeddings = n_num * (n_bins + 1)
        assert model.numeric_emb.num_embeddings == expected_num_embeddings
        assert model.numeric_emb.embedding_dim == emb_dim

    @pytest.mark.parametrize("with_emb", [False, True], ids=["no_emb", "with_emb"])
    def test_batch_size_one(self, with_emb):
        model, vocab_sizes, _ = _make_small_model(with_emb=with_emb)
        model.eval()
        bins = torch.randint(1, 8, (1, 10), dtype=torch.long)
        cats = torch.randint(0, 3, (1, len(vocab_sizes)), dtype=torch.long)
        emb = torch.randn(1, PRETRAINED_EMB_DIM * 2) if with_emb else None
        with torch.no_grad():
            logits = model(bins, cats, emb) if with_emb else model(bins, cats)
        assert logits.shape == (1,)
        assert torch.isfinite(logits).all()


class TestParameterCount:
    def test_no_product_buyer_id_embeddings(self):
        """Verify the model does NOT have product_id or buyer_id embedding tables."""
        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 10_000_000, f"Too many params: {total_params:,}"

        param_names = [name for name, _ in model.named_parameters()]
        assert not any("product_id" in n for n in param_names)
        assert not any("unified_buyer_id" in n for n in param_names)
        assert any("shop_id" in n for n in param_names)
