"""Tests for feature schema definitions."""

from __future__ import annotations

from reranking_dcn import ALL_FEATURES, CAT_FEATURES, NUMERIC_FEATURES, PRETRAINED_EMB_DIM


class TestFeatureLists:
    def test_numeric_count(self):
        assert len(NUMERIC_FEATURES) == 185

    def test_cat_features(self):
        assert "page_type" in CAT_FEATURES
        assert len(CAT_FEATURES) == 6

    def test_all_features(self):
        assert len(ALL_FEATURES) == 185 + 6

    def test_pretrained_emb_dim(self):
        assert PRETRAINED_EMB_DIM == 768
