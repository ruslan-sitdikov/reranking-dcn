"""Tests for configuration dataclasses."""

from __future__ import annotations

from reranking_dcn import Config


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.num_emb_dim == 4
        assert cfg.num_quantile_bins == 64
        assert cfg.pretrained_emb_dim == 768
        assert cfg.projected_emb_dim == 64
        assert cfg.dcn_num_cross_layers == 3
        assert cfg.dcn_num_experts == 4
        assert cfg.dcn_expert_rank == 64
        assert cfg.dcn_deep_dims == (512, 256, 128)
        assert cfg.head_dims == (64, 32)
        assert cfg.dropout == 0.1

    def test_id_hash_config_only_broker(self):
        cfg = Config()
        assert list(cfg.id_hash_config.keys()) == ["broker_id"]
        assert cfg.id_hash_config["broker_id"] == (50_000, 16)
        assert "product_id" not in cfg.id_hash_config
        assert "unified_buyer_id" not in cfg.id_hash_config

    def test_cat_emb_dim_overrides(self):
        cfg = Config()
        assert cfg.cat_emb_dim_overrides == {}
