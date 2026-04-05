"""Tests for configuration dataclasses."""

from __future__ import annotations

from reranking_dcn import Config
from reranking_dcn.config import (
    DataConfig,
    InfraConfig,
    OptimizerConfig,
    TrainingConfig,
)


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

    def test_id_hash_config_only_shop(self):
        cfg = Config()
        assert list(cfg.id_hash_config.keys()) == ["shop_id"]
        assert cfg.id_hash_config["shop_id"] == (50_000, 16)
        assert "product_id" not in cfg.id_hash_config
        assert "unified_buyer_id" not in cfg.id_hash_config

    def test_cat_emb_dim_overrides(self):
        cfg = Config()
        assert cfg.cat_emb_dim_overrides == {}

    def test_to_serving_arch_dict(self):
        cfg = Config()
        arch = cfg.to_serving_arch_dict(
            num_continuous=185,
            vocab_sizes={"cat_a": 10, "cat_b": 5},
        )
        assert arch["num_continuous"] == 185
        assert arch["num_cat_features"] == 2
        assert arch["cat_feature_order"] == ["cat_a", "cat_b"]
        assert arch["n_quantile_bins"] == 64


class TestTrainingConfigSubConfigs:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.mode == "finetune"
        assert cfg.data.user_sample_pct == 100
        assert cfg.infra.billing_project == "sdp-stg-shop-ml"
        assert cfg.optimizer.learning_rate == 1e-3
        assert cfg.runtime.batch_size == 2048
        assert cfg.monitor.comet_project == "dcn-v2-training"

    def test_from_flat_dict(self):
        cfg = TrainingConfig.from_flat_dict(
            {
                "mode": "base_train",
                "stop_date": "2026-03-24",
                "billing_project": "my-project",
                "user_sample_pct": 5,
                "learning_rate": 1e-4,
                "batch_size": 512,
                "comet_project": "my-exp",
            }
        )
        assert cfg.mode == "base_train"
        assert cfg.stop_date == "2026-03-24"
        assert cfg.infra.billing_project == "my-project"
        assert cfg.data.user_sample_pct == 5
        assert cfg.optimizer.learning_rate == 1e-4
        assert cfg.runtime.batch_size == 512
        assert cfg.monitor.comet_project == "my-exp"

    def test_from_flat_dict_ignores_unknown(self):
        cfg = TrainingConfig.from_flat_dict(
            {
                "stop_date": "2026-03-24",
                "unknown_field": 999,
            }
        )
        assert cfg.stop_date == "2026-03-24"

    def test_from_flat_dict_nested_passthrough(self):
        cfg = TrainingConfig.from_flat_dict(
            {
                "mode": "base_train",
                "infra": {"billing_project": "nested-proj"},
            }
        )
        assert cfg.infra.billing_project == "nested-proj"

    def test_to_flat_dict_roundtrip(self):
        original = TrainingConfig.from_flat_dict(
            {
                "mode": "base_train",
                "stop_date": "2026-03-20",
                "billing_project": "test-proj",
                "user_sample_pct": 10,
            }
        )
        flat = original.to_flat_dict()
        assert flat["mode"] == "base_train"
        assert flat["stop_date"] == "2026-03-20"
        assert flat["billing_project"] == "test-proj"
        assert flat["user_sample_pct"] == 10


class TestTrainingConfigDateArithmetic:
    def test_base_train_dates(self):
        cfg = TrainingConfig.from_flat_dict(
            {
                "stop_date": "2026-03-24",
                "nn_base_train_start": "2026-01-10",
                "nn_base_train_stop": "2026-02-10",
            }
        )
        assert cfg.data.nn_base_train_start < cfg.data.nn_base_train_stop

    def test_finetune_day_window_1_day(self):
        cfg = TrainingConfig.from_flat_dict(
            {
                "stop_date": "2026-03-24",
                "finetune_data_window_days": 1,
            }
        )
        assert cfg.finetune_day_start == "2026-03-23"
        assert cfg.finetune_day_stop == "2026-03-24"

    def test_finetune_day_window_3_days(self):
        cfg = TrainingConfig.from_flat_dict(
            {
                "stop_date": "2026-03-24",
                "finetune_data_window_days": 3,
            }
        )
        assert cfg.finetune_day_start == "2026-03-21"

    def test_checkpoint_gcs_path_default(self):
        cfg = TrainingConfig(
            infra=InfraConfig(gcs_bucket="my-bucket", output_gcs_prefix="dcn_v2"),
        )
        assert cfg.checkpoint_gcs_path == "gs://my-bucket/dcn_v2/latest/checkpoint.pt"

    def test_checkpoint_gcs_path_custom(self):
        cfg = TrainingConfig(
            infra=InfraConfig(checkpoint_gcs_prefix="gs://bucket/custom/path"),
        )
        assert cfg.checkpoint_gcs_path == "gs://bucket/custom/path/checkpoint.pt"

    def test_artifacts_gcs_dir(self):
        cfg = TrainingConfig(
            infra=InfraConfig(gcs_bucket="my-bucket", output_gcs_prefix="dcn_v2"),
        )
        assert cfg.artifacts_gcs_dir == "gs://my-bucket/dcn_v2/latest"

    def test_surface_filter_all(self):
        cfg = TrainingConfig(data=DataConfig(training_surface="all"))
        filt = cfg.data.surface_filter
        assert "home_feed" in filt
        assert "boosted_merchants" in filt

    def test_surface_filter_boosted_merchants(self):
        cfg = TrainingConfig(data=DataConfig(training_surface="boosted_merchants"))
        filt = cfg.data.surface_filter
        assert "page_type = 'boosted_merchants'" in filt
        assert "campaign_id IS NOT NULL" in filt


class TestDataConfigSourceProject:
    def test_default_tables_use_prd_project(self):
        cfg = DataConfig()
        assert "sdp-prd-shop-ml" in cfg.source_table
        assert "sdp-prd-shop-ml" in cfg.nomic_embedding_table
        assert "sdp-prd-shop-ml" in cfg.user_embedding_table
        assert "sdp-prd-shop-ml" in cfg.user_embedding_history_table

    def test_custom_project_propagates_to_tables(self):
        cfg = DataConfig(source_project="sdp-stg-shop-ml")
        assert "sdp-stg-shop-ml" in cfg.source_table
        assert "sdp-stg-shop-ml" in cfg.nomic_embedding_table
        assert "sdp-stg-shop-ml" in cfg.user_embedding_table
        assert "sdp-stg-shop-ml" in cfg.user_embedding_history_table
        assert "sdp-prd-shop-ml" not in cfg.source_table

    def test_explicit_table_overrides_project(self):
        cfg = DataConfig(
            source_project="sdp-stg-shop-ml",
            source_table="`custom.dataset.table`",
        )
        assert cfg.source_table == "`custom.dataset.table`"
        assert "sdp-stg-shop-ml" in cfg.nomic_embedding_table

    def test_from_flat_dict_routes_source_project(self):
        cfg = TrainingConfig.from_flat_dict({"source_project": "sdp-stg-shop-ml"})
        assert "sdp-stg-shop-ml" in cfg.data.source_table
