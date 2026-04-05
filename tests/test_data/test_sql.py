"""Tests for SQL query builders."""

from __future__ import annotations

import pytest

from reranking_dcn.config import DataConfig, TrainingConfig
from reranking_dcn.data.sql import (
    META_COLUMNS,
    _neg_ratio_filter,
    _select_cols,
    _user_sample_filter,
    sql_nn_base_train,
    sql_nn_finetune_day,
    sql_product_embeddings,
    sql_user_embeddings,
)
from reranking_dcn.features import CAT_FEATURES, NUMERIC_FEATURES


class TestSelectCols:
    def test_contains_all_meta_columns(self):
        cols = _select_cols()
        for mc in META_COLUMNS:
            assert mc in cols

    def test_contains_numeric_features(self):
        cols = _select_cols()
        for nf in NUMERIC_FEATURES[:5]:
            assert nf in cols

    def test_contains_cat_features(self):
        cols = _select_cols()
        for cf in CAT_FEATURES:
            assert cf in cols

    def test_no_duplicates(self):
        cols = _select_cols().split(", ")
        assert len(cols) == len(set(cols))

    def test_day_of_week_not_present(self):
        cols = _select_cols()
        assert "day_of_week" not in cols


class TestUserSampleFilter:
    @pytest.mark.parametrize(
        "user_sample_pct",
        [100, 200],
        ids=["full_100pct", "over_200pct"],
    )
    def test_high_pct_returns_true(self, user_sample_pct):
        cfg = TrainingConfig.from_flat_dict({"user_sample_pct": user_sample_pct})
        assert _user_sample_filter(cfg) == "TRUE"

    def test_partial_sample_uses_farm_fingerprint(self):
        cfg = TrainingConfig.from_flat_dict({"user_sample_pct": 5})
        result = _user_sample_filter(cfg)
        assert "FARM_FINGERPRINT" in result
        assert "< 5" in result

    def test_one_pct_sample(self):
        cfg = TrainingConfig.from_flat_dict({"user_sample_pct": 1})
        result = _user_sample_filter(cfg)
        assert "< 1" in result


class TestNegRatioFilter:
    @pytest.mark.parametrize(
        "alias, neg_scale, expected_substrings",
        [
            ("f", 20, ["FARM_FINGERPRINT"]),
            ("my_alias", 20, ["my_alias.unified_buyer_id", "my_alias.product_id"]),
            ("f", 15, ["15"]),
        ],
        ids=["farm_fingerprint", "alias_columns", "neg_scale_value"],
    )
    def test_neg_ratio_filter_content(self, alias, neg_scale, expected_substrings):
        result = _neg_ratio_filter(alias, neg_scale)
        for substr in expected_substrings:
            assert substr in result


class TestSqlNnBaseTrain:
    @pytest.fixture()
    def cfg(self) -> TrainingConfig:
        return TrainingConfig.from_flat_dict({"stop_date": "2026-03-24", "user_sample_pct": 100})

    def test_uses_source_table(self, cfg: TrainingConfig):
        sql = sql_nn_base_train(cfg)
        assert "unified_recs__product_ranking__recommended_pages_training_v1" in sql

    def test_date_window_matches_config(self, cfg: TrainingConfig):
        sql = sql_nn_base_train(cfg)
        assert cfg.data.nn_base_train_start in sql
        assert cfg.data.nn_base_train_stop in sql

    def test_contains_target_columns(self, cfg: TrainingConfig):
        sql = sql_nn_base_train(cfg)
        assert "target_is_native_order" in sql
        assert "target_is_high_quality_click" not in sql
        assert "target_is_clicked" not in sql

    def test_positive_sample_retained(self, cfg: TrainingConfig):
        sql = sql_nn_base_train(cfg)
        assert "f.is_native_order = TRUE" in sql

    def test_negative_downsampling_present(self, cfg: TrainingConfig):
        sql = sql_nn_base_train(cfg)
        assert "FARM_FINGERPRINT" in sql

    def test_surface_filter_applied(self, cfg: TrainingConfig):
        sql = sql_nn_base_train(cfg)
        assert "page_type" in sql
        assert "recently_viewed" in sql


class TestSqlNnFinetuneDay:
    def test_uses_daily_dates(self):
        cfg = TrainingConfig.from_flat_dict({"stop_date": "2026-03-24"})
        sql = sql_nn_finetune_day(cfg)
        assert cfg.finetune_day_start in sql
        assert cfg.finetune_day_stop in sql

    def test_single_day_window(self):
        cfg = TrainingConfig.from_flat_dict(
            {
                "stop_date": "2026-03-24",
                "finetune_data_window_days": 1,
            }
        )
        assert cfg.finetune_day_start == "2026-03-23"
        assert cfg.finetune_day_stop == "2026-03-24"


class TestSqlProductEmbeddings:
    @pytest.mark.parametrize(
        "expected_substrings",
        [
            ["product_emb_0", "product_emb_767"],
            ["nomic_product_document_embedding"],
            ["my_project.my_dataset.product_ids"],
        ],
        ids=["emb_columns", "nomic_table", "ids_table_ref"],
    )
    def test_product_embedding_sql(self, expected_substrings):
        cfg = TrainingConfig.from_flat_dict({"stop_date": "2026-03-24"})
        sql = sql_product_embeddings(cfg, "`my_project.my_dataset.product_ids`")
        for substr in expected_substrings:
            assert substr in sql


class TestSqlUserEmbeddings:
    @pytest.mark.parametrize(
        "expected_substrings",
        [
            ["user_emb_0", "user_emb_767"],
            ["user_date_key"],
            ["user_average_embedding_history"],
            ["valid_from", "valid_to", "ORDER"],
            ["my_project.my_dataset.user_days"],
            ["unified_buyer_id"],
        ],
        ids=[
            "emb_columns",
            "user_date_key",
            "history_table",
            "scd2_filters",
            "ids_table_ref",
            "direct_ubi_join",
        ],
    )
    def test_user_embedding_sql(self, expected_substrings):
        cfg = TrainingConfig.from_flat_dict({"stop_date": "2026-03-24"})
        sql = sql_user_embeddings(cfg, "`my_project.my_dataset.user_days`")
        for substr in expected_substrings:
            assert substr in sql

    def test_no_2hop_join(self):
        """Direct UBI join should not use ubi_mapping or users tables."""
        cfg = TrainingConfig.from_flat_dict({"stop_date": "2026-03-24"})
        sql = sql_user_embeddings(cfg, "`my_project.my_dataset.user_days`")
        assert "identifier_ubi_mapping" not in sql
        assert "base__shop_server__users" not in sql
        assert "identifier_type" not in sql


class TestSqlBoostedMerchantsSurface:
    def test_boosted_merchants_filter(self):
        cfg = TrainingConfig.from_flat_dict(
            {
                "stop_date": "2026-03-24",
                "training_surface": "boosted_merchants",
            }
        )
        sql = sql_nn_base_train(cfg)
        assert "page_type = 'boosted_merchants'" in sql
        assert "campaign_id IS NOT NULL" in sql
