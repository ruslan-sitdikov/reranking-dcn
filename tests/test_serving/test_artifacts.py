"""Tests for preprocessing artifact serialization."""

from __future__ import annotations

import json

from conftest import make_dummy_preprocessor
from reranking_dcn import Config
from reranking_dcn.serving.artifacts import serialize_preprocessing_artifacts


class TestArtifactSerialization:
    def test_artifacts_saved(self, tmp_path):
        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        serialize_preprocessing_artifacts(prep, cfg, tmp_path)

        assert (tmp_path / "quantile_boundaries.npy").exists()
        assert (tmp_path / "cat_encoders.json").exists()
        assert (tmp_path / "id_hash_config.json").exists()
        assert (tmp_path / "model_config.json").exists()
        assert (tmp_path / "feature_list.json").exists()

    def test_id_hash_config_only_broker(self, tmp_path):
        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        serialize_preprocessing_artifacts(prep, cfg, tmp_path)

        with open(tmp_path / "id_hash_config.json") as f:
            id_hash = json.load(f)
        assert list(id_hash.keys()) == ["broker_id"]

    def test_feature_list_cat_features(self, tmp_path):
        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        serialize_preprocessing_artifacts(prep, cfg, tmp_path)

        with open(tmp_path / "feature_list.json") as f:
            features = json.load(f)
        assert "page_type" in features["cat_features"]
        assert len(features["cat_features"]) == 6
