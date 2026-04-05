"""Tests for train_cli -- config loading, arg parsing, and YAML integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from reranking_dcn.config import TrainingConfig
from reranking_dcn.train_cli import _load_config


def _namespace(**kwargs):
    """Build an argparse.Namespace-like object."""
    import argparse

    ns = argparse.Namespace()
    ns.config = kwargs.get("config")
    ns.config_str = kwargs.get("config_str")
    ns.dry_run = kwargs.get("dry_run", False)
    for field_name in [
        "mode",
        "stop_date",
        "billing_project",
        "local_dir",
        "output_dir",
        "user_sample_pct",
        "training_surface",
        "gcs_bucket",
        "checkpoint_gcs_prefix",
        "nn_base_train_start",
        "nn_base_train_stop",
    ]:
        setattr(ns, field_name, kwargs.get(field_name))
    return ns


class TestLoadConfig:
    def test_defaults_when_no_args(self):
        ns = _namespace()
        cfg = _load_config(ns)
        assert isinstance(cfg, TrainingConfig)
        assert cfg.mode == "finetune"

    def test_load_from_yaml(self):
        yaml_content = {
            "mode": "base_train",
            "stop_date": "2026-01-15",
            "user_sample_pct": 50,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            ns = _namespace(config=f.name)
            cfg = _load_config(ns)

        assert cfg.mode == "base_train"
        assert cfg.stop_date == "2026-01-15"
        assert cfg.data.user_sample_pct == 50

    def test_load_from_json_string(self):
        json_str = json.dumps(
            {
                "mode": "base_train",
                "stop_date": "2026-02-01",
            }
        )
        ns = _namespace(config_str=json_str)
        cfg = _load_config(ns)
        assert cfg.mode == "base_train"
        assert cfg.stop_date == "2026-02-01"

    def test_cli_overrides_yaml(self):
        yaml_content = {"stop_date": "2026-01-01", "user_sample_pct": 100}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            ns = _namespace(config=f.name, stop_date="2026-03-15")
            cfg = _load_config(ns)

        assert cfg.stop_date == "2026-03-15"
        assert cfg.data.user_sample_pct == 100

    def test_json_overrides_yaml(self):
        yaml_content = {"stop_date": "2026-01-01", "mode": "base_train"}
        json_str = json.dumps({"stop_date": "2026-02-02"})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            ns = _namespace(config=f.name, config_str=json_str)
            cfg = _load_config(ns)

        assert cfg.stop_date == "2026-02-02"
        assert cfg.mode == "base_train"

    def test_cli_overrides_json(self):
        json_str = json.dumps({"stop_date": "2026-01-01"})
        ns = _namespace(config_str=json_str, stop_date="2026-03-20")
        cfg = _load_config(ns)
        assert cfg.stop_date == "2026-03-20"

    def test_unknown_fields_ignored(self):
        json_str = json.dumps({"stop_date": "2026-03-24", "unknown_field": 999})
        ns = _namespace(config_str=json_str)
        cfg = _load_config(ns)
        assert cfg.stop_date == "2026-03-24"

    def test_base_training_yaml_loads(self):
        yaml_path = Path(__file__).resolve().parents[1] / "configs" / "base_training.yaml"
        if not yaml_path.exists():
            pytest.skip("base_training.yaml not found")
        ns = _namespace(config=str(yaml_path))
        cfg = _load_config(ns)
        assert cfg.mode == "base_train"
        assert cfg.data.nn_base_train_start == "2026-01-10"
        assert cfg.data.nn_base_train_stop == "2026-02-10"

    def test_daily_finetune_yaml_loads(self):
        yaml_path = Path(__file__).resolve().parents[1] / "configs" / "daily_finetune.yaml"
        if not yaml_path.exists():
            pytest.skip("daily_finetune.yaml not found")
        ns = _namespace(config=str(yaml_path))
        cfg = _load_config(ns)
        assert cfg.mode == "finetune"
        assert "embeddings" in cfg.optimizer.finetune_lrs
        assert cfg.optimizer.finetune_lrs["embeddings"] == 1e-5
