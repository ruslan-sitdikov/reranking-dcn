"""Tests for the end-to-end training pipeline (with mocked external services)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import torch

from conftest import make_dummy_preprocessor
from reranking_dcn import Config
from reranking_dcn.config import (
    DataConfig,
    InfraConfig,
    OptimizerConfig,
    RuntimeConfig,
    TrainingConfig,
)
from reranking_dcn.features import CAT_FEATURES, NUMERIC_FEATURES
from reranking_dcn.training.pipeline import (
    _download_checkpoint,
    _resolve_device,
    _save_checkpoint,
    _upload_to_gcs,
)
from reranking_dcn.training.train_steps import build_dcn_model
from reranking_dcn.training.trainer import DCNTrainer


def _make_synthetic_parquet(local_dir: Path, split: str, n_rows: int = 50) -> None:
    rng = np.random.default_rng(42)
    data: dict[str, list | np.ndarray] = {}
    for c in NUMERIC_FEATURES:
        data[c] = rng.standard_normal(n_rows).astype(np.float32).tolist()
    for c in CAT_FEATURES:
        data[c] = [f"cat_{rng.integers(0, 5)}" for _ in range(n_rows)]
    data["unified_buyer_id"] = [f"ubid_{i:08x}" for i in range(n_rows)]
    data["product_id"] = rng.integers(1, 5000, size=n_rows).tolist()
    data["shop_id"] = rng.integers(1, 100, size=n_rows).tolist()
    data["event_timestamp_day"] = ["2026-03-24"] * n_rows
    data["target_is_native_order"] = rng.integers(0, 2, size=n_rows).tolist()
    df = pl.DataFrame(data).with_columns(pl.col("event_timestamp_day").str.to_date())
    split_dir = local_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(split_dir / "shard_000.parquet")


class TestResolveDevice:
    def test_returns_cpu_when_cuda_unavailable(self):
        cfg = TrainingConfig()
        with patch("torch.cuda.is_available", return_value=False):
            device = _resolve_device(cfg)
        assert device == torch.device("cpu")


class TestSaveCheckpoint:
    def test_saves_v2_checkpoint(self):
        cfg_arch = Config(
            device="cpu",
            num_quantile_bins=4,
            dcn_deep_dims=(8,),
            head_dims=(4,),
            dcn_num_cross_layers=1,
            dcn_num_experts=1,
            dcn_expert_rank=4,
            use_pretrained_embeddings=False,
        )
        train_cfg = TrainingConfig(
            runtime=RuntimeConfig(batch_size=8, num_workers=0, epochs=1),
            optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
        )
        prep = make_dummy_preprocessor(cfg_arch)
        model = build_dcn_model(cfg_arch, prep, pretrained_emb_dim=0)
        trainer = DCNTrainer(model, prep, train_cfg, torch.device("cpu"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = _save_checkpoint(trainer, prep, output_dir, 0, "base_train")

            assert path.exists()
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            assert ckpt["checkpoint_version"] == 2
            assert "model_state_dict" in ckpt
            assert "preprocessor_state" in ckpt
            assert ckpt["mode"] == "base_train"


class TestUploadToGcs:
    def test_skips_invalid_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _upload_to_gcs(Path(tmpdir), "not-a-gs-path", "project")

    @patch("reranking_dcn.training.pipeline.time.sleep")
    @patch("google.cloud.storage.Client")
    def test_retries_on_upload_failure(self, mock_client_cls, mock_sleep):
        mock_blob = MagicMock()
        mock_blob.upload_from_filename.side_effect = [
            Exception("transient"),
            None,
        ]
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_cls.return_value.bucket.return_value = mock_bucket

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("data")

            _upload_to_gcs(Path(tmpdir), "gs://bucket/prefix", "proj")

        assert mock_blob.upload_from_filename.call_count == 2

    @patch("reranking_dcn.training.pipeline.time.sleep")
    @patch("google.cloud.storage.Client")
    def test_raises_after_max_retries(self, mock_client_cls, mock_sleep):
        mock_blob = MagicMock()
        mock_blob.upload_from_filename.side_effect = Exception("persistent failure")
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_cls.return_value.bucket.return_value = mock_bucket

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("data")

            with pytest.raises(RuntimeError, match="Failed to upload"):
                _upload_to_gcs(Path(tmpdir), "gs://bucket/prefix", "proj")


class TestDownloadCheckpoint:
    def test_local_path_passthrough(self):
        result = _download_checkpoint("/local/path/checkpoint.pt", Path("/tmp/ckpt.pt"), "proj")
        assert result == Path("/local/path/checkpoint.pt")
