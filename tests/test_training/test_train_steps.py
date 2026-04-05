"""Integration tests for training steps: base training + checkpoint + finetune."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from conftest import make_dummy_preprocessor
from reranking_dcn import Config
from reranking_dcn.config import OptimizerConfig, RuntimeConfig, TrainingConfig
from reranking_dcn.features import CAT_FEATURES, NUMERIC_FEATURES
from reranking_dcn.training.train_steps import (
    build_dcn_model,
    fit_preprocessor,
    run_base_training,
    run_finetune,
)


def _make_synthetic_parquet(local_dir: Path, split: str, n_rows: int = 100) -> None:
    """Write a synthetic Parquet split that looks like exported BQ data."""
    rng = np.random.default_rng(42)
    data: dict[str, list | np.ndarray] = {}

    for c in NUMERIC_FEATURES:
        data[c] = rng.standard_normal(n_rows).astype(np.float32).tolist()
    for c in CAT_FEATURES:
        data[c] = [f"cat_{rng.integers(0, 5)}" for _ in range(n_rows)]

    data["unified_buyer_id"] = [f"ubid_v3_ap:3_ios:false_{i:064x}" for i in range(n_rows)]
    data["product_id"] = rng.integers(1, 5000, size=n_rows).tolist()
    data["shop_id"] = rng.integers(1, 100, size=n_rows).tolist()
    data["event_timestamp_day"] = ["2026-03-24"] * n_rows
    data["target_is_native_order"] = rng.integers(0, 2, size=n_rows).tolist()

    df = pl.DataFrame(data).with_columns(pl.col("event_timestamp_day").str.to_date())

    split_dir = local_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(split_dir / "shard_000.parquet")


@pytest.fixture()
def small_cfg() -> Config:
    return Config(
        device="cpu",
        num_quantile_bins=8,
        num_emb_dim=4,
        dcn_num_cross_layers=1,
        dcn_num_experts=1,
        dcn_expert_rank=4,
        dcn_deep_dims=(16, 8),
        head_dims=(8,),
        dropout=0.0,
        use_pretrained_embeddings=False,
    )


@pytest.fixture()
def train_cfg() -> TrainingConfig:
    return TrainingConfig(
        mode="base_train",
        stop_date="2026-03-24",
        optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
        runtime=RuntimeConfig(batch_size=16, num_workers=0, epochs=1),
    )


class TestBuildDcnModel:
    def test_model_is_created(self, small_cfg: Config):
        prep = make_dummy_preprocessor(small_cfg)
        model = build_dcn_model(small_cfg, prep, pretrained_emb_dim=0)
        assert isinstance(model, torch.nn.Module)

    def test_model_parameter_count_reasonable(self, small_cfg: Config):
        prep = make_dummy_preprocessor(small_cfg)
        model = build_dcn_model(small_cfg, prep, pretrained_emb_dim=0)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        assert n_params < 1_000_000


class TestFitPreprocessor:
    def test_fit_returns_preprocessor(self, small_cfg: Config):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            _make_synthetic_parquet(local_dir, "nn_base_train")
            df = pl.read_parquet(local_dir / "nn_base_train" / "shard_000.parquet")
            prep = fit_preprocessor(small_cfg, df)
            assert prep.num_continuous == len(NUMERIC_FEATURES)
            assert len(prep.vocab_sizes) > 0


class TestRunBaseTraining:
    def test_base_training_produces_trainer_and_preprocessor(
        self,
        small_cfg: Config,
        train_cfg: TrainingConfig,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            _make_synthetic_parquet(local_dir, "nn_base_train")

            trainer, prep = run_base_training(
                train_cfg,
                small_cfg,
                local_dir,
                device=torch.device("cpu"),
            )

            assert trainer is not None
            assert prep is not None
            assert prep.num_continuous == len(NUMERIC_FEATURES)


class TestPreprocessorSaveState:
    def test_save_and_restore_roundtrip(self, small_cfg: Config):
        """Verify preprocessor state survives save/restore via v2 checkpoint format."""
        from reranking_dcn.preprocessing.gpu_preprocessor import GPUFeaturePreprocessor

        prep = make_dummy_preprocessor(small_cfg)
        state = prep.save_state()
        restored = GPUFeaturePreprocessor.from_state(state)

        assert restored.num_continuous == prep.num_continuous
        assert restored.vocab_sizes == prep.vocab_sizes
        assert restored.embedding_dims == prep.embedding_dims
        assert restored.cat_encoders == prep.cat_encoders
        torch.testing.assert_close(restored._quantile_boundaries, prep._quantile_boundaries)


class TestRunFinetune:
    def test_finetune_loads_v2_checkpoint_and_trains(
        self,
        small_cfg: Config,
        train_cfg: TrainingConfig,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            _make_synthetic_parquet(local_dir, "nn_base_train")
            _make_synthetic_parquet(local_dir, "nn_finetune")

            trainer, prep = run_base_training(
                train_cfg,
                small_cfg,
                local_dir,
                device=torch.device("cpu"),
            )

            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            base_model = getattr(trainer.model, "_orig_mod", trainer.model)
            torch.save(
                {
                    "checkpoint_version": 2,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "preprocessor_state": prep.save_state(),
                    "pretrained_emb_dim": 0,
                },
                checkpoint_path,
            )

            ft_cfg = TrainingConfig(
                mode="finetune",
                stop_date="2026-03-24",
                optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
                runtime=RuntimeConfig(batch_size=16, num_workers=0, epochs=1),
            )

            ft_trainer, ft_prep = run_finetune(
                ft_cfg,
                small_cfg,
                local_dir,
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )

            assert ft_trainer is not None
            assert ft_prep is not None

    def test_rejects_legacy_v1_checkpoint_without_preprocessor_state(
        self,
        small_cfg: Config,
        train_cfg: TrainingConfig,
    ):
        """v1 checkpoints (pickled preprocessor) must be rejected for security."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            _make_synthetic_parquet(local_dir, "nn_base_train")
            _make_synthetic_parquet(local_dir, "nn_finetune")

            trainer, prep = run_base_training(
                train_cfg,
                small_cfg,
                local_dir,
                device=torch.device("cpu"),
            )

            checkpoint_path = Path(tmpdir) / "legacy_checkpoint.pt"
            base_model = getattr(trainer.model, "_orig_mod", trainer.model)
            torch.save(
                {
                    "model_state_dict": base_model.state_dict(),
                    "preprocessor": prep,
                    "pretrained_emb_dim": 0,
                },
                checkpoint_path,
            )

            ft_cfg = TrainingConfig(
                mode="finetune",
                stop_date="2026-03-24",
                optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
                runtime=RuntimeConfig(batch_size=16, num_workers=0, epochs=1),
            )

            with pytest.raises((ValueError, RuntimeError, pickle.UnpicklingError)):
                run_finetune(
                    ft_cfg,
                    small_cfg,
                    local_dir,
                    device=torch.device("cpu"),
                    checkpoint_path=checkpoint_path,
                )
