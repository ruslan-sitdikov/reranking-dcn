"""Tests for DCNTrainer -- forward/backward pass, LR scheduling, and inference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import make_dummy_preprocessor
from reranking_dcn import Config
from reranking_dcn.config import TrainingConfig
from reranking_dcn.data.dataset import make_dataloader
from reranking_dcn.training.trainer import DCNTrainer
from reranking_dcn.training.train_steps import build_dcn_model


def _make_tiny_trainer(
    with_emb: bool = False,
) -> tuple[DCNTrainer, torch.utils.data.DataLoader]:
    """Build a tiny model + dataloader for CPU-based testing."""
    rng = np.random.default_rng(42)
    cfg = Config(
        device="cpu",
        num_quantile_bins=8,
        num_emb_dim=4,
        dcn_num_cross_layers=1,
        dcn_num_experts=1,
        dcn_expert_rank=4,
        dcn_deep_dims=(16, 8),
        head_dims=(8,),
        dropout=0.0,
        use_pretrained_embeddings=with_emb,
        projected_emb_dim=16,
    )
    train_cfg = TrainingConfig.from_flat_dict(
        {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 8,
            "num_workers": 0,
            "epochs": 1,
        }
    )

    prep = make_dummy_preprocessor(cfg)
    n_num = prep.num_continuous
    n_cat = len(prep.vocab_sizes)

    N = 32
    numeric = rng.standard_normal((N, n_num)).astype(np.float32)
    cat_indices = rng.integers(0, 3, size=(N, n_cat)).astype(np.int32)
    targets = rng.integers(0, 2, size=(N,)).astype(np.float32)

    emb_dim = cfg.pretrained_emb_dim * 2 if with_emb else 0
    emb = rng.standard_normal((N, emb_dim)).astype(np.float32) if with_emb else None

    model = build_dcn_model(cfg, prep, emb_dim)
    device = torch.device("cpu")
    trainer = DCNTrainer(model, prep, train_cfg, device)

    loader = make_dataloader(
        numeric,
        cat_indices,
        targets,
        batch_size=train_cfg.runtime.batch_size,
        shuffle=False,
        use_cuda=False,
        pretrained_emb=emb,
    )
    return trainer, loader


class TestDCNTrainer:
    @pytest.mark.parametrize("with_emb", [False, True], ids=["no_emb", "with_emb"])
    def test_train_epoch_finite_loss(self, with_emb):
        trainer, loader = _make_tiny_trainer(with_emb=with_emb)
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert np.isfinite(loss)
        assert loss > 0.0

    def test_loss_decreases_over_epochs(self):
        trainer, loader = _make_tiny_trainer()
        losses = []
        for _ in range(5):
            losses.append(trainer.train_epoch(loader))
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.5f} -> {losses[-1]:.5f}"

    @pytest.mark.parametrize("with_emb", [False, True], ids=["no_emb", "with_emb"])
    def test_predict_shape_and_finiteness(self, with_emb):
        trainer, loader = _make_tiny_trainer(with_emb=with_emb)
        trainer.train_epoch(loader)
        ds = loader.dataset
        kwargs = {}
        if with_emb:
            kwargs["pretrained_emb"] = ds.pretrained_emb.numpy()
        scores = trainer.predict(
            ds.numeric.numpy(),
            ds.cat_indices.numpy(),
            batch_size=16,
            **kwargs,
        )
        assert scores.shape == (len(ds),)
        assert np.isfinite(scores).all()

    def test_model_in_train_mode_after_train_epoch(self):
        trainer, loader = _make_tiny_trainer()
        trainer.train_epoch(loader)
        assert trainer.model.training

    def test_model_in_eval_mode_after_predict(self):
        trainer, loader = _make_tiny_trainer()
        trainer.train_epoch(loader)
        ds = loader.dataset
        trainer.predict(ds.numeric.numpy(), ds.cat_indices.numpy(), batch_size=16)
        assert not trainer.model.training


class TestCosineSchedule:
    def test_create_cosine_schedule(self):
        trainer, _ = _make_tiny_trainer()
        trainer.create_cosine_schedule(total_steps=100)
        assert trainer.scheduler is not None

    def test_warmup_lr_starts_low(self):
        trainer, _ = _make_tiny_trainer()
        trainer.create_cosine_schedule(total_steps=100)
        initial_lr = trainer.scheduler.get_last_lr()[0]
        assert initial_lr < trainer.cfg.optimizer.learning_rate

    def test_lr_recovers_after_warmup(self):
        trainer, loader = _make_tiny_trainer()
        trainer.create_cosine_schedule(total_steps=len(loader))
        trainer.train_epoch(loader)
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr > 0.0


class TestFinetuneLR:
    def test_set_finetune_lr_creates_multiple_groups(self):
        trainer, _ = _make_tiny_trainer()
        lr_config = {
            "embeddings": 1e-5,
            "cross_net": 5e-5,
            "head": 5e-4,
            "default": 1e-4,
        }
        trainer.set_finetune_lr(lr_config)
        assert len(trainer.optimizer.param_groups) >= 2

    def test_set_finetune_lr_all_params_assigned(self):
        trainer, _ = _make_tiny_trainer()
        total_before = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        lr_config = {"default": 1e-4}
        trainer.set_finetune_lr(lr_config)
        total_after = sum(p.numel() for pg in trainer.optimizer.param_groups for p in pg["params"])
        assert total_after == total_before

    def test_scheduler_cleared_after_finetune_lr(self):
        trainer, _ = _make_tiny_trainer()
        trainer.create_cosine_schedule(100)
        assert trainer.scheduler is not None
        trainer.set_finetune_lr({"default": 1e-4})
        trainer.scheduler = None
        assert trainer.scheduler is None
