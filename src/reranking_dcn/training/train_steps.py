"""Training steps for base training and daily fine-tune.

Loads data, fits the preprocessor, builds the model, and runs the
training loop.  Returns (trainer, preprocessor) for downstream
ONNX export and checkpointing.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import polars as pl
import torch

from ..config import Config, TrainingConfig
from ..data.dataset import make_dataloader
from ..data.embeddings import (
    load_embedding_lookups,
    lookup_pretrained_embeddings,
)
from ..data.exporter import Exporter
from ..export.checkpoint import build_model_from_config
from ..nn.ranker import DCNv2Ranker
from ..preprocessing.gpu_preprocessor import GPUFeaturePreprocessor
from .monitoring import CometMonitor
from .trainer import DCNTrainer

log = logging.getLogger(__name__)


def _free_memory(label: str = "", *, cuda: bool = False) -> None:
    gc.collect()
    if cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if label:
        log.debug("  Memory freed: %s", label)


def _prep_split(local_dir: Path, split: str) -> pl.DataFrame:
    return Exporter.prepare(Exporter.load(local_dir, split))


def build_dcn_model(
    arch_cfg: Config,
    preprocessor: GPUFeaturePreprocessor,
    pretrained_emb_dim: int,
) -> DCNv2Ranker:
    """Construct a DCNv2Ranker from architecture config and fitted preprocessor."""
    return build_model_from_config(preprocessor, arch_cfg, pretrained_emb_dim=pretrained_emb_dim)


def fit_preprocessor(
    arch_cfg: Config,
    df: pl.DataFrame,
) -> GPUFeaturePreprocessor:
    """Create and fit a GPUFeaturePreprocessor on the given DataFrame."""
    preprocessor = GPUFeaturePreprocessor(
        id_hash_config=arch_cfg.id_hash_config,
        cat_emb_dim_overrides=arch_cfg.cat_emb_dim_overrides,
        n_quantile_bins=arch_cfg.num_quantile_bins,
    ).fit(df)
    return preprocessor


def run_base_training(
    train_cfg: TrainingConfig,
    arch_cfg: Config,
    local_dir: Path,
    device: torch.device,
    monitor: CometMonitor | None = None,
) -> tuple[DCNTrainer, GPUFeaturePreprocessor]:
    """Base training: fit preprocessor, train 1 epoch on full base-train window.

    Returns ``(trainer, preprocessor)`` for downstream ONNX export or checkpointing.
    """
    use_cuda = device.type == "cuda"

    log.info("=" * 70)
    log.info(
        "BASE TRAINING: [%s, %s)",
        train_cfg.data.nn_base_train_start,
        train_cfg.data.nn_base_train_stop,
    )
    log.info("=" * 70)

    df_train = _prep_split(local_dir, "nn_base_train")
    log.info("  nn_base_train: %s rows", f"{df_train.shape[0]:,}")

    preprocessor = fit_preprocessor(arch_cfg, df_train)
    product_lookup, user_lookup = load_embedding_lookups(local_dir)

    numeric, cat_indices, _, targets = preprocessor.transform(df_train)
    emb = lookup_pretrained_embeddings(df_train, product_lookup, user_lookup)
    pretrained_emb_dim = emb.shape[1] if emb is not None else 0
    del df_train
    _free_memory(cuda=use_cuda)

    model = build_dcn_model(arch_cfg, preprocessor, pretrained_emb_dim)
    trainer = DCNTrainer(model, preprocessor, train_cfg, device)
    if monitor is not None:
        trainer.set_monitor(monitor)

    loader = make_dataloader(
        numeric,
        cat_indices,
        targets,
        batch_size=train_cfg.runtime.batch_size,
        shuffle=True,
        num_workers=train_cfg.runtime.num_workers,
        use_cuda=use_cuda,
        pretrained_emb=emb,
    )

    if train_cfg.optimizer.use_cosine_lr:
        trainer.create_cosine_schedule(len(loader))

    for epoch in range(train_cfg.runtime.epochs):
        loss = trainer.train_epoch(loader)
        log.info("  Epoch %d/%d loss: %.5f", epoch + 1, train_cfg.runtime.epochs, loss)

    del loader, numeric, cat_indices, emb, targets
    _free_memory("after base training", cuda=use_cuda)

    return trainer, preprocessor


def run_finetune(
    train_cfg: TrainingConfig,
    arch_cfg: Config,
    local_dir: Path,
    device: torch.device,
    checkpoint_path: Path,
    monitor: CometMonitor | None = None,
) -> tuple[DCNTrainer, GPUFeaturePreprocessor]:
    """Daily fine-tune: load checkpoint, train 1 epoch on fresh data with layer-wise LRs.

    Returns ``(trainer, preprocessor)`` for ONNX export and GCS upload.
    """
    use_cuda = device.type == "cuda"

    log.info("=" * 70)
    log.info(
        "FINE-TUNE: [%s, %s)  %dd from checkpoint %s",
        train_cfg.finetune_day_start,
        train_cfg.finetune_day_stop,
        train_cfg.data.finetune_data_window_days,
        checkpoint_path,
    )
    log.info("=" * 70)

    checkpoint = _load_training_checkpoint(checkpoint_path, device)

    if "preprocessor_state" not in checkpoint:
        raise ValueError(
            "Checkpoint missing 'preprocessor_state'. Legacy v1 (pickled) checkpoints "
            "are no longer supported. Re-run base training to produce a v2 checkpoint."
        )
    preprocessor = GPUFeaturePreprocessor.from_state(checkpoint["preprocessor_state"])
    preprocessor = preprocessor.to(device)

    product_lookup, user_lookup = load_embedding_lookups(local_dir)
    pretrained_emb_dim = checkpoint.get("pretrained_emb_dim", 0)

    model = build_dcn_model(arch_cfg, preprocessor, pretrained_emb_dim)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(
            f"Checkpoint missing 'model_state_dict' (and legacy 'state_dict'). "
            f"Found keys: {sorted(checkpoint.keys())}."
        )
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    trainer = DCNTrainer(model, preprocessor, train_cfg, device)
    if monitor is not None:
        trainer.set_monitor(monitor)

    if "optimizer_state_dict" in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    trainer.set_finetune_lr(train_cfg.optimizer.finetune_lrs)
    trainer.scheduler = None

    df_finetune = _prep_split(local_dir, "nn_finetune")
    log.info("  finetune data: %s rows", f"{df_finetune.shape[0]:,}")

    numeric, cat_indices, _, targets = preprocessor.transform(df_finetune)
    emb = lookup_pretrained_embeddings(df_finetune, product_lookup, user_lookup)
    del df_finetune
    _free_memory(cuda=use_cuda)

    loader = make_dataloader(
        numeric,
        cat_indices,
        targets,
        batch_size=train_cfg.runtime.batch_size,
        shuffle=True,
        num_workers=train_cfg.runtime.num_workers,
        use_cuda=use_cuda,
        pretrained_emb=emb,
    )

    for epoch in range(train_cfg.runtime.epochs):
        loss = trainer.train_epoch(loader)
        log.info("  Fine-tune epoch %d/%d loss: %.5f", epoch + 1, train_cfg.runtime.epochs, loss)

    del loader, numeric, cat_indices, emb, targets
    _free_memory("after finetune", cuda=use_cuda)

    return trainer, preprocessor


def _load_training_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """Load a v2 training checkpoint with ``weights_only=True``.

    Only safe (non-pickle) v2 checkpoints are supported.  Legacy v1
    checkpoints that contain pickled objects will raise an error --
    re-run base training to produce a v2 checkpoint.
    """
    return torch.load(checkpoint_path, map_location=device, weights_only=True)
