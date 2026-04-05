"""Model construction and checkpoint loading utilities."""

from __future__ import annotations

import logging

import torch

from ..config import Config
from ..nn.ranker import DCNv2Ranker
from ..preprocessing.gpu_preprocessor import GPUFeaturePreprocessor

log = logging.getLogger(__name__)

PREPROCESSOR_STATE_KEYS = frozenset({"state_dict", "cat_encoders", "vocab_sizes", "embedding_dims"})


def build_model_from_config(
    preprocessor: GPUFeaturePreprocessor,
    cfg: Config,
    pretrained_emb_dim: int | None = None,
) -> DCNv2Ranker:
    """Construct a DCNv2Ranker from config and preprocessor state.

    Args:
        pretrained_emb_dim: Override for the pretrained embedding dimension.
            If None, derived from ``cfg.use_pretrained_embeddings`` and
            ``cfg.pretrained_emb_dim``.
    """
    if pretrained_emb_dim is None:
        pretrained_emb_dim = cfg.pretrained_emb_dim * 2 if cfg.use_pretrained_embeddings else 0

    return DCNv2Ranker(
        num_continuous=preprocessor.num_continuous,
        vocab_sizes=preprocessor.vocab_sizes,
        embedding_dims=preprocessor.embedding_dims,
        n_quantile_bins=cfg.num_quantile_bins,
        num_emb_dim=cfg.num_emb_dim,
        pretrained_emb_dim=pretrained_emb_dim,
        projected_emb_dim=cfg.projected_emb_dim,
        num_cross_layers=cfg.dcn_num_cross_layers,
        num_experts=cfg.dcn_num_experts,
        expert_rank=cfg.dcn_expert_rank,
        deep_dims=cfg.dcn_deep_dims,
        head_dims=cfg.head_dims,
        dropout=cfg.dropout,
    )


def load_checkpoint(model: DCNv2Ranker, checkpoint_path: str) -> DCNv2Ranker:
    """Load weights from a training checkpoint, stripping torch.compile prefixes."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
        raise ValueError(
            f"Checkpoint missing 'model_state_dict' (and legacy 'state_dict'). "
            f"Found keys: {sorted(ckpt.keys())}. "
            "File may be corrupted or from an incompatible version."
        )
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    log.info(
        "  Loaded checkpoint: mode=%s, version=%s",
        ckpt.get("mode"),
        ckpt.get("checkpoint_version", 1),
    )
    return model
