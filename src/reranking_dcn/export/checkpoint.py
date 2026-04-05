"""Model construction and checkpoint loading utilities."""

from __future__ import annotations

import logging

import torch

from ..config import Config
from ..nn.ranker import DCNv2Ranker
from ..preprocessing.gpu_preprocessor import GPUFeaturePreprocessor

log = logging.getLogger(__name__)

PREPROCESSOR_CHECKPOINT_KEYS = frozenset(
    {"state_dict", "cat_encoders", "vocab_sizes", "embedding_dims"}
)

CHECKPOINT_REQUIRED_KEY = "state_dict"


def build_model_from_config(
    preprocessor: GPUFeaturePreprocessor,
    cfg: Config,
) -> DCNv2Ranker:
    """Construct a DCNv2Ranker from config and preprocessor state."""
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
    if CHECKPOINT_REQUIRED_KEY not in ckpt:
        raise ValueError(
            f"Checkpoint missing '{CHECKPOINT_REQUIRED_KEY}' key. "
            f"Found keys: {sorted(ckpt.keys())}. "
            "File may be corrupted or from an incompatible version."
        )
    state_dict = ckpt["state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    log.info(
        "  Loaded checkpoint: variant=%s, walk_forward_weeks=%s",
        ckpt.get("variant"),
        ckpt.get("walk_forward_weeks"),
    )
    return model
