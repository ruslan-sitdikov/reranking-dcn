"""Configuration dataclasses for DCN-v2 architecture and export."""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import PRETRAINED_EMB_DIM


@dataclass
class Config:
    """DCN-v2 architecture and serving configuration.

    Training-only settings (GCP tables, walk-forward windows, etc.)
    live in the training script, not here.
    """

    # Quantile binning
    num_quantile_bins: int = 64
    num_emb_dim: int = 4

    # Pre-trained embedding dimensions (768d Nomic product + 768d user-avg)
    pretrained_emb_dim: int = PRETRAINED_EMB_DIM
    use_pretrained_embeddings: bool = True
    projected_emb_dim: int = 64

    # DCN-v2 architecture
    dcn_num_cross_layers: int = 3
    dcn_num_experts: int = 4
    dcn_expert_rank: int = 64
    dcn_deep_dims: tuple[int, ...] = (512, 256, 128)
    head_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1

    # Hashed ID embeddings (only broker_id; product/user use Nomic embeddings)
    #   broker_id: ~723K unique → 50K buckets, dim 16
    id_hash_config: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "broker_id": (50_000, 16),
        }
    )

    cat_emb_dim_overrides: dict[str, int] = field(default_factory=dict)

    device: str = "cuda"


@dataclass
class ModelArchConfig:
    """Minimal config needed to reconstruct DCNv2Ranker for ONNX loading."""

    num_continuous: int
    n_quantile_bins: int
    num_emb_dim: int
    pretrained_emb_dim: int
    projected_emb_dim: int
    num_cross_layers: int
    num_experts: int
    expert_rank: int
    deep_dims: list[int]
    head_dims: list[int]
    dropout: float
    num_cat_features: int
    cat_feature_order: list[str]
