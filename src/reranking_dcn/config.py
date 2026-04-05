"""Configuration dataclasses for DCN-v2 architecture, training, and export.

TrainingConfig is composed from domain-specific sub-configs (DataConfig,
InfraConfig, OptimizerConfig, RuntimeConfig, MonitorConfig) for clear
separation of concerns.  Use ``TrainingConfig.from_flat_dict()`` for
backward compatibility with flat YAML / CLI / Airflow config dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from datetime import date as dt_date, timedelta
from typing import Any, Literal

from .features import PRETRAINED_EMB_DIM


@dataclass
class Config:
    """DCN-v2 architecture and serving configuration."""

    num_quantile_bins: int = 64
    num_emb_dim: int = 4

    pretrained_emb_dim: int = PRETRAINED_EMB_DIM
    use_pretrained_embeddings: bool = True
    projected_emb_dim: int = 64

    dcn_num_cross_layers: int = 3
    dcn_num_experts: int = 4
    dcn_expert_rank: int = 64
    dcn_deep_dims: tuple[int, ...] = (512, 256, 128)
    head_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1

    id_hash_config: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shop_id": (50_000, 16),
        }
    )

    cat_emb_dim_overrides: dict[str, int] = field(default_factory=dict)

    device: str = "cuda"

    def to_serving_arch_dict(
        self,
        num_continuous: int,
        vocab_sizes: dict[str, int],
    ) -> dict[str, Any]:
        """Build the architecture dict for serving-time model_config.json.

        Replaces the standalone ModelArchConfig dataclass by deriving
        all serving-relevant fields from Config + preprocessor metadata.
        """
        return {
            "num_continuous": num_continuous,
            "n_quantile_bins": self.num_quantile_bins,
            "num_emb_dim": self.num_emb_dim,
            "pretrained_emb_dim": (
                self.pretrained_emb_dim if self.use_pretrained_embeddings else 0
            ),
            "projected_emb_dim": self.projected_emb_dim,
            "num_cross_layers": self.dcn_num_cross_layers,
            "num_experts": self.dcn_num_experts,
            "expert_rank": self.dcn_expert_rank,
            "deep_dims": list(self.dcn_deep_dims),
            "head_dims": list(self.head_dims),
            "dropout": self.dropout,
            "num_cat_features": len(vocab_sizes),
            "cat_feature_order": sorted(vocab_sizes.keys()),
        }


# ---------------------------------------------------------------------------
# Training configuration — composed from domain-specific sub-configs
# ---------------------------------------------------------------------------


def _table(project: str, dataset: str, table: str) -> str:
    return f"`{project}.{dataset}.{table}`"


_PRD_PROJECT = "sdp-prd-shop-ml"


@dataclass
class DataConfig:
    """What data to fetch: source tables, filters, date windows.

    Table defaults use the production project. Override ``source_project``
    to point at a staging project for dev/test runs without accidentally
    querying production data.
    """

    source_project: str = _PRD_PROJECT

    source_table: str = ""
    nomic_embedding_table: str = ""
    user_embedding_table: str = ""
    user_embedding_history_table: str = ""

    negative_sample_scale: int = 20
    user_sample_pct: int = 100
    training_surface: str = "all"

    nn_base_train_start: str = ""
    nn_base_train_stop: str = ""
    finetune_data_window_days: int = 1

    def __post_init__(self) -> None:
        p = self.source_project
        if not self.source_table:
            self.source_table = _table(
                p,
                "advertising_unified_recs",
                "unified_recs__product_ranking__recommended_pages_training_v1",
            )
        if not self.nomic_embedding_table:
            self.nomic_embedding_table = _table(
                p,
                "feature_store",
                "feature_store__inference__feature__nomic_product_document_embedding_v1",
            )
        if not self.user_embedding_table:
            self.user_embedding_table = _table(
                p,
                "advertising_feature_store_v2",
                "feature_store_v2__inference__user_average_embedding",
            )
        if not self.user_embedding_history_table:
            self.user_embedding_history_table = _table(
                p,
                "advertising_feature_store_v2",
                "feature_store_v2__primitive__user_average_embedding_history",
            )

    @property
    def surface_filter(self) -> str:
        if self.training_surface == "boosted_merchants":
            return (
                "page_type = 'boosted_merchants' AND campaign_id IS NOT NULL "
                "AND section_id NOT LIKE '%recently_viewed%'"
            )
        return (
            "page_type IN ('home_feed', 'boosted_merchants') "
            "AND section_id NOT LIKE '%recently_viewed%'"
        )


@dataclass
class InfraConfig:
    """Where to run: GCP project, GCS paths, local directories."""

    billing_project: str = "sdp-stg-shop-ml"
    local_dir: str = "./data"
    output_dir: str = "./outputs"
    gcs_bucket: str = "sdp-stg-shop-ml-temp"
    output_gcs_prefix: str = "dcn_v2"
    checkpoint_gcs_prefix: str = ""
    update_latest_checkpoint: bool = True


@dataclass
class OptimizerConfig:
    """How to train: loss, learning rates, schedule."""

    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0

    use_cosine_lr: bool = True
    cosine_warmup_fraction: float = 0.05
    cosine_min_lr_fraction: float = 0.01

    finetune_lrs: dict[str, float] = field(
        default_factory=lambda: {
            "embeddings": 1e-5,
            "numeric_emb": 1e-5,
            "cross_net": 5e-5,
            "product_proj": 1e-4,
            "user_proj": 1e-4,
            "deep_net": 1e-4,
            "head": 5e-4,
            "default": 1e-4,
        }
    )


@dataclass
class RuntimeConfig:
    """Hardware and batch configuration."""

    batch_size: int = 2048
    num_workers: int = 16
    num_threads: int = 128
    num_interop_threads: int = 8
    epochs: int = 1


@dataclass
class MonitorConfig:
    """Experiment tracking configuration."""

    comet_project: str = "dcn-v2-training"
    comet_workspace: str = ""
    comet_secret_project: str = "shopify-comet-production"
    comet_secret_id: str = "sa-shop-ml-personalization"


@dataclass
class TrainingConfig:
    """Top-level configuration for DCN-v2 training / daily fine-tune pipeline.

    Composed from domain-specific sub-configs for clear separation of concerns.
    Use ``from_flat_dict()`` for backward compatibility with flat YAML/CLI configs.
    """

    mode: Literal["base_train", "finetune"] = "finetune"
    stop_date: str = "2026-03-08"

    data: DataConfig = field(default_factory=DataConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    _BASE_TRAIN_WINDOW_DAYS: int = 30

    # ── Computed date anchors ──

    @property
    def _stop(self) -> dt_date:
        return dt_date.fromisoformat(self.stop_date)

    @property
    def base_train_start(self) -> str:
        """Start of the base-training window. Falls back to stop_date - 30d."""
        return (
            self.data.nn_base_train_start
            or (self._stop - timedelta(days=self._BASE_TRAIN_WINDOW_DAYS)).isoformat()
        )

    @property
    def base_train_stop(self) -> str:
        """End of the base-training window. Falls back to stop_date."""
        return self.data.nn_base_train_stop or self.stop_date

    @property
    def finetune_day_start(self) -> str:
        """Start of the single-day fine-tune window (daily mode)."""
        d = self._stop - timedelta(days=self.data.finetune_data_window_days)
        return d.isoformat()

    @property
    def finetune_day_stop(self) -> str:
        return self.stop_date

    @property
    def checkpoint_gcs_path(self) -> str:
        prefix = self.infra.checkpoint_gcs_prefix or (
            f"gs://{self.infra.gcs_bucket}/{self.infra.output_gcs_prefix}/latest"
        )
        return f"{prefix}/checkpoint.pt"

    @property
    def artifacts_gcs_dir(self) -> str:
        return f"gs://{self.infra.gcs_bucket}/{self.infra.output_gcs_prefix}/latest"

    @property
    def artifacts_gcs_dated_dir(self) -> str:
        return f"gs://{self.infra.gcs_bucket}/{self.infra.output_gcs_prefix}/{self.stop_date}"

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten config to a single-level dict for logging/monitoring."""
        result: dict[str, Any] = {"mode": self.mode, "stop_date": self.stop_date}
        for group_name in ("data", "infra", "optimizer", "runtime", "monitor"):
            group = getattr(self, group_name)
            for f in dc_fields(type(group)):
                result[f.name] = getattr(group, f.name)
        return result

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Build from a flat dict for backward compatibility with YAML/CLI configs.

        Automatically routes each key to the correct sub-config based on field
        membership.  Also accepts already-nested dicts (pass-through).
        Unknown keys are silently ignored.
        """
        _sub_config_classes: dict[str, type] = {
            "data": DataConfig,
            "infra": InfraConfig,
            "optimizer": OptimizerConfig,
            "runtime": RuntimeConfig,
            "monitor": MonitorConfig,
        }

        top_field_names = {f.name for f in dc_fields(cls)} - set(_sub_config_classes)
        sub_field_map: dict[str, str] = {}
        for group_name, klass in _sub_config_classes.items():
            for f in dc_fields(klass):
                sub_field_map[f.name] = group_name

        top_kwargs: dict[str, Any] = {}
        group_kwargs: dict[str, dict[str, Any]] = {g: {} for g in _sub_config_classes}

        for key, val in d.items():
            if key in _sub_config_classes and isinstance(val, dict):
                group_kwargs[key].update(val)
            elif key in top_field_names:
                top_kwargs[key] = val
            elif key in sub_field_map:
                group_kwargs[sub_field_map[key]][key] = val

        return cls(
            **top_kwargs,
            **{name: klass(**group_kwargs[name]) for name, klass in _sub_config_classes.items()},
        )
