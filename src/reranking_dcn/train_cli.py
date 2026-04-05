"""CLI entry point for DCN-v2 training (SkyPilot / local).

Usage:
    # Base training from YAML config
    python -m reranking_dcn.train_cli --config configs/base_training.yaml

    # Daily finetune from YAML config
    python -m reranking_dcn.train_cli --config configs/daily_finetune.yaml

    # Override via CLI args
    python -m reranking_dcn.train_cli \
        --config configs/daily_finetune.yaml \
        --stop_date 2026-03-24 \
        --user_sample_pct 5

    # From JSON config string (Airflow / SkyPilot env var)
    python -m reranking_dcn.train_cli --config-str '{"mode":"finetune",...}'

    # Dry-run (print SQL only)
    python -m reranking_dcn.train_cli --config configs/base_training.yaml --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import torch
import yaml

from .config import TrainingConfig
from .training.pipeline import run_base_training_pipeline, run_daily_finetune_pipeline

log = logging.getLogger("reranking_dcn")

_CLI_PASSTHROUGH_ARGS = frozenset({"config", "config_str", "dry_run"})


def _load_config(args: argparse.Namespace) -> TrainingConfig:
    """Build TrainingConfig from YAML file, JSON string, and CLI overrides.

    All sources use a flat key namespace routed to the correct sub-config
    via ``TrainingConfig.from_flat_dict()``.
    """
    merged: dict = {}

    if args.config:
        with open(args.config) as f:
            merged.update(yaml.safe_load(f))

    if args.config_str:
        merged.update(json.loads(args.config_str))

    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in _CLI_PASSTHROUGH_ARGS and v is not None and v != ""
    }
    merged.update(cli_overrides)

    return TrainingConfig.from_flat_dict(merged)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _setup_torch(cfg: TrainingConfig) -> None:
    torch.set_num_threads(cfg.runtime.num_threads)
    torch.set_num_interop_threads(cfg.runtime.num_interop_threads)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))
        log.info(
            "VRAM: %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DCN-v2 Training Pipeline")

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--config-str", type=str, help="JSON config string (from Airflow)")
    parser.add_argument("--dry_run", action="store_true", help="Print SQL and exit")

    parser.add_argument("--mode", type=str, choices=["base_train", "finetune"])
    parser.add_argument("--stop_date", type=str)
    parser.add_argument("--billing_project", type=str)
    parser.add_argument("--local_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--user_sample_pct", type=int)
    parser.add_argument("--training_surface", type=str)
    parser.add_argument("--gcs_bucket", type=str)
    parser.add_argument("--checkpoint_gcs_prefix", type=str)
    parser.add_argument("--nn_base_train_start", type=str)
    parser.add_argument("--nn_base_train_stop", type=str)
    parser.add_argument(
        "--update_latest_checkpoint",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=None,
        help="Upload artifacts to 'latest' GCS path (default: true)",
    )

    args = parser.parse_args()

    _setup_logging()
    cfg = _load_config(args)
    _setup_torch(cfg)

    log.info("=" * 70)
    log.info("DCN-v2 Training Pipeline")
    log.info("  Mode:         %s", cfg.mode)
    log.info("  Stop date:    %s", cfg.stop_date)
    log.info("  Base train:   [%s, %s)", cfg.base_train_start, cfg.base_train_stop)
    log.info("  Project:      %s", cfg.infra.billing_project)
    log.info("  GCS bucket:   %s", cfg.infra.gcs_bucket)
    log.info("  Local dir:    %s", cfg.infra.local_dir)
    log.info("  Output dir:   %s", cfg.infra.output_dir)
    log.info("  User sample:  %d%%", cfg.data.user_sample_pct)
    log.info("  Update latest: %s", cfg.infra.update_latest_checkpoint)
    log.info("=" * 70)

    t0 = time.monotonic()

    if cfg.mode == "base_train":
        run_base_training_pipeline(cfg, dry_run=args.dry_run)
    elif cfg.mode == "finetune":
        run_daily_finetune_pipeline(cfg, dry_run=args.dry_run)
    else:
        log.error("Unknown mode: %s", cfg.mode)
        sys.exit(1)

    elapsed = time.monotonic() - t0
    log.info("Pipeline finished in %.1f minutes.", elapsed / 60)


if __name__ == "__main__":
    main()
