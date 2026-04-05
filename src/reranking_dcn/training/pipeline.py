"""Top-level training orchestrators for base training and daily fine-tune.

Both modes share a single ``_run_pipeline`` core that handles:
  data export -> train -> checkpoint -> ONNX export -> validate -> upload

The only differences are injected via callables (Strategy pattern):
  - which data export function to run
  - which training function to call
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import torch

from ..config import Config, TrainingConfig
from ..data.exporter import Exporter
from ..data.sql import sql_nn_base_train, sql_nn_finetune_day
from ..export.onnx_export import export_onnx
from ..export.validation import validate_parity
from ..preprocessing.gpu_preprocessor import GPUFeaturePreprocessor
from ..serving.artifacts import serialize_preprocessing_artifacts
from .monitoring import CometMonitor
from .train_steps import run_base_training, run_finetune
from .trainer import DCNTrainer

log = logging.getLogger(__name__)

TrainFn = Callable[
    [TrainingConfig, Config, Path, torch.device, CometMonitor | None],
    tuple[DCNTrainer, GPUFeaturePreprocessor],
]


def _resolve_device(cfg: TrainingConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_checkpoint(
    trainer: DCNTrainer,
    preprocessor: GPUFeaturePreprocessor,
    output_dir: Path,
    pretrained_emb_dim: int,
    mode: str,
) -> Path:
    """Save model checkpoint with safe preprocessor serialization (v2 format).

    Uses ``GPUFeaturePreprocessor.save_state()`` instead of pickling the
    entire nn.Module, enabling ``weights_only=True`` on load and
    version-independent deserialization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"

    base_model = getattr(trainer.model, "_orig_mod", trainer.model)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in base_model.state_dict().items()}

    torch.save(
        {
            "checkpoint_version": 2,
            "model_state_dict": state_dict,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "preprocessor_state": preprocessor.save_state(),
            "pretrained_emb_dim": pretrained_emb_dim,
            "mode": mode,
            "timestamp": time.time(),
        },
        checkpoint_path,
    )
    log.info("  Checkpoint saved (v2): %s", checkpoint_path)
    return checkpoint_path


def _upload_to_gcs(
    local_dir: Path,
    gcs_dir: str,
    project: str,
    max_retries: int = 4,
) -> None:
    """Upload all files in local_dir to the given GCS directory with retry."""
    from google.cloud import storage

    if not gcs_dir.startswith("gs://"):
        log.warning("  Skipping GCS upload: invalid prefix %s", gcs_dir)
        return

    parts = gcs_dir.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    for local_file in local_dir.iterdir():
        if not local_file.is_file():
            continue
        blob_name = f"{prefix}/{local_file.name}" if prefix else local_file.name
        blob = bucket.blob(blob_name)

        for attempt in range(1, max_retries + 1):
            try:
                blob.upload_from_filename(str(local_file))
                log.info("  Uploaded %s -> gs://%s/%s", local_file.name, bucket_name, blob_name)
                break
            except Exception as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to upload {local_file.name} to gs://{bucket_name}/{blob_name} "
                        f"after {max_retries} attempts"
                    ) from exc
                wait = min(2**attempt, 30)
                log.warning(
                    "  Upload attempt %d/%d for %s failed (%s), retrying in %ds...",
                    attempt,
                    max_retries,
                    local_file.name,
                    exc,
                    wait,
                )
                time.sleep(wait)


def _download_checkpoint(gcs_path: str, local_path: Path, project: str) -> Path:
    """Download a single checkpoint file from GCS."""
    from google.cloud import storage

    if not gcs_path.startswith("gs://"):
        return Path(gcs_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))
    log.info("  Downloaded checkpoint: %s -> %s", gcs_path, local_path)
    return local_path


# ---------------------------------------------------------------------------
# Unified pipeline core (Strategy pattern)
# ---------------------------------------------------------------------------


def _run_pipeline(
    train_cfg: TrainingConfig,
    arch_cfg: Config,
    *,
    export_data_fn: Callable[[Exporter], None],
    train_fn: TrainFn,
    dry_run_sql_fn: Callable[[TrainingConfig], str],
    dry_run: bool = False,
) -> Path:
    """Shared pipeline: export -> train -> checkpoint -> ONNX -> validate -> upload.

    All mode-specific behaviour is injected via the callable arguments,
    eliminating the duplication between base training and fine-tune.
    """
    device = _resolve_device(train_cfg)
    local_dir = Path(train_cfg.infra.local_dir)
    output_dir = Path(train_cfg.infra.output_dir)

    if dry_run:
        log.info("DRY RUN: SQL for %s:", train_cfg.mode)
        log.info(dry_run_sql_fn(train_cfg))
        return output_dir

    monitor = CometMonitor(
        project_name=train_cfg.monitor.comet_project,
        experiment_name=f"{train_cfg.mode}-{train_cfg.stop_date}",
        tags=[train_cfg.mode, train_cfg.stop_date],
        workspace=train_cfg.monitor.comet_workspace or None,
        secret_project=train_cfg.monitor.comet_secret_project,
        secret_id=train_cfg.monitor.comet_secret_id,
    )
    if monitor.active:
        monitor.log_params(train_cfg.to_flat_dict())

    exporter = Exporter(train_cfg)
    log.info("Exporting %s data...", train_cfg.mode)
    export_data_fn(exporter)

    trainer, preprocessor = train_fn(
        train_cfg,
        arch_cfg,
        local_dir,
        device,
        monitor,
    )

    raw_model = getattr(trainer.model, "_orig_mod", trainer.model)
    pretrained_emb_dim = raw_model._pretrained_emb_dim
    _save_checkpoint(trainer, preprocessor, output_dir, pretrained_emb_dim, train_cfg.mode)

    log.info("Exporting ONNX model...")
    onnx_paths = export_onnx(
        model=raw_model,
        preprocessor=preprocessor,
        cfg=arch_cfg,
        output_dir=output_dir,
    )

    log.info("Validating ONNX parity...")
    validate_parity(
        pytorch_model=raw_model,
        preprocessor=preprocessor,
        onnx_paths=onnx_paths,
        cfg=arch_cfg,
    )

    log.info("Serializing preprocessing artifacts for serving...")
    serialize_preprocessing_artifacts(preprocessor, arch_cfg, output_dir)

    billing = train_cfg.infra.billing_project
    log.info("Uploading artifacts to GCS (dated: %s)...", train_cfg.stop_date)
    _upload_to_gcs(output_dir, train_cfg.artifacts_gcs_dated_dir, billing)

    if train_cfg.infra.update_latest_checkpoint:
        log.info("Updating latest artifacts at %s ...", train_cfg.artifacts_gcs_dir)
        _upload_to_gcs(output_dir, train_cfg.artifacts_gcs_dir, billing)
    else:
        log.info(
            "Skipping latest upload (update_latest_checkpoint=false). " "Dated artifacts: %s",
            train_cfg.artifacts_gcs_dated_dir,
        )

    monitor.end()
    log.info("%s pipeline complete. Artifacts in %s", train_cfg.mode, output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_base_training_pipeline(
    train_cfg: TrainingConfig,
    arch_cfg: Config | None = None,
    dry_run: bool = False,
) -> Path:
    """Full base-training pipeline: export -> train -> ONNX -> upload.

    Returns the local output directory containing checkpoint and ONNX model.
    """
    return _run_pipeline(
        train_cfg,
        arch_cfg or Config(),
        export_data_fn=lambda exp: exp.run_base_training(),
        train_fn=run_base_training,
        dry_run_sql_fn=sql_nn_base_train,
        dry_run=dry_run,
    )


def run_daily_finetune_pipeline(
    train_cfg: TrainingConfig,
    arch_cfg: Config | None = None,
    dry_run: bool = False,
) -> Path:
    """Full daily fine-tune pipeline: download checkpoint -> export -> train -> ONNX -> upload.

    Returns the local output directory containing updated checkpoint and ONNX model.
    """
    local_dir = Path(train_cfg.infra.local_dir)

    def _finetune_with_checkpoint(
        t_cfg: TrainingConfig,
        a_cfg: Config,
        l_dir: Path,
        device: torch.device,
        monitor: CometMonitor | None = None,
    ) -> tuple[DCNTrainer, GPUFeaturePreprocessor]:
        checkpoint_local = local_dir / "checkpoint.pt"
        _download_checkpoint(
            t_cfg.checkpoint_gcs_path,
            checkpoint_local,
            t_cfg.infra.billing_project,
        )
        return run_finetune(
            t_cfg,
            a_cfg,
            l_dir,
            device,
            checkpoint_path=checkpoint_local,
            monitor=monitor,
        )

    return _run_pipeline(
        train_cfg,
        arch_cfg or Config(),
        export_data_fn=lambda exp: exp.run_finetune(),
        train_fn=_finetune_with_checkpoint,
        dry_run_sql_fn=sql_nn_finetune_day,
        dry_run=dry_run,
    )
