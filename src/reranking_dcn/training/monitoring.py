"""Comet ML integration for DCN-v2 training metrics."""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)

_DEFAULT_SECRET_PROJECT = "shopify-comet-production"
_DEFAULT_SECRET_ID = "sa-shop-ml-personalization"


def _resolve_api_key(
    secret_project: str = _DEFAULT_SECRET_PROJECT,
    secret_id: str = _DEFAULT_SECRET_ID,
) -> str | None:
    """Resolve Comet API key: env var first, then GCP Secret Manager.

    Uses ``google.cloud.secretmanager`` directly instead of comet_ml's
    lazy-reference wrapper which returns a base64 placeholder on newer
    comet_ml versions (3.57+) when it cannot resolve the secret itself.
    """
    api_key = os.environ.get("COMET_API_KEY")
    if api_key:
        log.info("  Comet API key found in COMET_API_KEY env var")
        return api_key

    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{secret_project}/secrets/{secret_id}/versions/latest"
        log.info("  Fetching Comet API key from Secret Manager (%s)...", secret_name)
        response = client.access_secret_version(name=secret_name)
        api_key = response.payload.data.decode("UTF-8").strip()
        if api_key:
            return api_key
    except ImportError:
        log.info(
            "  google.cloud.secretmanager not available — pip install google-cloud-secret-manager"
        )
    except Exception as exc:
        log.warning("  Failed to fetch Comet API key from Secret Manager: %s", exc)

    return None


class CometMonitor:
    """Lightweight wrapper around ``comet_ml.Experiment`` for training metrics.

    Logs per-batch loss and LR, per-epoch aggregate metrics, and GPU
    utilisation (if available).  Falls back gracefully when ``comet_ml``
    is not installed or the API key is absent.

    API key resolution order:
      1. ``COMET_API_KEY`` environment variable
      2. GCP Secret Manager (``shopify-comet-production`` project)
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str | None = None,
        tags: list[str] | None = None,
        workspace: str | None = None,
        secret_project: str = _DEFAULT_SECRET_PROJECT,
        secret_id: str = _DEFAULT_SECRET_ID,
        disabled: bool = False,
    ) -> None:
        self._experiment = None
        self._step = 0
        self._epoch = 0

        if disabled:
            log.info("  CometMonitor: explicitly disabled")
            return

        api_key = _resolve_api_key(secret_project, secret_id)
        if not api_key:
            log.info("  CometMonitor: disabled (no API key found)")
            return

        try:
            import comet_ml  # type: ignore[import-untyped]

            self._experiment = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
                auto_metric_logging=False,
                auto_param_logging=False,
            )
            if experiment_name:
                self._experiment.set_name(experiment_name)
            if tags:
                self._experiment.add_tags(tags)
            log.info("  CometMonitor: experiment %s", self._experiment.get_key())
        except Exception as exc:
            log.warning("  CometMonitor: failed to initialise (%s)", exc)

    @property
    def active(self) -> bool:
        return self._experiment is not None

    def log_params(self, params: dict) -> None:
        if self._experiment is not None:
            self._experiment.log_parameters(params)

    def log_batch(self, batch_idx: int, loss: float, lr: float) -> None:
        if self._experiment is None:
            return
        self._step += 1
        self._experiment.log_metric("batch_loss", loss, step=self._step)
        self._experiment.log_metric("learning_rate", lr, step=self._step)

        if self._step % 200 == 0 and torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            self._experiment.log_metric("gpu_mem_gb", mem_gb, step=self._step)

    def log_epoch(self, loss: float, **extra: float) -> None:
        if self._experiment is None:
            return
        self._epoch += 1
        self._experiment.log_metric("epoch_loss", loss, epoch=self._epoch)
        for key, val in extra.items():
            self._experiment.log_metric(key, val, epoch=self._epoch)

    def end(self) -> None:
        if self._experiment is not None:
            self._experiment.end()
