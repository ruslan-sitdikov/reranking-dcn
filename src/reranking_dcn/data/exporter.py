"""BigQuery -> local Parquet exporter via GCS staging.

Pipeline per split:
  1. Export training data first (with negative sampling)
  2. Extract unique product_id / (unified_buyer_id, day) from downloaded Parquet
  3. Upload IDs to BQ temp tables
  4. Export product + user embeddings in parallel (only for IDs in training data)
  5. ``transfer_manager`` parallel download to local disk + background GCS cleanup
"""

from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from ..config import TrainingConfig
from ..features import CAT_FEATURES, NUMERIC_FEATURES, PRETRAINED_EMB_DIM
from .bq_staging import BQStagingManager
from .gcs_transfer import cleanup_gcs_prefix, download_from_gcs, make_gcs_client
from .sql import (
    sql_nn_base_train,
    sql_nn_finetune_day,
    sql_product_embeddings,
    sql_user_embeddings,
)

log = logging.getLogger(__name__)

ALL_PRETRAINED_EMB_COLS = tuple(f"product_emb_{i}" for i in range(PRETRAINED_EMB_DIM)) + tuple(
    f"user_emb_{i}" for i in range(PRETRAINED_EMB_DIM)
)


class Exporter:
    """BQ -> local Parquet via EXPORT DATA to GCS + parallel download."""

    _GCS_TMP_ROOT = "pconv_export_tmp"
    _EXPORT_WORKERS = 3

    def __init__(self, cfg: TrainingConfig) -> None:
        from google.cloud import bigquery

        self.cfg = cfg
        self.client = bigquery.Client(project=cfg.infra.billing_project)
        self.gcs_client = make_gcs_client(cfg.infra.billing_project)
        self.local_dir = Path(cfg.infra.local_dir)
        self._staging = BQStagingManager(self.client, cfg.infra.billing_project)

    # -- GCS helpers -------------------------------------------------------

    def _gcs_prefix(self, name: str) -> str:
        ts = int(time.time())
        return f"{self._GCS_TMP_ROOT}/{name}_{ts}"

    def _gcs_uri(self, prefix: str) -> str:
        return f"gs://{self.cfg.infra.gcs_bucket}/{prefix}"

    # -- Core export -------------------------------------------------------

    def query_to_parquet(self, sql: str, name: str) -> None:
        """Run a BQ EXPORT DATA query and download results to local Parquet."""
        dest = self.local_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        prefix = self._gcs_prefix(name)
        gcs_uri = self._gcs_uri(prefix)

        export_sql = (
            f"EXPORT DATA OPTIONS(\n"
            f"  uri='{gcs_uri}/*.parquet',\n"
            f"  format='PARQUET',\n"
            f"  compression='SNAPPY',\n"
            f"  overwrite=true\n"
            f") AS\n{sql}"
        )

        log.info("Exporting '%s' -> %s ...", name, gcs_uri)
        t0 = time.monotonic()

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            job = self.client.query(export_sql)
            while not job.done():
                time.sleep(5)
            if not job.errors:
                break
            if attempt < max_retries:
                wait = 30 * attempt
                log.warning(
                    "  '%s': BQ EXPORT attempt %d/%d failed (%s), retrying in %ds...",
                    name,
                    attempt,
                    max_retries,
                    job.errors,
                    wait,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"BQ EXPORT job {job.job_id} failed after {max_retries} attempts: {job.errors}"
                )

        export_secs = time.monotonic() - t0
        log.info("  '%s': BQ EXPORT done in %.1fs", name, export_secs)

        t1 = time.monotonic()
        n_shards = download_from_gcs(
            self.cfg.infra.gcs_bucket,
            prefix,
            dest,
            self.cfg.infra.billing_project,
        )
        dl_secs = time.monotonic() - t1

        total_bytes = sum(f.stat().st_size for f in dest.glob("*.parquet"))
        log.info(
            "  '%s': %d shards (%.1f MB) downloaded in %.1fs (%.1f MB/s)",
            name,
            n_shards,
            total_bytes / 1e6,
            dl_secs,
            total_bytes / 1e6 / max(dl_secs, 0.01),
        )
        cleanup_gcs_prefix(self.cfg.infra.gcs_bucket, prefix, self.cfg.infra.billing_project)

    # -- High-level export -------------------------------------------------

    def _export_embeddings(self, training_split: str) -> None:
        """Extract IDs from training data, upload to BQ, export embeddings in parallel."""
        cfg = self.cfg
        product_ids, buyer_days = self._staging.extract_ids_from_parquet(
            self.local_dir, training_split
        )

        log.info("Uploading training IDs to BQ temp tables...")
        product_table = self._staging.upload_product_ids(product_ids)
        buyer_table = self._staging.upload_buyer_days(buyer_days)
        del product_ids, buyer_days

        emb_splits: dict[str, str] = {
            "product_embeddings": sql_product_embeddings(cfg, product_table),
            "user_embeddings": sql_user_embeddings(cfg, buyer_table),
        }

        try:
            self._export_parallel(emb_splits)
        finally:
            self._staging.cleanup_table(product_table)
            self._staging.cleanup_table(buyer_table)

    def run_base_training(self) -> None:
        """Export all splits needed for base training."""
        log.info("Step 1/2: Exporting training data...")
        self.query_to_parquet(sql_nn_base_train(self.cfg), "nn_base_train")

        log.info("Step 2/2: Exporting embeddings for training IDs...")
        self._export_embeddings("nn_base_train")

    def run_finetune(self) -> None:
        """Export data for daily fine-tune."""
        log.info("Step 1/2: Exporting finetune data...")
        self.query_to_parquet(sql_nn_finetune_day(self.cfg), "nn_finetune")

        log.info("Step 2/2: Exporting embeddings for finetune IDs...")
        self._export_embeddings("nn_finetune")

    def _export_parallel(self, splits: dict[str, str]) -> None:
        log.info(
            "Launching %d parallel BQ export jobs (%d splits)...",
            self._EXPORT_WORKERS,
            len(splits),
        )
        with ThreadPoolExecutor(
            max_workers=self._EXPORT_WORKERS,
            thread_name_prefix="bq_export",
        ) as pool:
            futures = {
                pool.submit(self.query_to_parquet, sql, name): name for name, sql in splits.items()
            }
            for future in as_completed(futures):
                future.result()

    # -- Loading -----------------------------------------------------------

    @staticmethod
    def load(local_dir: Path, split: str) -> pl.DataFrame:
        split_dir = local_dir / split
        files = sorted(split_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {split_dir}")
        log.info("Loading '%s': %d shard(s)...", split, len(files))
        t0 = time.monotonic()
        df = pl.scan_parquet(files).collect()
        log.info(
            "  %s rows x %d cols in %.1fs",
            f"{df.shape[0]:,}",
            df.shape[1],
            time.monotonic() - t0,
        )
        return df

    @staticmethod
    def prepare(df: pl.DataFrame) -> pl.DataFrame:
        """Fill nulls/NaNs for numeric, categorical, and embedding features."""
        num = [c for c in NUMERIC_FEATURES if c in df.columns]
        cat = [c for c in CAT_FEATURES if c in df.columns]
        emb = [c for c in ALL_PRETRAINED_EMB_COLS if c in df.columns]

        return df.with_columns(
            [pl.col(c).fill_null(0.0).fill_nan(0.0).cast(pl.Float32) for c in num]
            + [pl.col(c).cast(pl.Utf8).fill_null("__MISSING__") for c in cat]
            + [pl.col(c).fill_null(0.0).fill_nan(0.0).cast(pl.Float32) for c in emb]
        )
