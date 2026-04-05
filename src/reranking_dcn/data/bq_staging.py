"""BigQuery temp table management for embedding export.

Handles ID extraction from Parquet, upload to BQ temp tables, and
background cleanup of those temp tables.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


class BQStagingManager:
    """Manages BQ temp tables for embedding ID staging."""

    def __init__(self, client, billing_project: str) -> None:
        self._client = client
        self._billing_project = billing_project
        self._tmp_dataset_ref: str | None = None

    def extract_ids_from_parquet(
        self, local_dir: Path, split: str
    ) -> tuple[np.ndarray, pl.DataFrame]:
        """Extract unique product_ids and (buyer, day) pairs from exported training data."""
        split_dir = local_dir / split
        files = sorted(split_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {split_dir}")

        t0 = time.monotonic()
        df = (
            pl.scan_parquet(files)
            .select("product_id", "unified_buyer_id", "event_timestamp_day")
            .collect()
        )

        product_ids = df["product_id"].unique().drop_nulls().to_numpy()
        buyer_days = df.select("unified_buyer_id", "event_timestamp_day").unique()

        log.info(
            "  Extracted %s unique product_ids, %s unique (buyer, day) pairs (%.1fs)",
            f"{len(product_ids):,}",
            f"{buyer_days.shape[0]:,}",
            time.monotonic() - t0,
        )
        del df
        return product_ids, buyer_days

    def _ensure_tmp_dataset(self) -> str:
        if self._tmp_dataset_ref is not None:
            return self._tmp_dataset_ref
        dataset_ref = f"{self._billing_project}.pconv_tmp"
        self._client.query(
            f"CREATE SCHEMA IF NOT EXISTS `{dataset_ref}` "
            f"OPTIONS(default_table_expiration_days=1)"
        ).result()
        self._tmp_dataset_ref = dataset_ref
        return dataset_ref

    def _upload_parquet_to_bq(
        self,
        df: pl.DataFrame,
        table_id: str,
        schema: list,
    ) -> None:
        import os
        import tempfile

        from google.cloud import bigquery

        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        try:
            tmp.close()
            df.write_parquet(tmp.name)
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                source_format=bigquery.SourceFormat.PARQUET,
                schema=schema,
            )
            with open(tmp.name, "rb") as f:
                job = self._client.load_table_from_file(f, table_id, job_config=job_config)
                job.result()
        finally:
            os.unlink(tmp.name)

    def upload_product_ids(self, product_ids: np.ndarray) -> str:
        """Upload product IDs to a BQ temp table, return fully-qualified name."""
        from google.cloud import bigquery

        dataset_ref = self._ensure_tmp_dataset()
        table_id = f"{dataset_ref}.product_ids_{int(time.time())}"

        df = pl.DataFrame({"product_id": product_ids.astype(np.int64)})
        self._upload_parquet_to_bq(
            df,
            table_id,
            [bigquery.SchemaField("product_id", "INT64")],
        )
        log.info("  Uploaded %s product IDs to %s", f"{len(product_ids):,}", table_id)
        return f"`{table_id}`"

    def upload_buyer_days(self, buyer_days: pl.DataFrame) -> str:
        """Upload (unified_buyer_id, event_timestamp_day) pairs to a BQ temp table."""
        from google.cloud import bigquery

        dataset_ref = self._ensure_tmp_dataset()
        table_id = f"{dataset_ref}.buyer_days_{int(time.time())}"

        self._upload_parquet_to_bq(
            buyer_days,
            table_id,
            [
                bigquery.SchemaField("unified_buyer_id", "STRING"),
                bigquery.SchemaField("event_timestamp_day", "DATE"),
            ],
        )
        log.info(
            "  Uploaded %s (buyer, day) pairs to %s",
            f"{buyer_days.shape[0]:,}",
            table_id,
        )
        return f"`{table_id}`"

    def cleanup_table(self, table_ref: str) -> None:
        """Drop a BQ temp table in a background thread."""
        bare = table_ref.strip("`")

        def _bg() -> None:
            try:
                self._client.query(f"DROP TABLE IF EXISTS `{bare}`").result()
                log.info("  [bg] Dropped BQ temp table %s", bare)
            except Exception as exc:
                log.warning("  [bg] Failed to drop %s (non-fatal): %s", bare, exc)

        threading.Thread(target=_bg, daemon=True).start()
