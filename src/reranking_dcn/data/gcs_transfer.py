"""GCS download, upload, and cleanup with connection pooling and retry."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

_DOWNLOAD_WORKERS = 64


def make_gcs_client(project: str, pool_size: int = _DOWNLOAD_WORKERS):
    """Create a GCS client with a large HTTP connection pool."""
    import requests.adapters as _requests_adapters
    from google.cloud import storage

    client = storage.Client(project=project)
    adapter = _requests_adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    client._http.mount("https://", adapter)
    client._http.mount("http://", adapter)
    return client


def download_from_gcs(
    bucket_name: str,
    prefix: str,
    local_dest: Path,
    project: str,
    max_retries: int = 6,
) -> int:
    """Download Parquet shards from GCS with parallel threads and retry.

    Returns the number of successfully downloaded shards.
    """
    from google.cloud.storage import transfer_manager

    local_dest.mkdir(parents=True, exist_ok=True)
    dl_client = make_gcs_client(project)
    bucket = dl_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise RuntimeError(f"No blobs found at gs://{bucket_name}/{prefix}")

    blob_dest_pairs = []
    for blob in blobs:
        filename = Path(blob.name).name
        if not filename:
            continue
        blob_dest_pairs.append((blob, str(local_dest / filename)))

    log.info(
        "  Downloading %d shards (%d workers) from gs://%s/%s ...",
        len(blob_dest_pairs),
        _DOWNLOAD_WORKERS,
        bucket_name,
        prefix,
    )

    results = transfer_manager.download_many(
        blob_dest_pairs,
        max_workers=_DOWNLOAD_WORKERS,
        worker_type=transfer_manager.THREAD,
        raise_exception=False,
    )

    failed_pairs = [
        blob_dest_pairs[i] for i, result in enumerate(results) if isinstance(result, Exception)
    ]
    for attempt in range(1, max_retries + 1):
        if not failed_pairs:
            break
        wait = min(2**attempt, 60)
        log.warning(
            "  Retrying %d failed shard(s) (attempt %d/%d, backoff %ds)...",
            len(failed_pairs),
            attempt,
            max_retries,
            wait,
        )
        time.sleep(wait)
        retry_results = transfer_manager.download_many(
            failed_pairs,
            max_workers=min(_DOWNLOAD_WORKERS, len(failed_pairs)),
            worker_type=transfer_manager.THREAD,
            raise_exception=False,
        )
        failed_pairs = [
            failed_pairs[i]
            for i, result in enumerate(retry_results)
            if isinstance(result, Exception)
        ]

    if failed_pairs:
        fail_pct = len(failed_pairs) / len(blob_dest_pairs) * 100
        names = [pair[0].name for pair in failed_pairs[:10]]
        if fail_pct > 2.0:
            raise RuntimeError(
                f"Failed to download {len(failed_pairs)}/{len(blob_dest_pairs)} "
                f"shards ({fail_pct:.2f}%) after {max_retries} retries:\n"
                + "\n".join(f"  {n}" for n in names)
            )
        log.warning(
            "  Skipping %d/%d unreachable shard(s) (%.3f%%) after %d retries: %s",
            len(failed_pairs),
            len(blob_dest_pairs),
            fail_pct,
            max_retries,
            ", ".join(Path(n).name for n in names),
        )

    return len(blob_dest_pairs) - len(failed_pairs)


def cleanup_gcs_prefix(bucket_name: str, prefix: str, project: str) -> None:
    """Delete all objects under a GCS prefix in a background daemon thread."""
    from google.cloud import storage as _storage

    def _bg_delete() -> None:
        try:
            bg_client = _storage.Client(project=project)
            bucket = bg_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))
            if not blobs:
                return
            batch_size = 100
            for i in range(0, len(blobs), batch_size):
                bucket.delete_blobs(blobs[i : i + batch_size], on_error=lambda _: None)
            log.info(
                "  [bg] Cleaned up GCS: gs://%s/%s (%d objects)",
                bucket_name,
                prefix,
                len(blobs),
            )
        except Exception as exc:
            log.warning("  [bg] GCS cleanup failed (non-fatal): %s", exc)

    threading.Thread(target=_bg_delete, daemon=True).start()
