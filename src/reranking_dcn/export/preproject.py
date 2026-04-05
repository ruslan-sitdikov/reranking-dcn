"""Offline pre-projection of raw embeddings through trained projection layers."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..nn.ranker import DCNv2Ranker

log = logging.getLogger(__name__)


@torch.no_grad()
def preproject_embeddings(
    model: DCNv2Ranker,
    embedding_parquets_dir: Path,
    output_dir: Path,
    batch_size: int = 100_000,
) -> None:
    """Project raw 768d embeddings through trained projection layers, save as FP32.

    Reads product/user embedding parquets, runs through model.product_proj /
    model.user_proj, and writes FP32 parquets: {id, proj_emb_0..63, has_emb}.
    """
    import polars as pl

    raw_model = getattr(model, "_orig_mod", model)
    raw_model.eval()
    raw_model.cpu()

    if raw_model._pretrained_emb_dim == 0:
        log.warning("Model has no pretrained embeddings, skipping pre-projection")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    per_source_dim = raw_model._per_source_emb_dim
    proj_dim = raw_model._projected_emb_dim

    for source, proj_layer, id_col, emb_prefix in [
        ("product_embeddings", raw_model.product_proj, "product_id", "product_emb_"),
        ("user_embeddings", raw_model.user_proj, "user_id", "user_emb_"),
    ]:
        source_dir = embedding_parquets_dir / source
        if not source_dir.exists():
            log.warning("  Embedding source not found: %s", source_dir)
            continue

        files = sorted(source_dir.glob("*.parquet"))
        if not files:
            log.warning("  No parquet files in %s", source_dir)
            continue

        log.info("  Pre-projecting %s from %d files...", source, len(files))
        proj_layer.eval()
        total_entities = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_paths: list[Path] = []

            for f_idx, f in enumerate(files):
                df = pl.read_parquet(f)
                ids = df[id_col].to_numpy()
                emb_cols = [c for c in df.columns if c.startswith(emb_prefix)][:per_source_dim]
                if not emb_cols:
                    emb_cols = [c for c in df.columns if c != id_col][:per_source_dim]

                raw_emb = df.select(emb_cols).to_numpy().astype(np.float32)

                file_ids, file_projs, file_has = [], [], []
                for start in range(0, len(ids), batch_size):
                    end = min(start + batch_size, len(ids))
                    batch_raw = torch.from_numpy(raw_emb[start:end])
                    batch_proj = proj_layer(batch_raw).numpy()
                    has_emb = (np.abs(raw_emb[start:end]).sum(axis=1) > 0).astype(np.float32)
                    file_ids.append(ids[start:end])
                    file_projs.append(batch_proj)
                    file_has.append(has_emb)

                del df, raw_emb

                if not file_ids:
                    continue

                ids_arr = np.concatenate(file_ids)
                proj_arr = np.concatenate(file_projs)
                has_arr = np.concatenate(file_has)
                total_entities += len(ids_arr)

                chunk_data = {id_col: ids_arr}
                for d in range(proj_dim):
                    chunk_data[f"proj_emb_{d}"] = proj_arr[:, d]
                chunk_data["has_emb"] = has_arr

                chunk_path = Path(tmpdir) / f"chunk_{f_idx:04d}.parquet"
                pl.DataFrame(chunk_data).write_parquet(chunk_path)
                chunk_paths.append(chunk_path)

            out_path = output_dir / f"{source}_projected.parquet"
            if chunk_paths:
                pl.concat([pl.read_parquet(p) for p in chunk_paths]).write_parquet(out_path)

        if not chunk_paths:
            log.warning("  No data projected for %s", source)
            continue

        size_mb = out_path.stat().st_size / 1e6
        log.info(
            "  Saved %s: %d entities, %dd FP32, %.1f MB",
            out_path.name,
            total_entities,
            proj_dim,
            size_mb,
        )
