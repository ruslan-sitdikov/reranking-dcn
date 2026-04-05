"""ONNX Runtime CPU inference latency benchmark."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def benchmark_latency(
    onnx_path: Path,
    batch_sizes: list[int] | None = None,
    n_warmup: int = 50,
    n_iterations: int = 500,
    n_num_features: int | None = None,
    n_cat_features: int | None = None,
    proj_emb_dim: int = 64,
) -> dict[int, dict[str, float]]:
    """Measure ONNX Runtime CPU inference latency across batch sizes.

    Input names and feature counts are auto-detected from the ONNX model
    when not explicitly provided.

    Returns {batch_size: {mean_ms, p50_ms, p95_ms, p99_ms}}.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_names = [inp.name for inp in sess.get_inputs()]
    for inp in sess.get_inputs():
        if inp.name == "numeric_bins" and n_num_features is None:
            n_num_features = inp.shape[1]
        elif inp.name == "cat_indices" and n_cat_features is None:
            n_cat_features = inp.shape[1]
    if n_num_features is None:
        raise ValueError("Could not auto-detect n_num_features from ONNX model; pass explicitly.")
    if n_cat_features is None:
        raise ValueError("Could not auto-detect n_cat_features from ONNX model; pass explicitly.")

    if batch_sizes is None:
        batch_sizes = [1, 4, 10, 50, 100, 184, 200, 356]

    has_emb = "proj_product_emb" in input_names

    results = {}
    for B in batch_sizes:
        feeds = {
            "numeric_bins": np.ascontiguousarray(
                np.random.randint(1, 64, (B, n_num_features), dtype=np.int64)
            ),
            "cat_indices": np.ascontiguousarray(
                np.random.randint(1, 10, (B, n_cat_features), dtype=np.int64)
            ),
        }
        if has_emb:
            feeds["proj_product_emb"] = np.ascontiguousarray(
                np.random.randn(B, proj_emb_dim).astype(np.float32)
            )
            feeds["proj_user_emb"] = np.ascontiguousarray(
                np.random.randn(B, proj_emb_dim).astype(np.float32)
            )
            feeds["has_product_emb"] = np.ones((B, 1), dtype=np.float32)
            feeds["has_user_emb"] = np.ones((B, 1), dtype=np.float32)

        for _ in range(n_warmup):
            sess.run(None, feeds)

        latencies = np.empty(n_iterations, dtype=np.float64)
        for it in range(n_iterations):
            t0 = time.perf_counter()
            sess.run(None, feeds)
            latencies[it] = (time.perf_counter() - t0) * 1000

        results[B] = {
            "mean_ms": float(latencies.mean()),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
        log.info(
            "  B=%4d: mean=%.2fms  p50=%.2fms  p95=%.2fms  p99=%.2fms",
            B,
            results[B]["mean_ms"],
            results[B]["p50_ms"],
            results[B]["p95_ms"],
            results[B]["p99_ms"],
        )

    return results
