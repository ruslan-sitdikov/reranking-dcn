"""CLI entry point for DCN-v2 ONNX export, validation, and benchmarking."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .export.benchmark import benchmark_latency
from .export.checkpoint import (
    PREPROCESSOR_CHECKPOINT_KEYS,
    build_model_from_config,
    load_checkpoint,
)
from .export.onnx_export import export_onnx
from .export.preproject import preproject_embeddings
from .export.validation import validate_parity
from .features import CAT_FEATURES, NUMERIC_FEATURES
from .preprocessing.gpu_preprocessor import GPUFeaturePreprocessor
from .serving.artifacts import serialize_preprocessing_artifacts

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    p = argparse.ArgumentParser(description="DCN-v2 ONNX Export & Validation")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to trained DCN checkpoint (.pt). Empty = random weights.",
    )
    p.add_argument(
        "--preprocessor_path",
        type=str,
        default="",
        help="Path to saved preprocessor state. Empty = create dummy.",
    )
    p.add_argument("--output_dir", type=str, default="./onnx_artifacts")
    p.add_argument(
        "--validate", action="store_true", help="Run parity validation (PyTorch vs ONNX)"
    )
    p.add_argument(
        "--benchmark_only",
        action="store_true",
        help="Skip export, only run latency benchmark on existing ONNX",
    )
    p.add_argument(
        "--embedding_dir",
        type=str,
        default="",
        help="Path to raw embedding parquets for pre-projection",
    )
    p.add_argument("--opset_version", type=int, default=17)
    p.add_argument("--no_pretrained_embeddings", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        use_pretrained_embeddings=not args.no_pretrained_embeddings,
        device="cpu",
    )

    if args.benchmark_only:
        log.info("=" * 60)
        log.info("LATENCY BENCHMARK ONLY")
        log.info("=" * 60)
        for variant in ["dcn_model_no_emb.onnx", "dcn_model.onnx"]:
            onnx_path = output_dir / variant
            if onnx_path.exists():
                log.info("")
                log.info("Benchmarking %s", variant)
                benchmark_latency(onnx_path)
        return

    log.info("=" * 60)
    log.info("DCN-v2 ONNX EXPORT PIPELINE")
    log.info("=" * 60)

    preprocessor = GPUFeaturePreprocessor(
        id_hash_config=cfg.id_hash_config,
        cat_emb_dim_overrides=cfg.cat_emb_dim_overrides,
        n_quantile_bins=cfg.num_quantile_bins,
    )

    if args.preprocessor_path:
        state = torch.load(args.preprocessor_path, map_location="cpu", weights_only=False)
        missing = PREPROCESSOR_CHECKPOINT_KEYS - set(state.keys())
        if missing:
            raise ValueError(
                f"Preprocessor checkpoint is missing expected keys: {missing}. "
                "File may be corrupted or from an incompatible version."
            )
        preprocessor.load_state_dict(state["state_dict"])
        preprocessor.cat_encoders = state["cat_encoders"]
        preprocessor.vocab_sizes = state["vocab_sizes"]
        preprocessor.embedding_dims = state["embedding_dims"]
        log.info("  Loaded preprocessor from %s", args.preprocessor_path)
    else:
        log.warning("  No preprocessor path — using dummy preprocessor (random boundaries)")
        n_num = len(NUMERIC_FEATURES)
        boundaries = np.sort(
            np.random.randn(n_num, cfg.num_quantile_bins - 1).astype(np.float32), axis=1
        )
        preprocessor.register_buffer("_quantile_boundaries", torch.from_numpy(boundaries))
        for col in CAT_FEATURES:
            preprocessor.cat_encoders[col] = {f"val_{i}": i + 1 for i in range(10)}
            preprocessor.vocab_sizes[col] = 11
            preprocessor.embedding_dims[col] = cfg.cat_emb_dim_overrides.get(col, min(50, 11 // 2))
        for col, (n_buckets, emb_dim) in cfg.id_hash_config.items():
            preprocessor.vocab_sizes[col] = n_buckets + 1
            preprocessor.embedding_dims[col] = emb_dim

    model = build_model_from_config(preprocessor, cfg)
    if args.checkpoint_path:
        model = load_checkpoint(model, args.checkpoint_path)
    else:
        log.warning("  No checkpoint — using random weights (pipeline validation only)")

    total_params = sum(p.numel() for p in model.parameters())
    log.info("  Model: %s parameters (%.1f MB)", f"{total_params:,}", total_params * 4 / 1e6)

    log.info("")
    log.info("--- Serializing preprocessing artifacts ---")
    serialize_preprocessing_artifacts(preprocessor, cfg, output_dir)

    log.info("")
    log.info("--- Exporting ONNX models ---")
    onnx_paths = export_onnx(model, preprocessor, cfg, output_dir, args.opset_version)

    if args.validate:
        log.info("")
        log.info("--- Parity Validation ---")
        passed = validate_parity(model, preprocessor, onnx_paths, cfg)
        if passed:
            log.info("  All parity checks PASSED")
        else:
            log.error("  Parity checks FAILED — ONNX output diverges from PyTorch")

    log.info("")
    log.info("--- Latency Benchmark ---")
    for variant_name, onnx_path in onnx_paths.items():
        log.info("")
        log.info("Benchmark: %s", variant_name)
        results = benchmark_latency(onnx_path)

        results_path = output_dir / f"latency_{variant_name}.json"
        with open(results_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        log.info("  Saved latency results -> %s", results_path)

    if args.embedding_dir and cfg.use_pretrained_embeddings:
        log.info("")
        log.info("--- Pre-projecting Embeddings ---")
        emb_output = output_dir / "embeddings"
        preproject_embeddings(model, Path(args.embedding_dir), emb_output)

    log.info("")
    log.info("=" * 60)
    log.info("EXPORT COMPLETE")
    log.info("  Artifacts: %s", output_dir)
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            log.info("    %s (%.1f KB)", f.relative_to(output_dir), f.stat().st_size / 1024)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
