"""Tests for ONNX Runtime latency benchmark."""

from __future__ import annotations

from conftest import make_dummy_preprocessor
from reranking_dcn import Config, NUMERIC_FEATURES
from reranking_dcn.export.benchmark import benchmark_latency
from reranking_dcn.export.checkpoint import build_model_from_config
from reranking_dcn.export.onnx_export import export_onnx


class TestBenchmark:
    def test_benchmark_runs(self, tmp_path):
        cfg = Config(device="cpu", use_pretrained_embeddings=False)
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        paths = export_onnx(model, prep, cfg, tmp_path)

        results = benchmark_latency(
            paths["no_emb"],
            batch_sizes=[1, 4],
            n_warmup=2,
            n_iterations=5,
            n_num_features=len(NUMERIC_FEATURES),
            n_cat_features=len(prep.vocab_sizes),
        )
        assert 1 in results
        assert 4 in results
        assert results[1]["mean_ms"] > 0
