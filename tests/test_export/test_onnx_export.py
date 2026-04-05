"""Tests for ONNX export functionality."""

from __future__ import annotations

import pytest
import torch

from conftest import make_dummy_preprocessor
from reranking_dcn import Config, NUMERIC_FEATURES
from reranking_dcn.export.checkpoint import build_model_from_config
from reranking_dcn.export.onnx_export import export_onnx


class TestONNXExport:
    @pytest.fixture()
    def export_with_emb(self, tmp_path):
        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        paths = export_onnx(model, prep, cfg, tmp_path)
        return paths, model, prep, cfg, tmp_path

    @pytest.fixture()
    def export_no_emb(self, tmp_path):
        cfg = Config(device="cpu", use_pretrained_embeddings=False)
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        paths = export_onnx(model, prep, cfg, tmp_path)
        return paths, model, prep, cfg, tmp_path

    def test_export_with_emb_creates_onnx(self, export_with_emb):
        paths, *_ = export_with_emb
        assert "with_emb" in paths
        assert paths["with_emb"].exists()
        assert paths["with_emb"].stat().st_size > 0

    def test_export_no_emb_creates_onnx(self, export_no_emb):
        paths, *_ = export_no_emb
        assert "no_emb" in paths
        assert paths["no_emb"].exists()

    def test_onnx_loadable_with_emb(self, export_with_emb):
        import onnxruntime as ort

        paths, *_ = export_with_emb
        sess = ort.InferenceSession(str(paths["with_emb"]), providers=["CPUExecutionProvider"])
        input_names = [inp.name for inp in sess.get_inputs()]
        assert "numeric_bins" in input_names
        assert "proj_product_emb" in input_names

    def test_onnx_loadable_no_emb(self, export_no_emb):
        import onnxruntime as ort

        paths, *_ = export_no_emb
        sess = ort.InferenceSession(str(paths["no_emb"]), providers=["CPUExecutionProvider"])
        input_names = [inp.name for inp in sess.get_inputs()]
        assert "numeric_bins" in input_names
        assert "proj_product_emb" not in input_names


class TestDeviceRestoration:
    def test_export_restores_device(self, tmp_path):
        cfg = Config(device="cpu", use_pretrained_embeddings=False)
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        original_device = next(model.parameters()).device

        export_onnx(model, prep, cfg, tmp_path)

        restored_device = next(model.parameters()).device
        assert restored_device == original_device
