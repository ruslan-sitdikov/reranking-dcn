"""Parity tests between PyTorch and ONNX model outputs."""

from __future__ import annotations

import numpy as np
import torch

from conftest import make_dummy_preprocessor
from reranking_dcn import Config, DCNv2Serving, DCNv2ServingNoEmb, NUMERIC_FEATURES
from reranking_dcn.export.checkpoint import build_model_from_config
from reranking_dcn.export.onnx_export import export_onnx


class TestParity:
    def test_parity_with_emb(self, tmp_path):
        import onnxruntime as ort

        cfg = Config(device="cpu")
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        model.eval()
        paths = export_onnx(model, prep, cfg, tmp_path)

        B = 16
        n_num = len(NUMERIC_FEATURES)
        n_cat = len(prep.vocab_sizes)
        bins = torch.randint(1, cfg.num_quantile_bins, (B, n_num), dtype=torch.long)
        cats = torch.randint(1, 5, (B, n_cat), dtype=torch.long)
        proj_prod = torch.randn(B, cfg.projected_emb_dim)
        proj_user = torch.randn(B, cfg.projected_emb_dim)
        has_prod = torch.ones(B, 1)
        has_user = torch.ones(B, 1)

        serving = DCNv2Serving(model)
        serving.eval()
        with torch.no_grad():
            pt_out = serving(bins, cats, proj_prod, proj_user, has_prod, has_user).numpy()

        sess = ort.InferenceSession(str(paths["with_emb"]), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(
            ["dcn_score"],
            {
                "numeric_bins": bins.numpy(),
                "cat_indices": cats.numpy(),
                "proj_product_emb": proj_prod.numpy(),
                "proj_user_emb": proj_user.numpy(),
                "has_product_emb": has_prod.numpy(),
                "has_user_emb": has_user.numpy(),
            },
        )[0]

        np.testing.assert_allclose(pt_out, onnx_out, atol=1e-4, rtol=1e-4)

    def test_parity_no_emb(self, tmp_path):
        import onnxruntime as ort

        cfg = Config(device="cpu", use_pretrained_embeddings=False)
        prep = make_dummy_preprocessor(cfg)
        model = build_model_from_config(prep, cfg)
        model.eval()
        paths = export_onnx(model, prep, cfg, tmp_path)

        B = 16
        n_num = len(NUMERIC_FEATURES)
        n_cat = len(prep.vocab_sizes)
        bins = torch.randint(1, cfg.num_quantile_bins, (B, n_num), dtype=torch.long)
        cats = torch.randint(1, 5, (B, n_cat), dtype=torch.long)

        serving = DCNv2ServingNoEmb(model)
        serving.eval()
        with torch.no_grad():
            pt_out = serving(bins, cats).numpy()

        sess = ort.InferenceSession(str(paths["no_emb"]), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(
            ["dcn_score"],
            {
                "numeric_bins": bins.numpy(),
                "cat_indices": cats.numpy(),
            },
        )[0]

        np.testing.assert_allclose(pt_out, onnx_out, atol=1e-4, rtol=1e-4)
