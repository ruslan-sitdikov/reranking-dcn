"""Parity validation between PyTorch and ONNX model outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..features import NUMERIC_FEATURES
from ..nn.ranker import DCNv2Ranker
from ..preprocessing.gpu_preprocessor import GPUFeaturePreprocessor
from ..serving.wrappers import DCNv2Serving, DCNv2ServingNoEmb

log = logging.getLogger(__name__)


def validate_parity(
    pytorch_model: DCNv2Ranker,
    preprocessor: GPUFeaturePreprocessor,
    onnx_paths: dict[str, Path],
    cfg: Config,
    n_samples: int = 1000,
    atol: float = 1e-4,
) -> bool:
    """Compare PyTorch and ONNX outputs on random data.

    Returns True if max absolute difference < atol for all variants.
    """
    import onnxruntime as ort

    raw_model = getattr(pytorch_model, "_orig_mod", pytorch_model)
    raw_model.eval()
    raw_model.cpu()

    n_num = len(NUMERIC_FEATURES)
    cat_order = sorted(preprocessor.vocab_sizes.keys())
    cat_vocab_sizes = [preprocessor.vocab_sizes[col] for col in cat_order]
    B = n_samples
    all_passed = True

    dummy_bins = torch.randint(1, cfg.num_quantile_bins, (B, n_num), dtype=torch.long)
    dummy_cats = torch.stack(
        [torch.randint(0, max(2, vs), (B,)) for vs in cat_vocab_sizes], dim=1
    ).long()

    if "no_emb" in onnx_paths:
        log.info("  Validating parity: no_emb variant (%d samples)...", B)

        serving_no_emb = DCNv2ServingNoEmb(raw_model)
        serving_no_emb.eval()
        with torch.no_grad():
            pt_scores = serving_no_emb(dummy_bins, dummy_cats).numpy()

        sess = ort.InferenceSession(str(onnx_paths["no_emb"]), providers=["CPUExecutionProvider"])
        onnx_scores = sess.run(
            ["dcn_score"],
            {
                "numeric_bins": dummy_bins.numpy(),
                "cat_indices": dummy_cats.numpy(),
            },
        )[0]

        max_diff = np.abs(pt_scores - onnx_scores).max()
        mean_diff = np.abs(pt_scores - onnx_scores).mean()
        passed = max_diff < atol
        status = "PASS" if passed else "FAIL"
        log.info(
            "  [%s] no_emb: max_diff=%.2e, mean_diff=%.2e (threshold=%.2e)",
            status,
            max_diff,
            mean_diff,
            atol,
        )
        if not passed:
            all_passed = False

    if "with_emb" in onnx_paths:
        log.info("  Validating parity: with_emb variant (%d samples)...", B)
        proj_dim = raw_model._projected_emb_dim

        dummy_proj_product = torch.randn(B, proj_dim)
        dummy_proj_user = torch.randn(B, proj_dim)
        dummy_has_product = torch.ones(B, 1)
        dummy_has_user = torch.ones(B, 1)

        serving_emb = DCNv2Serving(raw_model)
        serving_emb.eval()
        with torch.no_grad():
            pt_scores = serving_emb(
                dummy_bins,
                dummy_cats,
                dummy_proj_product,
                dummy_proj_user,
                dummy_has_product,
                dummy_has_user,
            ).numpy()

        sess = ort.InferenceSession(str(onnx_paths["with_emb"]), providers=["CPUExecutionProvider"])
        onnx_scores = sess.run(
            ["dcn_score"],
            {
                "numeric_bins": dummy_bins.numpy(),
                "cat_indices": dummy_cats.numpy(),
                "proj_product_emb": dummy_proj_product.numpy(),
                "proj_user_emb": dummy_proj_user.numpy(),
                "has_product_emb": dummy_has_product.numpy(),
                "has_user_emb": dummy_has_user.numpy(),
            },
        )[0]

        max_diff = np.abs(pt_scores - onnx_scores).max()
        mean_diff = np.abs(pt_scores - onnx_scores).mean()
        passed = max_diff < atol
        status = "PASS" if passed else "FAIL"
        log.info(
            "  [%s] with_emb: max_diff=%.2e, mean_diff=%.2e (threshold=%.2e)",
            status,
            max_diff,
            mean_diff,
            atol,
        )
        if not passed:
            all_passed = False

    return all_passed
