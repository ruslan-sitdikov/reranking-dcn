"""ONNX graph export for DCN-v2 serving models."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..config import Config
from ..features import NUMERIC_FEATURES
from ..nn.ranker import DCNv2Ranker
from ..preprocessing.gpu_preprocessor import GPUFeaturePreprocessor
from ..serving.wrappers import DCNv2Serving, DCNv2ServingNoEmb

log = logging.getLogger(__name__)


def _simplify_onnx(onnx_path: Path) -> None:
    """Run onnxsim to strip unnecessary Cast/Gather/Identity nodes."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        log.info("  onnxsim not installed — skipping graph simplification")
        return

    model_proto = onnx.load(str(onnx_path))
    simplified, ok = onnxsim.simplify(model_proto)
    if ok:
        onnx.save(simplified, str(onnx_path))
        log.info("  Simplified %s", onnx_path.name)
    else:
        log.warning("  onnxsim validation failed for %s — keeping original", onnx_path.name)


def export_onnx(
    model: DCNv2Ranker,
    preprocessor: GPUFeaturePreprocessor,
    cfg: Config,
    output_dir: Path,
    opset_version: int = 17,
) -> dict[str, Path]:
    """Export DCN-v2 to ONNX.

    If the model was trained without pretrained embeddings, exports a no_emb
    variant. If trained with pretrained embeddings, exports a with_emb variant
    that accepts pre-projected 64d embeddings. The two are mutually exclusive
    because the network's weight dimensions depend on whether embeddings were
    included during training.

    Returns dict of {variant_name: onnx_path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    original_device = next(raw_model.parameters()).device
    raw_model.eval()
    raw_model.cpu()

    try:
        n_num = len(NUMERIC_FEATURES)
        n_cat = len(preprocessor.vocab_sizes)
        B = 4
        exported = {}

        dummy_bins = torch.randint(0, cfg.num_quantile_bins, (B, n_num), dtype=torch.long)
        dummy_cats = torch.zeros(B, n_cat, dtype=torch.long)

        if raw_model._pretrained_emb_dim > 0:
            serving_emb = DCNv2Serving(raw_model)
            serving_emb.eval()

            proj_dim = raw_model._projected_emb_dim
            dummy_proj_product = torch.randn(B, proj_dim)
            dummy_proj_user = torch.randn(B, proj_dim)
            dummy_has_product = torch.ones(B, 1)
            dummy_has_user = torch.ones(B, 1)

            onnx_path_emb = output_dir / "dcn_model.onnx"
            torch.onnx.export(
                serving_emb,
                (
                    dummy_bins,
                    dummy_cats,
                    dummy_proj_product,
                    dummy_proj_user,
                    dummy_has_product,
                    dummy_has_user,
                ),
                str(onnx_path_emb),
                opset_version=opset_version,
                input_names=[
                    "numeric_bins",
                    "cat_indices",
                    "proj_product_emb",
                    "proj_user_emb",
                    "has_product_emb",
                    "has_user_emb",
                ],
                output_names=["dcn_score"],
                dynamic_axes={
                    "numeric_bins": {0: "batch"},
                    "cat_indices": {0: "batch"},
                    "proj_product_emb": {0: "batch"},
                    "proj_user_emb": {0: "batch"},
                    "has_product_emb": {0: "batch"},
                    "has_user_emb": {0: "batch"},
                    "dcn_score": {0: "batch"},
                },
            )
            _simplify_onnx(onnx_path_emb)
            size_mb = onnx_path_emb.stat().st_size / 1e6
            log.info("  Exported dcn_model.onnx (%.1f MB, opset %d)", size_mb, opset_version)
            exported["with_emb"] = onnx_path_emb
        else:
            serving_no_emb = DCNv2ServingNoEmb(raw_model)
            serving_no_emb.eval()

            onnx_path_no_emb = output_dir / "dcn_model_no_emb.onnx"
            torch.onnx.export(
                serving_no_emb,
                (dummy_bins, dummy_cats),
                str(onnx_path_no_emb),
                opset_version=opset_version,
                input_names=["numeric_bins", "cat_indices"],
                output_names=["dcn_score"],
                dynamic_axes={
                    "numeric_bins": {0: "batch"},
                    "cat_indices": {0: "batch"},
                    "dcn_score": {0: "batch"},
                },
            )
            _simplify_onnx(onnx_path_no_emb)
            size_mb = onnx_path_no_emb.stat().st_size / 1e6
            log.info("  Exported dcn_model_no_emb.onnx (%.1f MB, opset %d)", size_mb, opset_version)
            exported["no_emb"] = onnx_path_no_emb
    finally:
        raw_model.to(original_device)

    return exported
