"""ONNX-compatible serving wrappers for DCN-v2."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..nn.ranker import DCNv2Ranker
from ..preprocessing.gpu_preprocessor import GPUFeaturePreprocessor


class _DCNv2ServingBase(nn.Module):
    """Shared base for ONNX-compatible serving wrappers.

    Uses nn.ModuleList (integer-indexed) instead of ModuleDict for
    compatibility with torch.export / ONNX tracing.
    """

    def __init__(self, trained_model: DCNv2Ranker) -> None:
        super().__init__()
        self.numeric_emb = trained_model.numeric_emb
        self.register_buffer("_num_bin_offsets", trained_model._num_bin_offsets)
        cat_order = trained_model._cat_feature_order
        self.cat_embeddings = nn.ModuleList([trained_model.embeddings[col] for col in cat_order])
        self.n_cat = len(cat_order)
        self.cross_net = trained_model.cross_net
        self.deep_net = trained_model.deep_net
        self.head_hidden = trained_model.head_hidden
        self.head_logit = trained_model.head_logit

    def _score(
        self,
        numeric_bins: torch.Tensor,
        cat_indices: torch.Tensor,
        extra_parts: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        offset_bins = numeric_bins + self._num_bin_offsets
        num_emb_flat = self.numeric_emb(offset_bins).view(numeric_bins.shape[0], -1)

        cat_embs = [self.cat_embeddings[i](cat_indices[:, i]) for i in range(self.n_cat)]

        combined = DCNv2Ranker._build_feature_vector(
            self.cross_net, self.deep_net, num_emb_flat, cat_embs, extra_parts
        )
        hidden = self.head_hidden(combined)
        return self.head_logit(hidden).squeeze(-1)


class DCNv2Serving(_DCNv2ServingBase):
    """Serving-optimized DCN-v2 that accepts pre-projected 64d embeddings.

    Removes the 768->64 projection layers from the inference graph.
    At serving time, embeddings are pre-projected offline and looked up
    from Bigtable as FP32 (~640MB for boosted_merchants users + products).

    Raises ValueError if the source model was trained without pretrained
    embeddings -- use DCNv2ServingNoEmb instead in that case.

    Inputs:
        numeric_bins:      (B, 185) int64  -- quantile bin indices
        cat_indices:       (B, N_cat) int64  -- encoded categorical indices
        proj_product_emb:  (B, projected_emb_dim) float32
        proj_user_emb:     (B, projected_emb_dim) float32
        has_product_emb:   (B, 1) float32
        has_user_emb:      (B, 1) float32

    Output:
        dcn_score: (B,) float32  -- raw logit
    """

    def __init__(self, trained_model: DCNv2Ranker) -> None:
        if trained_model._pretrained_emb_dim == 0:
            raise ValueError(
                "DCNv2Serving requires a model trained with pretrained embeddings. "
                "Use DCNv2ServingNoEmb instead."
            )
        super().__init__(trained_model)

    def forward(
        self,
        numeric_bins: torch.Tensor,
        cat_indices: torch.Tensor,
        proj_product_emb: torch.Tensor,
        proj_user_emb: torch.Tensor,
        has_product_emb: torch.Tensor,
        has_user_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self._score(
            numeric_bins,
            cat_indices,
            [proj_product_emb, proj_user_emb, has_product_emb, has_user_emb],
        )

    @staticmethod
    def n_cat_features(preprocessor: GPUFeaturePreprocessor) -> int:
        return len(preprocessor.vocab_sizes)


class DCNv2ServingNoEmb(_DCNv2ServingBase):
    """Serving wrapper WITHOUT pre-trained embeddings (v1 simplified path).

    Inputs:
        numeric_bins: (B, 185) int64  -- quantile bin indices
        cat_indices:  (B, N_cat) int64  -- encoded categorical indices

    Output:
        dcn_score: (B,) float32  -- raw logit
    """

    def forward(
        self,
        numeric_bins: torch.Tensor,
        cat_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self._score(numeric_bins, cat_indices)
