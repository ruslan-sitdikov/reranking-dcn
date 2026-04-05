"""DCN-v2 (MoE) ranker with quantile-binned numeric embeddings and SwiGLU MLPs.

Architecture:
    Quantile-binned numerics -> learned embeddings  -+
    Encoded categoricals -> learned embeddings        |-- concat -> x0
    Pre-projected product/user embeddings (optional) -+
                          |
           +--------------+---------------+
           v                              v
    MoE Cross Network            SwiGLU Deep Network
           |                              |
           +--------------+--------------+
                     concat -> SwiGLU Head -> logit
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SwiGLUBlock, build_swiglu_mlp
from .cross_network import CrossNetwork


class DCNv2Ranker(nn.Module):
    """DCN-v2 (MoE) with quantile-binned numeric embeddings and SwiGLU MLPs.

    Numeric features are represented as learned embeddings over quantile
    bins (distribution-agnostic, nonlinear encoding) rather than z-scored
    scalars.  All hidden layers use SwiGLU gated activations with LayerNorm,
    replacing the prior ReLU + BatchNorm stack.
    """

    LOGIT_CLAMP_MIN = -10.0
    LOGIT_CLAMP_MAX = 10.0

    def __init__(
        self,
        num_continuous: int,
        vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        n_quantile_bins: int = 64,
        num_emb_dim: int = 4,
        pretrained_emb_dim: int = 0,
        projected_emb_dim: int = 64,
        num_cross_layers: int = 3,
        num_experts: int = 4,
        expert_rank: int = 64,
        deep_dims: tuple[int, ...] = (512, 256, 128),
        head_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self._cat_feature_order = sorted(vocab_sizes.keys())
        self._pretrained_emb_dim = pretrained_emb_dim
        self._per_source_emb_dim = pretrained_emb_dim // 2 if pretrained_emb_dim > 0 else 0
        self._projected_emb_dim = projected_emb_dim
        self._num_continuous = num_continuous
        self._n_quantile_bins = n_quantile_bins
        self._num_emb_dim = num_emb_dim

        # Quantile-bin embeddings: feature i, bin j -> index = i * (Q+1) + j.
        # Index 0 within each feature's range is reserved for padding.
        total_num_bins = num_continuous * (n_quantile_bins + 1)
        self.numeric_emb = nn.Embedding(total_num_bins, num_emb_dim, padding_idx=0)
        offsets = torch.arange(num_continuous, dtype=torch.long) * (n_quantile_bins + 1)
        self.register_buffer("_num_bin_offsets", offsets)

        numeric_emb_total = num_continuous * num_emb_dim

        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(
                    vocab_sizes[col], embedding_dims[col], padding_idx=0, max_norm=1.0
                )
                for col in self._cat_feature_order
            }
        )

        total_cat_emb_dim = sum(embedding_dims[c] for c in self._cat_feature_order)

        if pretrained_emb_dim > 0:
            self.product_proj = nn.Sequential(
                nn.Linear(self._per_source_emb_dim, projected_emb_dim),
                nn.LayerNorm(projected_emb_dim),
                nn.GELU(),
            )
            self.user_proj = nn.Sequential(
                nn.Linear(self._per_source_emb_dim, projected_emb_dim),
                nn.LayerNorm(projected_emb_dim),
                nn.GELU(),
            )
            projected_total = projected_emb_dim * 2
            emb_indicator_dims = 2
        else:
            projected_total = 0
            emb_indicator_dims = 0

        input_dim = numeric_emb_total + total_cat_emb_dim + projected_total + emb_indicator_dims

        self.cross_net = CrossNetwork(input_dim, num_cross_layers, num_experts, expert_rank)
        self.deep_net = build_swiglu_mlp([input_dim] + list(deep_dims), dropout)

        combined_dim = input_dim + deep_dims[-1]
        self.head_hidden = build_swiglu_mlp([combined_dim] + list(head_dims), dropout)
        self.head_logit = nn.Linear(head_dims[-1], 1)
        self._init_weights()

    @staticmethod
    def _init_mlp(seq: nn.Sequential) -> None:
        for child in seq:
            if isinstance(child, SwiGLUBlock):
                nn.init.xavier_uniform_(child.w_gate.weight)
                nn.init.xavier_uniform_(child.w_val.weight)
                if child.w_val.bias is not None:
                    nn.init.zeros_(child.w_val.bias)
            elif isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def _init_weights(self) -> None:
        self._init_mlp(self.deep_net)
        self._init_mlp(self.head_hidden)

        nn.init.normal_(self.head_logit.weight, std=0.01)
        if self.head_logit.bias is not None:
            nn.init.zeros_(self.head_logit.bias)

        nn.init.normal_(self.numeric_emb.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.numeric_emb.weight[self.numeric_emb.padding_idx])

        for module in self.embeddings.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

        if self._pretrained_emb_dim > 0:
            for proj in (self.product_proj, self.user_proj):
                for module in proj.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    @staticmethod
    def _build_feature_vector(
        cross_net: CrossNetwork,
        deep_net: nn.Sequential,
        num_emb_flat: torch.Tensor,
        cat_embs: list[torch.Tensor],
        extra_parts: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Assemble feature parts and run through cross+deep networks.

        Canonical ordering: [numeric_embs, *categorical_embs, *extra_parts].
        Shared by training (DCNv2Ranker) and ONNX serving wrappers to
        guarantee identical tensor layout.
        """
        parts: list[torch.Tensor] = [num_emb_flat] + cat_embs
        if extra_parts is not None:
            parts.extend(extra_parts)
        x0 = torch.cat(parts, dim=1)
        cross_out = cross_net(x0)
        deep_out = deep_net(x0)
        return torch.cat([cross_out, deep_out], dim=1)

    def _compute_combined(
        self,
        numeric_bins: torch.Tensor,
        cat_indices: torch.Tensor,
        pretrained_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared feature extraction: inputs -> concat(cross_out, deep_out)."""
        offset_bins = numeric_bins + self._num_bin_offsets
        num_emb_flat = self.numeric_emb(offset_bins).view(numeric_bins.shape[0], -1)

        cat_embs = [
            self.embeddings[col](cat_indices[:, i]) for i, col in enumerate(self._cat_feature_order)
        ]

        extra_parts: list[torch.Tensor] | None = None
        if self._pretrained_emb_dim > 0:
            extra_parts = []
            if pretrained_emb is not None:
                product_raw = pretrained_emb[:, : self._per_source_emb_dim]
                user_raw = pretrained_emb[:, self._per_source_emb_dim :]
                has_product = (product_raw.abs().sum(dim=1, keepdim=True) > 0).float()
                has_user = (user_raw.abs().sum(dim=1, keepdim=True) > 0).float()
                extra_parts.extend(
                    [
                        self.product_proj(product_raw),
                        self.user_proj(user_raw),
                        has_product,
                        has_user,
                    ]
                )
            else:
                B = numeric_bins.shape[0]
                dev = num_emb_flat.device
                zero_raw = torch.zeros(B, self._per_source_emb_dim, device=dev)
                extra_parts.extend(
                    [
                        self.product_proj(zero_raw),
                        self.user_proj(zero_raw),
                        torch.zeros(B, 1, device=dev),
                        torch.zeros(B, 1, device=dev),
                    ]
                )

        return self._build_feature_vector(
            self.cross_net, self.deep_net, num_emb_flat, cat_embs, extra_parts
        )

    def forward(
        self,
        numeric_bins: torch.Tensor,
        cat_indices: torch.Tensor,
        pretrained_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns relevance logit per item clamped to [LOGIT_CLAMP_MIN, LOGIT_CLAMP_MAX] -- shape (B,)."""
        combined = self._compute_combined(numeric_bins, cat_indices, pretrained_emb)
        hidden = self.head_hidden(combined)
        return self.head_logit(hidden).squeeze(-1).clamp(self.LOGIT_CLAMP_MIN, self.LOGIT_CLAMP_MAX)
