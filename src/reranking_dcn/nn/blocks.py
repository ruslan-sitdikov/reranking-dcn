"""SwiGLU feedforward blocks for the DCN-v2 network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUBlock(nn.Module):
    """Gated feedforward block: LayerNorm(Linear(x) * SiLU(Gate(x))).

    Replaces the standard Linear -> BatchNorm -> ReLU pattern with a
    multiplicative gate.  SiLU (Swish) has non-zero gradients everywhere,
    eliminating dead-neuron issues in sparse ad-feature regimes.
    LayerNorm normalizes per-sample, removing BatchNorm's train/eval
    running-statistics discrepancy.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.w_gate = nn.Linear(in_dim, out_dim, bias=False)
        self.w_val = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(self.w_val(x) * F.silu(self.w_gate(x))))


def build_swiglu_mlp(dims: list[int], dropout: float) -> nn.Sequential:
    """Build a SwiGLU + LayerNorm MLP stack.

    All hidden layers use SwiGLUBlock (gated activation + LayerNorm +
    Dropout).  The final layer is a plain nn.Linear (no activation)
    to produce raw logits or the final hidden representation.
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        if i < len(dims) - 2:
            layers.append(SwiGLUBlock(dims[i], dims[i + 1], dropout))
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
    return nn.Sequential(*layers)
