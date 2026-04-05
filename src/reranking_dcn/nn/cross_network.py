"""DCN-v2 Mixture-of-Experts cross layers for explicit feature interaction."""

from __future__ import annotations

import torch
import torch.nn as nn


class MoECrossLayer(nn.Module):
    """DCN-v2 cross layer with Mixture of Low-Rank Experts (Section 3.2).

    x_{l+1} = x_0 * (E_l(x_l) + b_l) + x_l

    where the expert mixture replaces the full-rank weight matrix:
        E_l(x_l) = sum_{k=1}^{K} G_k(x_l) * (U_k V_k^T x_l)

    G(x_l) = softmax(W_g x_l) in R^K is the input-dependent gating function,
    and U_k in R^{d x r}, V_k in R^{r x d} are per-expert low-rank factors.
    """

    def __init__(self, dim: int, num_experts: int = 4, low_rank: int = 64) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.low_rank = low_rank
        self.U = nn.Parameter(torch.empty(num_experts, dim, low_rank))
        self.V = nn.Parameter(torch.empty(num_experts, low_rank, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for k in range(self.num_experts):
            nn.init.xavier_uniform_(self.U.data[k])
            nn.init.xavier_uniform_(self.V.data[k])
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        gate_scores = torch.softmax(self.gate(xl), dim=-1)
        vx = torch.einsum("krd,bd->bkr", self.V, xl)
        uvx = torch.einsum("kdr,bkr->bkd", self.U, vx)
        expert_out = torch.einsum("bk,bkd->bd", gate_scores, uvx)
        return x0 * (expert_out + self.bias) + xl


class CrossNetwork(nn.Module):
    """Stack of DCN-v2 MoE cross layers for explicit feature interaction learning."""

    def __init__(self, dim: int, num_layers: int, num_experts: int = 4, low_rank: int = 64) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [MoECrossLayer(dim, num_experts, low_rank) for _ in range(num_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl
