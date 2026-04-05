"""Tests for MoE cross layers and cross network."""

from __future__ import annotations

import torch
import torch.nn as nn

from reranking_dcn import CrossNetwork, MoECrossLayer


class TestMoECrossLayer:
    def test_output_shape(self):
        dim, K, r = 64, 4, 16
        layer = MoECrossLayer(dim, K, r)
        x0 = torch.randn(8, dim)
        out = layer(x0, x0)
        assert out.shape == (8, dim)

    def test_residual_connection(self):
        dim = 32
        layer = MoECrossLayer(dim, num_experts=2, low_rank=8)
        with torch.no_grad():
            layer.V.zero_()
        x0 = torch.randn(4, dim)
        with torch.no_grad():
            out = layer(x0, x0)
        torch.testing.assert_close(out, x0, atol=1e-5, rtol=1e-5)

    def test_gate_init_near_uniform(self):
        dim = 64
        layer = MoECrossLayer(dim, num_experts=4, low_rank=16)
        nn.init.normal_(layer.gate.weight, std=0.01)
        x = torch.randn(100, dim)
        scores = torch.softmax(layer.gate(x), dim=-1)
        mean_scores = scores.mean(dim=0)
        assert torch.allclose(mean_scores, torch.full_like(mean_scores, 0.25), atol=0.05)

    def test_standalone_init_finite(self):
        """MoECrossLayer should produce finite outputs without external init."""
        layer = MoECrossLayer(32, num_experts=2, low_rank=8)
        x0 = torch.randn(4, 32)
        with torch.no_grad():
            out = layer(x0, x0)
        assert torch.isfinite(out).all()


class TestCrossNetwork:
    def test_output_preserves_dim(self):
        dim = 64
        net = CrossNetwork(dim, num_layers=3, num_experts=4, low_rank=16)
        x0 = torch.randn(8, dim)
        out = net(x0)
        assert out.shape == (8, dim)
