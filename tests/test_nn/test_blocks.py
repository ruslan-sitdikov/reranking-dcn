"""Tests for SwiGLU feedforward blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from reranking_dcn import SwiGLUBlock
from reranking_dcn.nn.blocks import build_swiglu_mlp


class TestSwiGLUBlock:
    def test_output_shape(self):
        block = SwiGLUBlock(64, 32, dropout=0.0)
        x = torch.randn(8, 64)
        out = block(x)
        assert out.shape == (8, 32)

    def test_has_layernorm(self):
        block = SwiGLUBlock(64, 32, dropout=0.0)
        assert isinstance(block.norm, nn.LayerNorm)

    def test_gate_no_bias(self):
        block = SwiGLUBlock(64, 32, dropout=0.0)
        assert block.w_gate.bias is None

    def test_val_has_bias(self):
        block = SwiGLUBlock(64, 32, dropout=0.0)
        assert block.w_val.bias is not None


class TestBuildSwiGLUMLP:
    def test_hidden_layers_are_swiglu(self):
        mlp = build_swiglu_mlp([128, 64, 32, 1], dropout=0.1)
        assert isinstance(mlp[0], SwiGLUBlock)
        assert isinstance(mlp[1], SwiGLUBlock)
        assert isinstance(mlp[2], nn.Linear)

    def test_final_layer_is_linear(self):
        mlp = build_swiglu_mlp([128, 64, 1], dropout=0.1)
        assert isinstance(mlp[-1], nn.Linear)

    def test_output_shape(self):
        mlp = build_swiglu_mlp([128, 64, 32], dropout=0.0)
        x = torch.randn(4, 128)
        out = mlp(x)
        assert out.shape == (4, 32)
