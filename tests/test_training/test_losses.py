"""Tests for focal loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from reranking_dcn.training.losses import focal_loss


class TestFocalLoss:
    @pytest.mark.parametrize(
        "logits, targets",
        [
            (torch.randn(32), torch.randint(0, 2, (32,)).float()),
            (torch.randn(32), torch.ones(32)),
            (torch.randn(32), torch.zeros(32)),
            (torch.tensor([0.5]), torch.tensor([1.0])),
        ],
        ids=["random_batch", "all_positive", "all_negative", "single_sample"],
    )
    def test_basic_loss_properties(self, logits, targets):
        loss = focal_loss(logits, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_gamma_zero_approximates_bce(self):
        """With gamma=0 and alpha=0.5, focal loss should equal standard BCE."""
        torch.manual_seed(42)
        logits = torch.randn(1000)
        targets = torch.randint(0, 2, (1000,)).float()

        fl = focal_loss(logits, targets, gamma=0.0, alpha=0.5)
        bce = F.binary_cross_entropy_with_logits(logits, targets) * 0.5
        torch.testing.assert_close(fl, bce, atol=1e-5, rtol=1e-5)

    def test_higher_gamma_reduces_easy_example_loss(self):
        """Well-classified examples should contribute less with higher gamma."""
        logits = torch.tensor([5.0, -5.0])
        targets = torch.tensor([1.0, 0.0])
        loss_g0 = focal_loss(logits, targets, gamma=0.0)
        loss_g2 = focal_loss(logits, targets, gamma=2.0)
        assert loss_g2.item() < loss_g0.item()

    def test_hard_examples_get_more_weight(self):
        """Misclassified examples should get relatively more weight."""
        easy_logits = torch.tensor([5.0])
        easy_targets = torch.tensor([1.0])
        hard_logits = torch.tensor([0.0])
        hard_targets = torch.tensor([1.0])

        easy_loss = focal_loss(easy_logits, easy_targets, gamma=2.0)
        hard_loss = focal_loss(hard_logits, hard_targets, gamma=2.0)
        assert hard_loss.item() > easy_loss.item()

    def test_alpha_balances_classes(self):
        logits = torch.zeros(100)
        pos_targets = torch.ones(100)
        neg_targets = torch.zeros(100)

        loss_pos_high_alpha = focal_loss(logits, pos_targets, gamma=0.0, alpha=0.9)
        loss_pos_low_alpha = focal_loss(logits, pos_targets, gamma=0.0, alpha=0.1)
        assert loss_pos_high_alpha.item() > loss_pos_low_alpha.item()

        loss_neg_high_alpha = focal_loss(logits, neg_targets, gamma=0.0, alpha=0.9)
        loss_neg_low_alpha = focal_loss(logits, neg_targets, gamma=0.0, alpha=0.1)
        assert loss_neg_low_alpha.item() > loss_neg_high_alpha.item()

    def test_loss_is_differentiable(self):
        logits = torch.randn(16, requires_grad=True)
        targets = torch.randint(0, 2, (16,)).float()
        loss = focal_loss(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
