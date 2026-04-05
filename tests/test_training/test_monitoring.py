"""Tests for CometMonitor (disabled + active modes)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from reranking_dcn.training.monitoring import CometMonitor, _resolve_api_key


class TestCometMonitorDisabled:
    def test_inactive_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("COMET_API_KEY", None)
            monitor = CometMonitor(project_name="test")
            assert not monitor.active

    def test_explicitly_disabled(self):
        monitor = CometMonitor(project_name="test", disabled=True)
        assert not monitor.active

    @pytest.mark.parametrize(
        "method_call",
        [
            lambda m: m.log_batch(batch_idx=0, loss=0.5, lr=1e-3),
            lambda m: m.log_epoch(loss=0.5, extra_metric=0.1),
            lambda m: m.log_params({"lr": 0.001, "batch_size": 2048}),
            lambda m: m.end(),
        ],
        ids=["log_batch", "log_epoch", "log_params", "end"],
    )
    def test_noop_when_disabled(self, method_call):
        monitor = CometMonitor(project_name="test", disabled=True)
        method_call(monitor)

    @pytest.mark.parametrize(
        "action, attr, expected",
        [
            (lambda m: [m.log_batch(i, 0.5, 1e-3) for i in range(2)], "_step", 0),
            (lambda m: m.log_epoch(loss=0.5), "_epoch", 0),
        ],
        ids=["step_counter", "epoch_counter"],
    )
    def test_counter_stays_zero_when_disabled(self, action, attr, expected):
        monitor = CometMonitor(project_name="test", disabled=True)
        action(monitor)
        assert getattr(monitor, attr) == expected

    def test_experiment_name_and_tags_ignored_when_disabled(self):
        monitor = CometMonitor(
            project_name="test",
            experiment_name="my-experiment",
            tags=["tag1", "tag2"],
            disabled=True,
        )
        assert not monitor.active


class TestResolveApiKey:
    def test_returns_env_var_when_set(self):
        with patch.dict(os.environ, {"COMET_API_KEY": "test-key-123"}):
            assert _resolve_api_key() == "test-key-123"

    def test_returns_none_without_secret_manager(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("COMET_API_KEY", None)
            key = _resolve_api_key()
            assert key is None


class TestCometMonitorActive:
    def test_log_batch_increments_step(self):
        monitor = CometMonitor(project_name="test", disabled=True)
        monitor._experiment = MagicMock()

        monitor.log_batch(batch_idx=0, loss=0.5, lr=1e-3)
        monitor.log_batch(batch_idx=1, loss=0.4, lr=1e-3)

        assert monitor._step == 2
        assert monitor._experiment.log_metric.call_count == 4

    def test_log_epoch_increments_epoch(self):
        monitor = CometMonitor(project_name="test", disabled=True)
        monitor._experiment = MagicMock()

        monitor.log_epoch(loss=0.3)
        monitor.log_epoch(loss=0.2, auc=0.9)

        assert monitor._epoch == 2
        assert monitor._experiment.log_metric.call_count >= 3

    def test_log_params_delegates_to_experiment(self):
        monitor = CometMonitor(project_name="test", disabled=True)
        monitor._experiment = MagicMock()

        params = {"lr": 0.001, "batch_size": 2048}
        monitor.log_params(params)

        monitor._experiment.log_parameters.assert_called_once_with(params)

    def test_end_delegates_to_experiment(self):
        monitor = CometMonitor(project_name="test", disabled=True)
        monitor._experiment = MagicMock()

        monitor.end()

        monitor._experiment.end.assert_called_once()
