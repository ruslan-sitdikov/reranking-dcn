"""Tests for GCS transfer utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reranking_dcn.data.gcs_transfer import (
    cleanup_gcs_prefix,
    download_from_gcs,
    make_gcs_client,
)


class TestMakeGcsClient:
    @patch("google.cloud.storage.Client")
    @patch("requests.adapters.HTTPAdapter")
    def test_creates_client_with_pool(self, mock_adapter_cls, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        client = make_gcs_client("test-project", pool_size=32)

        mock_client_cls.assert_called_once_with(project="test-project")
        assert mock_adapter_cls.call_count == 1
        mock_client._http.mount.assert_any_call("https://", mock_adapter_cls.return_value)
        mock_client._http.mount.assert_any_call("http://", mock_adapter_cls.return_value)


class TestDownloadFromGcs:
    def test_raises_when_no_blobs(self):
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "reranking_dcn.data.gcs_transfer.make_gcs_client"
        ) as mock_make_client:
            mock_client = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_make_client.return_value = mock_client

            with pytest.raises(RuntimeError, match="No blobs found"):
                download_from_gcs("bucket", "prefix", Path(tmpdir), "project")


class TestCleanupGcsPrefix:
    @patch("reranking_dcn.data.gcs_transfer.threading.Thread")
    def test_starts_daemon_thread(self, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        cleanup_gcs_prefix("bucket", "prefix", "project")

        mock_thread_cls.assert_called_once()
        _, kwargs = mock_thread_cls.call_args
        assert kwargs["daemon"] is True
        mock_thread.start.assert_called_once()
