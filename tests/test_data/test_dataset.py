"""Tests for PointwiseDataset and make_dataloader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reranking_dcn.data.dataset import PointwiseDataset, make_dataloader


@pytest.fixture()
def dummy_arrays():
    rng = np.random.default_rng(42)
    N, n_num, n_cat = 64, 10, 3
    numeric = rng.standard_normal((N, n_num)).astype(np.float32)
    cat_indices = rng.integers(0, 5, size=(N, n_cat)).astype(np.int32)
    targets = rng.integers(0, 2, size=(N,)).astype(np.float32)
    emb = rng.standard_normal((N, 128)).astype(np.float32)
    return numeric, cat_indices, targets, emb


class TestPointwiseDataset:
    def test_length(self, dummy_arrays):
        numeric, cat_indices, targets, _ = dummy_arrays
        ds = PointwiseDataset(numeric, cat_indices, targets)
        assert len(ds) == 64

    @pytest.mark.parametrize(
        "use_emb, expected_len",
        [(False, 3), (True, 4)],
        ids=["no_emb_3", "with_emb_4"],
    )
    def test_getitem_tuple_length(self, dummy_arrays, use_emb, expected_len):
        numeric, cat_indices, targets, emb = dummy_arrays
        ds = PointwiseDataset(
            numeric,
            cat_indices,
            targets,
            pretrained_emb=emb if use_emb else None,
        )
        assert len(ds[0]) == expected_len

    @pytest.mark.parametrize(
        "index, expected_dtype",
        [(0, torch.float32), (1, torch.long), (2, torch.float32)],
        ids=["numeric_float32", "cat_long", "target_float32"],
    )
    def test_tensor_dtypes(self, dummy_arrays, index, expected_dtype):
        numeric, cat_indices, targets, _ = dummy_arrays
        ds = PointwiseDataset(numeric, cat_indices, targets)
        assert ds[0][index].dtype == expected_dtype

    def test_target_values_match(self, dummy_arrays):
        numeric, cat_indices, targets, _ = dummy_arrays
        ds = PointwiseDataset(numeric, cat_indices, targets)
        expected = torch.as_tensor(targets, dtype=torch.float32)
        torch.testing.assert_close(ds.targets, expected)

    def test_emb_shape(self, dummy_arrays):
        numeric, cat_indices, targets, emb = dummy_arrays
        ds = PointwiseDataset(numeric, cat_indices, targets, pretrained_emb=emb)
        assert ds[0][3].shape == (128,)

    def test_emb_is_none_attribute(self, dummy_arrays):
        numeric, cat_indices, targets, _ = dummy_arrays
        ds = PointwiseDataset(numeric, cat_indices, targets)
        assert ds.pretrained_emb is None


class TestMakeDataloader:
    def test_returns_dataloader(self, dummy_arrays):
        numeric, cat_indices, targets, _ = dummy_arrays
        loader = make_dataloader(
            numeric,
            cat_indices,
            targets,
            batch_size=16,
            shuffle=False,
            use_cuda=False,
        )
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_batch_shapes(self, dummy_arrays):
        numeric, cat_indices, targets, _ = dummy_arrays
        loader = make_dataloader(
            numeric,
            cat_indices,
            targets,
            batch_size=16,
            shuffle=False,
            use_cuda=False,
        )
        batch = next(iter(loader))
        assert batch[0].shape == (16, 10)
        assert batch[1].shape == (16, 3)
        assert batch[2].shape == (16,)

    def test_batch_with_emb(self, dummy_arrays):
        numeric, cat_indices, targets, emb = dummy_arrays
        loader = make_dataloader(
            numeric,
            cat_indices,
            targets,
            batch_size=16,
            shuffle=False,
            use_cuda=False,
            pretrained_emb=emb,
        )
        batch = next(iter(loader))
        assert len(batch) == 4
        assert batch[3].shape == (16, 128)

    def test_cpu_mode_no_workers(self, dummy_arrays):
        numeric, cat_indices, targets, _ = dummy_arrays
        loader = make_dataloader(
            numeric,
            cat_indices,
            targets,
            batch_size=16,
            shuffle=False,
            num_workers=8,
            use_cuda=False,
        )
        assert loader.num_workers == 0

    def test_iterates_full_dataset(self, dummy_arrays):
        numeric, cat_indices, targets, _ = dummy_arrays
        loader = make_dataloader(
            numeric,
            cat_indices,
            targets,
            batch_size=16,
            shuffle=False,
            use_cuda=False,
        )
        total = sum(b[0].shape[0] for b in loader)
        assert total == 64
