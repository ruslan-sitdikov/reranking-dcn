"""Tests for EmbeddingLookup and embedding helper functions."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from reranking_dcn.data.embeddings import (
    EmbeddingLookup,
    lookup_pretrained_embeddings,
    user_date_key,
)
from reranking_dcn.data.sql import USER_DATE_KEY_SCALE


class TestEmbeddingLookup:
    @pytest.fixture()
    def lookup(self) -> EmbeddingLookup:
        ids = np.array([10, 30, 20, 50, 40], dtype=np.int64)
        emb = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        return EmbeddingLookup(ids, emb)

    def test_known_ids_returned(self, lookup: EmbeddingLookup):
        query = np.array([10, 30, 50], dtype=np.int64)
        result = lookup.lookup(query)
        assert result.shape == (3, 8)
        assert not np.allclose(result[0], 0.0)
        assert not np.allclose(result[1], 0.0)

    def test_unknown_ids_return_zeros(self, lookup: EmbeddingLookup):
        query = np.array([999, 1000], dtype=np.int64)
        result = lookup.lookup(query)
        assert result.shape == (2, 8)
        np.testing.assert_array_equal(result, 0.0)

    def test_mixed_known_and_unknown(self, lookup: EmbeddingLookup):
        query = np.array([10, 999, 30], dtype=np.int64)
        result = lookup.lookup(query)
        assert not np.allclose(result[0], 0.0)
        np.testing.assert_array_equal(result[1], 0.0)
        assert not np.allclose(result[2], 0.0)

    def test_empty_query(self, lookup: EmbeddingLookup):
        query = np.array([], dtype=np.int64)
        result = lookup.lookup(query)
        assert result.shape == (0, 8)

    def test_output_is_float16(self, lookup: EmbeddingLookup):
        query = np.array([10], dtype=np.int64)
        result = lookup.lookup(query)
        assert result.dtype == np.float16

    def test_lookup_with_preallocated_out(self, lookup: EmbeddingLookup):
        query = np.array([10, 20], dtype=np.int64)
        out = np.ones((2, 8), dtype=np.float16)
        result = lookup.lookup(query, out=out)
        assert result is out
        assert not np.allclose(result[0], 1.0)

    def test_presorted_flag(self):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        emb = np.eye(5, dtype=np.float32)[:, :3]
        lk = EmbeddingLookup(ids, emb, presorted=True)
        result = lk.lookup(np.array([3], dtype=np.int64))
        assert result.shape == (1, 3)
        assert result[0, 2] != 0.0

    def test_emb_dim_attribute(self, lookup: EmbeddingLookup):
        assert lookup.emb_dim == 8

    def test_single_id_lookup(self, lookup: EmbeddingLookup):
        result = lookup.lookup(np.array([10], dtype=np.int64))
        assert result.shape == (1, 8)

    def test_duplicate_query_ids(self, lookup: EmbeddingLookup):
        query = np.array([10, 10, 10], dtype=np.int64)
        result = lookup.lookup(query)
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[1], result[2])


def _date_to_unix_days(year: int, month: int, day: int) -> int:
    """Convert a date to Unix epoch days (same as Polars Date -> Int32 cast)."""
    from datetime import date

    epoch = date(1970, 1, 1)
    return (date(year, month, day) - epoch).days


class TestUserDateKey:
    def test_basic_computation(self):
        df = pl.DataFrame(
            {
                "ubi_hash": [100, 200],
                "event_timestamp_day": ["2026-03-24", "2026-03-25"],
            }
        ).with_columns(pl.col("event_timestamp_day").str.to_date())

        keys = user_date_key(df)
        assert keys.shape == (2,)

        day1_unix = _date_to_unix_days(2026, 3, 24)
        day2_unix = _date_to_unix_days(2026, 3, 25)
        assert keys[0] == 100 * USER_DATE_KEY_SCALE + day1_unix
        assert keys[1] == 200 * USER_DATE_KEY_SCALE + day2_unix

    def test_null_ubi_hash_filled_to_zero(self):
        df = pl.DataFrame(
            {
                "ubi_hash": [None],
                "event_timestamp_day": ["2026-03-24"],
            }
        ).with_columns(
            pl.col("ubi_hash").cast(pl.Int64),
            pl.col("event_timestamp_day").str.to_date(),
        )
        keys = user_date_key(df)
        day_unix = _date_to_unix_days(2026, 3, 24)
        assert keys[0] == 0 * USER_DATE_KEY_SCALE + day_unix

    @pytest.mark.parametrize(
        "uuid_hashes, dates",
        [
            ([1, 2], ["2026-03-24", "2026-03-24"]),
            ([1, 1], ["2026-03-24", "2026-03-25"]),
        ],
        ids=["different_hashes_same_day", "same_hash_different_days"],
    )
    def test_unique_keys_for_different_inputs(self, uuid_hashes, dates):
        df = pl.DataFrame(
            {
                "ubi_hash": uuid_hashes,
                "event_timestamp_day": dates,
            }
        ).with_columns(pl.col("event_timestamp_day").str.to_date())
        keys = user_date_key(df)
        assert keys[0] != keys[1]


class TestLookupPretrainedEmbeddings:
    def test_returns_none_when_no_lookups(self):
        df = pl.DataFrame(
            {
                "product_id": [1, 2],
                "ubi_hash": [10, 20],
                "event_timestamp_day": ["2026-03-24", "2026-03-24"],
            }
        ).with_columns(pl.col("event_timestamp_day").str.to_date())
        result = lookup_pretrained_embeddings(df, None, None)
        assert result is None

    def test_product_only_lookup(self):
        rng = np.random.default_rng(42)
        product_ids = np.array([1, 2, 3], dtype=np.int64)
        product_emb = rng.standard_normal((3, 4)).astype(np.float32)
        product_lookup = EmbeddingLookup(product_ids, product_emb)

        df = pl.DataFrame(
            {
                "product_id": [1, 2, 999],
                "ubi_hash": [10, 20, 30],
                "event_timestamp_day": ["2026-03-24", "2026-03-24", "2026-03-24"],
            }
        ).with_columns(pl.col("event_timestamp_day").str.to_date())

        result = lookup_pretrained_embeddings(df, product_lookup, None)
        assert result is not None
        assert result.shape == (3, 4)
        assert not np.allclose(result[0], 0.0)
        np.testing.assert_array_equal(result[2], 0.0)

    def test_combined_product_and_user(self):
        rng = np.random.default_rng(42)
        product_lookup = EmbeddingLookup(
            np.array([1, 2], dtype=np.int64),
            rng.standard_normal((2, 4)).astype(np.float32),
        )

        day_unix = _date_to_unix_days(2026, 3, 24)
        user_keys = np.array(
            [10 * USER_DATE_KEY_SCALE + day_unix],
            dtype=np.int64,
        )
        user_lookup = EmbeddingLookup(
            user_keys,
            rng.standard_normal((1, 6)).astype(np.float32),
        )

        df = pl.DataFrame(
            {
                "product_id": [1],
                "ubi_hash": [10],
                "event_timestamp_day": ["2026-03-24"],
            }
        ).with_columns(pl.col("event_timestamp_day").str.to_date())

        result = lookup_pretrained_embeddings(df, product_lookup, user_lookup)
        assert result is not None
        assert result.shape == (1, 10)  # 4 product + 6 user
