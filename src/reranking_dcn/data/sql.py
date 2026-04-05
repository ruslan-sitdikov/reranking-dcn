"""SQL query builders for BQ data export.

All queries target the unified training table and produce Parquet-friendly
output suitable for DCN-v2 pointwise training.
"""

from __future__ import annotations

from ..config import TrainingConfig
from ..features import CAT_FEATURES, NUMERIC_FEATURES, PRETRAINED_EMB_DIM

META_COLUMNS = [
    "unified_buyer_id",
    "product_id",
    "shop_id",
    "event_timestamp_day",
]

USER_DATE_KEY_SCALE = 100_000
_UBI_HASH_MOD = 9_000_000_000_000


def _ubi_hash_sql(col: str) -> str:
    """Overflow-safe INT64 hash of a unified_buyer_id string.

    Raw FARM_FINGERPRINT values can overflow when multiplied by
    USER_DATE_KEY_SCALE, so we take MOD(ABS(...), 9e12) first.
    """
    return f"MOD(ABS(FARM_FINGERPRINT({col})), {_UBI_HASH_MOD})"


def _select_cols() -> str:
    seen: set[str] = set()
    cols: list[str] = []
    for c in META_COLUMNS + list(NUMERIC_FEATURES) + list(CAT_FEATURES):
        if c not in seen:
            cols.append(c)
            seen.add(c)
    return ", ".join(cols)


def _user_sample_filter(cfg: TrainingConfig) -> str:
    if cfg.data.user_sample_pct >= 100:
        return "TRUE"
    return f"MOD(ABS(FARM_FINGERPRINT(unified_buyer_id)), 100) " f"< {cfg.data.user_sample_pct}"


def _neg_ratio_filter(alias: str, neg_scale: int) -> str:
    """Deterministic ratio-based negative downsampling via FARM_FINGERPRINT."""
    hash_expr = (
        f"MOD(ABS(FARM_FINGERPRINT(CONCAT("
        f"{alias}.unified_buyer_id, '|', "
        f"CAST({alias}.product_id AS STRING), '|', "
        f"CAST({alias}.event_timestamp_day AS STRING)"
        f"))), 1000000)"
    )
    return (
        f"{hash_expr} <= CAST(1000000 * {neg_scale}"
        f" * {alias}._pos_rate / (1.0 - {alias}._pos_rate) AS INT64)"
    )


def _embedding_select_cols(
    alias: str,
    array_col: str,
    col_prefix: str,
    dim: int,
) -> str:
    return ",\n      ".join(
        f"COALESCE({alias}.{array_col}[OFFSET({i})], 0.0) AS {col_prefix}{i}" for i in range(dim)
    )


# ---------------------------------------------------------------------------
# Pointwise training data
# ---------------------------------------------------------------------------


def _sql_nn_pointwise(
    cfg: TrainingConfig,
    start_date: str,
    end_date: str,
) -> str:
    """Pointwise data with ratio-based negative downsampling."""
    neg_ratio = _neg_ratio_filter("f", cfg.data.negative_sample_scale)
    filt = (
        f"event_timestamp_day >= DATE('{start_date}')\n"
        f"      AND event_timestamp_day < DATE('{end_date}')\n"
        f"      AND ({cfg.data.surface_filter})\n"
        f"      AND ({_user_sample_filter(cfg)})"
    )
    cols = _select_cols()
    return f"""WITH
  _stats AS (
    SELECT AVG(IF(is_native_order, 1.0, 0.0)) AS _pos_rate
    FROM {cfg.data.source_table}
    WHERE {filt}
  ),
  base AS (
    SELECT src.*, _stats._pos_rate
    FROM {cfg.data.source_table} AS src
    CROSS JOIN _stats
    WHERE {filt}
  )
SELECT {cols},
  IF(f.is_native_order, 1, 0)       AS target_is_native_order,
  {_ubi_hash_sql("f.unified_buyer_id")} AS ubi_hash
FROM base AS f
WHERE f.is_native_order = TRUE
   OR {neg_ratio}"""


def sql_nn_base_train(cfg: TrainingConfig) -> str:
    """Base-training split: ``[base_train_start, base_train_stop)``."""
    return _sql_nn_pointwise(cfg, cfg.base_train_start, cfg.base_train_stop)


def sql_nn_finetune_day(cfg: TrainingConfig) -> str:
    """Single-day fine-tune (daily pipeline): ``[finetune_day_start, finetune_day_stop)``."""
    return _sql_nn_pointwise(cfg, cfg.finetune_day_start, cfg.finetune_day_stop)


# ---------------------------------------------------------------------------
# Embedding exports
# ---------------------------------------------------------------------------


def sql_product_embeddings(cfg: TrainingConfig, ids_table: str) -> str:
    """Nomic product embeddings for products present in the training data.

    ``ids_table`` is a fully-qualified BQ temp table containing a single
    ``product_id`` column with the distinct product IDs extracted from the
    already-exported (and neg-sampled) training Parquet.
    """
    pe = _embedding_select_cols(
        "pe",
        "product_search_document_embedding",
        "product_emb_",
        PRETRAINED_EMB_DIM,
    )
    return f"""SELECT pe.product_id,
      {pe}
FROM {cfg.data.nomic_embedding_table} AS pe
INNER JOIN {ids_table} AS ids
  ON pe.product_id = ids.product_id"""


def sql_user_embeddings(cfg: TrainingConfig, ids_table: str) -> str:
    """Point-in-time user-average embeddings for (user, day) pairs in the training data.

    ``ids_table`` is a fully-qualified BQ temp table containing
    ``unified_buyer_id`` and ``event_timestamp_day`` columns with the distinct
    (user, day) pairs extracted from the already-exported training Parquet.

    Joins directly on ``unified_buyer_id`` against the UBI-keyed SCD-2 history
    table (feature_store_v2), fetching the ORDER-based embedding valid at
    each training day.
    """
    ue = _embedding_select_cols(
        "ue",
        "user_average_product_search_document_embedding",
        "user_emb_",
        PRETRAINED_EMB_DIM,
    )
    ubi_hash = _ubi_hash_sql("td.unified_buyer_id")
    return f"""WITH
training_user_days AS (
    SELECT DISTINCT unified_buyer_id, event_timestamp_day
    FROM {ids_table}
),
date_range AS (
    SELECT MIN(event_timestamp_day) AS min_day, MAX(event_timestamp_day) AS max_day
    FROM training_user_days
),
user_emb_history AS (
    SELECT ue.*
    FROM {cfg.data.user_embedding_history_table} AS ue
    CROSS JOIN date_range AS dr
    WHERE ue.interaction_type = 'ORDER'
      AND ue.valid_from <= TIMESTAMP(dr.max_day)
      AND (ue.valid_to IS NULL
           OR ue.valid_to >= TIMESTAMP(dr.min_day))
)
SELECT
    ({ubi_hash} * {USER_DATE_KEY_SCALE}
     + UNIX_DATE(td.event_timestamp_day)) AS user_date_key,
    {ue}
FROM training_user_days AS td
INNER JOIN user_emb_history AS ue
    ON  ue.unified_buyer_id = td.unified_buyer_id
    AND ue.valid_from <= TIMESTAMP(td.event_timestamp_day)
    AND (ue.valid_to IS NULL
         OR ue.valid_to > TIMESTAMP(td.event_timestamp_day))"""
