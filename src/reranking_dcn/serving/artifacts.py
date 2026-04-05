"""Serialization of preprocessing artifacts for serving."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..config import Config
from ..features import ALL_FEATURES, CAT_FEATURES, NUMERIC_FEATURES
from ..preprocessing.gpu_preprocessor import GPUFeaturePreprocessor

log = logging.getLogger(__name__)


def serialize_preprocessing_artifacts(
    preprocessor: GPUFeaturePreprocessor,
    cfg: Config,
    output_dir: Path,
) -> None:
    """Save all preprocessing state needed for serving-time feature transform."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(preprocessor, "_quantile_boundaries"):
        raise ValueError(
            "Preprocessor has no _quantile_boundaries buffer. "
            "Was .fit() called or boundaries registered?"
        )
    boundaries = preprocessor._quantile_boundaries.cpu().numpy()
    np.save(output_dir / "quantile_boundaries.npy", boundaries)
    log.info("  Saved quantile_boundaries: %s", boundaries.shape)

    with open(output_dir / "cat_encoders.json", "w") as f:
        json.dump(preprocessor.cat_encoders, f, indent=2, default=str)
    log.info("  Saved cat_encoders: %d features", len(preprocessor.cat_encoders))

    id_hash = {col: list(v) for col, v in cfg.id_hash_config.items()}
    with open(output_dir / "id_hash_config.json", "w") as f:
        json.dump(id_hash, f, indent=2)
    log.info("  Saved id_hash_config: %s", id_hash)

    with open(output_dir / "vocab_sizes.json", "w") as f:
        json.dump(preprocessor.vocab_sizes, f, indent=2)

    with open(output_dir / "embedding_dims.json", "w") as f:
        json.dump(preprocessor.embedding_dims, f, indent=2)

    arch_dict = cfg.to_serving_arch_dict(
        num_continuous=preprocessor.num_continuous,
        vocab_sizes=preprocessor.vocab_sizes,
    )
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(arch_dict, f, indent=2)
    log.info("  Saved model_config")

    feature_list = {
        "numeric_features": list(NUMERIC_FEATURES),
        "cat_features": list(CAT_FEATURES),
        "all_features": list(ALL_FEATURES),
    }
    with open(output_dir / "feature_list.json", "w") as f:
        json.dump(feature_list, f, indent=2)
    log.info("  Saved feature_list (%d numeric, %d cat)", len(NUMERIC_FEATURES), len(CAT_FEATURES))
