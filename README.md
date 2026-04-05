# DCN-v2 Product Ranker

Training, ONNX export, and serving pipeline for the DCN-v2 (Deep & Cross Network v2)
product re-ranker used in Shop App home feed and boosted merchants surfaces.

## Pipeline Overview

```
BigQuery  →  GCS staging  →  Local Parquet  →  GPU Training  →  ONNX Export  →  GCS Upload
                                                                                    ↓
                                                              CPU Serving  ←  Bigtable / Feature Store
```

**Two operational modes:**

| Mode | Trigger | Data window | What it does |
|------|---------|-------------|--------------|
| `base_train` | Manual / one-off | 4-8 weeks | Fit preprocessor + train from scratch with cosine LR |
| `finetune` | Airflow DAG (daily) | 1 day | Load latest checkpoint, train 1 epoch with layer-wise LRs |

Both modes share a single unified pipeline (`training/pipeline.py`) that handles:
export → train → checkpoint → ONNX export → parity validation → artifact serialization → GCS upload.

## Directory Structure

```
src/reranking_dcn/
├── config.py              # Config (arch) + TrainingConfig (composed sub-configs)
├── features.py            # Feature schema: NUMERIC_FEATURES, CAT_FEATURES, constants
├── cli.py                 # ONNX export CLI (dcn-export entry point)
├── train_cli.py           # Training CLI (python -m reranking_dcn.train_cli)
│
├── nn/                    # Neural network layers
│   ├── blocks.py          # SwiGLU feedforward blocks
│   ├── cross_network.py   # DCN-v2 MoE cross layers
│   └── ranker.py          # DCNv2Ranker — full model
│
├── data/                  # Data loading & BigQuery export
│   ├── sql.py             # SQL query builders
│   ├── exporter.py        # BQ → GCS → local Parquet pipeline
│   ├── dataset.py         # PyTorch Dataset / DataLoader
│   └── embeddings.py      # Vectorized embedding lookup (binary search)
│
├── training/              # Training orchestration
│   ├── pipeline.py        # Top-level pipeline (Strategy pattern)
│   ├── orchestration.py   # Base training + fine-tune loops
│   ├── trainer.py         # DCNTrainer: training loop, LR schedules
│   ├── losses.py          # Focal loss
│   └── monitoring.py      # Comet ML integration
│
├── export/                # ONNX export & validation
│   ├── checkpoint.py      # Model construction + checkpoint loading
│   ├── onnx_export.py     # ONNX graph export
│   ├── validation.py      # PyTorch ↔ ONNX parity checks
│   ├── benchmark.py       # ONNX Runtime latency benchmark
│   └── preproject.py      # Offline embedding pre-projection
│
├── preprocessing/         # Feature transforms
│   ├── gpu_preprocessor.py    # GPU quantile binning (training)
│   ├── serving_preprocessor.py # CPU transform (serving)
│   └── quantile.py            # Shared quantile encoding (NumPy)
│
└── serving/               # Serving wrappers & artifacts
    ├── wrappers.py        # ONNX-compatible model wrappers
    └── artifacts.py       # Serialization for serving
```

## Quick Start

### Base Training (SkyPilot)

```bash
sky launch -c dcn-train vertexai-training/reranking_dcn/sky_configs/dcn_train_dev.yaml \
  --env MODE=base_train \
  --env STOP_DATE=2026-03-24
```

### Base Training (Local)

```bash
python -m reranking_dcn.train_cli --config configs/base_training.yaml
```

### Daily Fine-Tune

```bash
python -m reranking_dcn.train_cli --config configs/daily_finetune.yaml \
  --stop_date 2026-03-24
```

### ONNX Export Only

```bash
dcn-export --checkpoint_path ./outputs/checkpoint.pt \
           --preprocessor_path ./outputs/checkpoint.pt \
           --validate
```

### Dry Run (Print SQL)

```bash
python -m reranking_dcn.train_cli --config configs/base_training.yaml --dry_run
```

## Configuration

`TrainingConfig` is composed from domain-specific sub-configs:

| Sub-config | What it controls |
|-----------|-----------------|
| `DataConfig` | Source tables, negative sampling, date windows, surface filters |
| `InfraConfig` | GCP project, GCS bucket, local/output directories |
| `OptimizerConfig` | Learning rates, focal loss, cosine schedule, fine-tune LRs |
| `RuntimeConfig` | Batch size, workers, threads, epochs |
| `MonitorConfig` | Comet ML project and credentials |

YAML configs use a flat key format for simplicity — `TrainingConfig.from_flat_dict()`
automatically routes each key to the correct sub-config.

## Checkpoint Format (v2)

Checkpoints use safe serialization (no pickle of nn.Module objects):

```python
{
    "checkpoint_version": 2,
    "model_state_dict": {...},           # torch tensors
    "optimizer_state_dict": {...},       # torch tensors
    "preprocessor_state": {              # plain dict + tensors
        "quantile_boundaries": tensor,
        "cat_encoders": {...},
        "vocab_sizes": {...},
        "embedding_dims": {...},
        "id_hash_config": {...},
        "n_quantile_bins": int,
    },
    "pretrained_emb_dim": int,
    "mode": str,
}
```

This enables `torch.load(weights_only=True)` and version-independent deserialization.
Legacy v1 checkpoints (pickled preprocessor) are auto-detected and loaded with a
deprecation warning.

## Testing

```bash
cd vertexai-training/reranking_dcn
pip install -e ".[export]"
pytest tests/
```

Key test suites:
- `test_preprocessing/test_parity.py` — Validates GPU ↔ CPU preprocessing parity
- `test_training/test_orchestration.py` — End-to-end base train + fine-tune
- `test_export/test_parity.py` — PyTorch ↔ ONNX output parity

## Architecture Decisions

- **Quantile binning** over z-scores: distribution-agnostic, nonlinear encoding that handles
  the heavy-tailed feature distributions common in ad-rank features.
- **SwiGLU** over ReLU: non-zero gradients everywhere eliminate dead-neuron issues in
  sparse ad-feature regimes. LayerNorm replaces BatchNorm to avoid train/eval discrepancy.
- **MoE cross layers**: input-dependent gating with low-rank experts provides explicit
  feature interaction learning at manageable parameter cost.
- **Focal loss**: down-weights well-classified examples at ~0.04% positive rate.
- **Pre-projected embeddings**: 768d → 64d projection is done offline and served from
  Bigtable, keeping the ONNX inference graph small and fast.
