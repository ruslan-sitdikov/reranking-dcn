"""DCN-v2 training loop with focal loss, cosine LR, and layer-wise fine-tune."""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainingConfig
from .losses import focal_loss
from .monitoring import CometMonitor

log = logging.getLogger(__name__)


def _check_triton() -> bool:
    """Return True if a working triton installation exists (needed by torch.compile/inductor)."""
    try:
        import importlib

        importlib.import_module("triton")
        return True
    except ModuleNotFoundError:
        return False


class DCNTrainer:
    """Manages one DCN-v2 model's training lifecycle.

    Supports:
    * Base training with cosine LR + linear warmup
    * Fine-tune with per-layer learning rates and momentum transfer
    * bfloat16 mixed precision on CUDA
    * ``torch.compile`` for H200 throughput
    """

    _LOG_EVERY_N_BATCHES = 50

    def __init__(
        self,
        model: nn.Module,
        preprocessor: nn.Module,
        cfg: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.preprocessor = preprocessor.to(device)
        self.cfg = cfg
        self.device = device
        self._use_amp = device.type == "cuda"

        if device.type == "cuda":
            _triton_ok = _check_triton()
            if _triton_ok:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                log.info("  torch.compile(mode='reduce-overhead') applied")
            else:
                log.warning("  triton not available, skipping torch.compile (eager mode)")

        use_fused = device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            fused=use_fused,
        )
        self._use_fused_optimizer = use_fused
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._monitor: CometMonitor | None = None

    def set_monitor(self, monitor: CometMonitor) -> None:
        """Attach an optional monitor for metrics logging."""
        self._monitor = monitor

    # -- LR scheduling -----------------------------------------------------

    def create_cosine_schedule(self, total_steps: int) -> None:
        """Per-step cosine LR with linear warmup."""
        warmup_steps = max(int(total_steps * self.cfg.optimizer.cosine_warmup_fraction), 1)
        min_frac = self.cfg.optimizer.cosine_min_lr_fraction

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return min_frac + (1.0 - min_frac) * step / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return min_frac + 0.5 * (1.0 - min_frac) * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        log.info(
            "  Cosine LR schedule: %d total steps, %d warmup, min_factor=%.3f",
            total_steps,
            warmup_steps,
            min_frac,
        )

    def set_finetune_lr(self, lr_config: dict[str, float]) -> None:
        """Rebuild optimizer with layer-wise LRs, preserving AdamW momentum."""
        default_lr = lr_config.get("default", 1e-4)
        param_groups: list[dict] = []
        assigned: set[int] = set()

        for group_prefix, lr in lr_config.items():
            if group_prefix == "default":
                continue
            group_params: list[nn.Parameter] = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                clean = name.replace("_orig_mod.", "")
                if clean.startswith(group_prefix + "."):
                    if id(param) not in assigned:
                        group_params.append(param)
                        assigned.add(id(param))
            if group_params:
                param_groups.append({"params": group_params, "lr": lr})
                log.info(
                    "  LR group %-15s: %3d params, lr=%.1e",
                    group_prefix,
                    len(group_params),
                    lr,
                )

        remaining = [
            p for p in self.model.parameters() if p.requires_grad and id(p) not in assigned
        ]
        if remaining:
            param_groups.append({"params": remaining, "lr": default_lr})
            log.info(
                "  LR group %-15s: %3d params, lr=%.1e",
                "default",
                len(remaining),
                default_lr,
            )

        old_state = dict(self.optimizer.state)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.optimizer.weight_decay,
            fused=self._use_fused_optimizer,
        )
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                if p in old_state:
                    self.optimizer.state[p] = old_state[p]
        log.info(
            "  Layer-wise LR applied (%d groups, momentum preserved)",
            len(param_groups),
        )

    # -- Training ----------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one pointwise training epoch with focal loss.

        Returns the average loss over the epoch.
        """
        self.model.train()
        total_loss, total_items = 0.0, 0
        window_loss, window_items = 0.0, 0
        has_emb = getattr(loader.dataset, "pretrained_emb", None) is not None
        gamma = self.cfg.optimizer.focal_loss_gamma
        alpha = self.cfg.optimizer.focal_loss_alpha

        pbar = tqdm(
            loader,
            desc="  train",
            leave=False,
            unit="batch",
            miniters=self._LOG_EVERY_N_BATCHES,
        )
        for batch_idx, batch in enumerate(pbar):
            tensors = [t.to(self.device, non_blocking=True) for t in batch]
            if has_emb:
                numeric, cat_idx, targets, emb = tensors
            else:
                numeric, cat_idx, targets = tensors
                emb = None

            numeric = self.preprocessor.quantile_encode_on_device(numeric)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
                logits = self.model(numeric, cat_idx, emb)

            logits = logits.float()
            cls_loss = focal_loss(logits, targets, gamma=gamma, alpha=alpha)
            logit_penalty = 1e-2 * (logits**2).mean()
            loss = cls_loss + logit_penalty

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.optimizer.grad_clip_norm
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            B = numeric.shape[0]
            batch_loss = loss.item()
            total_loss += batch_loss * B
            total_items += B
            window_loss += batch_loss * B
            window_items += B

            if (batch_idx + 1) % self._LOG_EVERY_N_BATCHES == 0:
                avg_window = window_loss / max(window_items, 1)
                avg_total = total_loss / max(total_items, 1)
                pbar.set_postfix(
                    last50_loss=f"{avg_window:.5f}",
                    avg_loss=f"{avg_total:.5f}",
                )
                window_loss, window_items = 0.0, 0

                if self._monitor is not None:
                    self._monitor.log_batch(
                        batch_idx=batch_idx,
                        loss=batch_loss,
                        lr=self.optimizer.param_groups[0]["lr"],
                    )

            if (batch_idx + 1) % 2000 == 0:
                log.info(
                    "    batch %d/%d  avg_loss=%.5f",
                    batch_idx + 1,
                    len(loader),
                    total_loss / max(total_items, 1),
                )

        avg_loss = total_loss / max(total_items, 1)

        if self._monitor is not None:
            self._monitor.log_epoch(loss=avg_loss)

        return avg_loss

    # -- Inference ---------------------------------------------------------

    def predict(
        self,
        numeric: np.ndarray,
        cat_indices: np.ndarray,
        pretrained_emb: np.ndarray | None = None,
        batch_size: int = 65_536,
    ) -> np.ndarray:
        """Item-level inference. Returns raw logit scores ``(N,)``."""
        self.model.eval()
        score_parts: list[np.ndarray] = []
        n = numeric.shape[0]

        with torch.inference_mode():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                num_t = torch.from_numpy(numeric[start:end]).to(self.device, non_blocking=True)
                num_t = self.preprocessor.quantile_encode_on_device(num_t)
                cat_t = (
                    torch.from_numpy(np.ascontiguousarray(cat_indices[start:end]))
                    .long()
                    .to(self.device, non_blocking=True)
                )

                emb_t = None
                if pretrained_emb is not None:
                    chunk = pretrained_emb[start:end]
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32)
                    emb_t = torch.from_numpy(chunk).to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
                    raw = self.model(num_t, cat_t, emb_t)
                score_parts.append(raw.float().cpu().numpy())

        return np.concatenate(score_parts)
