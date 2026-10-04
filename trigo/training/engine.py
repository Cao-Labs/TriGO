from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from trigo.go_term import GOEncodingBundle
from trigo.training.losses import MultiBranchBCELoss
from trigo.training.metrics import evaluate_multibranch

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


@dataclass
class EngineConfig:
    grad_clip_norm: float = 1.0
    show_progress: bool = True
    use_amp: bool = False
    amp_dtype: str = "float16"
    metrics_backend: str = "fast"
    metrics_propagate: bool = True
    train_progress_desc: str = "train"
    eval_progress_desc: str = "eval"
    active_namespaces: tuple[str, ...] = ("bp", "mf", "cc")


def _resolve_amp_dtype(name: str) -> torch.dtype:
    lowered = name.strip().lower()
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype='{name}'. Use one of: float16, bfloat16")


def _build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "entries": batch["entries"],
        "sequence_embeddings": batch["sequence_embeddings"].to(device, non_blocking=True),
        "contact_map": batch["contact_map"].to(device, non_blocking=True),
        "residue_mask": batch["residue_mask"].to(device, non_blocking=True),
        "labels": {ns: y.to(device, non_blocking=True) for ns, y in batch["labels"].items()},
    }


def _filter_namespaces(
    values: Dict[str, Any],
    active_namespaces: tuple[str, ...],
) -> Dict[str, Any]:
    return {ns: values[ns] for ns in active_namespaces if ns in values}


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    go_bundle: GOEncodingBundle,
    criterion: MultiBranchBCELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: EngineConfig | None = None,
    scaler: Optional[Any] = None,
    ema: Optional[Any] = None,
) -> Dict[str, float]:
    cfg = cfg or EngineConfig()
    model.train()
    running = {"total": 0.0, "bp": 0.0, "mf": 0.0, "cc": 0.0}
    steps = 0
    amp_enabled = cfg.use_amp and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
    scaler = scaler or _build_grad_scaler(enabled=amp_enabled)

    batch_iter = dataloader
    if cfg.show_progress:
        batch_iter = tqdm(dataloader, desc=cfg.train_progress_desc, unit="batch", leave=False)

    for batch in batch_iter:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                sequence_embeddings=batch["sequence_embeddings"],
                contact_map=batch["contact_map"],
                go_bundle=go_bundle,
                residue_mask=batch["residue_mask"],
            )
            logits = outputs["logits"]
            active_logits = _filter_namespaces(logits, cfg.active_namespaces)
            active_labels = _filter_namespaces(batch["labels"], cfg.active_namespaces)
            loss, loss_items = criterion(active_logits, active_labels, aux=outputs)

        if amp_enabled:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        steps += 1
        for k in running:
            if k in loss_items:
                running[k] += float(loss_items[k].detach().cpu().item())

    if steps == 0:
        return {k: 0.0 for k in running}
    return {k: v / steps for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    go_bundle: GOEncodingBundle,
    criterion: MultiBranchBCELoss,
    device: torch.device,
    cfg: EngineConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or EngineConfig()
    model.eval()
    running = {"total": 0.0, "bp": 0.0, "mf": 0.0, "cc": 0.0}
    steps = 0
    amp_enabled = cfg.use_amp and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)

    y_true = {ns: [] for ns in ("bp", "mf", "cc")}
    y_pred = {ns: [] for ns in ("bp", "mf", "cc")}
    proteins = []

    batch_iter = dataloader
    if cfg.show_progress:
        batch_iter = tqdm(dataloader, desc=cfg.eval_progress_desc, unit="batch", leave=False)

    for batch in batch_iter:
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                sequence_embeddings=batch["sequence_embeddings"],
                contact_map=batch["contact_map"],
                go_bundle=go_bundle,
                residue_mask=batch["residue_mask"],
            )
            logits = outputs["logits"]
            probs = {ns: torch.sigmoid(x) for ns, x in logits.items()}
            active_logits = _filter_namespaces(logits, cfg.active_namespaces)
            active_labels = _filter_namespaces(batch["labels"], cfg.active_namespaces)
            loss, loss_items = criterion(active_logits, active_labels, aux=outputs)
        steps += 1
        for k in running:
            if k in loss_items:
                running[k] += float(loss_items[k].detach().cpu().item())

        proteins.extend(batch["entries"])
        for ns in cfg.active_namespaces:
            if ns not in probs:
                continue
            y_true[ns].append(batch["labels"][ns].detach().float().cpu().numpy())
            # numpy conversion on CPU does not support bf16 in this runtime path.
            y_pred[ns].append(probs[ns].detach().float().cpu().numpy())

    mean_loss = {k: (v / steps if steps > 0 else 0.0) for k, v in running.items()}

    y_true_np = {
        ns: (np.concatenate(y_true[ns], axis=0) if y_true[ns] else np.empty((0, 0)))
        for ns in ("bp", "mf", "cc")
    }
    y_pred_np = {
        ns: (np.concatenate(y_pred[ns], axis=0) if y_pred[ns] else np.empty((0, 0)))
        for ns in ("bp", "mf", "cc")
    }
    metrics = evaluate_multibranch(
        y_true_np,
        y_pred_np,
        go_bundle=go_bundle,
        proteins=proteins,
        propagate=cfg.metrics_propagate,
        drop_root=True,
        show_progress=cfg.show_progress,
        metrics_device=str(device),
        backend=cfg.metrics_backend,
        active_namespaces=cfg.active_namespaces,
    )
    return {
        "loss": mean_loss,
        "metrics": metrics,
        "proteins": proteins,
        "y_true": y_true_np,
        "y_pred": y_pred_np,
    }
