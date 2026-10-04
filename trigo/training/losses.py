from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MultiBranchLossConfig:
    branch_weights: Dict[str, float] | None = None
    use_asymmetric_focal: bool = True
    asym_gamma_pos: float = 0.0
    asym_gamma_neg: float = 2.0
    asym_clip: float = 0.05
    gate_entropy_weight: float = 0.002
    use_pos_weight: bool = False
    use_frequency_weights: bool = True
    frequency_group_count: int = 3
    dynamic_branch_weight: bool = True
    branch_balance_momentum: float = 0.98
    branch_balance_temperature: float = 1.0
    namespace_size_loss_alpha: float = 0.0
    namespace_size_loss_reference: str = "mean"


def compute_pos_weight_from_label_indices(
    label_indices_list: Iterable[Dict[str, list[int]]],
    num_terms: Dict[str, int],
    smoothing: float = 1.0,
) -> Dict[str, Tensor]:
    pos_counts = {ns: torch.zeros(num_terms[ns], dtype=torch.float32) for ns in ("bp", "mf", "cc")}
    total = 0
    for item in label_indices_list:
        total += 1
        for ns in ("bp", "mf", "cc"):
            idxs = item.get(ns, [])
            if idxs:
                pos_counts[ns][idxs] += 1.0

    pos_weight = {}
    for ns in ("bp", "mf", "cc"):
        pos = pos_counts[ns] + smoothing
        neg = (total - pos_counts[ns]) + smoothing
        pos_weight[ns] = neg / pos
    return pos_weight


def compute_effective_number_weights_from_label_indices(
    label_indices_list: Iterable[Dict[str, list[int]]],
    num_terms: Dict[str, int],
    beta: float = 0.999,
    weight_min: float = 0.5,
    weight_max: float = 8.0,
    weight_max_by_namespace: Dict[str, float] | None = None,
) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Build clipped effective-number term weights and frequency groups."""
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0, 1).")
    if weight_min <= 0 or weight_max < weight_min:
        raise ValueError("Require 0 < weight_min <= weight_max.")

    pos_counts = {ns: torch.zeros(num_terms[ns], dtype=torch.float32) for ns in ("bp", "mf", "cc")}
    for item in label_indices_list:
        for ns in ("bp", "mf", "cc"):
            idxs = item.get(ns, [])
            if idxs:
                pos_counts[ns][idxs] += 1.0

    weights: Dict[str, Tensor] = {}
    counts: Dict[str, Tensor] = {}
    for ns in ("bp", "mf", "cc"):
        namespace_weight_max = float((weight_max_by_namespace or {}).get(ns, weight_max))
        if namespace_weight_max < weight_min:
            raise ValueError(f"Require weight_max_by_namespace['{ns}'] >= weight_min.")
        counts_ns = pos_counts[ns]
        # Terms unseen in train cannot use n=0 in the effective-number formula.
        # Keep them finite and neutral, because negative-only supervision would
        # otherwise over-suppress labels the model never observes as positives.
        effective_counts = counts_ns.clamp_min(1.0)
        if beta == 0.0:
            raw = torch.ones_like(effective_counts)
        else:
            raw = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), effective_counts))
        raw = torch.where(counts_ns > 0, raw, torch.ones_like(raw))
        raw = raw / raw.mean().clamp_min(1e-8)
        weights[ns] = raw.clamp(min=float(weight_min), max=namespace_weight_max)
        counts[ns] = counts_ns
    return weights, counts


class MultiBranchBCELoss(nn.Module):
    """
    Branch-wise loss with:
    - positive-row masking (same behavior as baseline)
    - optional asymmetric focal modulation
    - optional gate entropy bonus (prevents gate collapse)
    """

    def __init__(
        self,
        pos_weight: Dict[str, Tensor] | None = None,
        term_weight: Dict[str, Tensor] | None = None,
        term_counts: Dict[str, Tensor] | None = None,
        config: MultiBranchLossConfig | None = None,
    ):
        super().__init__()
        self.pos_weight = pos_weight or {}
        self.term_weight = term_weight or {}
        self.term_counts = term_counts or {}
        self.config = config or MultiBranchLossConfig()
        self.branch_weights = self.config.branch_weights or {"bp": 1.0, "mf": 1.0, "cc": 1.0}
        self.branch_loss_ema: Dict[str, Tensor] = {}

    def _asymmetric_focal_bce_elementwise(
        self,
        logits: Tensor,
        labels: Tensor,
        pos_weight: Optional[Tensor] = None,
    ) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
        if not self.config.use_asymmetric_focal:
            return bce

        prob = torch.sigmoid(logits)
        prob_pos = prob.clamp(min=1e-8, max=1.0 - 1e-8)
        prob_neg = (1.0 - prob).clamp(min=1e-8, max=1.0 - 1e-8)
        if self.config.asym_clip > 0:
            prob_neg = (prob_neg + self.config.asym_clip).clamp(max=1.0)

        pt = prob_pos * labels + prob_neg * (1.0 - labels)
        gamma = self.config.asym_gamma_pos * labels + self.config.asym_gamma_neg * (1.0 - labels)
        focal_weight = torch.pow((1.0 - pt).clamp(min=0.0), gamma)
        return bce * focal_weight

    def _branch_reduction(self, namespace: str, element_loss: Tensor) -> Tensor:
        per_term = element_loss.mean(dim=0)
        if self.config.use_frequency_weights and namespace in self.term_weight:
            weights = self.term_weight[namespace].to(per_term.device)
            if weights.numel() != per_term.numel():
                raise ValueError(
                    f"Term weight size mismatch for branch '{namespace}': "
                    f"weights={weights.numel()} terms={per_term.numel()}"
                )
            per_term = per_term * weights

        group_count = int(self.config.frequency_group_count)
        if group_count <= 1 or namespace not in self.term_counts:
            return per_term.mean()

        counts = self.term_counts[namespace].to(per_term.device)
        if counts.numel() != per_term.numel():
            raise ValueError(
                f"Term count size mismatch for branch '{namespace}': "
                f"counts={counts.numel()} terms={per_term.numel()}"
            )

        order = torch.argsort(counts, stable=True)
        group_losses = []
        for group_idx in range(group_count):
            start = group_idx * per_term.numel() // group_count
            end = (group_idx + 1) * per_term.numel() // group_count
            if end <= start:
                continue
            term_idx = order[start:end]
            group_losses.append(per_term.index_select(0, term_idx).mean())
        if not group_losses:
            return per_term.mean()
        return torch.stack(group_losses).mean()

    def _dynamic_branch_weights(
        self,
        branch_losses: Dict[str, Tensor],
        active_namespaces: list[str],
    ) -> Dict[str, Tensor]:
        device = next(iter(branch_losses.values())).device
        if (not self.config.dynamic_branch_weight) or len(active_namespaces) <= 1:
            return {ns: torch.ones((), device=device) for ns in active_namespaces}

        detached_losses = {ns: branch_losses[ns].detach() for ns in active_namespaces}
        if torch.is_grad_enabled():
            momentum = float(self.config.branch_balance_momentum)
            for ns, value in detached_losses.items():
                if ns not in self.branch_loss_ema:
                    self.branch_loss_ema[ns] = value.clone()
                else:
                    self.branch_loss_ema[ns] = (
                        self.branch_loss_ema[ns].to(device) * momentum + value * (1.0 - momentum)
                    )

        ratios = []
        for ns in active_namespaces:
            ema = self.branch_loss_ema.get(ns, detached_losses[ns]).to(device).clamp_min(1e-8)
            ratios.append((detached_losses[ns] / ema).clamp(min=0.25, max=4.0))
        ratio_tensor = torch.stack(ratios)
        alpha = torch.softmax(float(self.config.branch_balance_temperature) * ratio_tensor, dim=0)
        alpha = alpha * float(len(active_namespaces))
        return {ns: alpha[i] for i, ns in enumerate(active_namespaces)}

    def _namespace_size_weights(
        self,
        logits: Dict[str, Tensor],
        active_namespaces: list[str],
        device: torch.device,
    ) -> Dict[str, Tensor]:
        alpha = float(self.config.namespace_size_loss_alpha)
        if alpha == 0.0 or len(active_namespaces) <= 1:
            return {ns: torch.ones((), device=device) for ns in active_namespaces}

        sizes = {
            ns: float(logits[ns].shape[1])
            for ns in active_namespaces
            if ns in logits and logits[ns].ndim == 2 and logits[ns].shape[1] > 0
        }
        if len(sizes) != len(active_namespaces):
            return {ns: torch.ones((), device=device) for ns in active_namespaces}

        reference_name = str(self.config.namespace_size_loss_reference).lower()
        if reference_name in sizes:
            reference = sizes[reference_name]
        elif reference_name == "min":
            reference = min(sizes.values())
        elif reference_name == "max":
            reference = max(sizes.values())
        else:
            reference = sum(sizes.values()) / float(len(sizes))
        reference = max(reference, 1.0)

        raw = torch.tensor(
            [(sizes[ns] / reference) ** alpha for ns in active_namespaces],
            device=device,
            dtype=torch.float32,
        )
        raw = raw / raw.mean().clamp_min(1e-8)
        return {ns: raw[i] for i, ns in enumerate(active_namespaces)}

    @staticmethod
    def _entropy_of_weights(w: Tensor) -> Tensor:
        p = w.clamp(min=1e-8)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return -(p * torch.log(p)).sum(dim=-1).mean()

    def _gate_entropy_bonus(self, aux: Optional[Dict[str, object]], device: torch.device) -> Tensor:
        if aux is None:
            return torch.zeros((), device=device)

        tensors: list[Tensor] = []
        for key in ("branch_gate_weights", "protein_gate_weights"):
            value = aux.get(key, None)
            if isinstance(value, dict):
                for w in value.values():
                    if isinstance(w, torch.Tensor) and w.ndim >= 2 and w.size(-1) > 1:
                        tensors.append(w)

        if not tensors:
            return torch.zeros((), device=device)

        ents = [self._entropy_of_weights(w.to(device)) for w in tensors]
        return torch.stack(ents).mean()

    def forward(
        self,
        logits: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        aux: Optional[Dict[str, object]] = None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        device = next(iter(logits.values())).device
        branch_loss_values: Dict[str, Tensor] = {}
        losses: Dict[str, Tensor] = {}
        active_namespaces: list[str] = []

        for ns in ("bp", "mf", "cc"):
            if ns not in logits or ns not in labels:
                continue
            static_w = float(self.branch_weights.get(ns, 1.0))
            if static_w <= 0:
                continue
            branch_logits = logits[ns]
            branch_labels = labels[ns].to(branch_logits.device)

            if branch_logits.shape != branch_labels.shape:
                raise ValueError(
                    f"Shape mismatch for branch '{ns}': "
                    f"logits={tuple(branch_logits.shape)} labels={tuple(branch_labels.shape)}"
                )
            if branch_labels.ndim != 2:
                raise ValueError(f"Expected 2D labels for branch '{ns}', got ndim={branch_labels.ndim}")

            sample_mask = branch_labels.sum(dim=1) > 0
            if not bool(sample_mask.any().item()):
                losses[ns] = branch_logits.sum() * 0.0
                continue

            active_logits = branch_logits[sample_mask]
            active_labels = branch_labels[sample_mask]

            branch_pos_weight = self.pos_weight.get(ns)
            if self.config.use_pos_weight and branch_pos_weight is not None:
                branch_pos_weight = branch_pos_weight.to(branch_logits.device)
            else:
                branch_pos_weight = None

            element_loss = self._asymmetric_focal_bce_elementwise(
                logits=active_logits,
                labels=active_labels,
                pos_weight=branch_pos_weight,
            )
            branch_loss = self._branch_reduction(ns, element_loss)
            losses[ns] = branch_loss
            branch_loss_values[ns] = branch_loss
            active_namespaces.append(ns)

        if branch_loss_values:
            static_weights = torch.tensor(
                [float(self.branch_weights.get(ns, 1.0)) for ns in active_namespaces],
                device=device,
                dtype=torch.float32,
            )
            static_weights = static_weights * (float(len(active_namespaces)) / static_weights.sum().clamp_min(1e-8))
            dynamic_weights = self._dynamic_branch_weights(branch_loss_values, active_namespaces)
            namespace_size_weights = self._namespace_size_weights(logits, active_namespaces, device)
            total_loss = torch.zeros((), device=device)
            for i, ns in enumerate(active_namespaces):
                alpha = dynamic_weights[ns].to(device)
                size_w = namespace_size_weights[ns].to(device)
                losses[f"alpha_{ns}"] = alpha.detach()
                losses[f"namespace_size_weight_{ns}"] = size_w.detach()
                total_loss = total_loss + static_weights[i] * alpha * size_w * branch_loss_values[ns]
            total_loss = total_loss / float(len(active_namespaces))
        else:
            total_loss = sum(v.sum() * 0.0 for v in logits.values())

        gate_entropy = self._gate_entropy_bonus(aux=aux, device=device)
        losses["gate_entropy"] = gate_entropy
        if self.config.gate_entropy_weight > 0:
            total_loss = total_loss - float(self.config.gate_entropy_weight) * gate_entropy

        losses["total"] = total_loss
        return total_loss, losses


class SingleBranchBCELoss(nn.Module):
    """
    Single-task loss over concatenated BP/MF/CC logits.

    This is used for clean single-branch ablations: no branch weights,
    no frequency weighting, no dynamic branch balancing, and no gate
    regularization. It can still retain asymmetric focal modulation so the
    ablation removes only multi-branch training rather than the positive/
    negative imbalance treatment.
    """

    def __init__(
        self,
        use_asymmetric_focal: bool = True,
        asym_gamma_pos: float = 0.0,
        asym_gamma_neg: float = 2.0,
        asym_clip: float = 0.05,
    ):
        super().__init__()
        self.use_asymmetric_focal = bool(use_asymmetric_focal)
        self.asym_gamma_pos = float(asym_gamma_pos)
        self.asym_gamma_neg = float(asym_gamma_neg)
        self.asym_clip = float(asym_clip)

    def _elementwise_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if not self.use_asymmetric_focal:
            return bce

        prob = torch.sigmoid(logits)
        prob_pos = prob.clamp(min=1e-8, max=1.0 - 1e-8)
        prob_neg = (1.0 - prob).clamp(min=1e-8, max=1.0 - 1e-8)
        if self.asym_clip > 0:
            prob_neg = (prob_neg + self.asym_clip).clamp(max=1.0)

        pt = prob_pos * labels + prob_neg * (1.0 - labels)
        gamma = self.asym_gamma_pos * labels + self.asym_gamma_neg * (1.0 - labels)
        focal_weight = torch.pow((1.0 - pt).clamp(min=0.0), gamma)
        return bce * focal_weight

    def forward(
        self,
        logits: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        aux: Optional[Dict[str, object]] = None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        del aux
        active_logits = []
        active_labels = []
        active_namespaces = []
        device = next(iter(logits.values())).device
        for ns in ("bp", "mf", "cc"):
            if ns not in logits or ns not in labels:
                continue
            branch_logits = logits[ns]
            branch_labels = labels[ns].to(branch_logits.device)
            if branch_logits.shape != branch_labels.shape:
                raise ValueError(
                    f"Shape mismatch for branch '{ns}': "
                    f"logits={tuple(branch_logits.shape)} labels={tuple(branch_labels.shape)}"
                )
            active_logits.append(branch_logits)
            active_labels.append(branch_labels)
            active_namespaces.append(ns)

        if not active_logits:
            total_loss = torch.zeros((), device=device)
            return total_loss, {"total": total_loss}

        unified_logits = torch.cat(active_logits, dim=1)
        unified_labels = torch.cat(active_labels, dim=1)
        total_loss = self._elementwise_loss(unified_logits, unified_labels).mean()
        losses = {"total": total_loss}

        offset = 0
        for ns, branch_logits in zip(active_namespaces, active_logits):
            width = branch_logits.size(1)
            ns_logits = unified_logits[:, offset : offset + width]
            ns_labels = unified_labels[:, offset : offset + width]
            losses[ns] = self._elementwise_loss(ns_logits, ns_labels).mean()
            offset += width
        return total_loss, losses
