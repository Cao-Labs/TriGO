from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from trigo.go_term import GOEncodingBundle

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


GO_ROOT_BY_NAMESPACE = {
    "bp": "GO:0008150",
    "mf": "GO:0003674",
    "cc": "GO:0005575",
}
NAMESPACES = ("bp", "mf", "cc")
SMIN_PROTOCOL = {
    "type": "normalized semantic distance",
    "thresholds": "0.00..1.00 step 0.01",
    "ru": "mean over branch-positive test proteins of IC(false_negative) / IC(true_union_pred)",
    "mi": "mean over branch-positive test proteins of IC(false_positive) / IC(true_union_pred)",
    "ic": "IC=-log2(term_frequency among branch-positive propagated training proteins), train split only",
    "propagation": "y_true and y_pred propagated to ancestors, namespace root dropped",
}


def _pick_metrics_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _build_parent_adjacency(edge_index: np.ndarray, num_terms: int) -> List[List[int]]:
    parents: List[List[int]] = [[] for _ in range(num_terms)]
    if edge_index.size == 0:
        return parents
    for child, parent in edge_index.T:
        c = int(child)
        p = int(parent)
        if 0 <= c < num_terms and 0 <= p < num_terms:
            parents[c].append(p)
    return parents


def _build_topological_order(parents: List[List[int]]) -> List[int]:
    num_terms = len(parents)
    indegree = np.zeros(num_terms, dtype=np.int32)
    for child in range(num_terms):
        for parent in parents[child]:
            indegree[parent] += 1

    queue = deque(np.where(indegree == 0)[0].tolist())
    topo: List[int] = []
    while queue:
        child = queue.popleft()
        topo.append(child)
        for parent in parents[child]:
            indegree[parent] -= 1
            if indegree[parent] == 0:
                queue.append(parent)

    if len(topo) != num_terms:
        return list(range(num_terms))
    return topo


def _propagate_to_ancestors_dag(
    scores: np.ndarray,
    parents: List[List[int]],
    topo_order: List[int],
    show_progress: bool = False,
    progress_desc: str = "propagate",
) -> np.ndarray:
    if scores.size == 0:
        return scores
    out = scores.copy()
    node_iter = topo_order
    if show_progress:
        node_iter = tqdm(node_iter, desc=progress_desc, unit="node", leave=False)
    for child_idx in node_iter:
        direct_parents = parents[child_idx]
        if not direct_parents:
            continue
        child_col = out[:, child_idx][:, None]
        out[:, direct_parents] = np.maximum(out[:, direct_parents], child_col)
    return out


def _drop_namespace_root(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    namespace: str,
    term_ids: List[str],
) -> tuple[np.ndarray, np.ndarray]:
    root_go = GO_ROOT_BY_NAMESPACE.get(namespace)
    if root_go is None:
        return y_true, y_pred
    try:
        root_idx = term_ids.index(root_go)
    except ValueError:
        return y_true, y_pred
    keep = np.ones(len(term_ids), dtype=bool)
    keep[root_idx] = False
    return y_true[:, keep], y_pred[:, keep]


def maybe_propagate_branch_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    namespace: str,
    go_bundle: GOEncodingBundle | None,
    drop_root: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if go_bundle is None or namespace not in go_bundle.encodings:
        return y_true, y_pred

    ns_enc = go_bundle.encodings[namespace]
    num_terms = len(ns_enc.term_ids)
    if num_terms == 0 or y_true.size == 0 or y_pred.size == 0:
        return y_true, y_pred

    propagation_plan = getattr(ns_enc, "_propagation_plan_cache", None)
    if propagation_plan is None or len(propagation_plan[0]) != num_terms:
        edge_index = ns_enc.edge_index.detach().cpu().numpy()
        parents = _build_parent_adjacency(edge_index=edge_index, num_terms=num_terms)
        topo_order = _build_topological_order(parents)
        propagation_plan = (parents, topo_order)
        setattr(ns_enc, "_propagation_plan_cache", propagation_plan)
    parents, topo_order = propagation_plan

    y_true_prop = _propagate_to_ancestors_dag(
        y_true,
        parents=parents,
        topo_order=topo_order,
        show_progress=show_progress,
        progress_desc=f"prop-y_true-{namespace}",
    )
    y_pred_prop = _propagate_to_ancestors_dag(
        y_pred,
        parents=parents,
        topo_order=topo_order,
        show_progress=show_progress,
        progress_desc=f"prop-y_pred-{namespace}",
    )

    if drop_root:
        y_true_prop, y_pred_prop = _drop_namespace_root(
            y_true=y_true_prop,
            y_pred=y_pred_prop,
            namespace=namespace,
            term_ids=ns_enc.term_ids,
        )
    return y_true_prop, y_pred_prop


def prepare_branch_for_smin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    namespace: str,
    go_bundle: GOEncodingBundle,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_prop, y_pred_prop = maybe_propagate_branch_arrays(
        y_true=y_true.astype(np.float32, copy=False),
        y_pred=y_pred.astype(np.float32, copy=False),
        namespace=namespace,
        go_bundle=go_bundle,
        drop_root=True,
        show_progress=False,
    )
    keep = y_true_prop.sum(axis=1) > 0
    return (
        y_true_prop[keep].astype(bool, copy=False),
        y_pred_prop[keep].astype(np.float32, copy=False),
    )


def compute_ic_by_namespace(
    y_train: Dict[str, np.ndarray],
    go_bundle: GOEncodingBundle,
) -> Dict[str, np.ndarray]:
    ic_by_namespace = {}
    for namespace in NAMESPACES:
        y_true, _ = prepare_branch_for_smin(
            y_train[namespace],
            y_train[namespace],
            namespace,
            go_bundle,
        )
        if y_true.size == 0:
            ic_by_namespace[namespace] = np.zeros((0,), dtype=np.float32)
            continue
        counts = y_true.sum(axis=0).astype(np.float64)
        num_proteins = max(int(y_true.shape[0]), 1)
        probabilities = np.clip(counts / float(num_proteins), 1.0 / float(num_proteins), 1.0)
        ic_by_namespace[namespace] = (-np.log2(probabilities)).astype(np.float32)
    return ic_by_namespace


def normalized_smin_for_branch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ic: np.ndarray,
) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0 or ic.size == 0:
        return {
            "smin_normalized": 0.0,
            "smin_threshold": 0.0,
            "smin_ru": 0.0,
            "smin_mi": 0.0,
        }

    y_true_bool = y_true.astype(bool, copy=False)
    ic_float64 = ic.astype(np.float64, copy=False)
    best = (float("inf"), 0.0, 0.0, 0.0)
    for threshold in np.linspace(0.0, 1.0, 101):
        y_pred_bool = y_pred >= float(threshold)
        union = np.logical_or(y_true_bool, y_pred_bool)
        denominator = union.astype(np.float32).dot(ic_float64)
        denominator = np.where(denominator > 0.0, denominator, 1.0)
        ru_each = (
            np.logical_and(y_true_bool, ~y_pred_bool).astype(np.float32).dot(ic_float64)
            / denominator
        )
        mi_each = (
            np.logical_and(~y_true_bool, y_pred_bool).astype(np.float32).dot(ic_float64)
            / denominator
        )
        ru = float(np.mean(ru_each))
        mi = float(np.mean(mi_each))
        smin = math.sqrt(ru * ru + mi * mi)
        if smin < best[0]:
            best = (smin, float(threshold), ru, mi)
    return {
        "smin_normalized": float(best[0]),
        "smin_threshold": float(best[1]),
        "smin_ru": float(best[2]),
        "smin_mi": float(best[3]),
    }


def normalized_smin(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    ic_by_namespace: Dict[str, np.ndarray],
    go_bundle: GOEncodingBundle,
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for namespace in NAMESPACES:
        y_true_prop, y_pred_prop = prepare_branch_for_smin(
            y_true[namespace],
            y_pred[namespace],
            namespace,
            go_bundle,
        )
        metrics[namespace] = normalized_smin_for_branch(
            y_true_prop,
            y_pred_prop,
            ic_by_namespace[namespace],
        )
    return metrics


def _filter_branch_to_positive_proteins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proteins: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """
    Keep only proteins that have at least one known positive annotation
    in the current branch.
    """
    if y_true.size == 0 or y_pred.size == 0:
        return y_true, y_pred, proteins
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Row mismatch between y_true and y_pred: "
            f"{y_true.shape[0]} vs {y_pred.shape[0]}"
        )
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays for branch metrics, got y_true.ndim={y_true.ndim}, "
            f"y_pred.ndim={y_pred.ndim}"
        )

    positive_row_mask = np.sum(y_true, axis=1) > 0
    if not np.any(positive_row_mask):
        return (
            np.empty((0, y_true.shape[1]), dtype=y_true.dtype),
            np.empty((0, y_pred.shape[1]), dtype=y_pred.dtype),
            [] if proteins is not None else proteins,
        )

    y_true_keep = y_true[positive_row_mask]
    y_pred_keep = y_pred[positive_row_mask]

    proteins_keep = proteins
    if proteins is not None and len(proteins) == int(positive_row_mask.shape[0]):
        proteins_keep = [p for p, keep in zip(proteins, positive_row_mask.tolist()) if keep]
    return y_true_keep, y_pred_keep, proteins_keep


def _compute_prf_curve_torch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    device: torch.device,
    show_progress: bool = False,
    progress_desc: str = "fmax",
    recall_normalizer: str = "all",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if y_true.size == 0 or y_pred.size == 0:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, empty, empty

    yt = torch.as_tensor(y_true, dtype=torch.float32, device=device)
    yp = torch.as_tensor(y_pred, dtype=torch.float32, device=device)
    n = int(yt.size(0))
    if n == 0:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, empty, empty

    threshold_values = [t / 100.0 for t in range(1, 100)]
    thr_iter = threshold_values
    if show_progress:
        thr_iter = tqdm(threshold_values, desc=progress_desc, unit="thr", leave=False)

    recalls: List[float] = []
    precisions: List[float] = []
    fscores: List[float] = []
    thresholds: List[float] = []

    y_true_bool = yt > 0.5

    for thr in thr_iter:
        pred = yp > thr
        num_pred = pred.sum(dim=1)
        num_true = y_true_bool.sum(dim=1)
        overlap = torch.logical_and(pred, y_true_bool).sum(dim=1)

        pred_mask = num_pred > 0
        m = int(pred_mask.sum().item())
        if m == 0:
            continue

        avg_pr = (overlap[pred_mask].float() / num_pred[pred_mask].float()).sum() / float(m)

        true_mask = num_true > 0
        recall_terms = torch.zeros_like(num_true, dtype=torch.float32)
        recall_terms[true_mask] = overlap[true_mask].float() / num_true[true_mask].float()

        if recall_normalizer == "true_positive":
            denom = max(int(true_mask.sum().item()), 1)
        else:
            denom = n
        avg_rc = recall_terms.sum() / float(denom)

        pr = float(avg_pr.item())
        rc = float(avg_rc.item())
        if pr + rc <= 0:
            continue
        f = 2.0 * pr * rc / (pr + rc)

        precisions.append(pr)
        recalls.append(rc)
        fscores.append(float(f))
        thresholds.append(float(thr))

    return (
        np.asarray(recalls, dtype=np.float32),
        np.asarray(precisions, dtype=np.float32),
        np.asarray(fscores, dtype=np.float32),
        np.asarray(thresholds, dtype=np.float32),
    )


def _compute_micro_aupr_torch(y_true: np.ndarray, y_pred: np.ndarray, device: torch.device) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    yt = torch.as_tensor(y_true, dtype=torch.float32, device=device).reshape(-1)
    yp = torch.as_tensor(y_pred, dtype=torch.float32, device=device).reshape(-1)

    pos = torch.sum(yt)
    if float(pos.item()) <= 0:
        return 0.0

    order = torch.argsort(yp, descending=True)
    y_sorted = yt[order]
    tp = torch.cumsum(y_sorted, dim=0)
    ranks = torch.arange(1, y_sorted.numel() + 1, device=device, dtype=torch.float32)
    precision = tp / ranks
    ap = torch.sum(precision * y_sorted) / pos
    return float(ap.item())


def _compute_macro_aupr_torch(y_true: np.ndarray, y_pred: np.ndarray, device: torch.device) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    yt = torch.as_tensor(y_true, dtype=torch.float32, device=device)
    yp = torch.as_tensor(y_pred, dtype=torch.float32, device=device)
    positive_cols = torch.sum(yt, dim=0) > 0
    if not bool(positive_cols.any().item()):
        return 0.0

    yt = yt[:, positive_cols]
    yp = yp[:, positive_cols]
    order = torch.argsort(yp, dim=0, descending=True)
    y_sorted = torch.gather(yt, dim=0, index=order)
    tp = torch.cumsum(y_sorted, dim=0)
    ranks = torch.arange(1, y_sorted.size(0) + 1, device=device, dtype=torch.float32).unsqueeze(1)
    pos = torch.sum(yt, dim=0).clamp_min(1.0)
    ap_col = torch.sum((tp / ranks) * y_sorted, dim=0) / pos
    return float(ap_col.mean().item())


def _compute_micro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    if np.unique(yt).size < 2:
        return 0.0
    try:
        return float(roc_auc_score(yt, yp))
    except Exception:
        return 0.0


def _compute_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if y_true.ndim != 2 or y_pred.ndim != 2:
        return 0.0

    auc_values: List[float] = []
    for col in range(y_true.shape[1]):
        yt_col = y_true[:, col]
        if np.unique(yt_col).size < 2:
            continue
        try:
            auc_values.append(float(roc_auc_score(yt_col, y_pred[:, col])))
        except Exception:
            continue
    if not auc_values:
        return 0.0
    return float(np.mean(auc_values))


def _evaluate_namespace_fast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_device: str = "cpu",
    show_progress: bool = False,
    progress_desc: str = "fmax",
    recall_normalizer: str = "all",
    ) -> tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if y_true.size == 0 or y_pred.size == 0:
        empty = np.asarray([], dtype=np.float32)
        return (
            {
                "fmax": 0.0,
                "best_threshold": 0.5,
                "micro_aupr": 0.0,
                "macro_aupr": 0.0,
                "micro_auc": 0.0,
                "macro_auc": 0.0,
                "auc": 0.0,
            },
            empty,
            empty,
            empty,
            empty,
        )

    device = _pick_metrics_device(metrics_device)
    recall, precision, fscore, thresholds = _compute_prf_curve_torch(
        y_true=y_true,
        y_pred=y_pred,
        device=device,
        show_progress=show_progress,
        progress_desc=progress_desc,
        recall_normalizer=recall_normalizer,
    )

    if fscore.size == 0:
        fmax = 0.0
        best_threshold = 0.5
    else:
        idx = int(np.argmax(fscore))
        fmax = float(fscore[idx])
        best_threshold = float(thresholds[idx])

    micro_aupr = _compute_micro_aupr_torch(y_true, y_pred, device=device)
    macro_aupr = _compute_macro_aupr_torch(y_true, y_pred, device=device)
    micro_auc = _compute_micro_auc(y_true, y_pred)
    macro_auc = _compute_macro_auc(y_true, y_pred)
    return (
        {
            "fmax": fmax,
            "best_threshold": best_threshold,
            "micro_aupr": float(micro_aupr),
            "macro_aupr": float(macro_aupr),
            "micro_auc": float(micro_auc),
            "macro_auc": float(macro_auc),
            "auc": float(micro_auc),
        },
        recall,
        precision,
        fscore,
        thresholds,
    )


def save_eval_pickle(
    save_path: str | Path,
    ontology: str,
    proteins: List[str],
    goterms: List[str],
    gonames: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    payload = {
        "Y_true": y_true,
        "Y_pred": y_pred,
        "goterms": goterms,
        "gonames": gonames,
        "proteins": proteins,
        "ontology": ontology,
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)



def get_branch_pr_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    namespace: str,
    go_bundle: GOEncodingBundle,
    proteins: Optional[List[str]] = None,
    backend: str = "fast",
    metrics_device: str = "cpu",
    recall_normalizer: str = "all",
) -> Dict[str, object]:
    if backend != "fast":
        raise ValueError("TriGO supports only the built-in 'fast' metrics backend.")
    y_true, y_pred, proteins = _filter_branch_to_positive_proteins(
        y_true=y_true,
        y_pred=y_pred,
        proteins=proteins,
    )

    y_true_used, y_pred_used = maybe_propagate_branch_arrays(
        y_true=y_true,
        y_pred=y_pred,
        namespace=namespace,
        go_bundle=go_bundle,
        drop_root=True,
        show_progress=False,
    )
    metrics, recall, precision, fscore, thresholds = _evaluate_namespace_fast(
        y_true=y_true_used,
        y_pred=y_pred_used,
        metrics_device=metrics_device,
        show_progress=False,
        progress_desc=f"fmax-{namespace}",
        recall_normalizer=recall_normalizer,
    )

    return {
        **metrics,
        "recall": recall,
        "precision": precision,
        "fscore": fscore,
        "thresholds": thresholds,
    }


def evaluate_branch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    namespace: str,
    go_bundle: GOEncodingBundle,
    proteins: Optional[List[str]] = None,
    show_progress: bool = False,
    progress_desc: str = "fmax",
    metrics_device: str = "cpu",
    recall_normalizer: str = "all",
    backend: str = "fast",
) -> Dict[str, float]:
    if backend != "fast":
        raise ValueError("TriGO supports only the built-in 'fast' metrics backend.")
    y_true, y_pred, proteins = _filter_branch_to_positive_proteins(
        y_true=y_true,
        y_pred=y_pred,
        proteins=proteins,
    )

    metrics, _, _, _, _ = _evaluate_namespace_fast(
        y_true=y_true,
        y_pred=y_pred,
        metrics_device=metrics_device,
        show_progress=show_progress,
        progress_desc=progress_desc,
        recall_normalizer=recall_normalizer,
    )
    return metrics


def evaluate_multibranch(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    go_bundle: GOEncodingBundle | None = None,
    proteins: Optional[List[str]] = None,
    propagate: bool = True,
    drop_root: bool = True,
    show_progress: bool = False,
    metrics_device: str = "cpu",
    recall_normalizer: str = "all",
    backend: str = "fast",
    active_namespaces: tuple[str, ...] = ("bp", "mf", "cc"),
) -> Dict[str, Dict[str, float]]:
    if backend != "fast":
        raise ValueError("TriGO supports only the built-in 'fast' metrics backend.")
    out: Dict[str, Dict[str, float]] = {}
    ns_iter = tuple(ns for ns in active_namespaces if ns in {"bp", "mf", "cc"})
    if show_progress:
        ns_iter = tqdm(ns_iter, desc="metrics", unit="branch", leave=False)

    for ns in ns_iter:
        branch_y_true = y_true.get(ns, np.empty((0, 0)))
        branch_y_pred = y_pred.get(ns, np.empty((0, 0)))
        if go_bundle is None:
            out[ns] = {
                "fmax": 0.0,
                "best_threshold": 0.5,
                "micro_aupr": 0.0,
                "macro_aupr": 0.0,
                "micro_auc": 0.0,
                "macro_auc": 0.0,
                "auc": 0.0,
            }
            continue

        if propagate:
            branch_y_true, branch_y_pred = maybe_propagate_branch_arrays(
                y_true=branch_y_true,
                y_pred=branch_y_pred,
                namespace=ns,
                go_bundle=go_bundle,
                drop_root=drop_root,
                show_progress=show_progress,
            )

        out[ns] = evaluate_branch(
            branch_y_true,
            branch_y_pred,
            namespace=ns,
            go_bundle=go_bundle,
            proteins=proteins,
            show_progress=show_progress,
            progress_desc=f"fmax-{ns}",
            metrics_device=metrics_device,
            recall_normalizer=recall_normalizer,
            backend=backend,
        )
    return out

