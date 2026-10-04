from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trigo.data_pipeline import (
    ProteinGOBatchCollator,
    ProteinGODataset,
    ProteinGODatasetConfig,
    ProteinAnnotationRecord,
    load_annotation_records,
    load_go_namespace_and_name,
    random_split_records,
)
from trigo.go_term import GOEncodingBundle
from trigo.multi_branch import MultiBranchProteinGOConfig, MultiBranchProteinGOModel
from trigo.protein_ple import ProteinPLEConfig
from trigo.training import (
    EngineConfig,
    MultiBranchBCELoss,
    MultiBranchLossConfig,
    SingleBranchBCELoss,
    compute_effective_number_weights_from_label_indices,
    compute_pos_weight_from_label_indices,
    evaluate,
    train_one_epoch,
)
from trigo.training.metrics import (
    SMIN_PROTOCOL,
    compute_ic_by_namespace,
    maybe_propagate_branch_arrays,
    normalized_smin,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train the TriGO protein function predictor")
    p.add_argument("--xlsx-path", type=str, required=True)
    p.add_argument("--pdb-dir", type=str, required=True)
    p.add_argument("--obo-path", type=str, required=True)
    p.add_argument("--feature-cache-dir", type=str, required=True)
    p.add_argument("--go-embedding-path", type=str, required=True)
    p.add_argument(
        "--require-split",
        action="store_true",
        help=(
            "Require every annotation record to have a valid train/val/test value in the Split column. "
            "Use this for all reported dataset configurations."
        ),
    )
    p.add_argument(
        "--use-one-hot-go-embeddings",
        action="store_true",
        help=(
            "Replace each namespace GO embedding matrix with an identity matrix after optional "
            "GO-space filtering. This keeps term identity but removes pretrained GO embedding geometry."
        ),
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--init-checkpoint-path", type=str, default="", help="Optional checkpoint used to initialize model weights before training.")
    p.add_argument(
        "--active-namespaces",
        type=str,
        default="bp,mf,cc",
        help=(
            "Comma-separated namespaces that participate in training loss, validation selection, "
            "and final reported metrics. Use bp, mf, or cc for single-branch training."
        ),
    )
    p.add_argument(
        "--train-only-namespace",
        type=str,
        default="",
        choices=("", "bp", "mf", "cc"),
        help="Freeze all parameters except one namespace-specific branch. Empty keeps full-model training.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--eval-interval",
        type=int,
        default=1,
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lr-patience", type=int, default=3, help="Epochs with no val improvement before reducing LR")
    p.add_argument("--lr-reduce-factor", type=float, default=0.6, help="Multiply LR by this factor on plateau")
    p.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate for scheduler")
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without a validation-score improvement; 0 disables early stopping.",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum weighted validation-score increase that resets early-stopping patience.",
    )
    p.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=0,
        help="Do not stop before this epoch even if early-stopping patience has been exhausted.",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--max-residues", type=int, default=1022)
    p.add_argument("--contact-threshold", type=float, default=8.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--branch-loss-weights", type=str, default='{"bp":1,"mf":1,"cc":1}')
    p.add_argument("--bp-loss-weight", type=float, default=None, help="Override bp loss weight")
    p.add_argument("--mf-loss-weight", type=float, default=None, help="Override mf loss weight")
    p.add_argument("--cc-loss-weight", type=float, default=None, help="Override cc loss weight")
    p.add_argument("--bp-lr-multiplier", type=float, default=1.6, help="Learning-rate multiplier for BP-specific params")
    p.add_argument("--mf-lr-multiplier", type=float, default=1.0, help="Learning-rate multiplier for MF-specific params")
    p.add_argument("--cc-lr-multiplier", type=float, default=1.0, help="Learning-rate multiplier for CC-specific params")
    p.add_argument(
        "--val-score-weights",
        type=str,
        default='{"bp":0.5,"mf":0.25,"cc":0.25}',
        help="Weighted validation score for best-checkpoint selection.",
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=None,
        help=(
            "Unified dropout for protein PLE, GO branches, gates, predictors, and cross-namespace "
            "refinement. Defaults to 0.10. Prefer this for new experiments."
        ),
    )
    p.add_argument(
        "--protein-dropout",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility option that only overrides protein PLE dropout. "
            "Use --dropout for a single model-wide value."
        ),
    )
    p.add_argument("--protein-expert-hidden-dim", type=int, default=256)
    p.add_argument("--protein-expert-output-dim", type=int, default=256)
    p.add_argument("--model-expert-dim", type=int, default=256)
    p.add_argument("--shared-expert-hidden-dim", type=int, default=512)
    p.add_argument("--namespace-expert-hidden-dim", type=int, default=512)
    p.add_argument("--namespace-context-hidden-dim", type=int, default=256)
    p.add_argument("--gate-hidden-dim", type=int, default=128)
    p.add_argument(
        "--normalize-sequence-input",
        action="store_true",
        help="Apply LayerNorm to residue embeddings before protein-side experts.",
    )
    p.add_argument("--use-strong-protein-ple", action="store_true")
    p.add_argument("--clean-expert-gate", action="store_true")
    p.add_argument("--strong-ple-gcn-layers", type=int, default=3)
    p.add_argument("--namespace-attention-topk", type=int, default=256)
    p.add_argument("--bp-namespace-attention-topk", type=int, default=0, help="Override GO context top-k for BP; 0 uses --namespace-attention-topk.")
    p.add_argument("--mf-namespace-attention-topk", type=int, default=0, help="Override GO context top-k for MF; 0 uses --namespace-attention-topk.")
    p.add_argument("--cc-namespace-attention-topk", type=int, default=0, help="Override GO context top-k for CC; 0 uses --namespace-attention-topk.")
    p.add_argument(
        "--cross-namespace-refinement-repeats",
        type=int,
        default=0,
        help=(
            "Append this many lightweight cross-namespace semantic refinement blocks after the "
            "standard BP/MF/CC branch representations. 0 keeps the original model unchanged."
        ),
    )
    p.add_argument(
        "--cross-namespace-refinement-residual-init",
        type=float,
        default=0.1,
        help="Initial residual scale for each cross-namespace refinement block.",
    )
    p.add_argument(
        "--share-cross-namespace-refinement-weights",
        action="store_true",
        help=(
            "Reuse one cross-namespace refinement block for every repeat instead of creating "
            "one independently parameterized block per repeat."
        ),
    )
    p.add_argument(
        "--detach-cross-namespace-sources",
        action="store_true",
        help=(
            "Stop target-branch losses from backpropagating through source branch representations "
            "inside cross-namespace refinement. Cross-namespace information is still used, but the "
            "source branch representation is treated as read-only for that target branch."
        ),
    )
    p.add_argument(
        "--use-namespace-private-adapter",
        action="store_true",
        help=(
            "Add a lightweight residual adapter after each namespace branch fusion. This keeps "
            "BP/MF/CC jointly trained while giving each branch a private correction path before "
            "prediction and cross-namespace refinement."
        ),
    )
    p.add_argument(
        "--namespace-private-adapter-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension of each namespace private adapter.",
    )
    p.add_argument(
        "--namespace-private-adapter-residual-init",
        type=float,
        default=0.1,
        help="Default initial residual scale for namespace private adapters.",
    )
    p.add_argument(
        "--bp-namespace-private-adapter-residual-init",
        type=float,
        default=None,
        help="Override initial private-adapter residual scale for BP.",
    )
    p.add_argument(
        "--mf-namespace-private-adapter-residual-init",
        type=float,
        default=None,
        help="Override initial private-adapter residual scale for MF.",
    )
    p.add_argument(
        "--cc-namespace-private-adapter-residual-init",
        type=float,
        default=None,
        help="Override initial private-adapter residual scale for CC.",
    )
    p.add_argument(
        "--use-go-context-residual-gate",
        action="store_true",
        help="Apply GO context modulation as a learnable residual update instead of full FiLM replacement.",
    )
    p.add_argument(
        "--go-context-residual-init",
        type=float,
        default=0.1,
        help="Initial residual scale for GO context modulation when --use-go-context-residual-gate is enabled.",
    )
    p.add_argument(
        "--ablate-go-semantic-modulation",
        action="store_true",
        help=(
            "Paper ablation 'w/o GO-semantic modulation': retain the retrieved namespace GO context "
            "but replace feature-wise scale-and-shift modulation with direct protein-context concatenation."
        ),
    )
    p.add_argument(
        "--use-go-context-concat",
        dest="ablate_go_semantic_modulation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ablate-protein-ple",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ablate-modality-specific-skip-paths",
        action="store_true",
        help=(
            "Paper ablation 'w/o modality-specific skip paths': remove the separately projected "
            "sequence and structure skip representations from namespace routing, together with "
            "their associated protein-routing paths."
        ),
    )
    p.add_argument(
        "--ablate-modality-skip-paths",
        dest="ablate_modality_specific_skip_paths",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ablate-hmr",
        action="store_true",
        help=(
            "Paper ablation 'w/o HMR': replace the shared protein expert and hierarchical modality "
            "routing with the fixed fusion z=0.5*z_seq+0.5*z_str while retaining downstream paths."
        ),
    )
    p.add_argument(
        "--ablate-adaptive-protein-fusion",
        dest="ablate_hmr",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--ablate-go-context", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--ablate-go-modulation", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--ablate-go-semantic-branch",
        action="store_true",
        help=(
            "Paper ablation 'w/o GO-semantic branch': remove protein-conditioned GO-context "
            "retrieval and the GO-modulated branch pathway while retaining GO embeddings in "
            "the final similarity predictor."
        ),
    )
    p.add_argument(
        "--ablate-go-semantics",
        dest="ablate_go_semantic_branch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ablate-go-semantics-mlp-head",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ablate-go-semantics-mlp-hidden-dim",
        type=int,
        default=256,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ablate-go-semantics-skip-gate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--use-shared-go-branch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ablate-namespace-specific-pathways",
        action="store_true",
        help=(
            "Paper ablation 'w/o namespace-specific pathways': replace the separate BP, MF, and CC "
            "GO retrievers, GO-modulated experts, and GO-similarity predictors with shared counterparts."
        ),
    )
    p.add_argument(
        "--use-shared-go-branch-go-predictor",
        dest="ablate_namespace_specific_pathways",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--keep-multibranch-loss-for-shared-go-ablation",
        action="store_true",
        help=(
            "Keep the standard frequency-aware multi-branch loss and validation weights when a shared-GO "
            "ablation is enabled, isolating the architecture change in matched component studies."
        ),
    )
    p.add_argument(
        "--ablate-loss-function",
        action="store_true",
        help="Use a plain BCE-style loss by disabling focal, frequency weights, dynamic branch balancing, pos_weight, and gate entropy.",
    )

    p.add_argument("--disable-asym-focal", action="store_true")
    p.add_argument("--asym-focal-gamma-pos", type=float, default=0.0)
    p.add_argument("--asym-focal-gamma-neg", type=float, default=2.0)
    p.add_argument("--asym-focal-clip", type=float, default=0.05)
    p.add_argument("--gate-entropy-weight", type=float, default=0.002)
    p.add_argument("--term-prior-smoothing", type=float, default=1.0)
    p.add_argument("--freeze-term-bias", action="store_true")
    p.add_argument("--use-pos-weight", action="store_true", help="Also apply BCE pos_weight inside the new loss")
    p.add_argument(
        "--pos-weight-namespaces",
        type=str,
        default="bp,mf,cc",
        help="Comma-separated namespaces that use pos_weight when --use-pos-weight is enabled",
    )
    p.add_argument("--bp-pos-weight-cap", type=float, default=None, help="Optional maximum BP pos_weight")
    p.add_argument("--mf-pos-weight-cap", type=float, default=None, help="Optional maximum MF pos_weight")
    p.add_argument("--cc-pos-weight-cap", type=float, default=None, help="Optional maximum CC pos_weight")
    p.add_argument("--disable-frequency-weights", action="store_true", help="Disable effective-number term weighting")
    p.add_argument("--freq-loss-beta", type=float, default=0.999, help="Effective-number beta for term weights")
    p.add_argument("--freq-loss-weight-min", type=float, default=0.5, help="Minimum clipped term weight")
    p.add_argument("--freq-loss-weight-max", type=float, default=8.0, help="Maximum clipped term weight")
    p.add_argument("--bp-freq-loss-weight-max", type=float, default=None, help="Override maximum clipped BP term weight")
    p.add_argument("--mf-freq-loss-weight-max", type=float, default=None, help="Override maximum clipped MF term weight")
    p.add_argument("--cc-freq-loss-weight-max", type=float, default=None, help="Override maximum clipped CC term weight")
    p.add_argument("--freq-loss-groups", type=int, default=3, help="Frequency groups per branch; 3 means rare/mid/frequent")
    p.add_argument(
        "--namespace-size-loss-alpha",
        type=float,
        default=0.0,
        help=(
            "Scale each branch loss by (num_namespace_terms / reference_terms) ** alpha, "
            "then normalize active branch scales to mean 1. Use 0 to disable."
        ),
    )
    p.add_argument(
        "--namespace-size-loss-reference",
        type=str,
        default="mean",
        choices=("mean", "min", "max", "bp", "mf", "cc"),
        help="Reference term count used by --namespace-size-loss-alpha.",
    )
    p.add_argument("--disable-dynamic-branch-weight", action="store_true", help="Disable EMA dynamic branch balancing")
    p.add_argument("--branch-balance-momentum", type=float, default=0.98, help="EMA momentum for dynamic branch balancing")
    p.add_argument("--branch-balance-temperature", type=float, default=1.0, help="Softmax temperature multiplier for branch weights")

    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision on CUDA")
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16"),
        help="AMP autocast dtype when AMP is enabled",
    )
    p.add_argument(
        "--metrics-backend",
        type=str,
        default="fast",
        choices=("fast",),
        help="Built-in protein-centric evaluation backend.",
    )
    p.add_argument(
        "--go-space",
        type=str,
        default="filtered",
        choices=("full", "filtered"),
        help="Prediction GO space. 'full' keeps all terms in go-embedding file; "
        "'filtered' keeps only GO terms observed in the input annotation file.",
    )
    p.add_argument(
        "--no-eval-propagation",
        action="store_true",
        help="Disable GO ancestor propagation when computing validation/test metrics and PR curves.",
    )
    p.add_argument(
        "--save-epoch-start",
        type=int,
        default=0,
        help="Save per-epoch checkpoint/validation metrics starting at this epoch; 0 disables range saving.",
    )
    p.add_argument(
        "--save-epoch-end",
        type=int,
        default=0,
        help="Save per-epoch checkpoint/validation metrics up to this epoch, inclusive.",
    )
    p.add_argument(
        "--save-epoch-interval",
        type=int,
        default=1,
        help="Save one epoch snapshot every N epochs within the configured save range.",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def move_go_bundle_to_device(bundle: GOEncodingBundle, device: torch.device) -> GOEncodingBundle:
    for _, enc in bundle.encodings.items():
        enc.embeddings = enc.embeddings.to(device)
        enc.edge_index = enc.edge_index.to(device)
        enc.edge_type = enc.edge_type.to(device)
    return bundle


def collect_go_terms_from_records(
    records: list[ProteinAnnotationRecord],
) -> dict[str, set[str]]:
    kept = {"bp": set(), "mf": set(), "cc": set()}
    for rec in records:
        for ns in ("bp", "mf", "cc"):
            kept[ns].update(rec.go_by_namespace.get(ns, []))
    return kept


def prune_go_bundle_to_terms(
    bundle: GOEncodingBundle,
    kept_terms: dict[str, set[str]],
) -> GOEncodingBundle:
    from trigo.go_term import GONamespaceEncoding

    out = {}
    for ns in ("bp", "mf", "cc"):
        enc = bundle.encodings[ns]
        keep_set = kept_terms.get(ns, set())
        keep_ids = [go_id for go_id in enc.term_ids if go_id in keep_set]
        if not keep_ids:
            raise RuntimeError(
                f"No GO terms left in namespace '{ns}' after applying --go-space filtered. "
                "Check your annotation file and filtering output."
            )

        old_idx = {go_id: i for i, go_id in enumerate(enc.term_ids)}
        new_idx = {go_id: i for i, go_id in enumerate(keep_ids)}
        keep_tensor = torch.tensor([old_idx[go] for go in keep_ids], dtype=torch.long)
        embeddings = enc.embeddings.index_select(0, keep_tensor)

        edge_index = enc.edge_index.detach().cpu()
        edge_type = enc.edge_type.detach().cpu()
        if edge_index.numel() > 0:
            remapped_edges = []
            remapped_types = []
            for edge_i, (child_old, parent_old) in enumerate(edge_index.t().tolist()):
                child_go = enc.term_ids[int(child_old)]
                parent_go = enc.term_ids[int(parent_old)]
                if child_go in new_idx and parent_go in new_idx:
                    remapped_edges.append([new_idx[child_go], new_idx[parent_go]])
                    remapped_types.append(int(edge_type[edge_i].item()))
            if remapped_edges:
                new_edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
                new_edge_type = torch.tensor(remapped_types, dtype=torch.long)
            else:
                new_edge_index = torch.empty((2, 0), dtype=torch.long)
                new_edge_type = torch.empty((0,), dtype=torch.long)
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long)
            new_edge_type = torch.empty((0,), dtype=torch.long)

        out[ns] = GONamespaceEncoding(
            namespace=ns,
            term_ids=keep_ids,
            embeddings=embeddings,
            edge_index=new_edge_index,
            edge_type=new_edge_type,
            relation_vocab=enc.relation_vocab,
        )

    return GOEncodingBundle(encodings=out)


def replace_go_embeddings_with_one_hot(bundle: GOEncodingBundle) -> GOEncodingBundle:
    from trigo.go_term import GONamespaceEncoding

    out = {}
    for ns, enc in bundle.encodings.items():
        num_terms = len(enc.term_ids)
        out[ns] = GONamespaceEncoding(
            namespace=enc.namespace,
            term_ids=list(enc.term_ids),
            embeddings=torch.eye(num_terms, dtype=enc.embeddings.dtype),
            edge_index=enc.edge_index,
            edge_type=enc.edge_type,
            relation_vocab=enc.relation_vocab,
        )
    return GOEncodingBundle(encodings=out)


def parse_branch_loss_weights(raw: str) -> dict[str, float]:
    if raw is None or raw.strip() == "":
        return {"bp": 1.0, "mf": 1.0, "cc": 1.0}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--branch-loss-weights must be valid JSON, e.g. "
            '\'{"bp":1.0,"mf":1.0,"cc":1.0}\''
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("--branch-loss-weights must be a JSON object.")
    out = {"bp": 1.0, "mf": 1.0, "cc": 1.0}
    for ns in ("bp", "mf", "cc"):
        if ns in parsed:
            out[ns] = float(parsed[ns])
    return out


def parse_val_score_weights(raw: str) -> dict[str, float]:
    if raw is None or raw.strip() == "":
        return {"bp": 0.5, "mf": 0.25, "cc": 0.25}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--val-score-weights must be a JSON object.")
    out = {"bp": 0.5, "mf": 0.25, "cc": 0.25}
    for ns in ("bp", "mf", "cc"):
        if ns in parsed:
            out[ns] = float(parsed[ns])
    return out


def parse_active_namespaces(raw: str) -> tuple[str, ...]:
    names = tuple(ns.strip().lower() for ns in raw.split(",") if ns.strip())
    valid = {"bp", "mf", "cc"}
    invalid = [ns for ns in names if ns not in valid]
    if invalid:
        raise ValueError(f"--active-namespaces contains unknown values: {invalid}. Use bp,mf,cc.")
    deduped = tuple(dict.fromkeys(names))
    if not deduped:
        raise ValueError("--active-namespaces must contain at least one of bp,mf,cc.")
    return deduped


def weighted_val_score(
    val_metrics: dict[str, dict[str, float]],
    weights: dict[str, float],
    active_namespaces: tuple[str, ...] = ("bp", "mf", "cc"),
) -> float:
    numer = 0.0
    denom = 0.0
    available = [ns for ns in active_namespaces if ns in val_metrics]
    for ns in available:
        w = float(weights.get(ns, 0.0))
        numer += w * float(val_metrics[ns]["fmax"])
        denom += w
    if denom <= 0:
        return float(np.mean([val_metrics[ns]["fmax"] for ns in available]))
    return numer / denom


def should_save_epoch(epoch: int, start: int, end: int, interval: int = 1) -> bool:
    if start <= 0 or end <= 0:
        return False
    if end < start:
        return False
    return start <= epoch <= end and (epoch - start) % interval == 0


def compute_term_prior_logits_from_label_indices(
    label_indices_list: list[dict[str, list[int]]],
    num_terms: dict[str, int],
    smoothing: float = 1.0,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    total = max(len(label_indices_list), 1)
    for ns in ("bp", "mf", "cc"):
        counts = torch.zeros(num_terms[ns], dtype=torch.float32)
        for item in label_indices_list:
            idxs = item.get(ns, [])
            if idxs:
                counts[idxs] += 1.0
        p = (counts + smoothing) / (total + 2.0 * smoothing)
        p = p.clamp(min=1e-5, max=1.0 - 1e-5)
        out[ns] = torch.log(p / (1.0 - p)).clamp(min=-8.0, max=8.0)
    return out


def split_namespace_param_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {"base": [], "bp": [], "mf": [], "cc": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        namespace = next((ns for ns in ("bp", "mf", "cc") if f".{ns}." in name), "base")
        groups[namespace].append(p)
    return groups


def freeze_except_namespace(model: torch.nn.Module, namespace: str) -> None:
    token = f".{namespace}."
    for name, p in model.named_parameters():
        p.requires_grad = token in name


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for k, v in model.state_dict().items():
            if isinstance(v, UninitializedParameter):
                continue
            if torch.is_tensor(v) and torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for k, v in state.items():
            if isinstance(v, UninitializedParameter):
                continue
            if (not torch.is_tensor(v)) or (not torch.is_floating_point(v)):
                continue
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for k, v in state.items():
            if isinstance(v, UninitializedParameter):
                continue
            if k in self.shadow and torch.is_tensor(v) and torch.is_floating_point(v):
                self.backup[k] = v.detach().clone()
                v.copy_(self.shadow[k].to(v.device, non_blocking=True))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if not self.backup:
            return
        state = model.state_dict()
        for k, v in state.items():
            if isinstance(v, UninitializedParameter):
                continue
            if k in self.backup and torch.is_tensor(v) and torch.is_floating_point(v):
                v.copy_(self.backup[k].to(v.device, non_blocking=True))
        self.backup = {}


def split_records_by_annotation(
    records: list[ProteinAnnotationRecord],
) -> tuple[list[ProteinAnnotationRecord], list[ProteinAnnotationRecord], list[ProteinAnnotationRecord], bool]:
    train_records = [r for r in records if (r.split or "").strip().lower() == "train"]
    val_records = [r for r in records if (r.split or "").strip().lower() in {"val", "valid", "validation"}]
    test_records = [r for r in records if (r.split or "").strip().lower() == "test"]
    ok = len(train_records) > 0 and len(val_records) > 0 and len(test_records) > 0
    return train_records, val_records, test_records, ok


def validate_required_annotation_split(records: list[ProteinAnnotationRecord]) -> None:
    """Require a complete, explicit train/validation/test assignment."""
    aliases = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
    counts = {"train": 0, "val": 0, "test": 0}
    invalid_entries = []
    for record in records:
        raw_split = (record.split or "").strip().lower()
        normalized = aliases.get(raw_split)
        if normalized is None:
            invalid_entries.append(record.entry)
        else:
            counts[normalized] += 1
    if invalid_entries:
        preview = ", ".join(invalid_entries[:10])
        raise ValueError(
            "--require-split found records with missing or invalid Split values "
            f"({len(invalid_entries)} records; first entries: {preview})."
        )
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"--require-split requires non-empty partitions; missing: {missing}.")


def save_aupr_curve_figure(
    y_true_by_ns: dict[str, np.ndarray],
    y_pred_by_ns: dict[str, np.ndarray],
    go_bundle: GOEncodingBundle,
    save_path: Path,
    propagate: bool = True,
) -> None:
    plt.figure(figsize=(8, 6))
    ns_colors = {"bp": "#1f77b4", "mf": "#ff7f0e", "cc": "#2ca02c"}
    has_curve = False

    for ns in ("bp", "mf", "cc"):
        y_true = y_true_by_ns.get(ns, np.empty((0, 0)))
        y_pred = y_pred_by_ns.get(ns, np.empty((0, 0)))
        if propagate:
            y_true, y_pred = maybe_propagate_branch_arrays(
                y_true=y_true,
                y_pred=y_pred,
                namespace=ns,
                go_bundle=go_bundle,
                drop_root=True,
                show_progress=False,
            )

        if y_true.size == 0 or y_pred.size == 0:
            continue

        # Keep plotting/eval figure logic consistent with branch metric logic:
        # only proteins with at least one known positive label in this branch.
        positive_rows = y_true.sum(axis=1) > 0
        if not np.any(positive_rows):
            continue
        y_true = y_true[positive_rows]
        y_pred = y_pred[positive_rows]

        yt = y_true.reshape(-1)
        yp = y_pred.reshape(-1)
        if yt.sum() <= 0:
            continue

        precision, recall, _ = precision_recall_curve(yt, yp)
        ap = float(average_precision_score(yt, yp))

        plt.plot(
            recall,
            precision,
            color=ns_colors[ns],
            lw=2,
            label=f"{ns.upper()} AUPR={ap:.4f}",
        )
        has_curve = True

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test Micro-AUPR Curves by GO Branch")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.01)
    plt.grid(alpha=0.25)
    if has_curve:
        plt.legend(loc="lower left")
    else:
        plt.text(0.5, 0.5, "No positive labels available to draw PR curves.", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def dense_labels_from_label_indices(
    label_indices_list: list[dict[str, list[int]]],
    num_terms: dict[str, int],
) -> dict[str, np.ndarray]:
    labels = {
        ns: np.zeros((len(label_indices_list), num_terms[ns]), dtype=np.float32)
        for ns in ("bp", "mf", "cc")
    }
    for row_idx, label_indices in enumerate(label_indices_list):
        for ns in ("bp", "mf", "cc"):
            indices = label_indices.get(ns, [])
            if indices:
                labels[ns][row_idx, indices] = 1.0
    return labels


def prediction_payload(test_out: dict[str, object]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "proteins": np.asarray(test_out["proteins"], dtype=object),
    }
    for ns in ("bp", "mf", "cc"):
        payload[f"y_true_{ns}"] = test_out["y_true"][ns].astype(np.float32, copy=False)
        payload[f"y_pred_{ns}"] = test_out["y_pred"][ns].astype(np.float32, copy=False)
    return payload


def cross_namespace_refinement_residual_scales(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    scales = {}
    for key, value in state_dict.items():
        if "refinement_residual_logit" not in key:
            continue
        scales[key] = float(torch.sigmoid(value.detach().float()).cpu().item())
    return scales


def save_best_model_test_metrics(
    output_dir: Path,
    test_out: dict[str, object],
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    checkpoint_epoch: int,
    checkpoint_val_score: float,
) -> None:
    prediction_path = output_dir / "best_model_predictions.npz"
    np.savez_compressed(prediction_path, **prediction_payload(test_out))

    payload = {
        "checkpoint": "best_model",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_val_score": float(checkpoint_val_score),
        "test_loss": test_out["loss"],
        "test_metrics": test_out["metrics"],
        "normalized_smin_protocol": SMIN_PROTOCOL,
        "prediction_path": str(prediction_path),
        "cross_namespace_refinement_residual_scales": (
            cross_namespace_refinement_residual_scales(checkpoint["model_state_dict"])
        ),
    }
    metrics_path = output_dir / "best_model_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved best model test metrics with normalized Smin -> {metrics_path}")
    print(f"saved best model predictions -> {prediction_path}")


def save_test_outputs(
    output_dir: Path,
    test_out: dict[str, object],
    go_bundle: GOEncodingBundle,
    propagate: bool,
    best_epoch: int | None = None,
    best_val_score: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload: dict[str, object]
    metrics = test_out["metrics"]
    if best_epoch is None or best_val_score is None:
        metrics_payload = metrics
    else:
        metrics_payload = {
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_val_score),
            "metrics": metrics,
            "normalized_smin_protocol": SMIN_PROTOCOL,
        }

    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    aupr_fig_path = output_dir / "test_aupr_curves.png"
    save_aupr_curve_figure(
        y_true_by_ns=test_out["y_true"],
        y_pred_by_ns=test_out["y_pred"],
        go_bundle=go_bundle,
        save_path=aupr_fig_path,
        propagate=propagate,
    )
    print(f"saved best test metrics -> {metrics_path}")
    print(f"saved best AUPR figure -> {aupr_fig_path}")


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    extra: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "args": {k: getattr(args, k) for k in sorted(vars(args).keys())},
        "run_command": shlex.join([sys.executable, *sys.argv]),
        "argv": [sys.executable, *sys.argv],
    }
    if extra:
        payload.update(extra)
    config_path = output_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved run config -> {config_path}")


def build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> None:
    args = parse_args()
    active_namespaces = parse_active_namespaces(args.active_namespaces)
    args.active_namespaces = ",".join(active_namespaces)
    if args.dropout is not None and args.protein_dropout is not None and args.dropout != args.protein_dropout:
        raise ValueError("--dropout and --protein-dropout must match when both are provided.")
    args.dropout = 0.10 if args.dropout is None else float(args.dropout)
    args.protein_dropout = args.dropout if args.protein_dropout is None else float(args.protein_dropout)
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0, 1).")
    if not (0.0 <= args.protein_dropout < 1.0):
        raise ValueError("--protein-dropout must be in [0, 1).")
    if not (0.0 < args.lr_reduce_factor < 1.0):
        raise ValueError("--lr-reduce-factor must be in (0, 1).")
    for ns in ("bp", "mf", "cc"):
        if getattr(args, f"{ns}_lr_multiplier") <= 0:
            raise ValueError(f"--{ns}-lr-multiplier must be > 0.")
    if args.lr_patience < 1:
        raise ValueError("--lr-patience must be >= 1.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be >= 0.")
    if args.early_stopping_min_epochs < 0:
        raise ValueError("--early-stopping-min-epochs must be >= 0.")
    if args.eval_interval < 1:
        raise ValueError("--eval-interval must be >= 1.")
    if (args.save_epoch_start > 0 or args.save_epoch_end > 0) and args.save_epoch_end < args.save_epoch_start:
        raise ValueError("--save-epoch-end must be >= --save-epoch-start when epoch range saving is enabled.")
    if args.save_epoch_interval < 1:
        raise ValueError("--save-epoch-interval must be >= 1.")
    if args.namespace_attention_topk < 0:
        raise ValueError("--namespace-attention-topk must be >= 0.")
    if args.use_shared_go_branch and args.ablate_namespace_specific_pathways:
        raise ValueError("The paper namespace-pathway ablation cannot be combined with the legacy shared-MLP variant.")
    if args.ablate_go_semantic_modulation and args.ablate_go_modulation:
        raise ValueError("--ablate-go-semantic-modulation cannot be combined with the legacy modulation-disable flag.")
    if args.ablate_go_semantic_modulation and args.ablate_go_semantic_branch:
        raise ValueError("--ablate-go-semantic-modulation requires the GO-semantic branch.")
    if args.ablate_go_semantic_modulation and args.ablate_go_context:
        raise ValueError("--ablate-go-semantic-modulation requires retrieved GO contexts.")
    if args.ablate_go_semantic_modulation and (args.use_shared_go_branch or args.ablate_namespace_specific_pathways):
        raise ValueError("--ablate-go-semantic-modulation is defined for the standard namespace-specific pathways.")
    if (args.use_shared_go_branch or args.ablate_namespace_specific_pathways) and args.ablate_go_semantic_branch:
        raise ValueError("GO-semantic-branch and namespace-specific-pathway ablations must be run separately.")
    if len(active_namespaces) != 3 and (args.use_shared_go_branch or args.ablate_namespace_specific_pathways):
        raise ValueError("--active-namespaces single-branch training is not compatible with shared-branch ablations.")
    if args.keep_multibranch_loss_for_shared_go_ablation and not (
        args.use_shared_go_branch or args.ablate_namespace_specific_pathways
    ):
        raise ValueError(
            "--keep-multibranch-loss-for-shared-go-ablation requires a shared-GO ablation."
        )
    if args.ablate_go_semantics_skip_gate and not args.ablate_go_semantic_branch:
        raise ValueError("The legacy skip-gate variant requires --ablate-go-semantic-branch.")
    if args.ablate_modality_specific_skip_paths and args.ablate_go_semantics_skip_gate:
        raise ValueError(
            "--ablate-modality-specific-skip-paths is incompatible with the legacy skip-gate variant, "
            "which explicitly requires the downstream sequence/structure shared experts."
        )
    if args.ablate_hmr and args.ablate_protein_ple:
        raise ValueError(
            "--ablate-hmr and the legacy protein-encoder ablation define different "
            "protein-encoder ablations; enable only one."
        )
    if args.ablate_hmr and args.ablate_modality_specific_skip_paths:
        raise ValueError(
            "--ablate-hmr retains the modality-specific skip paths, whereas "
            "--ablate-modality-specific-skip-paths removes them; enable only one."
        )
    if args.ablate_hmr and args.clean_expert_gate:
        raise ValueError(
            "--ablate-hmr removes protein gates and is incompatible with "
            "--clean-expert-gate."
        )
    if args.protein_expert_hidden_dim <= 0 or args.protein_expert_output_dim <= 0:
        raise ValueError("--protein-expert-hidden-dim and --protein-expert-output-dim must be > 0.")
    model_width_args = (
        "model_expert_dim",
        "shared_expert_hidden_dim",
        "namespace_expert_hidden_dim",
        "namespace_context_hidden_dim",
        "gate_hidden_dim",
    )
    if any(getattr(args, name) <= 0 for name in model_width_args):
        raise ValueError("All multi-branch model width arguments must be > 0.")
    if args.strong_ple_gcn_layers < 1:
        raise ValueError("--strong-ple-gcn-layers must be >= 1.")
    if not (0.0 <= args.freq_loss_beta < 1.0):
        raise ValueError("--freq-loss-beta must be in [0, 1).")
    if args.freq_loss_weight_min <= 0 or args.freq_loss_weight_max < args.freq_loss_weight_min:
        raise ValueError("--freq-loss-weight-min/max must satisfy 0 < min <= max.")
    for ns in ("bp", "mf", "cc"):
        namespace_weight_max = getattr(args, f"{ns}_freq_loss_weight_max")
        if namespace_weight_max is not None and namespace_weight_max < args.freq_loss_weight_min:
            raise ValueError(f"--{ns}-freq-loss-weight-max must be >= --freq-loss-weight-min.")
    pos_weight_namespaces = {ns.strip() for ns in args.pos_weight_namespaces.split(",") if ns.strip()}
    invalid_pos_weight_namespaces = pos_weight_namespaces - {"bp", "mf", "cc"}
    if invalid_pos_weight_namespaces:
        raise ValueError(f"Unknown --pos-weight-namespaces values: {sorted(invalid_pos_weight_namespaces)}")
    for ns in ("bp", "mf", "cc"):
        namespace_pos_weight_cap = getattr(args, f"{ns}_pos_weight_cap")
        if namespace_pos_weight_cap is not None and namespace_pos_weight_cap <= 0:
            raise ValueError(f"--{ns}-pos-weight-cap must be > 0.")
    if args.freq_loss_groups < 1:
        raise ValueError("--freq-loss-groups must be >= 1.")
    if args.namespace_size_loss_alpha < 0:
        raise ValueError("--namespace-size-loss-alpha must be >= 0.")
    if not (0.0 <= args.branch_balance_momentum < 1.0):
        raise ValueError("--branch-balance-momentum must be in [0, 1).")
    if args.branch_balance_temperature <= 0:
        raise ValueError("--branch-balance-temperature must be > 0.")
    if args.ema_decay <= 0 or args.ema_decay >= 1:
        raise ValueError("--ema-decay must be in (0, 1).")
    set_seed(args.seed)
    print(f"active namespaces: {','.join(active_namespaces)}")
    device = pick_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    go_to_namespace, _ = load_go_namespace_and_name(args.obo_path)
    feature_cache_path = Path(args.feature_cache_dir)
    use_slim_feature_cache = all(
        (feature_cache_path / filename).exists()
        for filename in ("metadata.pt", "esm.f16.bin", "coordinates.f32.bin")
    )
    records = load_annotation_records(
        args.xlsx_path,
        args.pdb_dir,
        go_to_namespace,
        require_pdb=not use_slim_feature_cache,
    )
    if use_slim_feature_cache:
        print(f"feature cache mode: compact memory-mapped store ({feature_cache_path})")
    if args.max_records > 0:
        records = records[: args.max_records]

    go_bundle = GOEncodingBundle.load(args.go_embedding_path, map_location="cpu")
    full_sizes = {ns: len(go_bundle.encodings[ns].term_ids) for ns in ("bp", "mf", "cc")}
    if args.go_space == "filtered":
        kept_terms = collect_go_terms_from_records(records)
        go_bundle = prune_go_bundle_to_terms(go_bundle, kept_terms)
        filtered_sizes = {ns: len(go_bundle.encodings[ns].term_ids) for ns in ("bp", "mf", "cc")}
        print(
            "GO space mode=filtered:"
            f" bp {full_sizes['bp']}->{filtered_sizes['bp']},"
            f" mf {full_sizes['mf']}->{filtered_sizes['mf']},"
            f" cc {full_sizes['cc']}->{filtered_sizes['cc']}"
        )
    else:
        print(
            "GO space mode=full:"
            f" bp={full_sizes['bp']}, mf={full_sizes['mf']}, cc={full_sizes['cc']}"
        )
    if args.use_one_hot_go_embeddings:
        go_bundle = replace_go_embeddings_with_one_hot(go_bundle)
        one_hot_sizes = {ns: go_bundle.encodings[ns].embeddings.size(1) for ns in ("bp", "mf", "cc")}
        print(
            "GO embedding mode=one-hot:"
            f" bp_dim={one_hot_sizes['bp']}, mf_dim={one_hot_sizes['mf']}, cc_dim={one_hot_sizes['cc']}"
        )
    go_bundle = move_go_bundle_to_device(go_bundle, device)
    train_records, val_records, test_records, use_annotation_split = split_records_by_annotation(records)
    if args.require_split:
        validate_required_annotation_split(records)
        if not use_annotation_split:
            raise RuntimeError("Explicit Split validation passed but train/val/test extraction failed.")
    if use_annotation_split:
        print("records split mode: annotation column 'Split'")
    else:
        train_records, val_records, test_records = random_split_records(
            records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(
            "records split mode: random fallback "
            f"(train_ratio={args.train_ratio}, val_ratio={args.val_ratio})"
        )
    n_all = max(len(records), 1)
    print(
        "records split:"
        f" train={len(train_records)} ({len(train_records) / n_all:.2%})"
        f" val={len(val_records)} ({len(val_records) / n_all:.2%})"
        f" test={len(test_records)} ({len(test_records) / n_all:.2%})"
    )

    split_payload = {
        "train_entries": [r.entry for r in train_records],
        "val_entries": [r.entry for r in val_records],
        "test_entries": [r.entry for r in test_records],
    }
    with open(Path(args.output_dir) / "data_split.json", "w", encoding="utf-8") as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)

    # No pre-propagation before training.
    ds_cfg = ProteinGODatasetConfig(
        contact_distance_threshold=args.contact_threshold,
        max_residues=args.max_residues,
        propagate_labels_to_ancestors=False,
    )
    train_ds = ProteinGODataset(train_records, args.feature_cache_dir, go_bundle, cfg=ds_cfg)
    val_ds = ProteinGODataset(val_records, args.feature_cache_dir, go_bundle, cfg=ds_cfg)
    test_ds = ProteinGODataset(test_records, args.feature_cache_dir, go_bundle, cfg=ds_cfg)
    print(f"dataset size after filtering: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    collate_fn = ProteinGOBatchCollator(go_bundle)
    loader_common_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_common_kwargs["persistent_workers"] = True
        loader_common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_common_kwargs,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Run prepare_protein_features.py first.")

    first_item = train_ds[0]
    seq_input_dim = int(first_item["sequence_embeddings"].size(1))
    protein_cfg = ProteinPLEConfig(
        seq_input_dim=seq_input_dim,
        expert_hidden_dim=args.protein_expert_hidden_dim,
        expert_output_dim=args.protein_expert_output_dim,
        dropout=args.protein_dropout,
        max_residues=args.max_residues,
        ablate_protein_ple=args.ablate_protein_ple,
        ablate_modality_skip_paths=args.ablate_modality_specific_skip_paths,
        ablate_adaptive_protein_fusion=args.ablate_hmr,
        use_strong_protein_ple=args.use_strong_protein_ple,
        clean_expert_gate=args.clean_expert_gate,
        strong_gcn_layers=args.strong_ple_gcn_layers,
        normalize_sequence_input=args.normalize_sequence_input,
    )
    num_terms = {ns: len(go_bundle.encodings[ns].term_ids) for ns in ("bp", "mf", "cc")}
    model_cfg_kwargs = {
        "protein_dim": protein_cfg.expert_output_dim,
        "expert_dim": args.model_expert_dim,
        "shared_expert_hidden_dim": args.shared_expert_hidden_dim,
        "namespace_expert_hidden_dim": args.namespace_expert_hidden_dim,
        "gate_hidden_dim": args.gate_hidden_dim,
        "namespace_context_hidden_dim": args.namespace_context_hidden_dim,
        "namespace_attention_topk": args.namespace_attention_topk,
        "bp_namespace_attention_topk": args.bp_namespace_attention_topk,
        "mf_namespace_attention_topk": args.mf_namespace_attention_topk,
        "cc_namespace_attention_topk": args.cc_namespace_attention_topk,
        "dropout": args.dropout,
        "predictor_use_term_bias": True,
        "ablate_protein_ple": args.ablate_protein_ple,
        "ablate_modality_skip_paths": args.ablate_modality_specific_skip_paths,
        "ablate_go_context": args.ablate_go_context,
        "ablate_go_modulation": args.ablate_go_modulation,
        "ablate_go_semantics": args.ablate_go_semantic_branch,
        "ablate_go_semantics_mlp_head": args.ablate_go_semantics_mlp_head,
        "ablate_go_semantics_mlp_hidden_dim": args.ablate_go_semantics_mlp_hidden_dim,
        "ablate_go_semantics_skip_gate": args.ablate_go_semantics_skip_gate,
        "use_shared_go_branch": args.use_shared_go_branch,
        "use_shared_go_branch_go_predictor": args.ablate_namespace_specific_pathways,
        "use_go_context_residual_gate": args.use_go_context_residual_gate,
        "go_context_residual_init": args.go_context_residual_init,
        "use_go_context_concat": args.ablate_go_semantic_modulation,
        "cross_namespace_refinement_repeats": args.cross_namespace_refinement_repeats,
        "cross_namespace_refinement_residual_init": args.cross_namespace_refinement_residual_init,
        "share_cross_namespace_refinement_weights": args.share_cross_namespace_refinement_weights,
        "detach_cross_namespace_sources": args.detach_cross_namespace_sources,
        "use_namespace_private_adapter": args.use_namespace_private_adapter,
        "namespace_private_adapter_hidden_dim": args.namespace_private_adapter_hidden_dim,
        "namespace_private_adapter_residual_init": args.namespace_private_adapter_residual_init,
        "bp_namespace_private_adapter_residual_init": args.bp_namespace_private_adapter_residual_init,
        "mf_namespace_private_adapter_residual_init": args.mf_namespace_private_adapter_residual_init,
        "cc_namespace_private_adapter_residual_init": args.cc_namespace_private_adapter_residual_init,
        "bp_num_terms": num_terms["bp"],
        "mf_num_terms": num_terms["mf"],
        "cc_num_terms": num_terms["cc"],
    }
    supported_model_cfg_fields = {field.name for field in fields(MultiBranchProteinGOConfig)}
    required_width_fields = {
        "expert_dim",
        "shared_expert_hidden_dim",
        "namespace_expert_hidden_dim",
        "namespace_context_hidden_dim",
        "gate_hidden_dim",
    }
    missing_width_fields = required_width_fields - supported_model_cfg_fields
    if missing_width_fields:
        raise RuntimeError(
            "MultiBranchProteinGOConfig is too old for whole-model width scaling; missing fields: "
            f"{sorted(missing_width_fields)}"
        )
    unsupported_fields = sorted(set(model_cfg_kwargs) - supported_model_cfg_fields)
    if unsupported_fields:
        print(f"model config compatibility: skipping unsupported fields {unsupported_fields}")
    model_cfg = MultiBranchProteinGOConfig(
        **{key: value for key, value in model_cfg_kwargs.items() if key in supported_model_cfg_fields}
    )
    model = MultiBranchProteinGOModel(model_cfg, protein_encoder_cfg=protein_cfg).to(device)
    model_for_ckpt = model

    pos_weight = compute_pos_weight_from_label_indices(
        (s["label_indices"] for s in train_ds.samples),
        num_terms=num_terms,
        smoothing=args.term_prior_smoothing,
    )
    for ns in ("bp", "mf", "cc"):
        if ns not in pos_weight_namespaces:
            pos_weight[ns] = torch.ones_like(pos_weight[ns])
        namespace_pos_weight_cap = getattr(args, f"{ns}_pos_weight_cap")
        if namespace_pos_weight_cap is not None:
            pos_weight[ns] = pos_weight[ns].clamp(max=float(namespace_pos_weight_cap))
    train_label_indices = [s["label_indices"] for s in train_ds.samples]
    term_weight, term_counts = compute_effective_number_weights_from_label_indices(
        train_label_indices,
        num_terms=num_terms,
        beta=args.freq_loss_beta,
        weight_min=args.freq_loss_weight_min,
        weight_max=args.freq_loss_weight_max,
        weight_max_by_namespace={
            ns: value
            for ns in ("bp", "mf", "cc")
            if (value := getattr(args, f"{ns}_freq_loss_weight_max")) is not None
        },
    )
    term_prior_logits = compute_term_prior_logits_from_label_indices(
        label_indices_list=train_label_indices,
        num_terms=num_terms,
        smoothing=args.term_prior_smoothing,
    )
    for ns in ("bp", "mf", "cc"):
        if ns not in model.namespace_predictors:
            continue
        predictor = model.namespace_predictors[ns]
        if hasattr(predictor, "set_term_bias_from_prior"):
            predictor.set_term_bias_from_prior(
                term_prior_logits[ns].to(device),
                trainable=not args.freeze_term_bias,
            )

    if args.init_checkpoint_path:
        init_checkpoint = torch.load(args.init_checkpoint_path, map_location=device, weights_only=False)
        model_for_ckpt.load_state_dict(init_checkpoint["model_state_dict"])
        print(f"initialized model weights from checkpoint: {args.init_checkpoint_path}")
    if args.train_only_namespace:
        freeze_except_namespace(model_for_ckpt, args.train_only_namespace)
        print(f"training only namespace-specific parameters: {args.train_only_namespace}")

    branch_weights = parse_branch_loss_weights(args.branch_loss_weights)
    if args.bp_loss_weight is not None:
        branch_weights["bp"] = float(args.bp_loss_weight)
    if args.mf_loss_weight is not None:
        branch_weights["mf"] = float(args.mf_loss_weight)
    if args.cc_loss_weight is not None:
        branch_weights["cc"] = float(args.cc_loss_weight)
    val_score_weights = parse_val_score_weights(args.val_score_weights)
    use_single_branch_plain_bce = (
        args.use_shared_go_branch or args.ablate_namespace_specific_pathways
    ) and not args.keep_multibranch_loss_for_shared_go_ablation
    if use_single_branch_plain_bce:
        val_score_weights = {"bp": 1.0, "mf": 1.0, "cc": 1.0}
    print("branch loss weights:", branch_weights)
    print("val score weights:", val_score_weights)
    print(
        "new loss:",
        f"single_branch_loss={use_single_branch_plain_bce}",
        f"single_branch_asymmetric_focal={use_single_branch_plain_bce and not args.disable_asym_focal}",
        f"frequency_weights={not (args.disable_frequency_weights or args.ablate_loss_function or use_single_branch_plain_bce)}",
        f"groups={args.freq_loss_groups}",
        f"beta={args.freq_loss_beta}",
        f"clip=[{args.freq_loss_weight_min}, {args.freq_loss_weight_max}]",
        f"namespace_weight_max={{'bp': {args.bp_freq_loss_weight_max}, 'mf': {args.mf_freq_loss_weight_max}, 'cc': {args.cc_freq_loss_weight_max}}}",
        f"namespace_size_loss_alpha={0.0 if use_single_branch_plain_bce else args.namespace_size_loss_alpha}",
        f"namespace_size_loss_reference={args.namespace_size_loss_reference}",
        f"dynamic_branch_weight={not (args.disable_dynamic_branch_weight or args.ablate_loss_function or use_single_branch_plain_bce)}",
        f"branch_balance_momentum={args.branch_balance_momentum}",
        f"branch_balance_temperature={args.branch_balance_temperature}",
        f"use_pos_weight={False if (args.ablate_loss_function or use_single_branch_plain_bce) else args.use_pos_weight}",
        f"pos_weight_namespaces={sorted(pos_weight_namespaces)}",
        f"pos_weight_caps={{'bp': {args.bp_pos_weight_cap}, 'mf': {args.mf_pos_weight_cap}, 'cc': {args.cc_pos_weight_cap}}}",
    )
    loss_cfg = MultiBranchLossConfig(
        branch_weights=branch_weights,
        use_asymmetric_focal=not (args.disable_asym_focal or args.ablate_loss_function),
        asym_gamma_pos=args.asym_focal_gamma_pos,
        asym_gamma_neg=args.asym_focal_gamma_neg,
        asym_clip=args.asym_focal_clip,
        gate_entropy_weight=0.0 if (args.ablate_loss_function or use_single_branch_plain_bce) else args.gate_entropy_weight,
        use_pos_weight=False if (args.ablate_loss_function or use_single_branch_plain_bce) else args.use_pos_weight,
        use_frequency_weights=not (args.disable_frequency_weights or args.ablate_loss_function or use_single_branch_plain_bce),
        frequency_group_count=args.freq_loss_groups,
        namespace_size_loss_alpha=0.0 if use_single_branch_plain_bce else args.namespace_size_loss_alpha,
        namespace_size_loss_reference=args.namespace_size_loss_reference,
        dynamic_branch_weight=not (args.disable_dynamic_branch_weight or args.ablate_loss_function or use_single_branch_plain_bce),
        branch_balance_momentum=args.branch_balance_momentum,
        branch_balance_temperature=args.branch_balance_temperature,
    )
    if use_single_branch_plain_bce:
        criterion = SingleBranchBCELoss(
            use_asymmetric_focal=not args.disable_asym_focal,
            asym_gamma_pos=args.asym_focal_gamma_pos,
            asym_gamma_neg=args.asym_focal_gamma_neg,
            asym_clip=args.asym_focal_clip,
        )
    else:
        criterion = MultiBranchBCELoss(
            pos_weight=pos_weight,
            term_weight=term_weight,
            term_counts=term_counts,
            config=loss_cfg,
        )

    namespace_param_groups = split_namespace_param_groups(model_for_ckpt)
    lr_multipliers = {
        "base": 1.0,
        "bp": float(args.bp_lr_multiplier),
        "mf": float(args.mf_lr_multiplier),
        "cc": float(args.cc_lr_multiplier),
    }
    param_groups = []
    for group_name in ("base", "bp", "mf", "cc"):
        params = namespace_param_groups[group_name]
        if not params:
            continue
        multiplier = lr_multipliers[group_name]
        param_groups.append(
            {
                "params": params,
                "lr": args.lr * multiplier,
                "min_lr": args.min_lr * max(multiplier, 1.0),
                "group_name": group_name,
            }
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    print(
        "optimizer groups:",
        "; ".join(
            f"{group_name}_params={len(namespace_param_groups[group_name])} "
            f"lr={args.lr * lr_multipliers[group_name]:.2e}"
            for group_name in ("base", "bp", "mf", "cc")
            if namespace_param_groups[group_name]
        ),
    )
    print(
        "LR scheduler: epoch-based plateau"
        f" (patience={args.lr_patience} epochs, factor={args.lr_reduce_factor}, min_lr={args.min_lr})"
    )
    if args.early_stopping_patience > 0:
        print(
            "early stopping:"
            f" patience={args.early_stopping_patience}"
            f" min_delta={args.early_stopping_min_delta}"
            f" min_epochs={args.early_stopping_min_epochs}"
        )
    else:
        print("early stopping: disabled")
    use_amp = (not args.no_amp) and device.type == "cuda"
    if use_amp:
        print(f"AMP enabled (dtype={args.amp_dtype})")
    else:
        print("AMP disabled")
    scaler = build_grad_scaler(enabled=use_amp)
    use_ema = (not args.no_ema)
    ema = ExponentialMovingAverage(model=model_for_ckpt, decay=args.ema_decay) if use_ema else None
    if use_ema:
        print(f"EMA enabled (decay={args.ema_decay})")
    else:
        print("EMA disabled")
    engine_cfg = EngineConfig(
        grad_clip_norm=1.0,
        show_progress=not args.no_progress,
        use_amp=use_amp,
        amp_dtype=args.amp_dtype,
        metrics_backend=args.metrics_backend,
        metrics_propagate=not args.no_eval_propagation,
        active_namespaces=active_namespaces,
    )
    print(f"metrics backend: {args.metrics_backend}")
    print(f"eval propagation: {not args.no_eval_propagation}")

    best_val_score = -1.0
    best_epoch = 0
    last_lr_decay_epoch = 0
    early_stopping_best = float("-inf")
    early_stopping_bad_epochs = 0
    stopped_early = False
    completed_epochs = 0
    best_ckpt_path = Path(args.output_dir) / "best_model.pt"
    epoch_ckpt_dir = Path(args.output_dir) / "epoch_checkpoints"
    if args.save_epoch_start > 0 and args.save_epoch_end > 0:
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"epoch checkpoint saving enabled: {args.save_epoch_start}-{args.save_epoch_end} -> {epoch_ckpt_dir}")

    if args.eval_interval != 1:
        print("warning: --eval-interval is ignored; validation runs every epoch.")

    interrupted = False
    try:
        for epoch in range(1, args.epochs + 1):
            completed_epochs = epoch
            engine_cfg.train_progress_desc = f"train {epoch}/{args.epochs}"
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                go_bundle=go_bundle,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                cfg=engine_cfg,
                scaler=scaler,
                ema=ema,
            )

            if ema is not None:
                ema.apply_shadow(model)
            try:
                engine_cfg.eval_progress_desc = f"val {epoch}/{args.epochs}"
                val_out = evaluate(
                    model=model,
                    dataloader=val_loader,
                    go_bundle=go_bundle,
                    criterion=criterion,
                    device=device,
                    cfg=engine_cfg,
                )
            finally:
                if ema is not None:
                    ema.restore(model)
            val_metrics = val_out["metrics"]
            val_score_mean = float(np.mean([val_metrics[ns]["fmax"] for ns in active_namespaces]))
            val_score_weighted = weighted_val_score(
                val_metrics=val_metrics,
                weights=val_score_weights,
                active_namespaces=active_namespaces,
            )
            lr_info = " ".join(
                f"lr_{g.get('group_name', i)}={g['lr']:.2e}" for i, g in enumerate(optimizer.param_groups)
            )
            metric_info = " ".join(
                f"val_fmax_{ns}={val_metrics[ns]['fmax']:.4f}" for ns in active_namespaces
            )
            loss_info = (
                f"train_loss={train_loss.get('total', 0.0):.4f}"
                f" val_loss={val_out['loss'].get('total', 0.0):.4f}"
            )

            print(
                f"epoch={epoch}"
                f" {loss_info}"
                f" {metric_info}"
                f" val_fmax_mean={val_score_mean:.4f}"
                f" val_fmax_weighted={val_score_weighted:.4f}"
                f" {lr_info}"
            )

            if should_save_epoch(epoch, args.save_epoch_start, args.save_epoch_end, args.save_epoch_interval):
                epoch_tag = f"epoch_{epoch:03d}"
                if ema is not None:
                    ema.apply_shadow(model)
                try:
                    torch.save(
                        {
                            "model_state_dict": model_for_ckpt.state_dict(),
                            "protein_cfg": protein_cfg.__dict__,
                            "model_cfg": model_cfg.__dict__,
                            "epoch": epoch,
                            "val_score": float(val_score_weighted),
                            "val_score_mean": float(val_score_mean),
                            "val_score_weights": val_score_weights,
                            "active_namespaces": active_namespaces,
                        },
                        epoch_ckpt_dir / f"{epoch_tag}.pt",
                    )
                finally:
                    if ema is not None:
                        ema.restore(model)
                val_payload = {
                    "epoch": int(epoch),
                    "val_score": float(val_score_weighted),
                    "val_score_mean": float(val_score_mean),
                    "val_metrics": val_metrics,
                    "active_namespaces": active_namespaces,
                    "learning_rates": {
                        str(g.get("group_name", i)): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)
                    },
                }
                with open(epoch_ckpt_dir / f"{epoch_tag}_val_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(val_payload, f, ensure_ascii=False, indent=2)
                print(f"saved epoch snapshot: {epoch_ckpt_dir / f'{epoch_tag}.pt'}")

            if val_score_weighted > best_val_score:
                best_val_score = val_score_weighted
                best_epoch = epoch
                if ema is not None:
                    ema.apply_shadow(model)
                try:
                    torch.save(
                        {
                            "model_state_dict": model_for_ckpt.state_dict(),
                            "protein_cfg": protein_cfg.__dict__,
                            "model_cfg": model_cfg.__dict__,
                            "epoch": epoch,
                            "best_val_score": float(best_val_score),
                            "best_val_score_mean": float(val_score_mean),
                            "val_score_weights": val_score_weights,
                            "active_namespaces": active_namespaces,
                        },
                        best_ckpt_path,
                    )
                finally:
                    if ema is not None:
                        ema.restore(model)
                print(f"new best model at epoch={epoch}, checkpoint updated.")
            else:
                since_anchor = epoch - max(best_epoch, last_lr_decay_epoch)
                if since_anchor >= args.lr_patience:
                    prev_lrs = [float(g["lr"]) for g in optimizer.param_groups]
                    new_lrs = []
                    for group in optimizer.param_groups:
                        min_group_lr = float(group.get("min_lr", args.min_lr))
                        new_lrs.append(max(float(group["lr"]) * args.lr_reduce_factor, min_group_lr))
                    if any(n < p for p, n in zip(prev_lrs, new_lrs)):
                        for group, lr_new in zip(optimizer.param_groups, new_lrs):
                            group["lr"] = lr_new
                        print(
                            "lr reduced: "
                            + ", ".join(f"{p:.3e}->{n:.3e}" for p, n in zip(prev_lrs, new_lrs))
                            + " "
                            f"(no improvement for {since_anchor} epochs)"
                        )
                    last_lr_decay_epoch = epoch

            if val_score_weighted > early_stopping_best + args.early_stopping_min_delta:
                early_stopping_best = val_score_weighted
                early_stopping_bad_epochs = 0
            else:
                early_stopping_bad_epochs += 1

            if (
                args.early_stopping_patience > 0
                and epoch >= args.early_stopping_min_epochs
                and early_stopping_bad_epochs >= args.early_stopping_patience
            ):
                stopped_early = True
                print(
                    "early stopping triggered:"
                    f" epoch={epoch}"
                    f" patience={args.early_stopping_patience}"
                    f" min_delta={args.early_stopping_min_delta}"
                    f" best_validation_score={early_stopping_best:.6f}"
                )
                break
    except KeyboardInterrupt:
        interrupted = True
        print("training interrupted by user; running one final test from best checkpoint.")

    print(f"training completed: epochs={completed_epochs}/{args.epochs} stopped_early={stopped_early}")
    print(f"best checkpoint: {best_ckpt_path} (score={best_val_score:.4f})")
    if not best_ckpt_path.exists():
        print("no best checkpoint saved yet; saving current model as fallback best checkpoint.")
        if ema is not None:
            ema.apply_shadow(model)
        try:
            torch.save(
                {
                    "model_state_dict": model_for_ckpt.state_dict(),
                    "protein_cfg": protein_cfg.__dict__,
                    "model_cfg": model_cfg.__dict__,
                    "epoch": int(best_epoch),
                    "best_val_score": float(best_val_score),
                    "val_score_weights": val_score_weights,
                    "active_namespaces": active_namespaces,
                },
                best_ckpt_path,
            )
        finally:
            if ema is not None:
                ema.restore(model)

    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model_for_ckpt.load_state_dict(checkpoint["model_state_dict"])
    ckpt_epoch = int(checkpoint.get("epoch", best_epoch))
    ckpt_best_val = float(checkpoint.get("best_val_score", best_val_score))
    test_out = evaluate(
        model=model,
        dataloader=test_loader,
        go_bundle=go_bundle,
        criterion=criterion,
        device=device,
        cfg=engine_cfg,
    )
    print("computing normalized Smin from training-split annotation IC...")
    train_labels_for_smin = dense_labels_from_label_indices(
        label_indices_list=train_label_indices,
        num_terms=num_terms,
    )
    ic_by_namespace = compute_ic_by_namespace(train_labels_for_smin, go_bundle)
    smin_metrics = normalized_smin(
        y_true=test_out["y_true"],
        y_pred=test_out["y_pred"],
        ic_by_namespace=ic_by_namespace,
        go_bundle=go_bundle,
    )
    for ns in active_namespaces:
        test_out["metrics"][ns].update(smin_metrics[ns])
    del train_labels_for_smin, ic_by_namespace
    print("test metrics:", test_out["metrics"])
    save_best_model_test_metrics(
        output_dir=Path(args.output_dir),
        test_out=test_out,
        checkpoint_path=best_ckpt_path,
        checkpoint=checkpoint,
        checkpoint_epoch=ckpt_epoch,
        checkpoint_val_score=ckpt_best_val,
    )
    save_test_outputs(
        output_dir=Path(args.output_dir),
        test_out=test_out,
        go_bundle=go_bundle,
        propagate=not args.no_eval_propagation,
        best_epoch=ckpt_epoch,
        best_val_score=ckpt_best_val,
    )
    save_run_config(
        output_dir=Path(args.output_dir),
        args=args,
        extra={
            "resolved_device": str(device),
            "amp_enabled": bool(use_amp),
            "records_split_counts": {
                "train": int(len(train_records)),
                "val": int(len(val_records)),
                "test": int(len(test_records)),
            },
            "dataset_size_after_filtering": {
                "train": int(len(train_ds)),
                "val": int(len(val_ds)),
                "test": int(len(test_ds)),
            },
            "go_terms_after_filtering": {
                ns: int(len(go_bundle.encodings[ns].term_ids)) for ns in ("bp", "mf", "cc")
            },
            "branch_loss_weights": {ns: float(branch_weights[ns]) for ns in ("bp", "mf", "cc")},
            "val_score_weights": {ns: float(val_score_weights[ns]) for ns in ("bp", "mf", "cc")},
            "active_namespaces": list(active_namespaces),
            "ablation": {
                "hmr": bool(args.ablate_hmr),
                "modality_specific_skip_paths": bool(args.ablate_modality_specific_skip_paths),
                "go_semantic_branch": bool(args.ablate_go_semantic_branch),
                "go_semantic_modulation": bool(args.ablate_go_semantic_modulation),
                "namespace_specific_pathways": bool(args.ablate_namespace_specific_pathways),
                "loss_function": bool(args.ablate_loss_function),
            },
            "new_loss": {
                "ablate_loss_function": bool(args.ablate_loss_function),
                "single_branch_plain_bce": False,
                "single_branch_asymmetric_focal": bool(use_single_branch_plain_bce and not args.disable_asym_focal),
                "use_asymmetric_focal": bool(loss_cfg.use_asymmetric_focal),
                "gate_entropy_weight": float(loss_cfg.gate_entropy_weight),
                "use_frequency_weights": bool(loss_cfg.use_frequency_weights),
                "frequency_group_count": int(args.freq_loss_groups),
                "frequency_beta": float(args.freq_loss_beta),
                "frequency_weight_min": float(args.freq_loss_weight_min),
                "frequency_weight_max": float(args.freq_loss_weight_max),
                "namespace_size_loss_alpha": float(loss_cfg.namespace_size_loss_alpha),
                "namespace_size_loss_reference": str(loss_cfg.namespace_size_loss_reference),
                "dynamic_branch_weight": bool(loss_cfg.dynamic_branch_weight),
                "branch_balance_momentum": float(args.branch_balance_momentum),
                "branch_balance_temperature": float(args.branch_balance_temperature),
                "use_pos_weight": bool(loss_cfg.use_pos_weight),
            },
            "bp_lr_multiplier": float(args.bp_lr_multiplier),
            "mf_lr_multiplier": float(args.mf_lr_multiplier),
            "cc_lr_multiplier": float(args.cc_lr_multiplier),
            "ema_enabled": bool(use_ema),
            "ema_decay": float(args.ema_decay),
            "best_checkpoint": str(best_ckpt_path),
            "best_epoch": int(ckpt_epoch),
            "best_val_score": float(ckpt_best_val),
        },
    )
    if interrupted:
        print(f"saved interrupted-run test outputs -> {args.output_dir}")
    else:
        print(f"saved final test outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
