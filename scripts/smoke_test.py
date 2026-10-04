"""Run a small CPU forward pass without datasets or pretrained checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trigo.go_term import GOEncodingBundle, GONamespaceEncoding  # noqa: E402
from trigo.multi_branch import MultiBranchProteinGOConfig, MultiBranchProteinGOModel  # noqa: E402
from trigo.protein_ple import ProteinPLEConfig  # noqa: E402


def make_go_bundle() -> GOEncodingBundle:
    encodings = {}
    for namespace, count in (("bp", 5), ("mf", 4), ("cc", 3)):
        encodings[namespace] = GONamespaceEncoding(
            namespace=namespace,
            term_ids=[f"GO:{idx:07d}" for idx in range(count)],
            embeddings=torch.randn(count, 10),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_type=torch.empty((0,), dtype=torch.long),
        )
    return GOEncodingBundle(encodings)


def main() -> None:
    torch.manual_seed(7)
    protein_cfg = ProteinPLEConfig(
        seq_input_dim=12,
        expert_hidden_dim=16,
        expert_output_dim=16,
        gate_hidden_dim=8,
        max_residues=8,
        dropout=0.0,
    )
    model_cfg = MultiBranchProteinGOConfig(
        protein_dim=16,
        expert_dim=16,
        shared_expert_hidden_dim=24,
        namespace_expert_hidden_dim=24,
        namespace_context_hidden_dim=12,
        gate_hidden_dim=8,
        namespace_attention_topk=3,
        cross_namespace_refinement_repeats=2,
        dropout=0.0,
    )
    model = MultiBranchProteinGOModel(model_cfg, protein_encoder_cfg=protein_cfg).eval()

    sequence_embeddings = torch.randn(2, 8, 12)
    coordinates = torch.rand(2, 8, 3)
    distances = torch.cdist(coordinates, coordinates)
    contact_map = torch.where(
        distances <= 0.8,
        distances,
        torch.full_like(distances, float("inf")),
    )
    diagonal = torch.arange(8)
    contact_map[:, diagonal, diagonal] = float("inf")
    residue_mask = torch.ones(2, 8, dtype=torch.bool)

    with torch.no_grad():
        output = model(
            sequence_embeddings,
            contact_map,
            make_go_bundle(),
            residue_mask,
        )

    observed = {namespace: tuple(logits.shape) for namespace, logits in output["logits"].items()}
    expected = {"bp": (2, 5), "mf": (2, 4), "cc": (2, 3)}
    if observed != expected:
        raise RuntimeError(f"Unexpected output shapes: {observed}; expected {expected}")
    print(f"TriGO smoke test passed: {observed}")


if __name__ == "__main__":
    main()
