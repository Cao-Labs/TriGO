from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from trigo.go_term import GOEncodingBundle
from trigo.protein_ple import ProteinPLEConfig, ProteinPLEEncoder

from .components import (
    CrossNamespaceSemanticRefinementBlock,
    DirectMLPPredictor,
    GOSimilarityPredictor,
    MLPExpert,
    NamespaceConditionedExpert,
    NamespaceContextEncoder,
    NamespacePrivateAdapter,
    PLEVectorGate,
)


# 多分支模型的结构与维度超参数配置。
@dataclass
class MultiBranchProteinGOConfig:
    namespaces: Tuple[str, ...] = ("bp", "mf", "cc")
    protein_dim: int = 256
    expert_dim: int = 256
    shared_expert_hidden_dim: int = 512
    namespace_expert_hidden_dim: int = 512
    namespace_expert_layers: int = 3
    gate_hidden_dim: int = 128
    namespace_context_hidden_dim: int = 256
    namespace_attention_topk: int = 256
    bp_namespace_attention_topk: int = 0
    mf_namespace_attention_topk: int = 0
    cc_namespace_attention_topk: int = 0
    predictor_use_term_bias: bool = True
    dropout: float = 0.1
    ablate_protein_ple: bool = False
    ablate_modality_skip_paths: bool = False
    ablate_go_context: bool = False
    ablate_go_modulation: bool = False
    ablate_go_semantics: bool = False
    ablate_go_semantics_mlp_head: bool = False
    ablate_go_semantics_mlp_hidden_dim: int = 256
    ablate_go_semantics_skip_gate: bool = False
    use_shared_go_branch: bool = False
    use_shared_go_branch_go_predictor: bool = False
    use_go_context_residual_gate: bool = False
    go_context_residual_init: float = 0.1
    use_go_context_concat: bool = False
    cross_namespace_refinement_repeats: int = 0
    cross_namespace_refinement_residual_init: float = 0.1
    share_cross_namespace_refinement_weights: bool = False
    detach_cross_namespace_sources: bool = False
    use_namespace_private_adapter: bool = False
    namespace_private_adapter_hidden_dim: int = 128
    namespace_private_adapter_residual_init: float = 0.1
    bp_namespace_private_adapter_residual_init: Optional[float] = None
    mf_namespace_private_adapter_residual_init: Optional[float] = None
    cc_namespace_private_adapter_residual_init: Optional[float] = None
    bp_num_terms: int = 0
    mf_num_terms: int = 0
    cc_num_terms: int = 0


class MultiBranchProteinGOModel(nn.Module):
    """主模型：融合序列/结构/共享专家后对 bp/mf/cc 三个分支分别预测。"""

    def __init__(
        self,
        cfg: MultiBranchProteinGOConfig,
        protein_encoder: Optional[ProteinPLEEncoder] = None,
        protein_encoder_cfg: Optional[ProteinPLEConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg

        if protein_encoder is None:
            if protein_encoder_cfg is None:
                raise ValueError("Provide either `protein_encoder` or `protein_encoder_cfg`.")
            self.protein_encoder = ProteinPLEEncoder(protein_encoder_cfg)
        else:
            self.protein_encoder = protein_encoder

        if cfg.cross_namespace_refinement_repeats < 0:
            raise ValueError("cross_namespace_refinement_repeats must be >= 0.")
        if cfg.share_cross_namespace_refinement_weights and cfg.cross_namespace_refinement_repeats == 0:
            raise ValueError(
                "share_cross_namespace_refinement_weights requires cross_namespace_refinement_repeats > 0."
            )
        if cfg.cross_namespace_refinement_repeats > 0 and (
            cfg.ablate_go_semantics or cfg.use_shared_go_branch or cfg.use_shared_go_branch_go_predictor
        ):
            raise ValueError(
                "Cross-namespace refinement is defined for the standard three-branch model. "
                "Disable shared-branch/GO-semantics ablations when using it."
            )

        self.sequence_shared_expert: Optional[MLPExpert] = None
        self.structure_shared_expert: Optional[MLPExpert] = None
        if not cfg.ablate_modality_skip_paths:
            self.sequence_shared_expert = MLPExpert(
                input_dim=cfg.protein_dim,
                output_dim=cfg.expert_dim,
                hidden_dim=cfg.shared_expert_hidden_dim,
                dropout=cfg.dropout,
            )
            self.structure_shared_expert = MLPExpert(
                input_dim=cfg.protein_dim,
                output_dim=cfg.expert_dim,
                hidden_dim=cfg.shared_expert_hidden_dim,
                dropout=cfg.dropout,
            )

        self.namespace_context_encoders = nn.ModuleDict(
            {
                ns: NamespaceContextEncoder(
                    protein_dim=cfg.protein_dim,
                    dropout=cfg.dropout,
                    hidden_dim=cfg.namespace_context_hidden_dim,
                    top_k=self._namespace_attention_topk(ns),
                )
                for ns in cfg.namespaces
            }
        )
        if cfg.use_shared_go_branch or cfg.use_shared_go_branch_go_predictor:
            # Shared-path ablation: one shared context encoder is reused for
            # BP/MF/CC GO spaces, then the aligned contexts are summed before
            # one conditioned expert feeds all predictors.
            self.shared_go_context_encoder = NamespaceContextEncoder(
                protein_dim=cfg.protein_dim,
                dropout=cfg.dropout,
                hidden_dim=cfg.namespace_context_hidden_dim,
                top_k=0,
            )
            self.shared_go_expert = NamespaceConditionedExpert(
                input_dim=cfg.protein_dim,
                output_dim=cfg.expert_dim,
                hidden_dim=cfg.namespace_expert_hidden_dim,
                num_layers=cfg.namespace_expert_layers,
                dropout=cfg.dropout,
                use_context_residual_gate=cfg.use_go_context_residual_gate,
                context_residual_init=cfg.go_context_residual_init,
                use_context_concat=cfg.use_go_context_concat,
            )
            if cfg.use_shared_go_branch:
                self.shared_mlp_predictor = DirectMLPPredictor(
                    input_dim=cfg.expert_dim,
                    num_terms=cfg.bp_num_terms + cfg.mf_num_terms + cfg.cc_num_terms,
                    hidden_dim=cfg.namespace_expert_hidden_dim,
                    dropout=cfg.dropout,
                    use_term_bias=False,
                )
            if cfg.use_shared_go_branch_go_predictor:
                self.shared_go_predictor = GOSimilarityPredictor(
                    query_dim=cfg.expert_dim,
                    dropout=cfg.dropout,
                    # A single predictor is reused for BP/MF/CC term sets, so
                    # per-term bias must stay outside this shared module.
                    use_term_bias=False,
                )
        self.namespace_experts = nn.ModuleDict(
            {
                ns: NamespaceConditionedExpert(
                    input_dim=cfg.protein_dim,
                    output_dim=cfg.expert_dim,
                    hidden_dim=cfg.namespace_expert_hidden_dim,
                    num_layers=cfg.namespace_expert_layers,
                    dropout=cfg.dropout,
                    use_context_residual_gate=cfg.use_go_context_residual_gate,
                    context_residual_init=cfg.go_context_residual_init,
                    use_context_concat=cfg.use_go_context_concat,
                )
                for ns in cfg.namespaces
            }
        )
        self.protein_only_expert: Optional[MLPExpert] = None
        if cfg.ablate_go_semantics_skip_gate:
            self.protein_only_expert = MLPExpert(
                input_dim=cfg.protein_dim,
                output_dim=cfg.expert_dim,
                hidden_dim=cfg.shared_expert_hidden_dim,
                dropout=cfg.dropout,
            )
        self.namespace_branch_gates = nn.ModuleDict()
        if not cfg.ablate_modality_skip_paths:
            self.namespace_branch_gates = nn.ModuleDict(
                {
                    ns: PLEVectorGate(
                        input_dim=cfg.expert_dim,
                        num_inputs=3,
                        gate_input_dim=cfg.expert_dim,
                        hidden_dim=cfg.gate_hidden_dim,
                        dropout=cfg.dropout,
                    )
                    for ns in cfg.namespaces
                }
            )
        self.namespace_private_adapters = nn.ModuleDict()
        if cfg.use_namespace_private_adapter:
            self.namespace_private_adapters = nn.ModuleDict(
                {
                    ns: NamespacePrivateAdapter(
                        dim=cfg.expert_dim,
                        hidden_dim=cfg.namespace_private_adapter_hidden_dim,
                        dropout=cfg.dropout,
                        residual_init=self._namespace_private_adapter_init(ns),
                    )
                    for ns in cfg.namespaces
                }
            )
        if cfg.ablate_go_semantics and (cfg.ablate_go_semantics_mlp_head or cfg.ablate_go_semantics_skip_gate):
            predictor_input_dim = cfg.expert_dim if cfg.ablate_go_semantics_skip_gate else cfg.protein_dim
            self.namespace_predictors = nn.ModuleDict(
                {
                    ns: DirectMLPPredictor(
                        input_dim=predictor_input_dim,
                        num_terms=self._namespace_num_terms(ns),
                        hidden_dim=cfg.ablate_go_semantics_mlp_hidden_dim,
                        dropout=cfg.dropout,
                        use_term_bias=cfg.predictor_use_term_bias,
                    )
                    for ns in cfg.namespaces
                }
            )
        else:
            self.namespace_predictors = nn.ModuleDict(
                {
                    ns: GOSimilarityPredictor(
                        query_dim=cfg.protein_dim if cfg.ablate_go_semantics else cfg.expert_dim,
                        dropout=cfg.dropout,
                        use_term_bias=cfg.predictor_use_term_bias,
                    )
                    for ns in cfg.namespaces
                }
            )
        refinement_block_count = (
            1
            if cfg.share_cross_namespace_refinement_weights and cfg.cross_namespace_refinement_repeats > 0
            else cfg.cross_namespace_refinement_repeats
        )
        self.cross_namespace_refinement_blocks = nn.ModuleList(
            [
                CrossNamespaceSemanticRefinementBlock(
                    namespaces=cfg.namespaces,
                    expert_dim=cfg.expert_dim,
                    hidden_dim=cfg.namespace_expert_hidden_dim,
                    num_layers=cfg.namespace_expert_layers,
                    gate_hidden_dim=cfg.gate_hidden_dim,
                    context_hidden_dim=cfg.namespace_context_hidden_dim,
                    top_k=cfg.namespace_attention_topk,
                    dropout=cfg.dropout,
                    use_context_residual_gate=cfg.use_go_context_residual_gate,
                    context_residual_init=cfg.go_context_residual_init,
                    use_context_concat=cfg.use_go_context_concat,
                    refinement_residual_init=cfg.cross_namespace_refinement_residual_init,
                    detach_cross_namespace_sources=cfg.detach_cross_namespace_sources,
                )
                for _ in range(refinement_block_count)
            ]
        )

    def _namespace_attention_topk(self, namespace: str) -> int:
        per_namespace = {
            "bp": self.cfg.bp_namespace_attention_topk,
            "mf": self.cfg.mf_namespace_attention_topk,
            "cc": self.cfg.cc_namespace_attention_topk,
        }.get(namespace, 0)
        if int(per_namespace) > 0:
            return int(per_namespace)
        return int(self.cfg.namespace_attention_topk)

    def _namespace_num_terms(self, namespace: str) -> int:
        per_namespace = {
            "bp": self.cfg.bp_num_terms,
            "mf": self.cfg.mf_num_terms,
            "cc": self.cfg.cc_num_terms,
        }.get(namespace, 0)
        if int(per_namespace) <= 0:
            raise ValueError(
                f"num_terms for namespace '{namespace}' must be provided when ablate_go_semantics=True."
            )
        return int(per_namespace)

    def _namespace_private_adapter_init(self, namespace: str) -> float:
        per_namespace = {
            "bp": self.cfg.bp_namespace_private_adapter_residual_init,
            "mf": self.cfg.mf_namespace_private_adapter_residual_init,
            "cc": self.cfg.cc_namespace_private_adapter_residual_init,
        }.get(namespace)
        if per_namespace is not None:
            return float(per_namespace)
        return float(self.cfg.namespace_private_adapter_residual_init)

    def _run_protein_encoder(
        self,
        sequence_embeddings: Tensor,
        contact_map: Tensor,
        residue_mask: Optional[Tensor],
    ) -> Dict[str, Tensor | Dict[str, Tensor]]:
        return self.protein_encoder(sequence_embeddings, contact_map, residue_mask)

    def forward(
        self,
        sequence_embeddings: Tensor,
        contact_map: Tensor,
        go_bundle: GOEncodingBundle,
        residue_mask: Optional[Tensor] = None,
    ) -> Dict[str, Dict[str, Tensor] | Tensor]:
        protein_outputs = self._run_protein_encoder(sequence_embeddings, contact_map, residue_mask)
        expert_repr = protein_outputs["expert_protein_repr"]
        seq_repr = expert_repr["sequence"]
        struct_repr = expert_repr["structure"]
        protein_repr = protein_outputs["protein_repr"]

        if self.cfg.ablate_protein_ple or self.cfg.ablate_modality_skip_paths:
            seq_shared_out = protein_repr.new_zeros((protein_repr.size(0), self.cfg.expert_dim))
            struct_shared_out = protein_repr.new_zeros((protein_repr.size(0), self.cfg.expert_dim))
        else:
            if self.sequence_shared_expert is None or self.structure_shared_expert is None:
                raise RuntimeError("Modality shared experts were not initialized.")
            seq_shared_out = self.sequence_shared_expert(seq_repr)
            struct_shared_out = self.structure_shared_expert(struct_repr)

        branch_logits: Dict[str, Tensor] = {}
        branch_outputs: Dict[str, Tensor] = {}
        namespace_contexts: Dict[str, Tensor] = {}
        namespace_branch_gate_weights: Dict[str, Tensor] = {}
        first_round_branch_gate_weights: Dict[str, Tensor] = {}
        cross_namespace_contexts: Dict[str, Dict[str, Tensor]] = {}
        cross_namespace_gate_weights: Dict[str, Tensor] = {}
        cross_namespace_shared_outputs: Dict[str, Tensor] = {}

        shared_go_context: Optional[Tensor] = None
        shared_branch_out: Optional[Tensor] = None
        use_shared_branch = self.cfg.use_shared_go_branch or self.cfg.use_shared_go_branch_go_predictor
        if use_shared_branch and not self.cfg.ablate_go_semantics:
            if self.cfg.ablate_go_context:
                shared_go_context = torch.zeros_like(protein_repr)
            else:
                shared_contexts = []
                for ns in self.cfg.namespaces:
                    if ns not in go_bundle.encodings:
                        continue
                    go_embeddings = go_bundle.encodings[ns].embeddings.to(protein_repr.device)
                    if go_embeddings.numel() == 0:
                        continue
                    shared_contexts.append(self.shared_go_context_encoder(protein_repr, go_embeddings))
                if shared_contexts:
                    shared_go_context = torch.stack(shared_contexts, dim=0).sum(dim=0)
                else:
                    shared_go_context = torch.zeros_like(protein_repr)
            shared_branch_out = self.shared_go_expert(
                protein_repr,
                context=shared_go_context,
                use_context_modulation=not self.cfg.ablate_go_modulation,
            )

        for ns in self.cfg.namespaces:
            if ns not in go_bundle.encodings:
                continue
            ns_encoding = go_bundle.encodings[ns]
            go_embeddings = ns_encoding.embeddings.to(protein_repr.device)
            if go_embeddings.numel() == 0:
                continue

            if self.cfg.ablate_go_semantics:
                go_context = torch.zeros_like(protein_repr)
                if self.cfg.ablate_go_semantics_skip_gate:
                    if self.protein_only_expert is None:
                        raise RuntimeError("Protein-only skip-gate expert was not initialized.")
                    protein_only_out = self.protein_only_expert(protein_repr)
                    branch_out, branch_w = self.namespace_branch_gates[ns](
                        [protein_only_out, seq_shared_out, struct_shared_out],
                        gate_input=protein_only_out,
                    )
                else:
                    branch_out = protein_repr
                    branch_w = protein_repr.new_full((protein_repr.size(0), 1), 1.0)
                logits = self.namespace_predictors[ns](branch_out, go_embeddings)
            elif use_shared_branch:
                if shared_go_context is None or shared_branch_out is None:
                    raise RuntimeError("Shared GO branch was not initialized before namespace prediction.")
                go_context = shared_go_context
                branch_out = shared_branch_out
                branch_w = shared_branch_out.new_full((shared_branch_out.size(0), 1), 1.0)
                if self.cfg.use_shared_go_branch_go_predictor:
                    logits = self.shared_go_predictor(branch_out, go_embeddings)
                else:
                    # Logits are produced once by a single unified MLP predictor
                    # below, then split back by namespace for existing metrics.
                    logits = None
            elif self.cfg.ablate_go_context:
                go_context = torch.zeros_like(protein_repr)
            else:
                go_context = self.namespace_context_encoders[ns](protein_repr, go_embeddings)
            if not (self.cfg.ablate_go_semantics or use_shared_branch):
                ns_expert_out = self.namespace_experts[ns](
                    protein_repr,
                    context=go_context,
                    use_context_modulation=not self.cfg.ablate_go_modulation,
                )
                if self.cfg.ablate_protein_ple or self.cfg.ablate_modality_skip_paths:
                    branch_out = ns_expert_out
                    branch_w = ns_expert_out.new_full((ns_expert_out.size(0), 1), 1.0)
                else:
                    branch_out, branch_w = self.namespace_branch_gates[ns](
                        [ns_expert_out, seq_shared_out, struct_shared_out],
                        gate_input=ns_expert_out,
                    )
                if ns in self.namespace_private_adapters:
                    branch_out = self.namespace_private_adapters[ns](branch_out)
                logits = self.namespace_predictors[ns](branch_out, go_embeddings)

            if logits is not None:
                branch_logits[ns] = logits
            branch_outputs[ns] = branch_out
            namespace_contexts[ns] = go_context
            namespace_branch_gate_weights[ns] = branch_w
            first_round_branch_gate_weights[ns] = branch_w

        if self.cfg.use_shared_go_branch and shared_branch_out is not None:
            unified_logits = self.shared_mlp_predictor(shared_branch_out)
            offset = 0
            for ns in self.cfg.namespaces:
                width = self._namespace_num_terms(ns)
                if ns not in go_bundle.encodings:
                    offset += width
                    continue
                if width <= 0:
                    continue
                branch_logits[ns] = unified_logits[:, offset : offset + width]
                offset += width
        elif self.cross_namespace_refinement_blocks:
            branch_logits = {}
            go_embeddings_by_namespace = {
                ns: go_bundle.encodings[ns].embeddings.to(protein_repr.device)
                for ns in self.cfg.namespaces
                if ns in go_bundle.encodings and go_bundle.encodings[ns].embeddings.numel() > 0
            }
            for block_idx in range(self.cfg.cross_namespace_refinement_repeats):
                refinement_block = self.cross_namespace_refinement_blocks[
                    0 if self.cfg.share_cross_namespace_refinement_weights else block_idx
                ]
                branch_outputs, private_contexts, refine_gate_weights, shared_out = refinement_block(
                    branch_outputs,
                    go_embeddings_by_namespace,
                    ablate_go_context=self.cfg.ablate_go_context,
                    ablate_go_modulation=self.cfg.ablate_go_modulation,
                )
                cross_namespace_contexts[f"repeat_{block_idx}"] = private_contexts
                for ns, weights in refine_gate_weights.items():
                    cross_namespace_gate_weights[f"{ns}_repeat_{block_idx}"] = weights
                    namespace_branch_gate_weights[ns] = weights
                cross_namespace_shared_outputs[f"repeat_{block_idx}"] = shared_out
            for ns in self.cfg.namespaces:
                if ns not in go_embeddings_by_namespace or ns not in branch_outputs:
                    continue
                branch_logits[ns] = self.namespace_predictors[ns](branch_outputs[ns], go_embeddings_by_namespace[ns])

        return {
            "logits": branch_logits,
            "branch_outputs": branch_outputs,
            "namespace_contexts": namespace_contexts,
            "cross_namespace_contexts": cross_namespace_contexts,
            "branch_gate_weights": namespace_branch_gate_weights,
            "first_round_branch_gate_weights": first_round_branch_gate_weights,
            "cross_namespace_gate_weights": cross_namespace_gate_weights,
            "cross_namespace_shared_outputs": cross_namespace_shared_outputs,
            "sequence_shared_expert_out": seq_shared_out,
            "structure_shared_expert_out": struct_shared_out,
            "protein_repr": protein_outputs["protein_repr"],
            "protein_gate_weights": protein_outputs.get("gate_weights", {}),
        }
