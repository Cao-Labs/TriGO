from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import build_gate
from . import gates as _gates  # noqa: F401


@dataclass
class ProteinPLEConfig:
    seq_input_dim: int
    expert_hidden_dim: int = 256
    expert_output_dim: int = 256
    dropout: float = 0.1
    max_residues: int = 4096
    gate_hidden_dim: int = 128

    sequence_gate_name: str = "sequence_gate"
    structure_gate_name: str = "structure_gate"
    shared_gate_name: str = "shared_gate"
    final_gate_name: str = "final_gate"
    ablate_protein_ple: bool = False
    ablate_modality_skip_paths: bool = False
    ablate_adaptive_protein_fusion: bool = False
    use_strong_protein_ple: bool = False
    clean_expert_gate: bool = False
    strong_gcn_layers: int = 3
    normalize_sequence_input: bool = False


class ProteinPLEEncoder(nn.Module):
    """
    Lightweight protein encoder with PLE-style routing.

    The experts are deliberately simple: ESM residue features provide the
    sequence evidence, compact contact-map statistics provide structure
    evidence, and a small MLP forms the shared sequence/structure view.
    The model capacity is kept in the routing/gating path rather than in
    heavy internal expert blocks.
    """

    def __init__(self, cfg: ProteinPLEConfig):
        super().__init__()
        self.cfg = cfg
        self.sequence_input_norm = (
            nn.LayerNorm(cfg.seq_input_dim) if cfg.normalize_sequence_input else nn.Identity()
        )

        if cfg.ablate_protein_ple:
            self.simple_fusion_mlp = nn.Sequential(
                nn.Linear(cfg.seq_input_dim + 5, cfg.expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.expert_hidden_dim, cfg.expert_output_dim),
                nn.LayerNorm(cfg.expert_output_dim),
            )
            return

        self.strong_gcn_layers = nn.ModuleList()
        self.strong_gcn_norms = nn.ModuleList()
        self.shared_gcn_layers = nn.ModuleList()
        self.shared_gcn_norms = nn.ModuleList()
        if cfg.use_strong_protein_ple:
            self.strong_structure_node_mlp = nn.Sequential(
                nn.Linear(8, cfg.expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.expert_hidden_dim, cfg.expert_output_dim),
                nn.LayerNorm(cfg.expert_output_dim),
            )
            for _ in range(max(1, int(cfg.strong_gcn_layers))):
                self.strong_gcn_layers.append(
                    nn.ModuleDict(
                        {
                            "self": nn.Linear(cfg.expert_output_dim, cfg.expert_output_dim),
                            "neighbor": nn.Linear(cfg.expert_output_dim, cfg.expert_output_dim),
                        }
                    )
                )
                self.strong_gcn_norms.append(nn.LayerNorm(cfg.expert_output_dim))
            self.strong_structure_attention = nn.Linear(cfg.expert_output_dim, 1)
            self.strong_structure_out = nn.Sequential(
                nn.Linear(cfg.expert_output_dim, cfg.expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.expert_hidden_dim, cfg.expert_output_dim),
                nn.LayerNorm(cfg.expert_output_dim),
            )

            if not cfg.ablate_adaptive_protein_fusion:
                self.shared_node_proj = nn.Sequential(
                    nn.Linear(cfg.seq_input_dim, cfg.expert_output_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.LayerNorm(cfg.expert_output_dim),
                )
                for _ in range(max(1, int(cfg.strong_gcn_layers))):
                    self.shared_gcn_layers.append(
                        nn.ModuleDict(
                            {
                                "self": nn.Linear(cfg.expert_output_dim, cfg.expert_output_dim),
                                "neighbor": nn.Linear(cfg.expert_output_dim, cfg.expert_output_dim),
                            }
                        )
                    )
                    self.shared_gcn_norms.append(nn.LayerNorm(cfg.expert_output_dim))
                self.shared_graph_attention = nn.Linear(cfg.expert_output_dim, 1)
                self.shared_graph_out = nn.Sequential(
                    nn.Linear(cfg.expert_output_dim, cfg.expert_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.expert_hidden_dim, cfg.expert_output_dim),
                    nn.LayerNorm(cfg.expert_output_dim),
                )

        self.sequence_residue_mlp = nn.Sequential(
            nn.Linear(cfg.seq_input_dim, cfg.expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.expert_hidden_dim, cfg.expert_output_dim),
            nn.LayerNorm(cfg.expert_output_dim),
        )
        self.sequence_attention = nn.Linear(cfg.expert_output_dim, 1)

        self.structure_mlp = nn.Sequential(
            nn.Linear(5, cfg.expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.expert_hidden_dim, cfg.expert_output_dim),
            nn.LayerNorm(cfg.expert_output_dim),
        )

        self.shared_mlp: Optional[nn.Sequential] = None
        if not cfg.ablate_adaptive_protein_fusion:
            self.shared_mlp = nn.Sequential(
                nn.Linear(cfg.expert_output_dim * 2, cfg.expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.expert_hidden_dim, cfg.expert_output_dim),
                nn.LayerNorm(cfg.expert_output_dim),
            )

        self.sequence_gate = None
        self.structure_gate = None
        self.shared_gate = None
        self.final_gate = None
        if not cfg.ablate_adaptive_protein_fusion and not cfg.ablate_modality_skip_paths:
            self.sequence_gate = build_gate(
                cfg.sequence_gate_name,
                input_dim=cfg.expert_output_dim,
                gate_input_dim=cfg.expert_output_dim,
                hidden_dim=cfg.gate_hidden_dim,
                dropout=cfg.dropout,
            )
            self.structure_gate = build_gate(
                cfg.structure_gate_name,
                input_dim=cfg.expert_output_dim,
                gate_input_dim=cfg.expert_output_dim,
                hidden_dim=cfg.gate_hidden_dim,
                dropout=cfg.dropout,
            )
        if not cfg.ablate_adaptive_protein_fusion:
            self.shared_gate = build_gate(
                cfg.shared_gate_name,
                input_dim=cfg.expert_output_dim,
                gate_input_dim=cfg.expert_output_dim,
                hidden_dim=cfg.gate_hidden_dim,
                dropout=cfg.dropout,
            )
        if not cfg.ablate_adaptive_protein_fusion and not cfg.ablate_modality_skip_paths:
            self.final_gate = build_gate(
                cfg.final_gate_name,
                input_dim=cfg.expert_output_dim,
                gate_input_dim=cfg.expert_output_dim * 3,
                hidden_dim=cfg.gate_hidden_dim,
                dropout=cfg.dropout,
            )

    @staticmethod
    def _safe_mask(sequence_embeddings: Tensor, residue_mask: Optional[Tensor]) -> Tensor:
        bsz, num_residues, _ = sequence_embeddings.shape
        if residue_mask is None:
            return torch.ones((bsz, num_residues), device=sequence_embeddings.device, dtype=torch.bool)
        return residue_mask.to(device=sequence_embeddings.device, dtype=torch.bool)

    def _sequence_expert(self, sequence_embeddings: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        residue_repr = self.sequence_residue_mlp(sequence_embeddings)
        residue_repr = residue_repr * mask.unsqueeze(-1).to(dtype=residue_repr.dtype)
        attn_logits = self.sequence_attention(residue_repr).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = torch.where(mask, attn_weights, torch.zeros_like(attn_weights))
        denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn_weights = attn_weights / denom
        protein_repr = (attn_weights.unsqueeze(-1) * residue_repr).sum(dim=1)
        return residue_repr, protein_repr

    @staticmethod
    def _structure_stats(contact_map: Tensor, mask: Tensor, dtype: torch.dtype) -> Tensor:
        bsz, num_residues, _ = contact_map.shape
        device = contact_map.device
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(num_residues, device=device, dtype=torch.bool).unsqueeze(0)
        valid_pairs = pair_mask & ~eye
        contacts = torch.isfinite(contact_map) & valid_pairs

        degree = contacts.sum(dim=-1).to(dtype=dtype)
        residue_count = mask.sum(dim=1, keepdim=True).to(dtype=dtype).clamp_min(1.0)
        degree_denom = (residue_count - 1.0).clamp_min(1.0)
        degree_norm = degree / degree_denom
        mask_f = mask.to(dtype=dtype)

        degree_mean = (degree_norm * mask_f).sum(dim=1, keepdim=True) / residue_count
        degree_var = (((degree_norm - degree_mean) ** 2) * mask_f).sum(dim=1, keepdim=True) / residue_count
        degree_std = degree_var.clamp_min(0.0).sqrt()
        degree_max = degree_norm.masked_fill(~mask, 0.0).max(dim=1, keepdim=True).values

        contact_count = contacts.sum(dim=(1, 2), keepdim=False).to(dtype=dtype).unsqueeze(-1)
        contact_distance_sum = torch.where(contacts, contact_map, torch.zeros_like(contact_map)).to(dtype=dtype)
        contact_distance_mean = contact_distance_sum.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        contact_distance_mean = contact_distance_mean / contact_count.clamp_min(1.0)

        valid_pair_count = valid_pairs.sum(dim=(1, 2), keepdim=False).to(dtype=dtype).unsqueeze(-1).clamp_min(1.0)
        contact_density = contact_count / valid_pair_count
        length_fraction = residue_count / float(max(num_residues, 1))

        return torch.cat([contact_distance_mean, degree_std, degree_max, contact_density, length_fraction], dim=-1)

    @staticmethod
    def _contact_adjacency(contact_map: Tensor, mask: Tensor, dtype: torch.dtype) -> Tensor:
        bsz, num_residues, _ = contact_map.shape
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(num_residues, device=contact_map.device, dtype=torch.bool).unsqueeze(0)
        contacts = torch.isfinite(contact_map) & pair_mask & ~eye
        adj = contacts.to(dtype=dtype)
        adj = adj + eye.to(dtype=dtype) * mask.unsqueeze(1).to(dtype=dtype)
        return adj

    @staticmethod
    def _structure_node_features(contact_map: Tensor, mask: Tensor, dtype: torch.dtype) -> Tensor:
        bsz, num_residues, _ = contact_map.shape
        device = contact_map.device
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(num_residues, device=device, dtype=torch.bool).unsqueeze(0)
        valid_pairs = pair_mask & ~eye
        contacts = torch.isfinite(contact_map) & valid_pairs
        mask_f = mask.to(dtype=dtype)

        degree = contacts.sum(dim=-1).to(dtype=dtype)
        residue_count = mask.sum(dim=1, keepdim=True).to(dtype=dtype).clamp_min(1.0)
        degree_norm = degree / (residue_count - 1.0).clamp_min(1.0)
        contact_density = contacts.sum(dim=(1, 2), keepdim=True).to(dtype=dtype)
        valid_count = valid_pairs.sum(dim=(1, 2), keepdim=True).to(dtype=dtype).clamp_min(1.0)
        contact_density = (contact_density / valid_count).expand(-1, num_residues, -1)

        pos = torch.arange(num_residues, device=device, dtype=dtype)
        denom = (residue_count - 1.0).clamp_min(1.0).unsqueeze(-1)
        rel_pos = pos.view(1, num_residues, 1) / denom
        rel_pos = rel_pos.expand(bsz, -1, -1)

        sep = (pos.view(1, num_residues, 1) - pos.view(1, 1, num_residues)).abs()
        sep = sep.to(dtype=dtype) / float(max(num_residues - 1, 1))
        sep = sep.expand(bsz, -1, -1)
        contacts_f = contacts.to(dtype=dtype)
        degree_denom = degree.unsqueeze(-1).clamp_min(1.0)
        sep_mean = (contacts_f * sep).sum(dim=-1, keepdim=True) / degree_denom
        sep_var = (contacts_f * (sep - sep_mean).pow(2)).sum(dim=-1, keepdim=True) / degree_denom
        sep_std = sep_var.clamp_min(0.0).sqrt()

        weighted_dist = torch.where(torch.isfinite(contact_map), contact_map, torch.zeros_like(contact_map)).to(dtype=dtype)
        dist_mean = (contacts_f * weighted_dist).sum(dim=-1, keepdim=True) / degree_denom
        dist_mean = dist_mean / dist_mean.detach().amax(dim=1, keepdim=True).clamp_min(1.0)

        length_fraction = (residue_count / float(max(num_residues, 1))).unsqueeze(1).expand(-1, num_residues, -1)
        isolated = (degree <= 0).to(dtype=dtype).unsqueeze(-1)
        features = torch.cat(
            [
                degree_norm.unsqueeze(-1),
                contact_density,
                rel_pos,
                sep_mean,
                sep_std,
                dist_mean,
                length_fraction,
                isolated,
            ],
            dim=-1,
        )
        return features * mask_f.unsqueeze(-1)

    def _graph_conv_stack(
        self,
        residue_repr: Tensor,
        adj: Tensor,
        mask: Tensor,
        layers: nn.ModuleList,
        norms: nn.ModuleList,
    ) -> Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=residue_repr.dtype)
        h = residue_repr * mask_f
        adj = adj.to(dtype=h.dtype)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adj_norm = adj / degree
        for layer, norm in zip(layers, norms):
            neigh = torch.bmm(adj_norm, h)
            update = layer["self"](h) + layer["neighbor"](neigh)
            h = norm(h + torch.nn.functional.gelu(update))
            h = h * mask_f
        return h

    @staticmethod
    def _attention_pool(residue_repr: Tensor, mask: Tensor, attention: nn.Linear) -> Tensor:
        attn_logits = attention(residue_repr).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = torch.where(mask, attn_weights, torch.zeros_like(attn_weights))
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (attn_weights.unsqueeze(-1) * residue_repr).sum(dim=1)

    def _strong_structure_expert(self, contact_map: Tensor, mask: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        node_features = self._structure_node_features(contact_map, mask, dtype=dtype)
        adj = self._contact_adjacency(contact_map, mask, dtype=dtype)
        residue_repr = self.strong_structure_node_mlp(node_features)
        residue_repr = self._graph_conv_stack(residue_repr, adj, mask, self.strong_gcn_layers, self.strong_gcn_norms)
        pooled = self._attention_pool(residue_repr, mask, self.strong_structure_attention)
        protein_repr = self.strong_structure_out(pooled)
        return residue_repr, protein_repr

    def _strong_shared_expert(self, sequence_embeddings: Tensor, contact_map: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        adj = self._contact_adjacency(contact_map, mask, dtype=sequence_embeddings.dtype)
        residue_repr = self.shared_node_proj(sequence_embeddings)
        residue_repr = self._graph_conv_stack(residue_repr, adj, mask, self.shared_gcn_layers, self.shared_gcn_norms)
        pooled = self._attention_pool(residue_repr, mask, self.shared_graph_attention)
        protein_repr = self.shared_graph_out(pooled)
        return residue_repr, protein_repr

    def _simple_encoder(
        self,
        sequence_embeddings: Tensor,
        contact_map: Tensor,
        mask: Tensor,
    ) -> Dict[str, Tensor | Dict[str, Tensor]]:
        mask_f = mask.unsqueeze(-1).to(dtype=sequence_embeddings.dtype)
        residue_count = mask_f.sum(dim=1).clamp_min(1.0)
        seq_mean = (sequence_embeddings * mask_f).sum(dim=1) / residue_count
        struct_features = self._structure_stats(contact_map, mask, dtype=sequence_embeddings.dtype)
        protein_repr = self.simple_fusion_mlp(torch.cat([seq_mean, struct_features], dim=-1))
        residue_repr = protein_repr.unsqueeze(1).expand(-1, sequence_embeddings.size(1), -1) * mask_f
        zero_protein_repr = torch.zeros_like(protein_repr)
        zero_residue_repr = torch.zeros_like(residue_repr)
        batch_size = protein_repr.size(0)
        return {
            "protein_repr": protein_repr,
            "expert_protein_repr": {
                "sequence": zero_protein_repr,
                "structure": zero_protein_repr,
                "shared": zero_protein_repr,
            },
            "expert_residue_repr": {
                "sequence": zero_residue_repr,
                "structure": zero_residue_repr,
                "shared": zero_residue_repr,
            },
            "gate_outputs": {
                "sequence_gate": zero_protein_repr,
                "structure_gate": zero_protein_repr,
                "shared_gate": zero_protein_repr,
            },
            "gate_weights": {
                "sequence_gate": protein_repr.new_full((batch_size, 1), 1.0),
                "structure_gate": protein_repr.new_full((batch_size, 1), 1.0),
                "shared_gate": protein_repr.new_full((batch_size, 1), 1.0),
                "final_gate": protein_repr.new_full((batch_size, 1), 1.0),
            },
        }

    def forward(
        self,
        sequence_embeddings: Tensor,
        contact_map: Tensor,
        residue_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor | Dict[str, Tensor]]:
        num_residues = sequence_embeddings.size(1)
        if num_residues > self.cfg.max_residues:
            raise ValueError(
                f"Input length {num_residues} exceeds max_residues={self.cfg.max_residues}. "
                "Increase max_residues when building the encoder."
            )

        mask = self._safe_mask(sequence_embeddings, residue_mask)
        sequence_embeddings = self.sequence_input_norm(sequence_embeddings)
        if self.cfg.ablate_protein_ple:
            return self._simple_encoder(sequence_embeddings, contact_map, mask)

        seq_residue_out, seq_vec = self._sequence_expert(sequence_embeddings, mask)

        if self.cfg.use_strong_protein_ple:
            struct_residue_out, struct_vec = self._strong_structure_expert(
                contact_map,
                mask,
                dtype=sequence_embeddings.dtype,
            )
            if self.cfg.ablate_adaptive_protein_fusion:
                shared_vec = torch.zeros_like(seq_vec)
                shared_residue_out = torch.zeros_like(seq_residue_out)
            else:
                shared_residue_out, shared_vec = self._strong_shared_expert(
                    sequence_embeddings,
                    contact_map,
                    mask,
                )
        else:
            struct_features = self._structure_stats(contact_map, mask, dtype=sequence_embeddings.dtype)
            struct_vec = self.structure_mlp(struct_features)
            struct_residue_out = struct_vec.unsqueeze(1).expand(-1, num_residues, -1)
            struct_residue_out = struct_residue_out * mask.unsqueeze(-1).to(dtype=struct_residue_out.dtype)

            if self.cfg.ablate_adaptive_protein_fusion:
                shared_vec = torch.zeros_like(seq_vec)
                shared_residue_out = torch.zeros_like(seq_residue_out)
            else:
                if self.shared_mlp is None:
                    raise RuntimeError("Shared protein expert was not initialized.")
                shared_vec = self.shared_mlp(torch.cat([seq_vec, struct_vec], dim=-1))
                shared_residue_out = shared_vec.unsqueeze(1).expand(-1, num_residues, -1)
                shared_residue_out = shared_residue_out * mask.unsqueeze(-1).to(dtype=shared_residue_out.dtype)

        if self.cfg.ablate_adaptive_protein_fusion:
            final_out = 0.5 * seq_vec + 0.5 * struct_vec
            gate_outputs = {}
            gate_weights = {}
        elif self.cfg.ablate_modality_skip_paths:
            if self.shared_gate is None:
                raise RuntimeError("Protein shared gate was not initialized.")
            shared_gate_out, shared_gate_w = self.shared_gate(
                [seq_vec, struct_vec, shared_vec],
                gate_input=shared_vec,
            )
            final_out = shared_gate_out
            gate_outputs = {"shared_gate": shared_gate_out}
            gate_weights = {"shared_gate": shared_gate_w}
        elif self.cfg.clean_expert_gate:
            batch_size = seq_vec.size(0)
            sequence_gate_out = seq_vec
            structure_gate_out = struct_vec
            shared_gate_out = shared_vec
            sequence_gate_w = seq_vec.new_full((batch_size, 1), 1.0)
            structure_gate_w = seq_vec.new_full((batch_size, 1), 1.0)
            shared_gate_w = seq_vec.new_full((batch_size, 1), 1.0)
        else:
            structure_gate_out, structure_gate_w = self.structure_gate(
                [struct_vec, shared_vec],
                gate_input=struct_vec,
            )
            sequence_gate_out, sequence_gate_w = self.sequence_gate(
                [seq_vec, shared_vec],
                gate_input=seq_vec,
            )
            shared_gate_out, shared_gate_w = self.shared_gate(
                [seq_vec, struct_vec, shared_vec],
                gate_input=shared_vec,
            )

        if not self.cfg.ablate_adaptive_protein_fusion and not self.cfg.ablate_modality_skip_paths:
            if self.final_gate is None:
                raise RuntimeError("Protein final gate was not initialized.")
            final_gate_input = torch.cat([sequence_gate_out, structure_gate_out, shared_gate_out], dim=-1)
            final_out, final_gate_w = self.final_gate(
                [sequence_gate_out, structure_gate_out, shared_gate_out],
                gate_input=final_gate_input,
            )
            gate_outputs = {
                "sequence_gate": sequence_gate_out,
                "structure_gate": structure_gate_out,
                "shared_gate": shared_gate_out,
            }
            gate_weights = {
                "sequence_gate": sequence_gate_w,
                "structure_gate": structure_gate_w,
                "shared_gate": shared_gate_w,
                "final_gate": final_gate_w,
            }

        return {
            "protein_repr": final_out,
            "expert_protein_repr": {
                "sequence": seq_vec,
                "structure": struct_vec,
                "shared": shared_vec,
            },
            "expert_residue_repr": {
                "sequence": seq_residue_out,
                "structure": struct_residue_out,
                "shared": shared_residue_out,
            },
            "gate_outputs": gate_outputs,
            "gate_weights": gate_weights,
        }
