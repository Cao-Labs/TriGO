from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseGate, register_gate


"""通用 PLE gate：根据 gate 输入学习各 expert 的融合权重。"""
class PLEGate(BaseGate):
    """
    PLE-style gate:
    - gate logits are produced by a trainable gate network.
    - an additional trainable global bias is learned for each expert.
    - final expert weights = softmax(gate_logits + global_bias).
    """

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        gate_input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.gate_input_dim = gate_input_dim or input_dim
        self.gate_net = nn.Sequential(
            nn.Linear(self.gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_inputs),
        )
        self.global_bias = nn.Parameter(torch.zeros(num_inputs))

    def forward(
        self,
        expert_vectors: List[Tensor],
        gate_input: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if len(expert_vectors) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} vectors, but got {len(expert_vectors)}")

        if gate_input is None:
            gate_input = torch.mean(torch.stack(expert_vectors, dim=1), dim=1)

        logits = self.gate_net(gate_input) + self.global_bias.unsqueeze(0)
        weights = torch.softmax(logits, dim=-1)
        stacked = torch.stack(expert_vectors, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return fused, weights


@register_gate("sequence_gate")
# 序列路径 gate：融合 [sequence expert, shared expert]。
class SequenceGate(PLEGate):
    def __init__(
        self,
        input_dim: int,
        gate_input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            num_inputs=2,
            gate_input_dim=gate_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


@register_gate("structure_gate")
# 结构路径 gate：融合 [structure expert, shared expert]。
class StructureGate(PLEGate):
    def __init__(
        self,
        input_dim: int,
        gate_input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            num_inputs=2,
            gate_input_dim=gate_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


@register_gate("shared_gate")
# 共享路径 gate：融合 [sequence, structure, shared] 三路 expert。
class SharedGate(PLEGate):
    def __init__(
        self,
        input_dim: int,
        gate_input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            num_inputs=3,
            gate_input_dim=gate_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


@register_gate("final_gate")
# 最终 gate：把三路 gated 向量融合为最终蛋白表示。
class FinalGate(PLEGate):
    def __init__(
        self,
        input_dim: int,
        gate_input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            num_inputs=3,
            gate_input_dim=gate_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
