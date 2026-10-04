from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor


# 保存单个 expert 的残基级与蛋白级表示输出。
@dataclass
class ExpertOutput:
    residue_repr: Tensor
    protein_repr: Tensor


# expert 抽象接口：把序列与接触图编码成表示。
class BaseExpert(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        sequence_embeddings: Tensor,
        contact_map: Tensor,
        residue_mask: Tensor | None = None,
    ) -> ExpertOutput:
        raise NotImplementedError


# gate 抽象接口：融合多个 expert 向量。
class BaseGate(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        expert_vectors: List[Tensor],
        gate_input: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


EXPERT_REGISTRY: Dict[str, Type[BaseExpert]] = {}
GATE_REGISTRY: Dict[str, Type[BaseGate]] = {}


def register_expert(name: str) -> Callable[[Type[BaseExpert]], Type[BaseExpert]]:
    def decorator(cls: Type[BaseExpert]) -> Type[BaseExpert]:
        if name in EXPERT_REGISTRY:
            raise ValueError(f"Expert '{name}' is already registered.")
        EXPERT_REGISTRY[name] = cls
        return cls

    return decorator


def register_gate(name: str) -> Callable[[Type[BaseGate]], Type[BaseGate]]:
    def decorator(cls: Type[BaseGate]) -> Type[BaseGate]:
        if name in GATE_REGISTRY:
            raise ValueError(f"Gate '{name}' is already registered.")
        GATE_REGISTRY[name] = cls
        return cls

    return decorator


def build_expert(name: str, **kwargs) -> BaseExpert:
    if name not in EXPERT_REGISTRY:
        available = ", ".join(sorted(EXPERT_REGISTRY))
        raise KeyError(f"Unknown expert '{name}'. Available experts: {available}")
    return EXPERT_REGISTRY[name](**kwargs)


def build_gate(name: str, **kwargs) -> BaseGate:
    if name not in GATE_REGISTRY:
        available = ", ".join(sorted(GATE_REGISTRY))
        raise KeyError(f"Unknown gate '{name}'. Available gates: {available}")
    return GATE_REGISTRY[name](**kwargs)
