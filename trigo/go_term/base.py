from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Type

import torch
from torch import Tensor


NAMESPACE_NAME_MAP = {
    "bp": "biological_process",
    "mf": "molecular_function",
    "cc": "cellular_component",
}

FULL_TO_SHORT_NAMESPACE = {v: k for k, v in NAMESPACE_NAME_MAP.items()}


@dataclass
class GOTerm:
    go_id: str
    name: str
    namespace: str
    is_a_parents: List[str]
    rel_parents: Dict[str, List[str]]
    is_obsolete: bool = False


@dataclass
class GOSubgraph:
    namespace: str
    term_ids: List[str]
    edges: List[Tuple[str, str]]
    edge_types: List[str]

    @property
    def num_nodes(self) -> int:
        return len(self.term_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


@dataclass
class GONamespaceEncoding:
    namespace: str
    term_ids: List[str]
    embeddings: Tensor
    edge_index: Tensor
    edge_type: Tensor
    relation_vocab: Tuple[str, ...] = ("is_a",)

    def to(self, device: torch.device | str) -> "GONamespaceEncoding":
        return GONamespaceEncoding(
            namespace=self.namespace,
            term_ids=self.term_ids,
            embeddings=self.embeddings.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            relation_vocab=self.relation_vocab,
        )


@dataclass
class GOEncodingBundle:
    encodings: Dict[str, GONamespaceEncoding]

    def namespaces(self) -> List[str]:
        return sorted(self.encodings.keys())

    def get(self, namespace: str) -> GONamespaceEncoding:
        return self.encodings[namespace]

    def save(self, path: str) -> None:
        payload = {
            ns: {
                "namespace": item.namespace,
                "term_ids": item.term_ids,
                "embeddings": item.embeddings.cpu(),
                "edge_index": item.edge_index.cpu(),
                "edge_type": item.edge_type.cpu(),
                "relation_vocab": list(item.relation_vocab),
            }
            for ns, item in self.encodings.items()
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "GOEncodingBundle":
        raw = torch.load(path, map_location=map_location, weights_only=False)
        encodings = {}
        for ns, item in raw.items():
            edge_index = item["edge_index"]
            if "edge_type" in item:
                edge_type = item["edge_type"]
            else:
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
            relation_vocab = tuple(item.get("relation_vocab", ("is_a",)))
            encodings[ns] = GONamespaceEncoding(
                namespace=item["namespace"],
                term_ids=item["term_ids"],
                embeddings=item["embeddings"],
                edge_index=edge_index,
                edge_type=edge_type,
                relation_vocab=relation_vocab,
            )
        return cls(encodings=encodings)


class BaseGOParser(ABC):
    @abstractmethod
    def parse(self, obo_path: str) -> Dict[str, GOTerm]:
        raise NotImplementedError


class BaseSubgraphBuilder(ABC):
    @abstractmethod
    def build(self, terms: Dict[str, GOTerm]) -> Dict[str, GOSubgraph]:
        raise NotImplementedError


class BaseGOEncoder(ABC):
    @abstractmethod
    def fit_transform(self, subgraph: GOSubgraph) -> GONamespaceEncoding:
        raise NotImplementedError


PARSER_REGISTRY: Dict[str, Type[BaseGOParser]] = {}
SUBGRAPH_BUILDER_REGISTRY: Dict[str, Type[BaseSubgraphBuilder]] = {}
GO_ENCODER_REGISTRY: Dict[str, Type[BaseGOEncoder]] = {}


def register_parser(name: str) -> Callable[[Type[BaseGOParser]], Type[BaseGOParser]]:
    def decorator(cls: Type[BaseGOParser]) -> Type[BaseGOParser]:
        if name in PARSER_REGISTRY:
            raise ValueError(f"Parser '{name}' is already registered.")
        PARSER_REGISTRY[name] = cls
        return cls

    return decorator


def register_subgraph_builder(name: str) -> Callable[[Type[BaseSubgraphBuilder]], Type[BaseSubgraphBuilder]]:
    def decorator(cls: Type[BaseSubgraphBuilder]) -> Type[BaseSubgraphBuilder]:
        if name in SUBGRAPH_BUILDER_REGISTRY:
            raise ValueError(f"Subgraph builder '{name}' is already registered.")
        SUBGRAPH_BUILDER_REGISTRY[name] = cls
        return cls

    return decorator


def register_go_encoder(name: str) -> Callable[[Type[BaseGOEncoder]], Type[BaseGOEncoder]]:
    def decorator(cls: Type[BaseGOEncoder]) -> Type[BaseGOEncoder]:
        if name in GO_ENCODER_REGISTRY:
            raise ValueError(f"GO encoder '{name}' is already registered.")
        GO_ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def build_parser(name: str, **kwargs) -> BaseGOParser:
    if name not in PARSER_REGISTRY:
        available = ", ".join(sorted(PARSER_REGISTRY))
        raise KeyError(f"Unknown parser '{name}'. Available: {available}")
    return PARSER_REGISTRY[name](**kwargs)


def build_subgraph_builder(name: str, **kwargs) -> BaseSubgraphBuilder:
    if name not in SUBGRAPH_BUILDER_REGISTRY:
        available = ", ".join(sorted(SUBGRAPH_BUILDER_REGISTRY))
        raise KeyError(f"Unknown subgraph builder '{name}'. Available: {available}")
    return SUBGRAPH_BUILDER_REGISTRY[name](**kwargs)


def build_go_encoder(name: str, **kwargs) -> BaseGOEncoder:
    if name not in GO_ENCODER_REGISTRY:
        available = ", ".join(sorted(GO_ENCODER_REGISTRY))
        raise KeyError(f"Unknown GO encoder '{name}'. Available: {available}")
    return GO_ENCODER_REGISTRY[name](**kwargs)
