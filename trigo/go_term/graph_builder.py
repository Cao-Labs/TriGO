from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .base import (
    BaseSubgraphBuilder,
    FULL_TO_SHORT_NAMESPACE,
    GOSubgraph,
    GOTerm,
    NAMESPACE_NAME_MAP,
    register_subgraph_builder,
)


@dataclass
class SubgraphBuilderConfig:
    relation_types: Tuple[str, ...] = ("is_a", "part_of")
    restrict_in_namespace_parent: bool = True
    include_is_a: bool = True
    min_nodes_for_namespace: int = 1


@register_subgraph_builder("go_namespace_hierarchy")
class GONamespaceHierarchyBuilder(BaseSubgraphBuilder):
    """
    Build three GO subgraphs (bp/mf/cc) from parsed GOTerm dictionary.
    """

    def __init__(self, config: SubgraphBuilderConfig | None = None):
        self.config = config or SubgraphBuilderConfig()

    def _collect_namespace_terms(self, terms: Dict[str, GOTerm], short_ns: str) -> Set[str]:
        full_ns = NAMESPACE_NAME_MAP[short_ns]
        return {go_id for go_id, term in terms.items() if term.namespace == full_ns}

    def _collect_edges_for_namespace(
        self,
        terms: Dict[str, GOTerm],
        namespace_terms: Set[str],
    ) -> tuple[List[Tuple[str, str]], List[str]]:
        edges: List[Tuple[str, str]] = []
        edge_types: List[str] = []
        rel_types = set(self.config.relation_types)

        for go_id in sorted(namespace_terms):
            term = terms[go_id]

            if self.config.include_is_a:
                for parent in term.is_a_parents:
                    if parent not in terms:
                        continue
                    if self.config.restrict_in_namespace_parent and parent not in namespace_terms:
                        continue
                    edges.append((go_id, parent))
                    edge_types.append("is_a")

            for rel_name, rel_parents in term.rel_parents.items():
                if rel_name not in rel_types:
                    continue
                for parent in rel_parents:
                    if parent not in terms:
                        continue
                    if self.config.restrict_in_namespace_parent and parent not in namespace_terms:
                        continue
                    edges.append((go_id, parent))
                    edge_types.append(rel_name)
        return edges, edge_types

    def build(self, terms: Dict[str, GOTerm]) -> Dict[str, GOSubgraph]:
        subgraphs: Dict[str, GOSubgraph] = {}
        for short_ns in ("bp", "mf", "cc"):
            ns_terms = self._collect_namespace_terms(terms, short_ns)
            if len(ns_terms) < self.config.min_nodes_for_namespace:
                continue

            edges, edge_types = self._collect_edges_for_namespace(terms, ns_terms)
            term_ids = sorted(ns_terms)
            subgraphs[short_ns] = GOSubgraph(
                namespace=short_ns,
                term_ids=term_ids,
                edges=edges,
                edge_types=edge_types,
            )
        return subgraphs


def summarize_subgraphs(subgraphs: Dict[str, GOSubgraph]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for ns, graph in subgraphs.items():
        summary[ns] = {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
        }
    return summary

