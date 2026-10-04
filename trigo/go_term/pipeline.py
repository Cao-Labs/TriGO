from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .base import GOEncodingBundle, GOSubgraph, build_go_encoder, build_parser, build_subgraph_builder
from .graph_builder import SubgraphBuilderConfig, summarize_subgraphs
from .node2vec import Node2VecConfig

# Register built-in components.
from . import parser as _parser  # noqa: F401
from . import graph_builder as _graph_builder  # noqa: F401
from . import node2vec as _node2vec  # noqa: F401


@dataclass
class GOEncoderPipelineConfig:
    parser_name: str = "obo_basic"
    subgraph_builder_name: str = "go_namespace_hierarchy"
    encoder_name: str = "node2vec_torch"

    keep_obsolete_terms: bool = False
    subgraph_builder_config: SubgraphBuilderConfig = field(default_factory=SubgraphBuilderConfig)

    bp_node2vec: Node2VecConfig = field(default_factory=Node2VecConfig)
    mf_node2vec: Node2VecConfig = field(default_factory=Node2VecConfig)
    cc_node2vec: Node2VecConfig = field(default_factory=Node2VecConfig)


class GOEncoderPipeline:
    """
    End-to-end:
    OBO -> parsed terms -> bp/mf/cc subgraphs -> separate Node2Vec encodings.
    """

    def __init__(self, cfg: GOEncoderPipelineConfig):
        self.cfg = cfg
        self.parser = build_parser(cfg.parser_name, keep_obsolete=cfg.keep_obsolete_terms)
        self.subgraph_builder = build_subgraph_builder(
            cfg.subgraph_builder_name,
            config=cfg.subgraph_builder_config,
        )
        self.encoders = {
            "bp": build_go_encoder(cfg.encoder_name, config=cfg.bp_node2vec),
            "mf": build_go_encoder(cfg.encoder_name, config=cfg.mf_node2vec),
            "cc": build_go_encoder(cfg.encoder_name, config=cfg.cc_node2vec),
        }

    def parse_and_build_subgraphs(self, obo_path: str) -> Dict[str, GOSubgraph]:
        terms = self.parser.parse(obo_path)
        subgraphs = self.subgraph_builder.build(terms)
        return subgraphs

    def fit_transform(self, obo_path: str) -> GOEncodingBundle:
        subgraphs = self.parse_and_build_subgraphs(obo_path)
        encodings = {}
        for ns in ("bp", "mf", "cc"):
            if ns not in subgraphs:
                continue
            encodings[ns] = self.encoders[ns].fit_transform(subgraphs[ns])
        return GOEncodingBundle(encodings=encodings)

    def summarize(self, obo_path: str) -> Dict[str, Dict[str, int]]:
        subgraphs = self.parse_and_build_subgraphs(obo_path)
        return summarize_subgraphs(subgraphs)
