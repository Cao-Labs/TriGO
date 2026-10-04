from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from trigo.go_term.parser import OBOBasicParser


@dataclass
class ProteinAnnotationRecord:
    entry: str
    sequence: str
    go_ids: List[str]
    go_by_namespace: Dict[str, List[str]]
    pdb_path: str
    split: str | None = None


def parse_go_annotation_field(text: object) -> List[str]:
    if text is None:
        return []
    if isinstance(text, float) and pd.isna(text):
        return []
    raw = str(text).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(";") if item.strip().startswith("GO:")]


def load_go_namespace_and_name(obo_path: str) -> tuple[Dict[str, str], Dict[str, str]]:
    parser = OBOBasicParser(keep_obsolete=False)
    terms = parser.parse(obo_path)
    go_to_namespace = {}
    go_to_name = {}
    for go_id, term in terms.items():
        if term.namespace == "biological_process":
            go_to_namespace[go_id] = "bp"
        elif term.namespace == "molecular_function":
            go_to_namespace[go_id] = "mf"
        elif term.namespace == "cellular_component":
            go_to_namespace[go_id] = "cc"
        go_to_name[go_id] = term.name
    return go_to_namespace, go_to_name


def load_annotation_records(
    xlsx_path: str,
    pdb_dir: str,
    go_to_namespace: Dict[str, str],
    require_pdb: bool = True,
) -> List[ProteinAnnotationRecord]:
    df = pd.read_excel(xlsx_path)
    required = {"Entry", "Sequence", "Gene Ontology IDs"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in annotation file: {sorted(missing)}")

    pdb_dir_path = Path(pdb_dir)
    records: List[ProteinAnnotationRecord] = []
    has_split_col = "Split" in df.columns
    for _, row in df.iterrows():
        entry = str(row["Entry"]).strip()
        if not entry:
            continue
        pdb_path = pdb_dir_path / f"{entry}.pdb"
        if require_pdb and not pdb_path.exists():
            continue

        sequence = str(row["Sequence"]).strip()
        go_ids = parse_go_annotation_field(row["Gene Ontology IDs"])
        go_by_ns = {"bp": [], "mf": [], "cc": []}
        for go_id in go_ids:
            ns = go_to_namespace.get(go_id)
            if ns in go_by_ns:
                go_by_ns[ns].append(go_id)

        split_value: str | None = None
        if has_split_col:
            raw_split = row.get("Split")
            if raw_split is not None and not (isinstance(raw_split, float) and pd.isna(raw_split)):
                split_text = str(raw_split).strip().lower()
                if split_text:
                    split_value = split_text

        records.append(
            ProteinAnnotationRecord(
                entry=entry,
                sequence=sequence,
                go_ids=go_ids,
                go_by_namespace=go_by_ns,
                pdb_path=str(pdb_path),
                split=split_value,
            )
        )

    return records


def build_go_term_index(
    namespace_term_ids: Dict[str, List[str]],
) -> Dict[str, Dict[str, int]]:
    return {ns: {go_id: idx for idx, go_id in enumerate(ids)} for ns, ids in namespace_term_ids.items()}


def annotation_to_label_indices(
    record: ProteinAnnotationRecord,
    go_index: Dict[str, Dict[str, int]],
) -> Dict[str, List[int]]:
    labels = {}
    for ns in ("bp", "mf", "cc"):
        idxs = []
        ns_map = go_index.get(ns, {})
        for go_id in record.go_by_namespace.get(ns, []):
            if go_id in ns_map:
                idxs.append(ns_map[go_id])
        labels[ns] = sorted(set(idxs))
    return labels
