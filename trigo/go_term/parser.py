from __future__ import annotations

from typing import Dict, List

from .base import BaseGOParser, GOTerm, register_parser


def _parse_parent_go_id(text: str) -> str:
    # "GO:0009987 ! cellular process" -> "GO:0009987"
    return text.split("!")[0].strip().split()[0]


@register_parser("obo_basic")
class OBOBasicParser(BaseGOParser):
    """
    Parse GO OBO files into GOTerm objects.
    """

    def __init__(self, keep_obsolete: bool = False):
        self.keep_obsolete = keep_obsolete

    def parse(self, obo_path: str) -> Dict[str, GOTerm]:
        terms: Dict[str, GOTerm] = {}
        current: Dict[str, object] = {}
        in_term_block = False

        def flush_current() -> None:
            nonlocal current
            if not current:
                return
            go_id = current.get("id")
            namespace = current.get("namespace")
            if not go_id or not namespace:
                current = {}
                return

            term = GOTerm(
                go_id=str(go_id),
                name=str(current.get("name", "")),
                namespace=str(namespace),
                is_a_parents=list(current.get("is_a_parents", [])),
                rel_parents=dict(current.get("rel_parents", {})),
                is_obsolete=bool(current.get("is_obsolete", False)),
            )
            if self.keep_obsolete or not term.is_obsolete:
                terms[term.go_id] = term
            current = {}

        with open(obo_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    flush_current()
                    in_term_block = True
                    current = {
                        "is_a_parents": [],
                        "rel_parents": {},
                        "is_obsolete": False,
                    }
                    continue
                if line.startswith("[Typedef]"):
                    flush_current()
                    in_term_block = False
                    continue
                if not in_term_block:
                    continue

                if ": " not in line:
                    continue
                key, value = line.split(": ", 1)
                if key == "id":
                    current["id"] = value
                elif key == "name":
                    current["name"] = value
                elif key == "namespace":
                    current["namespace"] = value
                elif key == "is_a":
                    current["is_a_parents"].append(_parse_parent_go_id(value))
                elif key == "relationship":
                    # e.g. "part_of GO:0009987 ! ..."
                    parts = value.split()
                    if len(parts) >= 2:
                        rel_type = parts[0]
                        parent = _parse_parent_go_id(parts[1])
                        rel_parents: Dict[str, List[str]] = current["rel_parents"]  # type: ignore[assignment]
                        rel_parents.setdefault(rel_type, []).append(parent)
                elif key == "is_obsolete" and value.lower() == "true":
                    current["is_obsolete"] = True

        flush_current()
        return terms

