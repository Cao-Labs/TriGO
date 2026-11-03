#!/usr/bin/env python3
"""Reconstruct the TriGO annotation workbook from released FASTA/TSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from openpyxl import Workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build annotations.xlsx from a released TriGO dataset directory."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    entry: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if entry is not None:
                    records[entry] = "".join(chunks)
                entry = line[1:].split()[0]
                if entry in records:
                    raise ValueError(f"Duplicate FASTA entry {entry} in {path}")
                chunks = []
            elif entry is None:
                raise ValueError(f"Sequence before first FASTA header in {path}")
            else:
                chunks.append(line)
    if entry is not None:
        records[entry] = "".join(chunks)
    return records


def main() -> None:
    args = parse_args()
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("annotations")
    sheet.append(["Entry", "Sequence", "Gene Ontology IDs", "Split"])

    seen: set[str] = set()
    for split in ("train", "val", "test"):
        sequences = read_fasta(args.data_dir / f"{split}.fasta")
        annotation_path = args.data_dir / f"{split}_annotations.tsv"
        with annotation_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["Entry", "Gene Ontology IDs"]:
                raise ValueError(f"Unexpected columns in {annotation_path}: {reader.fieldnames}")
            annotation_entries: set[str] = set()
            for row in reader:
                entry = row["Entry"].strip()
                if entry in seen:
                    raise ValueError(f"Duplicate Entry across splits: {entry}")
                if entry not in sequences:
                    raise ValueError(f"{entry} is in {annotation_path} but not the matching FASTA")
                annotation_entries.add(entry)
                seen.add(entry)
                sheet.append([entry, sequences[entry], row["Gene Ontology IDs"].strip(), split])
            if annotation_entries != set(sequences):
                missing = sorted(set(sequences) - annotation_entries)[:5]
                raise ValueError(f"FASTA entries without annotations in {split}: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output)
    print(f"Wrote {len(seen)} records to {args.output}")


if __name__ == "__main__":
    main()
