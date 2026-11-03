#!/usr/bin/env python3
"""Export TriGO annotation workbooks as public FASTA and TSV split files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import textwrap
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook


REQUIRED_COLUMNS = ("Entry", "Sequence", "Gene Ontology IDs", "Split")
SPLIT_NAMES = {"train": "train", "val": "val", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the final TriGO workbooks to split FASTA/TSV files."
    )
    parser.add_argument("--cafa3-xlsx", type=Path, required=True)
    parser.add_argument("--goa-xlsx", type=Path, required=True)
    parser.add_argument("--cafa5-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sequence(value: object, entry: str) -> str:
    sequence = "".join(str(value or "").split()).upper()
    if not sequence:
        raise ValueError(f"Missing sequence for {entry}")
    return sequence


def export_dataset(dataset: str, workbook_path: Path, output_root: Path) -> list[dict[str, object]]:
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_paths = {split: output_dir / f"{split}.fasta" for split in ("train", "val", "test")}
    tsv_paths = {
        split: output_dir / f"{split}_annotations.tsv" for split in ("train", "val", "test")
    }
    fasta_handles = {split: path.open("w", encoding="utf-8", newline="\n") for split, path in fasta_paths.items()}
    tsv_handles = {split: path.open("w", encoding="utf-8", newline="") for split, path in tsv_paths.items()}
    tsv_writers = {
        split: csv.writer(handle, delimiter="\t", lineterminator="\n")
        for split, handle in tsv_handles.items()
    }
    for writer in tsv_writers.values():
        writer.writerow(["Entry", "Gene Ontology IDs"])

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    headers = tuple(next(rows))
    if headers != REQUIRED_COLUMNS:
        raise ValueError(f"Unexpected columns in {workbook_path}: {headers}")

    counts: Counter[str] = Counter()
    seen: set[str] = set()
    empty_annotations: Counter[str] = Counter()
    try:
        for entry_value, sequence_value, go_value, split_value in rows:
            entry = str(entry_value or "").strip()
            if not entry:
                raise ValueError(f"Blank Entry in {workbook_path}")
            if entry in seen:
                raise ValueError(f"Duplicate Entry {entry} in {workbook_path}")
            seen.add(entry)

            raw_split = str(split_value or "").strip().lower()
            if raw_split not in SPLIT_NAMES:
                raise ValueError(f"Invalid split {raw_split!r} for {entry}")
            split = SPLIT_NAMES[raw_split]
            sequence = normalize_sequence(sequence_value, entry)
            go_terms = str(go_value or "").strip()

            fasta_handles[split].write(f">{entry}\n")
            fasta_handles[split].write("\n".join(textwrap.wrap(sequence, width=80)) + "\n")
            tsv_writers[split].writerow([entry, go_terms])
            counts[split] += 1
            if not go_terms:
                empty_annotations[split] += 1
    finally:
        workbook.close()
        for handle in fasta_handles.values():
            handle.close()
        for handle in tsv_handles.values():
            handle.close()

    manifest_rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        manifest_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "proteins": counts[split],
                "empty_annotation_rows": empty_annotations[split],
                "fasta": fasta_paths[split].relative_to(output_root).as_posix(),
                "fasta_sha256": sha256(fasta_paths[split]),
                "annotations": tsv_paths[split].relative_to(output_root).as_posix(),
                "annotations_sha256": sha256(tsv_paths[split]),
            }
        )
    return manifest_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "cafa3": args.cafa3_xlsx,
        "goa": args.goa_xlsx,
        "cafa5": args.cafa5_xlsx,
    }
    manifest_rows: list[dict[str, object]] = []
    for dataset, workbook_path in inputs.items():
        manifest_rows.extend(export_dataset(dataset, workbook_path, args.output_dir))

    manifest_path = args.output_dir / "manifest.tsv"
    fields = [
        "dataset",
        "split",
        "proteins",
        "empty_annotation_rows",
        "fasta",
        "fasta_sha256",
        "annotations",
        "annotations_sha256",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    for row in manifest_rows:
        print(f"{row['dataset']:5s} {row['split']:5s}: {row['proteins']}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
