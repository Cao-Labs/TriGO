# TriGO processed dataset splits

This directory contains the processed protein sequences, propagated GO annotations, and fixed train/validation/test assignments used by TriGO. Its organization follows the lightweight FASTA-plus-annotation style used by protein-function repositories such as ARF-GNN, while keeping the three TriGO benchmarks separate.

```text
data/
├── cafa3/
│   ├── train.fasta
│   ├── val.fasta
│   ├── test.fasta
│   ├── train_annotations.tsv
│   ├── val_annotations.tsv
│   └── test_annotations.tsv
├── goa/                       # same six files
├── cafa5/                     # same six files
├── manifest.tsv               # counts and SHA-256 checksums
└── README.md
```

FASTA headers contain only the protein identifier:

```text
>P12345
MSEQUENCE...
```

Each annotation file is tab-separated and contains:

```text
Entry	Gene Ontology IDs
P12345	GO:0000001;GO:0000002
```

The GO terms have already been propagated through the dataset-specific ontology and filtered using the training-set frequency rules described in [DATA.md](../DATA.md). They should not be propagated or frequency-filtered again during model training.

## Dataset sizes

| Dataset | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| CAFA3 | 53,143 | 5,934 | 2,962 | 62,039 |
| GOA | 24,076 | 5,965 | 10,652 | 40,693 |
| CAFA5 | 22,882 | 2,532 | 2,729 | 28,143 |

The CAFA5 files correspond to the feature-available subset used in the manuscript, including the common 2,729-protein test set. The protein-level counts above are larger than some namespace-specific counts in the paper because the latter include only proteins with at least one retained BP, MF, or CC label.

CAFA3 has 29 records without a GO term after final global frequency filtering. They are retained to preserve the fixed protein-level split; branch-specific training and evaluation ignore proteins without a retained positive label in that namespace.

## Build the training workbook

TriGO's training interface uses a single workbook containing all three splits. Reconstruct it from the released files with:

```bash
python scripts/build_annotation_workbook.py \
  --data-dir data/cafa3 \
  --output data/cafa3/annotations.xlsx
```

Repeat the command with `goa` or `cafa5` as needed. Generated `.xlsx` files are ignored by Git and are not part of the public data release.

PDB structures, ontology snapshots, ESM-2 caches, GO embeddings, and model outputs are not included. Obtain structures and ontology files from the sources described in [DATA.md](../DATA.md), then generate cached inputs using the repository preprocessing scripts.

## Integrity check

The counts and SHA-256 hashes of all released FASTA and TSV files are recorded in `manifest.tsv`. For example:

```bash
sha256sum data/cafa3/train.fasta
```

The underlying CAFA, UniProt, Gene Ontology, and structure records remain subject to their original licenses and terms of use.
