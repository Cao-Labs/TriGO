# Data sources, construction, and input format

The processed sequences, propagated GO annotations, and fixed split assignments used by TriGO are released as FASTA and TSV files under [`data/`](data/). Large or third-party artifacts—including PDB structures, ontology snapshots, pretrained weights, feature caches, checkpoints, and experiment outputs—are intentionally excluded. Obtain those resources from their original providers and retain the provider's license and attribution information.

## Reported datasets

| Dataset | Primary source | Raw files used | Split policy |
| --- | --- | --- | --- |
| CAFA3 | [Official CAFA resource](https://biofunctionprediction.org/cafa/) and the CAFA3/AlphaFold benchmark package used in this study | `train_data_train_standardAA_1000_afdb.tsv`, `train_data_valid_standardAA_1000_afdb.tsv`, `test_data_standardAA_1000.tsv`, and `afdb_dir_cafa3.tar.gz` | Preserve the supplied train, validation, and test partitions. |
| GOA | Archived [UniProt-GOA](https://www.ebi.ac.uk/GOA/downloads.html) annotations and [UniProt](https://www.ebi.ac.uk/uniprot/download-center) sequences | GOA annotation records, UniProt FASTA sequences, and the matching GO ontology snapshot | Temporal split at 1 January 2023, followed by homology filtering and a cluster-aware validation split. |
| CAFA5 | [CAFA5 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data) | `Train/train_sequences.fasta`, `Train/train_terms.tsv`, and `go-basic.obo` | Seeded, label-safe 90/10 train/test split, followed by a 10% validation holdout from the training portion. |

The CAFA5 competition data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The released CAFA5 FASTA and TSV files are modified derivatives produced by the processing described below.

### CAFA3

The three CAFA3 TSV files are normalized to the common annotation schema while retaining their supplied split assignments. Only proteins represented in the accompanying `afdb_dir_cafa3.tar.gz` structure archive are kept, and the compressed structures are extracted as `{Entry}.pdb` files. The GO ontology used for this benchmark is the 16 January 2016 release.

Annotations are propagated to ancestors through within-namespace `is_a` and `part_of` relations. The BP, MF, and CC roots are removed, and a GO term is retained only if it occurs in at least 50 training proteins.

### Temporal GOA benchmark

The GOA benchmark is assembled from UniProt-GOA annotations and UniProt protein sequences. It retains only the experimental or high-throughput evidence codes `EXP`, `IDA`, `IPI`, `IMP`, `IGI`, `IEP`, `HTP`, `HDA`, `HMP`, `HGI`, and `HEP`; electronically inferred (`IEA`) annotations are excluded.

Proteins annotated before 1 January 2023 form the training-side pool, and proteins first experimentally annotated after that date form the prospective test pool. The following leakage controls are then applied:

1. MMseqs2 removes a prospective test protein if it has at least 30% sequence identity and at least 50% alignment coverage to a training-side protein.
2. The training-side proteins are clustered at 50% sequence identity with CD-HIT and distributed across five cluster-aware folds. Fold 0 supplies the reported validation partition and the other folds supply the training partition.
3. Accessions shared with CAFA3 are removed. CD-HIT-2D additionally removes GOA sequences matching CAFA3 at 50% or greater sequence identity.
4. Proteins without a usable AlphaFold DB structure are removed.

Annotations are propagated with the same within-namespace hierarchy rules used for CAFA3. Terms are counted only in the training split and retained at minimum frequencies of 30 proteins for BP, 15 for MF, and 10 for CC. The final reported GOA split contains 24,076 training, 5,965 validation, and 10,652 test proteins before namespace-specific removal of proteins with no retained label.

Keep the exact `go-basic.obo` used to build the processed GOA workbook and GO embedding bundle together with the dataset. Substituting a current ontology release can change term namespaces, ancestry, or obsolete-term handling.

### CAFA5

CAFA5 starts from the competition's `Train/train_sequences.fasta` and `Train/train_terms.tsv`; its supplied `go-basic.obo` is the 1 January 2023 GO release. UniProt accessions are parsed from the FASTA headers. Using seed 42, 10% of the proteins are assigned to the test split, and 10% of the remaining training portion is assigned to validation. The holdout procedure keeps at least one training occurrence of every retained term.

Exact sequence matches to the CAFA3 and GOA datasets are excluded. Structures are retrieved by UniProt accession from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/), and records without a readable local PDB file are excluded. Annotations are then propagated through `is_a` and `part_of`, namespace roots are removed, and training-set frequency thresholds of 50, 10, and 10 proteins are applied to BP, MF, and CC, respectively.

After feature-availability filtering, the released CAFA5 split contains 22,882 training, 2,532 validation, and 2,729 test proteins. These are the same proteins used for the reported TriGO evaluation, including the common 2,729-protein test set.

## Common preprocessing rules

For all three datasets:

- GO terms are assigned to BP, MF, or CC using the dataset-specific ontology snapshot.
- Annotation propagation follows only within-namespace `is_a` and `part_of` parent edges; obsolete terms and the three namespace roots are excluded.
- Frequency thresholds are estimated from training proteins only and then applied unchanged to validation and test proteins.
- A protein must retain at least one GO label and have a readable PDB structure to enter a model run.
- Sequence residues and matching C-alpha coordinates are truncated consistently to 1,022 residues during feature generation.
- PDB files use the name `{Entry}.pdb`. Structures for GOA and CAFA5 were queried from AlphaFold DB by UniProt accession; CAFA3 uses its accompanying structure archive.
- ESM-2 residue features are generated from `esm2_t33_650M_UR50D`, layer 33, and cached before training. GO Node2Vec embeddings are generated from the same ontology snapshot supplied to the model.

## Recommended layout

```text
data/
  cafa3/
    annotations.xlsx
    go-basic.obo
    pdb/
      P12345.pdb
    protein_feature_cache/
    go_node2vec_embeddings.pt
  goa/
    annotations.xlsx
    go-basic.obo
    pdb/
    protein_feature_cache/
    go_node2vec_embeddings.pt
  cafa5/
    annotations.xlsx
    go-basic.obo
    pdb/
    protein_feature_cache/
    go_node2vec_embeddings.pt
pretrained/
  esm2_t33_650M_UR50D.pt
```

Alternative paths can be passed directly to the command-line programs. The reproduction scripts also accept `DATA_ROOT`, `OUTPUT_ROOT`, and `DEVICE` environment variables.

The repository initially provides `train.fasta`, `val.fasta`, `test.fasta`, and the matching `*_annotations.tsv` files in each dataset directory. Generate `annotations.xlsx` with `scripts/build_annotation_workbook.py`; the resulting workbook is intentionally ignored by Git.

## Annotation workbook

TriGO reads an Excel workbook with these columns:

| Column | Required | Meaning |
| --- | --- | --- |
| `Entry` | yes | Unique protein identifier; the matching structure is `{Entry}.pdb`. |
| `Sequence` | yes | Protein amino-acid sequence. |
| `Gene Ontology IDs` | yes | Semicolon-separated GO identifiers, for example `GO:0008150;GO:0003674`. |
| `Split` | required for reported configurations | `train`, `val`, or `test`. General exploratory runs may omit it and use a seeded random split. |

The PDB directory must contain one readable structure file per retained entry. The feature builder extracts C-alpha coordinates, truncates sequence and structure consistently, and caches the residue-level ESM representation and contact map.

The CAFA3, GOA, and CAFA5 configuration scripts pass `--require-split`. Every retained record must therefore have a recognized split value, and all three partitions must be non-empty. This prevents an incomplete or misspelled split column from silently changing a reported experiment into a random-split experiment.

The reported workbooks already contain the propagated and frequency-filtered labels. These are dataset-construction steps and are not inferred automatically by the training program. With `--go-space filtered`, TriGO restricts a full GO embedding bundle to terms present in the supplied annotation workbook.
