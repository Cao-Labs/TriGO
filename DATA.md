# Data layout and input format

Data, pretrained weights, feature caches, checkpoints, and experiment outputs are intentionally excluded from this repository.

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

The paper datasets apply ontology propagation and frequency filtering before training. These preprocessing choices are dataset construction steps and are not inferred automatically by the training program. With `--go-space filtered`, TriGO restricts a full GO embedding bundle to terms present in the supplied annotation workbook.
