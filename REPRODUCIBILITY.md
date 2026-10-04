# Reproducibility notes

The files in `configs/` record the principal TriGO settings used for the three reported datasets:

| Dataset | Refinement repeats | Residual initialization | Configuration |
| --- | ---: | ---: | --- |
| CAFA3 | 8 | 0.5 | `configs/train_cafa3.sh` |
| GOA | 2 | 0.4 | `configs/train_goa.sh` |
| CAFA5 | 6 | 0.2 | `configs/train_cafa5.sh` |

Run a configuration from the repository root:

```bash
DEVICE=cuda:0 DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/outputs \
  ./configs/train_cafa3.sh
```

Each run stores its resolved arguments and command in `run_config.json`, the data partition in `data_split.json`, the validation-selected checkpoint in `best_model.pt`, and test predictions and metrics in the output directory.

All three reported-dataset configurations enable `--require-split`. Training terminates if any record has a missing or unrecognized split value or if a train, validation, or test partition is empty. The test set is evaluated only after the validation-selected checkpoint has been restored.

Exact floating-point results can vary with the GPU, CUDA, PyTorch, AMP, and driver versions. Use the same data partitions and cached ESM representations when comparing architectural variants. The five component variants are defined in `ABLATIONS.md`; change only the intended component while preserving the corresponding dataset configuration.

The repository does not include dataset files, ESM weights, cached protein features, trained checkpoints, or paper output directories.
