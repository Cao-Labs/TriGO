# Component ablations

The public command-line names follow the five component variants reported for TriGO. Each flag changes only the stated component; all other dataset-specific settings should remain identical to the corresponding base configuration.

| Paper setting | Command-line flag | Implemented change |
| --- | --- | --- |
| w/o modality-specific skip paths | `--ablate-modality-specific-skip-paths` | Removes the separately projected sequence and structure skip representations and their associated routing paths. |
| w/o HMR | `--ablate-hmr` | Replaces the shared expert and hierarchical modality routing with `0.5 * z_seq + 0.5 * z_str`. |
| w/o GO-semantic branch | `--ablate-go-semantic-branch` | Removes GO-context retrieval and the GO-modulated branch pathway while retaining GO embeddings in the similarity predictor. |
| w/o GO-semantic modulation | `--ablate-go-semantic-modulation` | Retains the retrieved GO context but replaces feature-wise scale-and-shift modulation with direct protein-context concatenation. |
| w/o namespace-specific pathways | `--ablate-namespace-specific-pathways` | Replaces the separate BP, MF, and CC retrievers, experts, and GO-similarity predictors with shared counterparts. |

The component study uses the jointly trained base model without cross-namespace refinement. Set `--cross-namespace-refinement-repeats 0` for both the base model and every component variant.

For example, run the CAFA3 base model and the five matched variants as follows. A distinct output directory prevents one run from overwriting another.

```bash
./configs/train_cafa3.sh \
  --cross-namespace-refinement-repeats 0 \
  --output-dir outputs/ablation_cafa3/base

./configs/train_cafa3.sh \
  --cross-namespace-refinement-repeats 0 \
  --ablate-modality-specific-skip-paths \
  --output-dir outputs/ablation_cafa3/wo_modality_specific_skip_paths

./configs/train_cafa3.sh \
  --cross-namespace-refinement-repeats 0 \
  --ablate-hmr \
  --output-dir outputs/ablation_cafa3/wo_hmr

./configs/train_cafa3.sh \
  --cross-namespace-refinement-repeats 0 \
  --ablate-go-semantic-branch \
  --output-dir outputs/ablation_cafa3/wo_go_semantic_branch

./configs/train_cafa3.sh \
  --cross-namespace-refinement-repeats 0 \
  --ablate-go-semantic-modulation \
  --output-dir outputs/ablation_cafa3/wo_go_semantic_modulation

./configs/train_cafa3.sh \
  --cross-namespace-refinement-repeats 0 \
  --ablate-namespace-specific-pathways \
  --output-dir outputs/ablation_cafa3/wo_namespace_specific_pathways
```

The same five flags can be appended to `train_goa.sh` and `train_cafa5.sh`. Do not combine component-ablation flags in a single run. Historical option names remain accepted as hidden aliases for backward compatibility, but new experiments should use the paper-facing names above.
