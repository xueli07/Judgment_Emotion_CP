# Experiment 6 reference outputs

`Infoval-results.csv` contains results of the informational-value comparison. `filtered_by_participant.csv` and `nofiltered_by_participant.csv` contain trial-level analytical variables and participant fields.

The cache export regenerates ten numerical tables covering group CI similarity, PCA and its association with intensity, the permutation test, and jackknife and mixed-model summaries. From the repository root:

```sh
Rscript --vanilla code/analysis/exp06/02_export_reference_outputs_from_cache.R
```

| Input or output | Default local path | Override |
| --- | --- | --- |
| CI cache | `results/exp06/ci_cache_all.RData` | `EXP06_CACHE_PATH` |
| Face mask | `code/analysis/exp06/input_pic/average_face_mask.png` | `EXP06_MASK_PATH` |
| Output folder | `results/exp06/cache_export/` | `EXP06_OUTPUT_DIR` |

Supply the matching local input files before running the export. The cache is stored separately from the Git-tracked files.

R packages: `dplyr`, `readr`, `purrr`, `lme4`, `lmerTest`, `combinat`, and `png`. The export was tested with R 4.5.0 on Windows. See [session information](sessionInfo_exp06_export.txt) for recorded package versions.

The cache is not distributed; the main script generates it from the included noise RData. Required export objects: ci_cat, ci_cont, ci_levels_list, cor_res_loo. Six intensity levels are assumed; the main analysis additionally requires participant CI objects.
