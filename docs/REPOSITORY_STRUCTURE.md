# Repository structure and naming

The repository follows the Exp1–Exp6 order in the revised manuscript.

| Repository path | Contents |
|---|---|
| `data/raw/exp01` to `data/raw/exp06` | De-identified participant-level exports, inclusion tables, and raw-column dictionaries |
| `data/processed/exp01` to `data/processed/exp06` | Analysis-ready trial-level data |
| `data/processed/hddm` | Combined Exp1–Exp2 HDDM input |
| `code/analysis/exp01` to `code/analysis/exp06` | Experiment-specific analyses |
| `code/analysis/hddm` | Combined Exp1–Exp2 HDDM notebooks and figure-export scripts |
| `code/experiments/exp01` to `code/experiments/exp06` | PsychoPy programs and condition workbooks |
| `source_data` | Figure and supplementary-figure source-data index |
| `reference_outputs` | Selected tables and diagnostic outputs used to check reported results |
| `environment` | Software versions and environment notes |
| `docs` | Release documentation and provenance |

Legacy study-number directories from the earlier manuscript are not used in the public repository. Acquisition-time filenames containing `training` remain only where PsychoPy task files reference them directly.
