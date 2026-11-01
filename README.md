# Judgment Experience Abolishes Categorical Perception of Facial Expressions

This repository contains de-identified behavioral data, analysis code, PsychoPy task programs, and supporting documentation for six experiments reported in the associated manuscript.

## Study overview

Experiments 1 and 2 tested whether a repeated continuous judgment phase changes later categorical perception in an XAB discrimination task. The XAB data also support the hierarchical drift diffusion model (HDDM) analyses. Experiments 3 to 5 tested immediate effects of judgment format and the processing stage at which they arise. Experiment 6 used reverse correlation to compare perceptual representations under categorical and continuous judgments.

| Experiment | Main question | Final sample |
| --- | --- | --- |
| Exp1 | Continuous versus categorical judgment followed by XAB | 60 |
| Exp2 | Continuous judgment versus identity-matching control followed by XAB | 60 |
| Exp3 | Immediate effect of judgment format | 45 |
| Exp4 | Informed blocked versus uninformed intermixed judgments | 45 |
| Exp5 | Informed versus uninformed judgments with both formats intermixed | 45 |
| Exp6 | Reverse-correlation comparison of categorical and continuous judgments | 60 |

## Repository layout

- `data/raw/`: de-identified participant-level PsychoPy exports, inclusion tables, and column dictionaries.
- `data/processed/`: analysis-ready trial-level tables and the combined HDDM input.
- `code/analysis/`: R, MATLAB, Python, and HDDM analysis code.
- `code/experiments/`: PsychoPy Builder programs, generated Python files, and condition workbooks.
- `source_data/`: figure-to-data and figure-to-code index. Complete processed trial-level tables are supplied rather than duplicative panel-level tables.
- `reference_outputs/`: selected model, quality-control, and statistical outputs.
- `environment/`: recorded software versions and reproducibility notes.
- `docs/`: exclusions, experiment renumbering, stimuli restrictions, ethics/preregistration links, and release records.

## Reproducing the analyses

Start with the relevant README files in `data/`, `code/`, and `environment/`. Run analyses from the repository root so scripts can resolve repository-relative paths. Experiments 1 and 2 use the processed tables in `data/processed/exp01` and `data/processed/exp02`; the HDDM analysis uses `data/processed/hddm/xab_trial_level_exp01_exp02.csv`. HDDM model fitting is computationally intensive. Outstanding reproducibility checks are recorded in [docs/release_checklist.md](docs/release_checklist.md).

## Data protection and restricted materials

Participant files use public IDs. The key connecting public IDs to original identifiers is not included. The repository does not include TFED face images, morphed faces, average faces, noise faces, or face-derived classification images. Condition workbooks retain stimulus labels only, so the task code documents the procedures but cannot run end-to-end without authorized local stimulus assets.

Some acquisition-time raw columns and task-workbook filenames retain legacy `training` or `identification` names because PsychoPy programs reference those exact fields. For Experiments 1 and 2, these names refer to the judgment phase described in the manuscript. The original names are retained for compatibility with the acquisition programs; the accompanying documentation uses judgment terminology.

## License and citation

Original materials in this repository are released under [CC0 1.0](LICENSE). This waiver does not grant rights to excluded third-party face materials. If you use these data or code, please cite this repository as described in [CITATION.cff](CITATION.cff). Article citation details will be added when available.

## Repository status

This repository has been uploaded to GitHub and is currently private while it is prepared for manuscript submission. The remaining checks are listed in [docs/release_checklist.md](docs/release_checklist.md). Public access arrangements have not yet been finalized.
