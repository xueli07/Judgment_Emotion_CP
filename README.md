# Judgment Experience Abolishes Categorical Perception of Facial Expressions

This repository contains de-identified behavioral data, analysis code, and supporting documentation for six experiments reported in the associated manuscript.

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
- `source_data/`: figure-to-data and figure-to-code index. Complete processed trial-level tables are supplied rather than duplicative panel-level tables.
- `reference_outputs/`: selected model, quality-control, and statistical outputs.
- `environment/`: recorded software versions and reproducibility notes.
- `docs/`: participant and trial exclusions, stimulus sources, and sharing restrictions.

## Reproducing the analyses

Start with the relevant README files in `data/`, `code/`, and `environment/`. Run analyses from the repository root so scripts can resolve repository-relative paths. Experiments 1 and 2 use the processed tables in `data/processed/exp01` and `data/processed/exp02`; the HDDM analysis uses `data/processed/hddm/xab_trial_level_exp01_exp02.csv`. HDDM model fitting is computationally intensive. Full end-to-end execution of all analyses in a clean environment has not been completed. See [environment/REPRODUCIBILITY.md](environment/REPRODUCIBILITY.md) for the reproducibility record and known environment limitations.

## Data protection and restricted materials

Participant files use public IDs. The key connecting public IDs to original identifiers is not included. The repository does not include TFED face images, morphed faces, average faces, noise faces, or face-derived classification images. Raw data retain stimulus filename and path labels as metadata. Experiment 6 image reconstruction requires authorized face and noise assets that are not included; analyses based on the released numerical tables remain available.

Some acquisition-time raw columns retain legacy `training` or `identification` names. In Experiments 1 and 2, these refer to the judgment phase described in the manuscript. See [data/raw/README.md](data/raw/README.md) and the experiment-specific column dictionaries for their interpretation.

## Experimental task programs

Experimental task programs are available from the authors upon request at 242312056@njnu.edu.cn. Facial stimuli are subject to the original database's access and redistribution conditions.

## License and citation

Original data, numerical results, documentation, figures, analysis scripts, and code notebooks in this repository are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0), except where separate third-party terms apply. Third-party materials retain their own terms; no rights to excluded face materials are granted. See [LICENSE](LICENSE) for scope and exceptions.

If you use these data or code, please cite this repository as described in [CITATION.cff](CITATION.cff). Article citation details will be added when available.

