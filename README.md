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

## System requirements

Author-confirmed environment: Windows 11, R 4.5.0, MATLAB R2023b with Image Processing Toolbox for Exp6, Python 3.9.13 and HDDM 1.0.1 in `hcp4715/hddm:1.0.1`. No explicit GPU requirement or verified minimum RAM is specified. See [environment](environment/README.md).

## Installation

Install the module-specific dependencies in the [analysis guide](code/analysis/README.md). For the demo, use `Rscript -e "install.packages(c('dplyr','readr','tidyr'), repos='https://cloud.r-project.org')"`. Allow approximately one hour for initial R dependencies; download and compilation times vary. PsychoPy is acquisition software and is unnecessary for CSV analyses.

## Quick demo

From the repository root, in PowerShell:

```powershell
$env:EXP03_OUTPUT_DIR = 'results/exp03/demo'
Rscript --vanilla code/analysis/exp03/03_export_reference_outputs.R
```

POSIX:

```sh
EXP03_OUTPUT_DIR=results/exp03/demo Rscript --vanilla code/analysis/exp03/03_export_reference_outputs.R
```

Input: `data/processed/exp03/exp03_trial_data.csv`. Outputs in `results/exp03/demo/`: `AIC_model_comparison.csv`, `AIC_best_model_by_condition.csv`, `psychometric_parameters_by_participant.csv`, `PSE_slope_paired_tests.csv`, `sessionInfo_exp03.txt`. Compare with [reference outputs](reference_outputs/exp03/README.md). This table-export demo takes approximately three seconds with dependencies installed; it is not the full Exp3 workflow.

## Reproducing the manuscript analyses

Run R from the repository root. For Exp1–2 run scripts 02–05 in each experiment directory in order: judgment fitting, QC, XAB preparation, frequentist analysis. Optional 01_data_preprocessing.R regenerates processed CSVs and overwrites them by default. OUTPUT_PATH and QC_DIR redirect output; Exp1 DEMOGRAPHICS_PATH supplies demographic fields. Synchronize later readers if paths change.

For Exp3–5 run `01_model_comparison.R` then `02_psychometric_analysis.R` in each experiment's directory path, using `Rscript --vanilla`. Included helpers require plyr. For Exp6 run `Rscript --vanilla code/analysis/exp06/01_reverse_correlation_analysis.R`. It uses the processed trials, distributed noise RData and matching mask; the average JPG supports PC1 visualization. The main script generates its cache under `results/exp06/`; the cache is not bundled. Optional table export: `Rscript --vanilla code/analysis/exp06/02_export_reference_outputs_from_cache.R`.

For MATLAB change working directory to `code/analysis/exp06/` and run `exp06_cluster_test`. It reads supplied category/continuous CI PNGs, mask and backgrounds, and writes local `result/` and `outputfig/`, distinct from root `results/`. Newly reconstructed R CIs require a manual grouped-image handoff with matching orientation, scaling and mask. Stat4CI uses random-field-theory cluster-level correction; the R six-intensity trend test uses permutation testing. The MATLAB method wording in the submitted manuscript will be corrected in revision.

For HDDM mount the repository root at `/home/jovyan/work`:

```sh
docker run -it --rm -p 127.0.0.1:8888:8888 -v "YOUR_PATH:/home/jovyan/work" hcp4715/hddm:1.0.1
```

Open notebooks in `work/code/analysis/hddm/`, run notebook 0 then 1–4. Formal fitting writes `results/hddm/formal_models/`, matching subsequent readers; test_model remains separate. Formal settings: four chains, 15,000 iterations, burn-in 5,000, thin 2. Quick check: 100 samples, burn-in 10, not for scientific conclusions.

Figure 4 can be exported without HDDM refitting:

```sh
Rscript --vanilla code/analysis/hddm/figure4_exports_2026-09/scripts/export_fig4_panels.R
Rscript --vanilla code/analysis/hddm/figure4_exports_2026-09/scripts/export_fig4_rt_density.R
```

These use existing reference tables and trial data and write `results/figure4/`. Full commands, dependencies and adaptation guidance: [analysis guide](code/analysis/README.md).

## Using the code with your own data

Follow the independent [own-data guide](code/analysis/README.md#using-the-code-with-your-own-data), using the supplied processed CSVs and raw dictionaries as schemas. Change script input paths and redirect output; preserve participant pairing, response coding, missing values and units. The Exp3 demo input requires editing data_path (EXP03_OUTPUT_DIR only changes output). Its intensity transform is (emotion_level - 50) / 10 and must match your scale. HDDM INPUT_PATH/OUTPUT_DIR are in notebook 0; update later readers too. Exp6 requires your matching noise parameters and aligned images/mask. Review model assumptions, thresholds, condition reference levels and exclusions for the new study; arbitrary CSVs are not interchangeable.

## Outputs and runtime

`results/` is normally generated by execution. Exp3 demo: approximately three seconds excluding installation. Exp6 CI reconstruction: approximately four hours (author-provided); the additional 10,000-iteration informational-value runtime is unrecorded. HDDM: approximately 6–8 hours per formal model on AMD Ryzen 9 7945HX, 32 logical processors (reference configuration, not a minimum). Runtime varies with hardware and settings. Other workflow runtimes are not recorded.

## Verification scope

Records support Exp3 demo export, Exp6 table export from a compatible cache, and both Figure 4 exporters. The author confirmed the two HDDM .nc loading calls in Docker. These checks do not establish a complete clean-environment rerun. See [verification record](environment/REPRODUCIBILITY.md).

## Data/material availability

Selected Experiment 6 resources are included to support reproducibility: the average face, facial mask, visualization backgrounds, participant-level classification images, and an RData file containing the base image and parameters used to reconstruct the original noise patterns.

Average-face permission has been confirmed. `code/analysis/exp06/input_pic/` contains `average_face.jpg`, `average_face_mask.png`, `thinkstyle1_autoscaled.png`, `thinkstyle2_autoscaled.png` (all four images are 512 × 512), and `exp06_noise_seed.RData`. The `category/` and `continuous/` folders each contain 30 participant CI PNGs. CI PNGs are derived classification images, not original stimulus images.

The shared RData contains `base_faces$chinese` (512 × 512), `base_face_files$chinese` (`average_face.jpg`), `stimuli_params$chinese` (600 × 4092), `n_trials = 600`, `seed = 1`, and `noise_type = sinusoid`. It was renamed from `face_seed_1_time_8月_28_2025_20_13_副本.Rdata` without changing its objects. The JPG cannot replace this noise resource. Object availability alone does not establish successful reconstruction of the manuscript results.

Individual noise-added stimulus face images, other original facial stimulus images, and morphed facial stimulus images are not distributed. Third-party materials and their derivatives remain subject to the applicable source permissions and license terms. Their inclusion does not grant additional rights or automatically license them under CC BY 4.0.

Participant files use public IDs; the identification key is excluded. Legacy training/identification labels in Exp1–2 refer to judgment phases; see [raw-data documentation](data/raw/README.md).

## Experimental task programs

Experimental task programs are available upon request at 242312056@njnu.edu.cn. Acquisition used PsychoPy 2024.2.4 for Exp1, Exp2 and Exp6, and 2023.2.3 for Exp3–5.

## License and citation

Analysis code and original materials for which the authors hold rights are licensed under [CC BY 4.0](LICENSE-CC-BY-4.0), subject to [third-party exceptions](LICENSE). Cite through [CITATION.cff](CITATION.cff). Article citation details will be added when available.
