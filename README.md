# Judgment Experience Abolishes Categorical Perception of Facial Expressions

This repository contains de-identified, preprocessed trial-level behavioral data, analysis code, and supporting documentation for six experiments reported in the associated manuscript.

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

- `data/processed/`: analysis-ready trial-level tables and the combined HDDM input.
- `code/analysis/`: R, MATLAB, Python, and HDDM analysis code.
- `source_data/`: index linking each figure to its data, code and reference outputs.
- `reference_outputs/`: selected model, quality-control, and statistical outputs.
- `environment/`: recorded software versions and reproducibility notes.
- `docs/`: participant and trial exclusions, stimulus sources, and sharing restrictions.

## System requirements

The analyses used Windows 11, R 4.5.0, MATLAB R2023b with Image Processing Toolbox for Exp6, and Python 3.9.13 with HDDM 1.0.1 in `hcp4715/hddm:1.0.1`. See the [software environment](environment/README.md).

## Installation

Install the three R packages used in the Exp3 demo:

```sh
Rscript -e "install.packages(c('dplyr', 'readr', 'tidyr'), repos='https://cloud.r-project.org')"
```

Allow approximately one hour for initial R dependency setup. See the [analysis guide](code/analysis/README.md) for dependencies by experiment.

## Quick demo

From the repository root, in PowerShell:

```powershell
$env:EXP03_OUTPUT_DIR = 'results/exp03/demo'
Rscript code/analysis/exp03/03_export_reference_outputs.R
```

POSIX:

```sh
EXP03_OUTPUT_DIR=results/exp03/demo Rscript code/analysis/exp03/03_export_reference_outputs.R
```

The demo reads `data/processed/exp03/exp03_trial_data.csv` and writes `AIC_model_comparison.csv`, `AIC_best_model_by_condition.csv`, `psychometric_parameters_by_participant.csv`, `PSE_slope_paired_tests.csv` and `sessionInfo_exp03.txt` to `results/exp03/demo/`. The export takes about three seconds with the dependencies installed. See the [reference outputs](reference_outputs/exp03/README.md).

## Reproducing the manuscript analyses

Run R from the repository root. For Exp1–2, use the supplied preprocessed data and run scripts `02`–`05` in order for judgment fitting, quality control, XAB preparation and frequentist analysis.

For Exp3–5, run `01_model_comparison.R` followed by `02_psychometric_analysis.R` in each experiment's directory with `Rscript`. For Exp6, run `Rscript code/analysis/exp06/01_reverse_correlation_analysis.R` to reconstruct the classification images and generate the cache under `results/exp06/`. The optional cache exporter is `code/analysis/exp06/02_export_reference_outputs_from_cache.R`.

In MATLAB, set the working directory to `code/analysis/exp06/` and run `exp06_cluster_test`. It reads the supplied participant CI images, mask and backgrounds and writes `result/` and `outputfig/` within that working directory. The submitted manuscript mislabeled the MATLAB spatial analysis as a cluster-based permutation test. The analysis used Stat4CI random-field-theory cluster-level correction.

For HDDM, replace `YOUR_PATH` with the absolute path to the repository root and mount it at `/home/jovyan/work`:

```sh
docker run -it --rm -p 127.0.0.1:8888:8888 -v "YOUR_PATH:/home/jovyan/work" hcp4715/hddm:1.0.1
```

Open the notebooks in `work/code/analysis/hddm/`. In notebook `0`, complete setup through section 17, then run section 24 onward for the full model fit. Sections 19–20 provide a quick check, and sections 22–23 provide an optional subset fit. Run notebooks `1`–`4` on the formal fits in `results/hddm/formal_models/`.

To export Figure 4 from the supplied reference tables and trial data:

```sh
Rscript code/analysis/hddm/figure4_exports_2026-09/scripts/export_fig4_panels.R
Rscript code/analysis/hddm/figure4_exports_2026-09/scripts/export_fig4_rt_density.R
```

These use existing reference tables and trial data and write `results/figure4/`. Full commands, dependencies and adaptation guidance: [analysis guide](code/analysis/README.md).

## Using the code with your own data

See the [own-data guide](code/analysis/README.md#using-the-code-with-your-own-data) for input fields, units, participant pairing and settings to adapt for your study.

## Outputs and runtime

`results/` is normally generated by execution.

| Experiment / workflow | Analysis step | Approximate runtime |
| --- | --- | --- |
| Exp1 | R: judgment-phase and XAB behavioral analyses | A few seconds |
| Exp2 | R: judgment-phase and XAB behavioral analyses | A few seconds |
| Exp3 | R: model comparison, psychometric fitting, and statistical analyses | A few seconds |
| Exp4 | R: model comparison, psychometric fitting, and statistical analyses | A few seconds |
| Exp5 | R: model comparison, psychometric fitting, and statistical analyses | A few seconds |
| Exp6 | R: classification-image reconstruction | Approximately 4 hours |
| Exp6 | R: subsequent representational analyses | Within approximately 5 minutes |
| Exp6 | MATLAB: spatial analysis | Within approximately 5 minutes |
| HDDM | Formal model fitting | Approximately 6–8 hours per model |
| HDDM | Posterior predictive simulation (PPC) | Approximately 1 hour |

Runtimes are approximate and vary with hardware, analysis settings and the resources allocated to Docker.

HDDM reference configuration: AMD Ryzen 9 7945HX with 32 logical processors. Each formal fit uses four chains, 15,000 iterations per chain, 5,000 burn-in iterations and a thinning interval of 2.

The Exp6 five-minute estimate covers representational analyses after reconstruction. The 10,000-iteration informational-value calculation is a separate step with no recorded runtime.

## Data/material availability

The Exp6 materials in `code/analysis/exp06/input_pic/` include the approved average face, mask, visualization backgrounds and `exp06_noise_seed.RData`. The average-face JPG supports visualization. The RData contains the base image and parameters for reconstructing the noise patterns. The `category/` and `continuous/` directories each contain 30 participant classification images.

The [stimulus availability notes](docs/stimulus_availability.md) describe the shared materials and their usage terms.

Behavioral data are shared as de-identified, preprocessed trial-level tables. See the [processed-data documentation](data/processed/README.md) and [column dictionaries](docs/data_dictionaries/README.md).

## Experimental task programs

Experimental task programs are available upon request at 242312056@njnu.edu.cn. Acquisition used PsychoPy 2024.2.4 for Exp1, Exp2 and Exp6, and 2023.2.3 for Exp3–5.

## License and citation

Analysis code and original materials for which the authors hold rights are licensed under [CC BY 4.0](LICENSE-CC-BY-4.0). See the [third-party terms](LICENSE) and [citation information](CITATION.cff).
