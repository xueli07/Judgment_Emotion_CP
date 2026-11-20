# Analysis code

Run R commands from the repository root. The MATLAB and HDDM sections specify their own working directories.

## Quick demo

Install the three R packages used in the Exp3 demo. The export takes about three seconds on Windows 11 with R 4.5.0 and the dependencies installed.

```sh
Rscript -e "install.packages(c('dplyr', 'readr', 'tidyr'), repos='https://cloud.r-project.org')"
```

In PowerShell:

```powershell
$env:EXP03_OUTPUT_DIR = 'results/exp03/demo'
Rscript code/analysis/exp03/03_export_reference_outputs.R
```

In a POSIX shell:

```sh
EXP03_OUTPUT_DIR=results/exp03/demo Rscript code/analysis/exp03/03_export_reference_outputs.R
```

The demo reads `data/processed/exp03/exp03_trial_data.csv` and writes four CSVs containing AIC comparisons, best models, participant PSEs and slopes, and paired tests. It also saves R session information. Compare the CSVs with the [reference tables](../../reference_outputs/exp03/README.md).

## Experiments 1 and 2

With the supplied trial tables under `data/processed/`, run scripts `02` through `05` in order for each experiment: judgment fitting, quality control, XAB preparation, and XAB frequentist analysis. For example, start Experiment 1 with `Rscript code/analysis/exp01/02_judgment_model_fitting.R`. Outputs and intermediate inputs are under `results/exp01/` and `results/exp02/`.

The `01_data_preprocessing.R` scripts document preprocessing from the original participant exports and inclusion tables. For reproduction with the supplied trial tables, start at script `02`.

R packages include `tidyverse`, `readr`, `dplyr`, `purrr`, `minpack.lm`, `broom`, `scales`, `Rmisc`, `conflicted`, `car`, `rstatix`, and `ez`.

## Experiments 3 to 5

Run `01_model_comparison.R` and `02_psychometric_analysis.R` in each experiment folder via `Rscript`. Inputs are under `data/processed/exp03/` through `data/processed/exp05/`. Outputs go to the corresponding experiment folder under `results/`.

The helper_functions.R files are included for Exp3–Exp5 and require plyr.

In addition to tidyverse packages, the scripts use `readxl`, `reshape2`, `gridExtra`, `RColorBrewer`, `quickpsy`, `bruceR`, `sjPlot`, `psych`, `lme4`, `devtools`, `car`, `ggpubr`, `effectsize`, and `rstatix`. See the individual scripts for their imports.

## Experiment 6

* Run `02_export_reference_outputs_from_cache.R` with R to regenerate ten tables from a CI cache and its matching face mask. See the [Exp6 command, paths, and packages](../../reference_outputs/exp06/README.md).

* Run `Rscript code/analysis/exp06/01_reverse_correlation_analysis.R` for the reconstruction workflow. It reads `data/processed/exp06/exp06_trial_data.csv` and requires `code/analysis/exp06/input_pic/exp06_noise_seed.RData` and the matching face mask. The optional rendering image is `code/analysis/exp06/input_pic/average_face.jpg`. Outputs go under `results/exp06/`. R packages include `rcicr`, `dplyr`, `purrr`, `ggplot2`, `lme4`, `lmerTest`, `corrplot`, `combinat`, `imager`, `png`, and `jpeg`.

* For the MATLAB cluster script, set the working directory to `code/analysis/exp06` and run `exp06_cluster_test`. It uses Image Processing Toolbox and Stat4CI functions under `subfunction/`. Inputs are the participant CI PNGs in `category/` and `continuous/` and the mask and background images in `input_pic/`. Outputs go to `result/` and `outputfig/` under the MATLAB working directory.

## HDDM

The original fitting environment was the [dockerHDDM image](https://hub.docker.com/r/hcp4715/hddm) `hcp4715/hddm:1.0.1`.

Bind-mount the **repository root**, the directory containing `README.md`, `code/`, and `data/`. Docker maps this directory to `/home/jovyan/work` in the container. The mount is writable so files written to `/home/jovyan/work/results/` are saved under `results/` in your local checkout.

Run this command in PowerShell. Replace `YOUR_PATH` with the absolute path to the repository root on your computer.

```powershell
docker run -it --rm -p 127.0.0.1:8888:8888 -v "YOUR_PATH:/home/jovyan/work" hcp4715/hddm:1.0.1
```

Open the Jupyter URL shown in the terminal and navigate to `work/code/analysis/hddm/`. In notebook `0`, complete setup through section 17, then run section 24 onward for the full model fit. Sections 22–23 provide an optional subset fit. The full fit writes to `results/hddm/formal_models/`, which notebooks `1`–`4` use for diagnostics, posterior summaries and posterior predictive checks.

See the [runtime table](../../README.md#outputs-and-runtime) for HDDM settings and estimated runtimes.

For a quick container check, run notebook `0` through section 20. Sections 19–20 fit and inspect one model using 100 samples and 10 burn-in iterations. Outputs go to `results/hddm/test_model/`.

To regenerate Figure 4 from existing tables in `reference_outputs/hddm/tables/`, use R with `ggplot2`, `dplyr`, `tidyr`, `readr`, `scales`, and Cairo support for PDF output. Run both commands from the repository root. Outputs go under `results/figure4/`.

```sh
Rscript code/analysis/hddm/figure4_exports_2026-09/scripts/export_fig4_panels.R
Rscript code/analysis/hddm/figure4_exports_2026-09/scripts/export_fig4_rt_density.R
```

## Execution details and dependencies

Dependencies are listed below by experiment.

- Exp1: `Rmisc`, `bayesplot`, `brms`, `broom`, `car`, `conflicted`, `dplyr`, `effectsize`, `ez`, `ggplot2`, `gridExtra`, `minpack.lm`, `plyr`, `purrr`, `readr`, `rstan`, `rstatix`, `scales`, `tidyr`, `tidyverse`.
- Exp2: `Rmisc`, `bayesplot`, `brms`, `broom`, `car`, `conflicted`, `dplyr`, `effectsize`, `ez`, `ggplot2`, `gridExtra`, `minpack.lm`, `plyr`, `purrr`, `readr`, `rstan`, `rstatix`, `scales`, `tidyr`, `tidyverse`.
- Exp3: `RColorBrewer`, `bruceR`, `dplyr`, `ggplot2`, `gridExtra`, `lsr`, `plyr`, `purrr`, `quickpsy`, `readr`, `readxl`, `reshape2`, `tidyr`.
- Exp4: `RColorBrewer`, `broom`, `bruceR`, `car`, `devtools`, `dplyr`, `effectsize`, `ggplot2`, `ggpubr`, `gridExtra`, `lmSupport`, `lme4`, `plyr`, `psych`, `purrr`, `readxl`, `reshape2`, `rstatix`, `sjPlot`, `stats`, `tibble`, `tidyverse`.
- Exp5: `base`, `bruceR`, `dplyr`, `ggplot2`, `plyr`, `rstatix`, `stats`, `tibble`.
- Exp6: `bruceR`, `combinat`, `corrplot`, `dplyr`, `ggplot2`, `grid`, `imager`, `jpeg`, `lme4`, `lmerTest`, `png`, `purrr`, `rcicr`, `readr`.

### Experiment 6 handoff and methods

The main R script generates `results/exp06/ci_cache_all.RData` from the included noise resource. Run the cache exporter after that file is available.

The cache exporter uses `ci_cat`, `ci_cont`, `ci_levels_list` and `cor_res_loo` across six intensity levels. See the [Exp6 reference-output guide](../../reference_outputs/exp06/README.md) for default paths and environment variables.

To analyze newly reconstructed CIs in MATLAB, group the participant images by judgment format or update the input folder settings. Match the image orientation, scaling and mask to the MATLAB inputs.

The MATLAB spatial analysis uses Stat4CI random-field-theory cluster-level correction. The R trend test across six intensity levels uses permutation testing.

## Using the code with your own data

Use the supplied CSV headers and the [acquisition-column dictionaries](../../docs/data_dictionaries/README.md) as format references. Set input and output paths for your study and adapt the coding, units, pairing and model settings as described below.

| Module | Input and required adaptations |
| --- | --- |
| Exp1–2 | `data/processed/exp01/exp01_trial_data.csv` and `exp02/exp02_trial_data.csv`; Exp1 judgment_condition = categorical/continuous; Exp2 = control/continuous. Preserve judgment/XAB linkage and participant IDs. Change script input paths; redirect results and synchronize downstream paths. |
| Exp3 | `data/processed/exp03/exp03_trial_data.csv`: participant, block_name, face_stim, emotion_level, resp. block_name = categorical/continuous; resp in 0–1, binary for categorical trials. Pair conditions by participant. The demo uses (emotion_level - 50) / 10: adapt this transform to a different scale. Edit data_path for new input; EXP03_OUTPUT_DIR changes output only. |
| Exp4–5 | Respective processed CSVs, including thinking = categorical/continuous and task_awareness = Informed/Uninformed. Preserve participant IDs and within-participant format comparisons; retain the scripts' awareness/design grouping rather than treating all rows as independent. Inspect each script's grouping and model formula before changing the design. |
| Exp6 | `data/processed/exp06/exp06_trial_data.csv`: participant, ExpCondition, num, Resp, Raw_chosen and other supplied fields. ExpCondition = category/continuous; Resp = −1/+1. num must correspond one-to-one to the original noise parameters. Supply your own matching noise resource, not this study's parameters for unrelated stimuli. Match mask, image dimensions/orientation and intensity bins. Edit main-script input/resource/results paths. Cache overrides and six-level assumptions are described above. |
| HDDM | `data/processed/hddm/xab_trial_level_exp01_exp02.csv`: judgment_condition, pair_type (between/within), xab_resp (1 correct, 0 error), xab_rt (seconds), subj_idx (unique consistent participant ID). In notebook 0 User settings change INPUT_PATH and OUTPUT_DIR; synchronize later notebook inputs. Inspect combined CSV and code condition mapping rather than copying Exp2 control labels unchecked. Default RT range is 0.20–3 seconds; adapt for your study. |

See [reproduction notes](../../environment/REPRODUCIBILITY.md) and [materials](../../docs/stimulus_availability.md).
