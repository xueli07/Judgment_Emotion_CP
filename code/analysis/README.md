# Analysis code

Run R commands from the repository root. The MATLAB and HDDM sections specify their own working directories.

## Quick demo

Install R and the three demo packages. In the tested Windows environment, R 4.5.0 with dplyr 1.2.1 and readr 2.1.5 exported the demo tables in about 3 seconds with the packages already installed. Installation time depends on the connection and availability of binary packages.

```sh
Rscript -e "install.packages(c('dplyr', 'readr', 'tidyr'), repos='https://cloud.r-project.org')"
```

In PowerShell:

```powershell
$env:EXP03_OUTPUT_DIR = 'results/exp03/demo'
Rscript --vanilla code/analysis/exp03/03_export_reference_outputs.R
```

In a POSIX shell:

```sh
EXP03_OUTPUT_DIR=results/exp03/demo Rscript --vanilla code/analysis/exp03/03_export_reference_outputs.R
```

The demo reads `data/processed/exp03/exp03_trial_data.csv` and writes four CSVs containing AIC comparisons, best models, participant PSEs and slopes, and paired tests. It also saves R session information. Compare the CSVs with the [reference tables](../../reference_outputs/exp03/README.md).

## Experiments 1 and 2

With the supplied trial tables under `data/processed/`, run scripts `02` through `05` in order for each experiment: judgment fitting, quality control, XAB preparation, and XAB frequentist analysis. For example, start Experiment 1 with `Rscript --vanilla code/analysis/exp01/02_judgment_model_fitting.R`. Outputs and intermediate inputs are under `results/exp01/` and `results/exp02/`.

Run `01_data_preprocessing.R` when regenerating trial tables from the public raw exports and inclusion files. It overwrites the supplied processed CSV by default. Use `OUTPUT_PATH` and `QC_DIR` to direct the outputs elsewhere. Exp1 reads the supplied processed CSV to preserve age and gender fields absent from its public raw exports. Set `DEMOGRAPHICS_PATH` to supply a different source for those fields.

R packages include `tidyverse`, `readr`, `dplyr`, `purrr`, `minpack.lm`, `broom`, `scales`, `Rmisc`, `conflicted`, `car`, `rstatix`, and `ez`.

## Experiments 3 to 5

Run `01_model_comparison.R` and `02_psychometric_analysis.R` in each experiment folder via `Rscript --vanilla`. Inputs are under `data/processed/exp03/` through `data/processed/exp05/`. Outputs go to the corresponding experiment folder under `results/`.

The helper_functions.R files are included for Exp3–Exp5 and require plyr.

In addition to tidyverse packages, the scripts use `readxl`, `reshape2`, `gridExtra`, `RColorBrewer`, `quickpsy`, `bruceR`, `sjPlot`, `psych`, `lme4`, `devtools`, `car`, `ggpubr`, `effectsize`, and `rstatix`. See the individual scripts for their imports.

## Experiment 6

* Run `02_export_reference_outputs_from_cache.R` with R to regenerate ten tables from a CI cache and its matching face mask. See the [Exp6 command, paths, and packages](../../reference_outputs/exp06/README.md).

* Run `Rscript --vanilla code/analysis/exp06/01_reverse_correlation_analysis.R` for the reconstruction workflow. It reads `data/processed/exp06/exp06_trial_data.csv` and requires `code/analysis/exp06/input_pic/exp06_noise_seed.RData` and the matching face mask. The optional rendering image is `code/analysis/exp06/input_pic/average_face.jpg`. Outputs go under `results/exp06/`. R packages include `rcicr`, `dplyr`, `purrr`, `ggplot2`, `lme4`, `lmerTest`, `corrplot`, `combinat`, `imager`, `png`, and `jpeg`.

* For the MATLAB cluster script, set the working directory to `code/analysis/exp06` and run `exp06_cluster_test`. It uses Image Processing Toolbox and Stat4CI functions under `subfunction/`. Inputs are the participant CI PNGs in `category/` and `continuous/` and the mask and background images in `input_pic/`. Outputs go to `result/` and `outputfig/` under the MATLAB working directory.

## HDDM

The original fitting environment was the [dockerHDDM image](https://hub.docker.com/r/hcp4715/hddm) `hcp4715/hddm:1.0.1`.

Bind-mount the **repository root**, the directory containing `README.md`, `code/`, and `data/`. Docker maps this directory to `/home/jovyan/work` in the container. The mount is writable so files written to `/home/jovyan/work/results/` are saved under `results/` in your local checkout.

Run this command in PowerShell. Replace `YOUR_PATH` with the absolute path to the repository root on your computer.

```powershell
docker run -it --rm -p 127.0.0.1:8888:8888 -v "YOUR_PATH":/home/jovyan/work hcp4715/hddm:1.0.1 
```

Open the local Jupyter URL shown in the terminal. Open the notebooks inside `work/code/analysis/hddm/` so their kernels use that working directory. Notebook `0` reads `data/processed/hddm/xab_trial_level_exp01_exp02.csv` and writes model fits under `results/hddm/`. Notebooks `1` through `4` use fitted models under `results/hddm/formal_models/` and save diagnostics, posterior summaries and posterior predictive checks under `results/hddm/`.

**Estimated model-fitting time.** Allow approximately 6–8 hours per HDDM model on the reference Windows machine (AMD Ryzen 9 7945HX, 32 logical processors). The full fit uses four chains with 15,000 MCMC iterations per chain, including 5,000 burn-in iterations, and a thinning interval of 2. This planning estimate covers fitting one model. Actual runtime varies with model complexity and the CPU resources allocated to Docker.

For a quick container check, run notebook 0 through **19. Fit one test model first** and **20. Inspect posterior stats for the test model**, then stop before the model-subset runs. The test uses 100 samples, a burn-in of 10, and writes to results/hddm/test_model/. A successful check loads the trial CSV, completes the M4 test fit, and displays posterior statistics and DIC. See the [reproducibility record](../../environment/REPRODUCIBILITY.md) for verification status.

To regenerate Figure 4 from existing tables in `reference_outputs/hddm/tables/`, use R with `ggplot2`, `dplyr`, `tidyr`, `readr`, `scales`, and Cairo support for PDF output. Run both commands from the repository root. Outputs go under `results/figure4/`.

```sh
Rscript --vanilla code/analysis/hddm/figure4_exports_2026-09/scripts/export_fig4_panels.R
Rscript --vanilla code/analysis/hddm/figure4_exports_2026-09/scripts/export_fig4_rt_density.R
```


## Execution details and dependencies

Run R scripts from the repository root using `Rscript --vanilla <script-path>`. For Exp1 and Exp2, run `02_judgment_model_fitting.R`, `03_judgment_quality_control_descriptives.R`, `04_xab_data_preparation.R`, then `05_xab_frequentist_analysis.R` in their respective folders. Optional `01_data_preprocessing.R` runs first and overwrites processed CSVs by default; if OUTPUT_PATH changes, update downstream readers too. QC_DIR controls QC output; Exp1 DEMOGRAPHICS_PATH supplies demographic fields.

For Exp3–Exp5 run `01_model_comparison.R`, then `02_psychometric_analysis.R` from each experiment folder path. Inputs are `data/processed/exp0N/exp0N_trial_data.csv`; outputs are `results/exp0N/`. Dependencies below are scanned imports, not a single jointly validated environment. Install needed packages with `install.packages(c("package1", "package2"))`; consult source requirements for packages unavailable through your configured repository.


- Exp1: `Rmisc`, `bayesplot`, `brms`, `broom`, `car`, `conflicted`, `dplyr`, `effectsize`, `ez`, `ggplot2`, `gridExtra`, `minpack.lm`, `plyr`, `purrr`, `readr`, `rstan`, `rstatix`, `scales`, `tidyr`, `tidyverse`.
- Exp2: `Rmisc`, `bayesplot`, `brms`, `broom`, `car`, `conflicted`, `dplyr`, `effectsize`, `ez`, `ggplot2`, `gridExtra`, `minpack.lm`, `plyr`, `purrr`, `readr`, `rstan`, `rstatix`, `scales`, `tidyr`, `tidyverse`.
- Exp3: `RColorBrewer`, `bruceR`, `dplyr`, `ggplot2`, `gridExtra`, `lsr`, `plyr`, `purrr`, `quickpsy`, `readr`, `readxl`, `reshape2`, `tidyr`.
- Exp4: `RColorBrewer`, `broom`, `bruceR`, `car`, `devtools`, `dplyr`, `effectsize`, `ggplot2`, `ggpubr`, `gridExtra`, `lmSupport`, `lme4`, `plyr`, `psych`, `purrr`, `readxl`, `reshape2`, `rstatix`, `sjPlot`, `stats`, `tibble`, `tidyverse`.
- Exp5: `base`, `bruceR`, `dplyr`, `ggplot2`, `plyr`, `rstatix`, `stats`, `tibble`.
- Exp6: `bruceR`, `combinat`, `corrplot`, `dplyr`, `ggplot2`, `grid`, `imager`, `jpeg`, `lme4`, `lmerTest`, `png`, `purrr`, `rcicr`, `readr`.

### Experiment 6 handoff and methods

CI reconstruction takes approximately four hours (author-provided runtime). Informational value uses 10,000 iterations; its additional runtime was not recorded. Other full-workflow runtimes are not recorded. The main script writes `results/exp06/ci_cache_all.RData` when no cache exists. The cache is not distributed; the original noise RData is distributed and enables reconstruction. These are different resources.

The cache exporter requires `ci_cat`, `ci_cont`, `ci_levels_list`, and `cor_res_loo` and assumes six intensity levels. The main analysis additionally needs participant-level CI objects; an export-compatible cache need not satisfy the main analysis. Defaults: cache `results/exp06/ci_cache_all.RData`, mask `code/analysis/exp06/input_pic/average_face_mask.png`, output `results/exp06/cache_export/`. Override with EXP06_CACHE_PATH, EXP06_MASK_PATH and EXP06_OUTPUT_DIR. It exports ten numerical tables and sessionInfo.

MATLAB defaults to the supplied participant PNGs in `code/analysis/exp06/category/` and `continuous/`; these can be analyzed independently. R reconstruction outputs reside under `results/exp06/`. To use newly reconstructed CIs, correctly group the participant images for MATLAB or change its image-folder settings. Preserve image orientation, scaling, group assignment and mask alignment. This handoff is not automated.

MATLAB uses Stat4CI random-field-theory cluster-level correction, not cluster-based permutation testing. The R trend test across six intensity levels does use permutation testing. The submitted manuscript's MATLAB method description is to be corrected in revision; statistical methods and parameters are unchanged here.

## Using the code with your own data

Use the supplied CSV headers and experiment-specific dictionaries in `data/raw/` as format references. Preserve missing-value handling, response coding, units, participant pairing and exclusions used by the script, or deliberately adapt them to your design. Arbitrary CSVs cannot be used without adaptation. Change each script's input/data path and results directory to avoid overwriting supplied or prior outputs. Model forms, thresholds, condition reference levels, intensity scales and exclusion rules need review for a different study. This is format guidance, not validation on external researchers' data.

| Module | Input and required adaptations |
| --- | --- |
| Exp1–2 | `data/processed/exp01/exp01_trial_data.csv` and `exp02/exp02_trial_data.csv`; Exp1 judgment_condition = categorical/continuous; Exp2 = control/continuous. Preserve judgment/XAB linkage and participant IDs. Change script input paths; redirect results and synchronize downstream paths. |
| Exp3 | `data/processed/exp03/exp03_trial_data.csv`: participant, block_name, face_stim, emotion_level, resp. block_name = categorical/continuous; resp in 0–1, binary for categorical trials. Pair conditions by participant. The demo uses (emotion_level - 50) / 10: adapt this transform to a different scale. Edit data_path for new input; EXP03_OUTPUT_DIR changes output only. |
| Exp4–5 | Respective processed CSVs, including thinking = categorical/continuous and task_awareness = Informed/Uninformed. Preserve participant IDs and within-participant format comparisons; retain the scripts' awareness/design grouping rather than treating all rows as independent. Inspect each script's grouping and model formula before changing the design. |
| Exp6 | `data/processed/exp06/exp06_trial_data.csv`: participant, ExpCondition, num, Resp, Raw_chosen and other supplied fields. ExpCondition = category/continuous; Resp = −1/+1. num must correspond one-to-one to the original noise parameters. Supply your own matching noise resource, not this study's parameters for unrelated stimuli. Match mask, image dimensions/orientation and intensity bins. Edit main-script input/resource/results paths. Cache overrides and six-level assumptions are described above. |
| HDDM | `data/processed/hddm/xab_trial_level_exp01_exp02.csv`: judgment_condition, pair_type (between/within), xab_resp (1 correct, 0 error), xab_rt (seconds), subj_idx (unique consistent participant ID). In notebook 0 User settings change INPUT_PATH and OUTPUT_DIR; synchronize later notebook inputs. Inspect combined CSV and code condition mapping rather than copying Exp2 control labels unchecked. Default RT range is 0.20–3 seconds; adapt for your study. |

Formal model subsets and all-model fits write to `formal_models`; the independent `test_model` branch is a quick check, not scientific fitting. The 100-sample/10-burn-in check does not replace formal four-chain fitting.

The Exp3 demo writes `AIC_model_comparison.csv`, `AIC_best_model_by_condition.csv`, `psychometric_parameters_by_participant.csv`, `PSE_slope_paired_tests.csv`, and `sessionInfo_exp03.txt`. Its approximately three-second runtime excludes installation and covers table export only, not full Exp3 analysis or HDDM.

See [verification scope](../../environment/REPRODUCIBILITY.md) and [materials](../../docs/stimulus_availability.md).

### Filename case on case-sensitive systems

The current checkout contains `input_pic/exp06_noise_seed.Rdata`, while the R script requests `exp06_noise_seed.RData`. This capitalization differs. Before running on a case-sensitive filesystem, make the resource filename match the script exactly; this documentation update does not rename the resource or edit the code.
