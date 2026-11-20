# Analysis code

Run R commands from the repository root. The MATLAB and HDDM sections specify their own working directories.

## Quick demo

Install R and the two demo packages. In the tested Windows environment, R 4.5.0 with dplyr 1.2.1 and readr 2.1.5 exported the demo tables in about 3 seconds with the packages already installed. Installation time depends on the connection and availability of binary packages.

```sh
Rscript -e "install.packages(c('dplyr', 'readr'), repos='https://cloud.r-project.org')"
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

Exp3 scripts 01 and 02 require the original helper_functions.R next to the scripts in code/analysis/exp03/. Exp4 script 02 requires the same helper next to it in code/analysis/exp04/. Supply the corresponding original helper files before running these scripts. The helper also requires plyr. The Exp3 quick demo runs independently of these helpers.

In addition to tidyverse packages, the scripts use `readxl`, `reshape2`, `gridExtra`, `RColorBrewer`, `quickpsy`, `bruceR`, `sjPlot`, `psych`, `lme4`, `devtools`, `car`, `ggpubr`, `effectsize`, and `rstatix`. See the individual scripts for their imports.

## Experiment 6

* Run `02_export_reference_outputs_from_cache.R` with R to regenerate ten tables from a CI cache and its matching face mask. See the [Exp6 command, paths, and packages](../../reference_outputs/exp06/README.md).

* Run `Rscript --vanilla code/analysis/exp06/01_reverse_correlation_analysis.R` for the reconstruction workflow. It reads `data/processed/exp06/exp06_trial_data.csv` and requires `restricted_assets/exp06_noise_seed.RData` and the matching face mask. The optional rendering image is `restricted_assets/exp06_average_face.jpg`. Outputs go under `results/exp06/`. R packages include `rcicr`, `dplyr`, `purrr`, `ggplot2`, `lme4`, `lmerTest`, `corrplot`, `combinat`, `imager`, `png`, and `jpeg`.

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

