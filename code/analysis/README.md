# Analysis code

Analysis folders follow the revised manuscript order. Run scripts from the repository root so their repository-relative paths resolve correctly.

- `exp01` and `exp02`: `01_data_preprocessing.R` is a public-directory adaptation of the corresponding original Step1 workflow. It reads the de-identified participant files and public inclusion table, writes the analysis-ready trial table used by scripts `02` through `05`, and retains the original target-column selection and RT rules. The Exp1 public raw exports do not contain age or gender; the script therefore preserves those fields from the supplied processed table, where they were retained from the prior public package and are not used by the analyses. Validation reproduced the Exp1 core analysis data exactly and the Exp2 core data except for two machine-precision numeric display differences (approximately 1e-16).
- `exp03` to `exp05`: model-comparison and psychometric analyses.
- `exp06`: reverse-correlation analyses in R and MATLAB.
- `hddm`: combined Exp1–Exp2 hierarchical drift diffusion modeling notebooks, plotting scripts, and Figure 4 export helpers.
- `figure_generation`: illustration and figure-generation helpers.

Full end-to-end execution of all analyses in a clean environment has not been completed. See `environment/REPRODUCIBILITY.md` for the reproducibility record and known environment limitations.
