# Reference outputs

This directory contains numerical analysis outputs used to verify the manuscript. It complements the analysis code and processed data; it is not a substitute for either.

## Contents

- `exp01` and `exp02`: psychometric-fitting and XAB ANOVA summaries. The full source ANOVA tables and post-hoc tests remain available in the analysis code workflow.
- `exp03`: no final numerical results table was saved by the original analysis workflow. The source scripts and the reported values in the manuscript identify the analyses that must be rerun before a complete release.
- `exp04` and `exp05`: complete AIC comparisons, participant-level psychometric parameters, and the original PSE and slope ANOVA reports. The reports have the `.html` extension because the analysis package generated HTML-formatted tables despite the original `.doc` filename.
- `exp06`: participant-level reverse-correlation analysis data and informational-value results, using the same demographic-field policy as the prior public package.
- `hddm`: model comparison, posterior summaries, convergence diagnostics, and posterior predictive-check outputs.

The duplicate `Infovalpost-results.csv` was removed from `exp06` after verification that it was byte-identical to `Infoval-results.csv`.
