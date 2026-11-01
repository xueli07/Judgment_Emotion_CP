# Experiment 6 reference outputs

These numerical outputs support the informational-value and individual-level robustness analyses for the reverse-correlation experiment.

`filtered_by_participant.csv` and `nofiltered_by_participant.csv` retain pseudonymous participant IDs, participant demographic fields, and trial-level analytical variables, consistent with the prior public package.

`Infoval-results.csv` supports the informational-value comparison. The duplicate `Infovalpost-results.csv` was removed because it was byte-identical. `02_export_reference_outputs_from_cache.R` generated the PCA, permutation, LME, and jackknife numerical summaries from locally authorized restricted CI-cache and mask assets; neither restricted input is included here.
