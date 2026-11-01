# GitHub release checklist

This repository has been uploaded to https://github.com/xueli07/Judgment_Emotion_CP and is currently private. Complete or explicitly document the outcome of each applicable check before public release. Previously completed checks below are retained from the preparation record; this documentation update does not constitute a new validation of the data or code.

- [x] Rebuild the package around the revised manuscript's Exp1–Exp6 order.
- [x] Prepare GitHub-ready repository documentation and remove obsolete repository links.
- [x] Restore and validate public-directory Exp1-Exp2 preprocessing scripts that regenerate the analysis inputs for `02` through `05` from the de-identified raw files and inclusion tables.
- [x] Record verified AsPredicted registrations for Exp3–Exp5 and the public ethics statement.
- [x] Include a CC0 1.0 license consistent with the project's prior public-release standard.
- [x] Exclude TFED originals, morphs, average faces, noise faces, and face-derived classification images.
- [x] Confirm that no file exceeds GitHub's ordinary Git size limit.
- [ ] Complete a final human review for names, free text, dates, screenshots, and participant identifiers.
- [x] Verify the figure-to-data and figure-to-code index; complete processed trial-level tables are supplied instead of duplicative panel-level tables.
- [x] Export Exp3 AIC, PSE, and slope numerical reference tables from released trial-level data.
- [x] Export Exp6 PCA, permutation, jackknife, and LME numerical reference tables from locally authorized cache assets; do not release the cache or image assets.
- [x] Verify that `Infovalpost-results.csv` duplicates `Infoval-results.csv` and remove the duplicate.
- [ ] Run the R analyses in a clean environment and save `sessionInfo()` output.
- [ ] Verify the exact MATLAB and Stat4CI versions used for Experiment 6, if recoverable. The local MATLAB executable reports R2023b; the bundled Stat4CI source has no version tag.
- [ ] Verify the original Exp6 R version and record all package versions that can be recovered. Release-table exports record their current R 4.5.0 sessions.
- [ ] Test the PsychoPy programs using authorized local face assets, if a runnable task package is needed.
- [x] Retain legacy acquisition terminology where required for PsychoPy compatibility and explain its correspondence to the manuscript terminology in the root README.
- [x] Regenerate `MANIFEST.csv` from the current repository files after the documentation edits. The manifest excludes itself, Git metadata, and macOS metadata files.
- [x] Create the GitHub repository under xueli07 and upload the main branch; current visibility is Private.
- [ ] Finalize the public access arrangements for manuscript submission.
- [x] Update the manuscript title and add the GitHub URL to `CITATION.cff`.
- [ ] Add article citation details and DOI when available; an article DOI is not yet available.
- [ ] Create a versioned GitHub Release when the repository is made public.
