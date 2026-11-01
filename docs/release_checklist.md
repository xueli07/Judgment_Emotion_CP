# GitHub release checklist

This repository is prepared locally and must not be published until every applicable item below is complete.

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
- [ ] Replace legacy acquisition terminology inside generated task files only if doing so does not break PsychoPy references.
- [x] Regenerate `MANIFEST.csv` after the current release edits.
- [ ] Create the GitHub repository under the chosen account or organization and confirm its visibility.
- [ ] Add the final GitHub URL and article DOI to `CITATION.cff` after they exist.
- [ ] Create a versioned GitHub Release when the repository is made public.
