# Release-candidate validation

Validation completed on 22 September 2026 for the local GitHub release candidate.

- The source-data index contains 21 main and supplementary figure entries. Every referenced repository path exists.
- The raw-data release contains 328 participant files. The public files contain no age, gender, or handedness columns. The file-level transformation record is in `raw_data_privacy_transformation.csv`.
- All 15 PsychoPy Builder files parse as XML, all 5 HDDM notebooks parse as JSON, and all included Python files compile successfully.
- No repository file exceeds 50 MiB. Git Large File Storage is not required.
- The final file inventory and SHA-256 hashes are recorded in `MANIFEST.csv`.

The following checks require a confirmed analysis environment and remain documented in `release_checklist.md`: execution of the R analyses, capture of `sessionInfo()` output, recovery of the Exp6 R, MATLAB, and Stat4CI versions, and a final human review of the public release.
