# Processed data

This directory contains de-identified, preprocessed, analysis-ready trial-level data for Exp1–Exp6 and one combined HDDM input table for Exp1–Exp2.

- `exp01` to `exp06`: one trial-level CSV per experiment.
- `hddm/xab_trial_level_exp01_exp02.csv`: combined Exp1–Exp2 XAB table used for HDDM.

For Exp1 and Exp2, processed fields use revised terminology: `judgment_condition`, `emotion_level`, `face_identity`, `face_gender`, `judgment_response`, `judgment_rt`, and `xab_*`. Blank cells in task-specific columns mean that the field is not applicable to that trial.

Original participant exports and participant-inclusion tables are not distributed. The Exp1 and Exp2 preprocessing scripts remain as processing records but require separately held original inputs; public reproduction begins with script 02. Participant demographic fields retained in the released processed tables are not used by the released analyses.

Use the actual CSV headers and the [own-data guide](../../code/analysis/README.md#using-the-code-with-your-own-data) for input formats, response coding, units, missing values and participant pairing. See [exclusion criteria](../../docs/exclusion_criteria.md) for participant and trial filtering rules; downstream analyses may apply additional task-specific filters. Preserved [acquisition-column dictionaries](../../docs/data_dictionaries/README.md) describe historical acquisition fields, not a complete processed-data schema.
