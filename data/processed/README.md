# Processed data

This directory contains analysis-ready trial-level data for Exp1–Exp6 and one combined HDDM input table for Exp1–Exp2.

- `exp01` to `exp06`: one trial-level CSV per experiment.
- `hddm/xab_trial_level_exp01_exp02.csv`: combined Exp1–Exp2 XAB table used for HDDM.

For Exp1 and Exp2, processed fields use revised terminology: `judgment_condition`, `emotion_level`, `face_identity`, `face_gender`, `judgment_response`, `judgment_rt`, and `xab_*`. Blank cells in task-specific columns mean that the field is not applicable to that trial.

The Exp1 and Exp2 preprocessing scripts recreate the analytical trial variables from the anonymous raw exports. Participant demographic fields retained in the released processed tables are not used by the released analyses.
