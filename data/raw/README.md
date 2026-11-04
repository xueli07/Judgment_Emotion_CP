# De-identified raw data

Each `exp01` to `exp06` directory contains one public CSV per participant, a participant-inclusion table, and a raw-column dictionary. Public participant identifiers use the revised manuscript order (`E01` to `E06`). Pilot sessions and software tests are not included.

The raw exports were de-identified before this release candidate was built. The `participant` field is replaced with the public ID in each filename. Original participant identifiers, exact collection timestamps, free-text notes, original filenames, and the private identifier key are not included. Age, gender, and handedness fields are retained where they were present in the prior public package. Stimulus-path labels remain as task metadata, but the corresponding face images are withheld.

Raw columns containing `training` or `identification` retain the names used during data acquisition. In Exp1 and Exp2, these refer to the judgment phase described in the manuscript, not an additional training intervention. The names are preserved to keep the raw records traceable to the acquisition workflow. Refer to each experiment's raw-column dictionary for individual field definitions. Analysis folders and participant IDs follow the current manuscript's Exp1–Exp6 order.
