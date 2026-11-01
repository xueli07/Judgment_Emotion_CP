# Reproducibility record

The repository contains the version information that can be verified from the manuscript and retained materials. It deliberately does not invent a dependency lock file for historical analyses.

Exp3 and Exp6 numerical-reference-table exports were rerun in local R 4.5.0, with `sessionInfo()` saved in their reference-output folders. Before public release, run the remaining R scripts in a clean installation and save their session information. Add an `renv.lock` file only after confirming that it reproduces the final analyses. The retained HDDM information is Python 3.9.13 with HDDM 1.0.1; create an environment file only from a confirmed working environment. The available MATLAB executable reports version 23.2.0.2344743 (R2023b), while the exact original Exp6 R/MATLAB and Stat4CI versions remain unverified.
