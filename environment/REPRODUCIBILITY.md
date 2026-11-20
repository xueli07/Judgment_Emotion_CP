# Reproducibility record

The repository contains the version information that can be verified from the manuscript and retained materials. It deliberately does not invent a dependency lock file for historical analyses.

Exp3 and Exp6 numerical-reference-table exports were rerun in local R 4.5.0, with `sessionInfo()` saved in their reference-output folders. Before public release, run the remaining R scripts in a clean installation and save their session information. Add an `renv.lock` file only after confirming that it reproduces the final analyses. The original HDDM environment was dockerHDDM hcp4715/hddm:1.0.1 (Python 3.9.13 and HDDM 1.0.1). The notebook workflow has not been rerun end to end in this image after adapting the repository paths. The available MATLAB executable reports version 23.2.0.2344743 (R2023b), while the exact original Exp6 R/MATLAB and Stat4CI versions remain unverified.

The two Figure 4 exporters were run from the repository root against the supplied CSV files. The separate HDDM overlay script produced its plots, but its empirical ANOVA export reported an error in the local R environment. The Figure 1 illustration script requires Windows font support and a graphics device configured to use Arial. The Exp6 MATLAB cluster workflow has not been verified end to end in this repository layout.
