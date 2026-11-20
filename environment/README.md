# Software environment

The author confirmed Windows 11, R 4.5.0 for all R analyses, MATLAB R2023b with Image Processing Toolbox for Exp6, Python 3.9.13, HDDM 1.0.1, and Docker image `hcp4715/hddm:1.0.1` for the original analyses. The authors have run the analyses for all six experiments.

PsychoPy was used for acquisition: 2024.2.4 (Exp1, Exp2, Exp6) and 2023.2.3 (Exp3–Exp5). It is unnecessary for analysis of existing CSVs. Stat4CI uses bundled source; no exact upstream version/commit was recorded. Its source header cites 20 August 2004.

Install module-specific R packages as listed in the [analysis guide](../code/analysis/README.md). Allow approximately one hour for initial R dependency preparation, depending on downloads and compilation. MATLAB installation and Docker image downloads depend on existing installations and network speed. No explicit GPU requirement or verified minimum RAM is specified. The HDDM runtime reference CPU is AMD Ryzen 9 7945HX, 32 logical processors; it is not a minimum requirement.

See [software versions](software_versions.csv) and [reproduction notes](REPRODUCIBILITY.md). Historical export package versions apply only to those exports.

## Analysis runtimes

See the [workflow runtime table](../README.md#outputs-and-runtime) for author-provided or confirmed analysis timings. These approximate runtimes were provided or confirmed by the authors; they are approximate timings rather than standardized benchmarks. They exclude software installation, dependency preparation, and file downloads. Actual runtime varies with hardware, analysis settings, and the computing resources allocated to Docker.

HDDM reference configuration: AMD Ryzen 9 7945HX, 32 logical processors; four chains, 15,000 iterations per chain, burn-in 5,000, thinning 2. This is a runtime reference, not a minimum hardware requirement.

Installation and dependency-preparation times above are separate from analysis runtimes. The Exp3 quick-demo timing covers example-table export only. The independent 10,000-iteration informational-value runtime is unrecorded.
