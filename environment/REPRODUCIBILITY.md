# Reproduction notes

The authors have run the analyses for all six experiments. The software used for the original analyses is documented in [software versions](software_versions.csv). Historical Exp3 and Exp6 sessionInfo files record Windows 11 x64 build 26200 and R 4.5.0 and the package versions used for those specific export workflows. They are retained unchanged. A complete historical package lock file is not available.

## Inputs and execution

The release distributes preprocessed trial-level behavioral CSVs rather than original participant exports or inclusion tables. Public reproduction begins with those processed CSVs (script 02 for Exp1–2). The retained preprocessing scripts document the original processing workflow but require inputs not distributed in this release.

Follow the [analysis guide](../code/analysis/README.md) for module-specific dependencies, commands, expected outputs and runtimes. Experiment 6 reconstruction uses the distributed noise RData and matching mask; its computed CI cache is a separate generated file. Newly reconstructed R CIs require the documented grouped-image handoff to MATLAB.

The author has confirmed successful loading of the HDDM .nc file with both `az.from_netcdf(nc_path)` and `hddm.load(str(nc_path))` in Docker.
