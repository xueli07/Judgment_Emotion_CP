# Verification scope

Retained records support the Exp3 demo export, Exp6 numerical exports from a compatible local cache, and both Figure 4 exporters. Historical Exp3 and Exp6 sessionInfo records describe Windows 11 x64 build 26200 and R 4.5.0; they establish only those export environments. The author confirmed that `az.from_netcdf(nc_path)` and `hddm.load(str(nc_path))` both loaded the .nc file successfully in Docker. This is a loading check, not a complete HDDM/PPC rerun.

There is no recorded clean-environment end-to-end rerun of the entire repository. No complete historical package lock file is available. Author-confirmed original software versions and export-specific session records are distinct evidence. Historical sessionInfo files are retained unchanged.

## Public reproduction entry point

The release distributes preprocessed trial-level behavioral CSVs rather than original participant exports or inclusion tables. Public reproduction begins with those processed CSVs (script 02 for Exp1–2). The retained preprocessing scripts document the original processing workflow but require inputs not distributed in this release.
