# ==== Complete Package Conflict Resolution ====
options(stringsAsFactors = FALSE)

# Install and load the 'conflicted' package
if (!require(conflicted)) { 
  install.packages("conflicted")
  library(conflicted) 
}

# Declare all preferences at once (including encountered and potential conflicts)
conflicts_prefer(
  # ==== dplyr series ====
  dplyr::filter,
  dplyr::select,
  dplyr::lag,
  dplyr::recode,
  dplyr::summarize,
  dplyr::mutate,
  dplyr::arrange,
  dplyr::group_by,
  dplyr::ungroup,
  dplyr::rename,
  dplyr::distinct,
  dplyr::pull,
  dplyr::combine,
  dplyr::id,
  
  # ==== purrr series ====
  purrr::compact,
  purrr::discard,
  purrr::some,
  purrr::map,
  purrr::walk,
  
  # ==== brms/rstan series ====
  brms::rhat,
  brms::dstudent_t,
  brms::pstudent_t,
  brms::qstudent_t,
  brms::rstudent_t,
  rstan::extract,        # Added: previously encountered conflict
  
  # ==== tidyr series ====
  tidyr::extract,        # Added: another 'extract' conflict
  tidyr::expand,
  
  # ==== Statistical tests series ====
  rstatix::cohens_d,
  rstatix::eta_squared,
  rstatix::desc,
  rstatix::t_test,
  rstatix::wilcox_test,
  
  # ==== Plotting and data processing ====
  scales::col_factor,
  bayesplot::rhat,
  gridExtra::combine,
  
  # ==== Other common conflicts ====
  car::recode,           # car vs. dplyr
  car::some,             # car vs. purrr
  plyr::summarize,       # plyr vs. dplyr
  plyr::desc,            # plyr vs. rstatix
  plyr::id,              # plyr vs. dplyr
  plyr::failwith,        # plyr vs. dplyr
  plyr::compact,         # plyr vs. purrr
  readr::col_factor,     # readr vs. scales
  effectsize::cohens_d,  # effectsize vs. rstatix
  effectsize::eta_squared # effectsize vs. rstatix
)

# Set global preferences (fallback settings)
conflict_prefer_all("dplyr", quiet = TRUE)
conflict_prefer_all("brms", quiet = TRUE)
conflict_prefer_all("rstan", quiet = TRUE)
conflict_prefer_all("tidyr", quiet = TRUE)
conflict_prefer_all("purrr", quiet = TRUE)

cat("✓ Complete package conflict configuration loaded\n")
cat("✓ Resolved 'extract' conflict (preferring rstan::extract)\n")