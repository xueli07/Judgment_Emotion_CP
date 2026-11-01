#!/usr/bin/env Rscript

# Export the numerical Exp6 reference tables from a locally authorized CI cache.
# The cache and face mask are restricted assets and are intentionally not included
# in this public repository. Before running, set EXP06_CACHE_PATH and EXP06_MASK_PATH.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(lme4)
  library(lmerTest)
  library(combinat)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
root_dir <- normalizePath(file.path(script_dir, "..", "..", ".."), mustWork = TRUE)
output_dir <- Sys.getenv("EXP06_OUTPUT_DIR", unset = file.path(root_dir, "reference_outputs", "exp06"))
cache_path <- Sys.getenv("EXP06_CACHE_PATH")
mask_path <- Sys.getenv("EXP06_MASK_PATH")
if (!nzchar(cache_path) || !file.exists(cache_path)) stop("Set EXP06_CACHE_PATH to the authorized ci_cache_all.RData file.")
if (!nzchar(mask_path) || !file.exists(mask_path)) stop("Set EXP06_MASK_PATH to the authorized average_face_mask.png file.")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cache <- new.env(parent = emptyenv())
load(cache_path, envir = cache)
required <- c("ci_cat", "ci_cont", "ci_levels_list", "cor_res_loo")
missing <- required[!vapply(required, exists, logical(1), envir = cache, inherits = FALSE)]
if (length(missing)) stop("Cache lacks required objects: ", paste(missing, collapse = ", "))
ci_cat <- cache$ci_cat
ci_cont <- cache$ci_cont
ci_levels_list <- cache$ci_levels_list
cor_res_loo <- cache$cor_res_loo
rm(cache)
invisible(gc())

mask <- png::readPNG(mask_path)[, , 1]
mask_vec <- as.vector(ifelse(mask > 0.5, 1, 0))
vec_cat_masked <- as.vector(ci_cat$ci) * mask_vec
vec_cont_masked <- as.vector(ci_cont$ci) * mask_vec
ci_mat <- do.call(rbind, ci_levels_list)
if (nrow(ci_mat) != 6L) stop("Expected six continuous-intensity CI levels in the cache.")
levels <- seq_len(nrow(ci_mat))

as_cor_test_table <- function(test, comparison) {
  tibble(comparison = comparison, estimate = unname(test$estimate), statistic = unname(test$statistic),
         df = if (!is.null(test$parameter)) unname(test$parameter) else NA_real_, p_value = test$p.value,
         conf_low = if (!is.null(test$conf.int)) test$conf.int[[1]] else NA_real_,
         conf_high = if (!is.null(test$conf.int)) test$conf.int[[2]] else NA_real_, method = test$method)
}

group_similarity <- cor.test(vec_cat_masked, vec_cont_masked)
write_csv(as_cor_test_table(group_similarity, "categorical_vs_continuous_group_CI"),
          file.path(output_dir, "group_CI_similarity_test.csv"))

pca_res <- prcomp(ci_mat, center = TRUE, scale. = FALSE)
variance <- pca_res$sdev^2 / sum(pca_res$sdev^2)
write_csv(tibble(component = paste0("PC", seq_along(variance)), variance_explained = variance,
                 cumulative_variance_explained = cumsum(variance)),
          file.path(output_dir, "PCA_variance_explained.csv"))
pc1_scores <- pca_res$x[, 1]
write_csv(tibble(level = levels, PC1_score = pc1_scores), file.path(output_dir, "PCA_PC1_scores_by_intensity.csv"))
pc1_spearman <- cor.test(levels, pc1_scores, method = "spearman", exact = FALSE)
write_csv(as_cor_test_table(pc1_spearman, "PC1_score_vs_intensity_level"),
          file.path(output_dir, "PCA_PC1_spearman_test.csv"))

r_vec <- map_dbl(ci_levels_list, ~cor(.x, vec_cat_masked))
trend_fit <- lm(r_vec ~ levels)
write_csv(tibble(level = levels, correlation_with_categorical_CI = r_vec),
          file.path(output_dir, "categorical_CI_similarity_by_intensity.csv"))
permutations <- combinat::permn(levels)
permutation_slopes <- map_dbl(permutations, ~unname(coef(lm(r_vec ~ .x))[[2]]))
observed_slope <- unname(coef(trend_fit)[[2]])
permutation_p <- (sum(abs(permutation_slopes) >= abs(observed_slope)) + 1) / (length(permutation_slopes) + 1)
write_csv(tibble(observed_slope = observed_slope, observed_intercept = unname(coef(trend_fit)[[1]]),
                 n_exact_permutations = length(permutation_slopes), two_sided_p_value = permutation_p),
          file.path(output_dir, "categorical_CI_similarity_permutation_test.csv"))

loo <- cor_res_loo %>% mutate(participant = as.factor(participant), level = as.numeric(level))
n_participants <- n_distinct(loo$participant)
if (n_participants < 3L) stop("The cache does not contain sufficient participant-level LOO results.")
cor_pseudo <- loo %>% mutate(group_correlation = r_vec[level],
                              jackknife_pseudovalue = n_participants * group_correlation -
                                (n_participants - 1) * loo_corWithCat)
write_csv(cor_pseudo, file.path(output_dir, "jackknife_pseudovalues.csv"))
jackknife_summary <- cor_pseudo %>% group_by(level) %>%
  summarise(n_participants = n(), jackknife_mean = mean(jackknife_pseudovalue),
            jackknife_sd = sd(jackknife_pseudovalue), jackknife_se = jackknife_sd / sqrt(n_participants),
            ci_lower = jackknife_mean - jackknife_se, ci_upper = jackknife_mean + jackknife_se, .groups = "drop")
write_csv(jackknife_summary, file.path(output_dir, "jackknife_summary_by_intensity.csv"))

jackknife_lme <- lmer(jackknife_pseudovalue ~ level + (level | participant), data = cor_pseudo, REML = FALSE)
fixed_effects <- as.data.frame(coef(summary(jackknife_lme)))
fixed_effects$term <- rownames(fixed_effects)
rownames(fixed_effects) <- NULL
write_csv(as_tibble(fixed_effects) %>% select(term, everything()), file.path(output_dir, "jackknife_LME_fixed_effects.csv"))
write_csv(tibble(n_observations = nrow(cor_pseudo), n_participants = n_participants,
                 AIC = AIC(jackknife_lme), BIC = BIC(jackknife_lme), logLik = as.numeric(logLik(jackknife_lme)),
                 singular_fit = lme4::isSingular(jackknife_lme, tol = 1e-4)),
          file.path(output_dir, "jackknife_LME_model_fit.csv"))

capture.output(sessionInfo(), file = file.path(output_dir, "sessionInfo_exp06_export.txt"))
message("Exp6 numerical reference outputs written to: ", output_dir)
