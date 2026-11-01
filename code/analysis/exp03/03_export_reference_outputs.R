#!/usr/bin/env Rscript

# Export numerical reference tables for Exp3 from the released trial-level data.
# This script follows the original analysis: group AIC comparisons are based on
# face-stimulus means; participant PSE/slope estimates use logistic (categorical)
# and logistic NLS (continuous) fits on the centered morph scale.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
root_dir <- normalizePath(file.path(script_dir, "..", "..", ".."), mustWork = TRUE)
data_path <- file.path(root_dir, "data", "processed", "exp03", "exp03_trial_data.csv")
output_dir <- Sys.getenv("EXP03_OUTPUT_DIR", unset = file.path(root_dir, "reference_outputs", "exp03"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

trial_data <- read_csv(data_path, show_col_types = FALSE) %>%
  transmute(participant = as.character(participant), block_name = as.character(block_name),
            face_stim, emotion_level = as.numeric(emotion_level), resp = as.numeric(resp),
            emotion_scaled = (emotion_level - 50) / 10)

expected_conditions <- c("categorical", "continuous")
if (!setequal(unique(trial_data$block_name), expected_conditions)) stop("Unexpected Exp3 conditions.")

logistic_fun <- function(x, p1, p2) 1 / (1 + exp(-(p1 + p2 * x)))
linear_fun <- function(x, p1, p2) p1 * x + p2
step_fun <- function(x, p1, p2) ifelse(x < 0, p1, p2)
rss_fun <- function(p, x, y, fun) sum((y - fun(x, p[1], p[2]))^2)

stimulus_means <- trial_data %>%
  group_by(face_stim, block_name, emotion_scaled) %>%
  summarise(resp = mean(resp), .groups = "drop")

fit_aic_models <- function(data, condition) {
  logistic_fit <- nls(resp ~ logistic_fun(emotion_scaled, p1, p2), data = data, start = list(p1 = 0, p2 = 1))
  linear_fit <- nls(resp ~ linear_fun(emotion_scaled, p1, p2), data = data, start = list(p1 = 1, p2 = 0))
  step_fit <- optim(par = c(0, 1), fn = rss_fun, x = data$emotion_scaled, y = data$resp, fun = step_fun)
  rss <- c(Logistic = sum(resid(logistic_fit)^2), Linear = sum(resid(linear_fit)^2),
           Step = rss_fun(step_fit$par, data$emotion_scaled, data$resp, step_fun))
  coefficients <- rbind(coef(logistic_fit), coef(linear_fit), step_fit$par)
  tibble(condition = condition, model = names(rss), n_observations = nrow(data),
         rss = as.numeric(rss), aic = nrow(data) * log(rss / nrow(data)) + 4,
         parameter_1 = coefficients[, 1], parameter_2 = coefficients[, 2])
}

aic_results <- bind_rows(lapply(expected_conditions, function(condition) {
  fit_aic_models(filter(stimulus_means, block_name == condition), condition)
})) %>% group_by(condition) %>% mutate(delta_aic = aic - min(aic),
                                        is_best_model = aic == min(aic)) %>% ungroup()
write_csv(aic_results, file.path(output_dir, "AIC_model_comparison.csv"))
write_csv(filter(aic_results, is_best_model), file.path(output_dir, "AIC_best_model_by_condition.csv"))

fit_participant <- function(data, condition) {
  if (condition == "categorical") {
    model <- glm(resp ~ emotion_scaled, family = binomial(link = "logit"), data = data)
    pars <- coef(model)
    tibble(pse = -pars[["(Intercept)"]] / pars[["emotion_scaled"]], slope = pars[["emotion_scaled"]], fit_status = "ok")
  } else {
    model <- tryCatch(nls(resp ~ 1 / (1 + exp(-(b0 + b1 * emotion_scaled))), data = data,
                          start = list(b0 = 0, b1 = 1)), error = function(e) NULL)
    if (is.null(model)) return(tibble(pse = NA_real_, slope = NA_real_, fit_status = "nls_failed"))
    pars <- coef(model)
    tibble(pse = -pars[["b0"]] / pars[["b1"]], slope = pars[["b1"]], fit_status = "ok")
  }
}

participant_parameters <- trial_data %>%
  group_by(participant, block_name) %>%
  group_modify(~fit_participant(.x, .y$block_name)) %>%
  ungroup() %>% rename(condition = block_name)
write_csv(participant_parameters, file.path(output_dir, "psychometric_parameters_by_participant.csv"))

wide <- participant_parameters %>% filter(fit_status == "ok") %>%
  select(participant, condition, pse, slope) %>%
  tidyr::pivot_wider(names_from = condition, values_from = c(pse, slope))
paired_summary <- function(x, y, outcome) {
  test <- t.test(x, y, paired = TRUE)
  tibble(outcome = outcome, n_pairs = length(x), mean_continuous = mean(x), mean_categorical = mean(y),
         mean_difference_continuous_minus_categorical = unname(test$estimate), statistic_t = unname(test$statistic),
         df = unname(test$parameter), p_value = test$p.value, conf_low = test$conf.int[[1]], conf_high = test$conf.int[[2]])
}
paired_tests <- bind_rows(paired_summary(wide$pse_continuous, wide$pse_categorical, "PSE"),
                          paired_summary(wide$slope_continuous, wide$slope_categorical, "slope"))
write_csv(paired_tests, file.path(output_dir, "PSE_slope_paired_tests.csv"))

capture.output(sessionInfo(), file = file.path(output_dir, "sessionInfo_exp03.txt"))
message("Exp3 reference outputs written to: ", output_dir)
