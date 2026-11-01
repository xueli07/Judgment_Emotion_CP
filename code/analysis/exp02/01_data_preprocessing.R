#!/usr/bin/env Rscript

# Exp2 public-release preprocessing.
#
# This is the public-directory adaptation of Scripts/study6/Step1_Data_Preprocessing.R.
# It applies the same target-column selection and RT rules, but reads the
# de-identified participant files and public inclusion table supplied here.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(purrr)
})

script_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)][1])
if (is.na(script_file) || !nzchar(script_file)) script_file <- file.path(getwd(), "01_data_preprocessing.R")
repo_dir <- normalizePath(file.path(dirname(script_file), "..", "..", ".."), mustWork = TRUE)

raw_dir <- file.path(repo_dir, "data", "raw", "exp02")
inclusion_path <- file.path(raw_dir, "participant_inclusion.csv")
output_path <- Sys.getenv("OUTPUT_PATH", unset = file.path(repo_dir, "data", "processed", "exp02", "exp02_trial_data.csv"))
qc_dir <- Sys.getenv("QC_DIR", unset = file.path(repo_dir, "results", "exp02", "preprocessing"))

inclusion <- read_csv(inclusion_path, show_col_types = FALSE) %>%
  filter(included_in_reported_analysis %in% c(TRUE, "True", "TRUE")) %>%
  transmute(subject_id = participant, judgment_condition = group)

if (nrow(inclusion) != 60L || anyDuplicated(inclusion$subject_id)) {
  stop("The public inclusion table must contain 60 unique included Exp2 participants.")
}
if (!setequal(unique(inclusion$judgment_condition), c("control", "continuous"))) {
  stop("Unexpected Exp2 conditions in participant_inclusion.csv.")
}

column_or_na <- function(data, column) {
  if (column %in% names(data)) data[[column]] else rep(NA, nrow(data))
}
as_number <- function(x) suppressWarnings(as.numeric(x))
as_text <- function(x) {
  x <- as.character(x)
  x[is.na(x) | x == "NA"] <- NA_character_
  x
}

read_participant <- function(subject_id, condition) {
  file_path <- file.path(raw_dir, "participant_files", paste0(subject_id, "_raw.csv"))
  if (!file.exists(file_path)) stop("Missing public raw file: ", basename(file_path))
  raw <- read_csv(file_path, show_col_types = FALSE, num_threads = 1)

  if (condition == "control") {
    emotion <- "formal_control_emotion"; actor <- "formal_control_actor"
    gender <- "formal_control_gender"; block <- "form_1back_block"
    trial <- "form_1back_trial"; response <- "form_control_correct"; rt <- "form_control_rt"
  } else {
    emotion <- "formal_training_emotion"; actor <- "formal_training_actor"
    gender <- "form_current_gender"; block <- "form_cont_block"
    trial <- "form_cont_trial"; response <- "form_cont_resp"; rt <- "form_cont_rt"
  }

  tibble(
    subject_id = subject_id,
    emotion_level = as_number(column_or_na(raw, emotion)),
    face_identity = as_text(column_or_na(raw, actor)),
    face_gender = as_text(column_or_na(raw, gender)),
    block_id = as_number(column_or_na(raw, block)),
    trial_id = as_number(column_or_na(raw, trial)),
    judgment_response = as_number(column_or_na(raw, response)),
    judgment_rt = as_number(column_or_na(raw, rt)),
    age = as_number(column_or_na(raw, "age")),
    gender = as_text(column_or_na(raw, "gender")),
    xab_pairs = as_text(column_or_na(raw, "formal_xab_pairs")),
    xab_x_emotion = as_number(column_or_na(raw, "formal_x_emotion")),
    xab_block = as_number(column_or_na(raw, "formal_xab_block")),
    xab_trial = as_number(column_or_na(raw, "formal_xab_trial")),
    xab_resp = as_number(column_or_na(raw, "form_xab_resp")),
    xab_rt = as_number(column_or_na(raw, "form_xab_rt")),
    judgment_condition = condition
  ) %>%
    filter(!is.na(judgment_rt) | !is.na(xab_rt)) %>%
    filter(is.na(judgment_rt) | judgment_rt >= 0.2) %>%
    filter(is.na(xab_rt) | (xab_rt >= 0.2 & xab_rt < 10))
}

trial_data <- map2_dfr(inclusion$subject_id, inclusion$judgment_condition, read_participant) %>%
  select(subject_id, emotion_level, face_identity, face_gender, block_id, trial_id,
         judgment_response, judgment_rt, age, gender, xab_pairs, xab_x_emotion,
         xab_block, xab_trial, xab_resp, xab_rt, judgment_condition)

if (n_distinct(trial_data$subject_id) != 60L) stop("Processed Exp2 data do not contain 60 included participants.")

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
dir.create(qc_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(trial_data, output_path, na = "")
write_csv(
  trial_data %>% count(judgment_condition, name = "n_rows") %>%
    mutate(n_participants = map_int(judgment_condition, ~ n_distinct(trial_data$subject_id[trial_data$judgment_condition == .x]))),
  file.path(qc_dir, "preprocessing_summary.csv")
)
write_csv(
  trial_data %>% group_by(subject_id, judgment_condition) %>%
    summarise(n_rows = n(), n_judgment_trials = sum(!is.na(judgment_rt)),
              n_xab_trials = sum(!is.na(xab_rt)), .groups = "drop"),
  file.path(qc_dir, "per_participant_trial_counts.csv")
)

message("Wrote ", nrow(trial_data), " Exp2 rows for ", n_distinct(trial_data$subject_id), " participants to ", output_path)
