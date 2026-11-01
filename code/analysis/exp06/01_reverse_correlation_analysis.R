# ==============================================================================
# Script 2: Main Analyses (Group CIs, Similarity, PCA, and Mixed-Effects)
# ==============================================================================
rm(list = ls())
library(rcicr)
library(dplyr)
library(purrr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(corrplot)
library(combinat) # For permutations
library(imager)   # Used for image rotation and saving

# ==============================================================================
# 0. Custom Plotting Theme
# ==============================================================================
theme_psychometric <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", margin = margin(b = 10)),
      axis.title = element_text(color = "black", size = rel(1.1)),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.ticks.length = unit(2, "pt"),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = rel(0.9)),
      plot.caption = element_text(hjust = 1, color = "gray30", margin = margin(t = 8))
    )
}
# ==============================================================================
# 1. Setup, Data Loading, and Pre-processing
# ==============================================================================
# Resolve the package root from this script's location.
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
root_dir <- normalizePath(file.path(script_dir, "..", ".."))
setwd(root_dir)

# Define explicit paths based on the root directory
rdata_path <- file.path(root_dir, "restricted_assets", "exp06_noise_seed.RData")
mask_png   <- file.path(root_dir, "data", "exp06", "average_face_mask.png")
data_path  <- file.path(root_dir, "data", "exp06", "exp06_trial_data.csv")
results_dir <- file.path(root_dir, "results", "exp06")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(rdata_path)) {
  stop("Experiment 6 noise seed is not included because its redistribution status requires confirmation. See restricted_assets/README.md.")
}

# Load and prepare facial mask
mask_matrix <- png::readPNG(mask_png)[,,1]
mask_matrix <- ifelse(mask_matrix > 0.5, 1, 0)
mask_matrix[is.na(mask_matrix)] <- 0
mask_vec <- as.vector(mask_matrix)


# Load filtered data and exclude specific participants
S <- read.csv(data_path)

S_cat  <- S %>% filter(ExpCondition == "category")
S_cont <- S %>% filter(ExpCondition == "continuous")

# Pre-calculate Continuous task emotion intensity levels (1 to 6)
norm_to_minus1_1 <- function(x) {
  if (length(unique(x)) == 1) return(rep(0, length(x)))
  2 * ((x - min(x)) / (max(x) - min(x))) - 1
}

S_cont_levels <- S_cont %>%
  group_by(participant) %>%
  mutate(
    raw_norm = norm_to_minus1_1(Raw_chosen),
    raw_abs  = abs(raw_norm),
    q16 = quantile(raw_abs, 0.167, na.rm = TRUE),
    q33 = quantile(raw_abs, 0.333, na.rm = TRUE),
    q50 = quantile(raw_abs, 0.500, na.rm = TRUE),
    q66 = quantile(raw_abs, 0.667, na.rm = TRUE),
    q83 = quantile(raw_abs, 0.833, na.rm = TRUE),
    inforLevel = case_when(
      raw_abs <= q16 ~ "Level1",
      raw_abs > q16 & raw_abs <= q33 ~ "Level2",
      raw_abs > q33 & raw_abs <= q50 ~ "Level3",
      raw_abs > q50 & raw_abs <= q66 ~ "Level4",
      raw_abs > q66 & raw_abs <= q83 ~ "Level5",
      raw_abs > q83 ~ "Level6"
    )
  ) %>% ungroup()

levels_map <- paste0("Level", 1:6)


# ==============================================================================
# 2. CACHING BLOCK: Generate all Classification Images (CIs)
# ==============================================================================
ci_cache_file <- file.path(results_dir, "ci_cache_all.RData")

if (file.exists(ci_cache_file)) {
  cat("Cache file detected. Loading pre-computed CIs...\n")
  load(ci_cache_file)
  cat("CIs successfully loaded!\n")
  
} else {
  cat("No cache detected. Generating all CIs (this will take considerable time)...\n")
  
  # 2.1 Overall Group CIs
  ci_cat <- generateCI(
    stimuli = S_cat$num, responses = S_cat$Resp, baseimage = "chinese",
    participants = S_cat$participant, rdata = rdata_path,
    targetpath = file.path(results_dir, "cis", "categorical"),
    save_individual_cis = TRUE, ,scaling = "independent", n_cores = 1
  )
  
  ci_cont <- generateCI(
    stimuli = S_cont$num, responses = S_cont$Resp, baseimage = "chinese",
    participants = S_cont$participant, rdata = rdata_path,
    targetpath = file.path(results_dir, "cis", "continuous"),
    save_individual_cis = TRUE, scaling = "independent", n_cores = 1
  )
  
  # Temporary reference vector needed to compute individual correlations later
  temp_vec_cat_masked <- as.vector(ci_cat$ci) * mask_vec
  
  # 2.2 Continuous Levels CIs (Group level, 1 to 6)
  raw_ci_levels_list <- list()
  # 第一步：生成 6 个原始 CI 并存入 list
  for (lvl in levels_map) {
    data_sub <- S_cont_levels %>% filter(inforLevel == lvl )
    ci_res <- generateCI(
      stimuli = data_sub$num, responses = data_sub$Resp,participants = data_sub$participant, baseimage = "chinese",
      rdata = rdata_path, save_individual_cis = TRUE,targetpath = file.path(results_dir, "continuous_compare", "cis", lvl), 
      individual_scaling   = "independent",
      scaling = "independent",
      n_cores=1
    )
    raw_ci_levels_list[[lvl]] <- ci_res
  }
  

  #masked
  ci_levels_list <- list()
  for (lvl in levels_map) {
    # 按照旧代码逻辑：提取缩放后的 $ci，转为向量，并乘上 mask
    scaled_obj <- raw_ci_levels_list[[lvl]]
    ci_levels_list[[lvl]] <- as.vector(scaled_obj$ci) * mask_vec
  }
  
  # 2.3 Individual CIs (for Informational Value analysis)
  ci_cat_participant <- batchGenerateCI(
    data = S_cat, by = "participant", stimuli = "num", responses = "Resp",
    baseimage = "chinese", rdata = rdata_path, save_as_png = FALSE
  )
  
  ci_cont_participant <- batchGenerateCI(
    data = S_cont, by = "participant", stimuli = "num", responses = "Resp",
    baseimage = "chinese", rdata = rdata_path, save_as_png = FALSE
  )
  
  # 2.4 LOO Group-level CIs and correlations (Corrected Jackknife Logic)
  cat("Calculating Leave-One-Out (LOO) group CIs for Jackknife...\n")
  loo_cor_list <- list()
  participants_cont <- unique(S_cont_levels$participant)
  
  for (lvl_idx in 1:6) {
    lvl_name <- levels_map[lvl_idx]
    data_lvl <- S_cont_levels %>% filter(inforLevel == lvl_name)
    
    for (pid in participants_cont) {
      # Leave one participant out
      data_loo <- data_lvl %>% filter(participant != pid)
      if (nrow(data_loo) == 0) next
      
      # Generate LOO Group CI (using N-1 participants)
      ci_loo <- generateCI(
        stimuli = data_loo$num, responses = data_loo$Resp,participants = data_loo$participant, baseimage = "chinese",
        rdata = rdata_path, save_individual_cis = FALSE, scaling = "independent",
        n_cores = 1
      )
      
      # Correlate this high-signal LOO Group CI with the Categorical Group CI
      loo_cor <- cor(as.vector(ci_loo$ci) * mask_vec, temp_vec_cat_masked)
      
      loo_cor_list[[length(loo_cor_list) + 1]] <- data.frame(
        participant = as.factor(pid), 
        level = lvl_idx, 
        loo_corWithCat = loo_cor
      )
    }
  }
  cor_res_loo <- bind_rows(loo_cor_list)
  
  # Save everything to the cache (Note: using new cor_res_loo)
  save(ci_cat, ci_cont, ci_levels_list, ci_cat_participant, ci_cont_participant, cor_res_loo, 
       file = ci_cache_file)
  cat("All CIs generated and saved to cache:", ci_cache_file, "\n")
}
  

# Apply masks for downstream global use
vec_cat_masked <- as.vector(ci_cat$ci) * mask_vec
vec_cont_masked <- as.vector(ci_cont$ci) * mask_vec


# ==============================================================================
# 3. Representational Similarity Between Task Conditions
# ==============================================================================
cat_cont_cor <- cor.test(vec_cat_masked, vec_cont_masked)
cat("\n--- Group-level similarity between Categorical and Continuous tasks ---\n")
print(cat_cont_cor)


# ==============================================================================
# 4. Informational Value & Individual CIs (Supplementary)
# ==============================================================================
scaled_ci_dir <- file.path(results_dir, "scaled_ci")
noise_dir     <- file.path(results_dir, "noisepattern")
infoval_file  <- file.path(results_dir, "Infoval-results.csv")

if (!dir.exists(scaled_ci_dir)) dir.create(scaled_ci_dir, recursive = TRUE)
if (!dir.exists(noise_dir)) dir.create(noise_dir, recursive = TRUE)

cis_all_participant <- c(ci_cat_participant, ci_cont_participant)
scaled_cis_participant <- autoscale(cis_all_participant, save_as_pngs = TRUE, targetpath = scaled_ci_dir)

infoval_results <- data.frame(participant = character(), infoval = numeric(), Condition = character(), stringsAsFactors = FALSE)
cat("\nCalculating Informational Value for each participant (10,000 iterations)...\n")

for (i in seq_along(scaled_cis_participant)) {
  p_name <- names(scaled_cis_participant)[i]
  p_infoval <- computeInfoVal2IFC(scaled_cis_participant[[p_name]], rdata_path, iter = 10000)
  pid_str <- sub(".*participant_(.*)", "\\1", p_name)
  cond <- ifelse(substr(pid_str, 1, 1) == "3", "Categorical", "Continuous")
  

  infoval_results <- rbind(infoval_results, data.frame(
    participant = pid_str, 
    infoval = p_infoval, 
    Condition = cond
  ))
  
  # Save noise pattern images
  p_data <- scaled_cis_participant[[i]]$scaled
  img <- as.cimg(p_data) %>% imrotate(angle = 90)
  imager::save.image(img, file.path(noise_dir, paste0(p_name, ".png")))
}

write.csv(infoval_results, infoval_file, row.names = FALSE)

cat("\n--- Informational Value: Descriptive Statistics ---\n")
infoVal_summary <- infoval_results %>%
  group_by(Condition) %>%
  summarise(M = mean(infoval), SE = sd(infoval) / sqrt(n()), N = n())
print(infoVal_summary)

cat("\n--- Informational Value: Independent Samples t-test ---\n")

print(t.test(infoval ~ Condition, data = infoval_results, var.equal = TRUE))
# library(bruceR)
# TTEST(infoval_results,y="infoval", x= "Condition",paired = FALSE,)

# ==============================================================================
# 5. Continuous Representation: Similarity Matrix & PCA
# ==============================================================================
ci_mat <- do.call(rbind, ci_levels_list)
sim_matrix_cor <- cor(t(ci_mat), method = "pearson")

# Visualizing Similarity Matrix 
col_redblue <- colorRampPalette(c("#1F77B4", "white", "#D62728"))(200)
jpeg(file.path(results_dir, "corr_sim.jpg"),
     width = 4.5,
     height = 4,
     units = "in",
     res = 300)

cor_sim<-corrplot(sim_matrix_cor, col = col_redblue, method = "color", type = "lower",
         order = "hclust", hclust.method = "ward.D2", addCoef.col = "black", 
         tl.col = "black", diag = FALSE)
dev.off()

pca_res <- prcomp(ci_mat, center = TRUE, scale. = FALSE)
var_explained <- pca_res$sdev^2 / sum(pca_res$sdev^2)
cat("\nVariance explained by PC1:", var_explained[1], "\n")

# Correlate PC1 scores with ordinal emotion intensity levels
pc1_scores <- pca_res$x[, 1]
print(pc1_scores)
bin_index <- 1:6
cat("\n--- Spearman Correlation: PC1 Scores vs Emotion Intensity Level ---\n")
print(cor.test(bin_index, pc1_scores, method = "spearman"))

# ------------------------------------------------------------------------------
# Figure 1(e) - PC1 Score Trend
# ------------------------------------------------------------------------------
df_pc <- data.frame(bin_index = bin_index, PC1 = pc1_scores)

p_pc1 <- ggplot(df_pc, aes(x = bin_index, y = PC1)) +
  geom_point(color = "black", size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = "#D62728", linewidth = 1.2) +
  labs(x = "Emotion intensity level", y = "PC1 score") +
  scale_x_continuous(breaks = sort(unique(bin_index)), expand = expansion(mult = 0.05)) +
  theme_psychometric(base_size = 16) +
  theme(plot.margin = margin(10, 10, 10, 10))
plot(p_pc1)
# Save Figure 1(e)
ggsave(file.path(results_dir, "p_pc1_score.jpg"), p_pc1, width = 4.5, height = 4, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 1(f) - PC1 Loading map superimposed on base image
# ------------------------------------------------------------------------------

library(jpeg)     # Used for loading the base face image
library(grid)
# Extract PC1 loadings and reshape into a 512x512 matrix
pc1_loading <- pca_res$rotation[, 1]
eigen_image <- matrix(pc1_loading, nrow = 512, ncol = 512)
eigen_image <- t(apply(eigen_image, 2, rev)) # Correct orientation

# Convert to long format for ggplot
img_df <- expand.grid(x = 1:512, y = 1:512)
img_df$value <- as.vector(eigen_image)
img_df$z_value <- scale(img_df$value) # Standardize (z-score) to optimize color mapping

# Extract base image (ensure average_face.jpg is in the root directory)
base_face <- file.path(root_dir, "restricted_assets", "exp06_average_face.jpg")
if (file.exists(base_face)) {
  base_img <- readJPEG(base_face)  
  base_grob <- rasterGrob(base_img, width = unit(1, "npc"), height = unit(1, "npc"))
  
  PC1_load_plot <- ggplot(img_df, aes(x = x, y = y, fill = z_value)) +
    annotation_custom(base_grob, xmin = 1, xmax = 512, ymin = 1, ymax = 512) +
    geom_raster(alpha = 0.25) +
    scale_fill_gradient2(low = "#1F77B4", mid = "grey", high = "#D62728", midpoint = 0) +
    theme_void() + coord_fixed() + 
    labs(fill = "PC1 loading (z)") +  
    theme(legend.title = element_text(size = 14), legend.text = element_text(size = 12)) +
    guides(fill = guide_colorbar(barheight = unit(8, "cm"), barwidth = unit(0.8, "cm"), ticks.colour = "black"))
  
  plot(PC1_load_plot)
  # Save Figure 1(f)
  ggsave(file.path(results_dir, "pc1_load_with_base.png"), PC1_load_plot, width = 8, height = 8, dpi = 300)
} else {
  cat("\nWarning: Base image average_face.jpg not found. Skipping loading plot overlay.\n")
}
# ------------------------------------------------------------------------------


# ==============================================================================
# 6. Correlation with Categorical Prototype & Permutation Test
# ==============================================================================
r_vec <- map_dbl(ci_levels_list, ~cor(.x, vec_cat_masked))
obs_slope <- coef(lm(r_vec ~ bin_index))[2]

perm_levels <- combinat::permn(6)
perm_stats <- map_dbl(perm_levels, ~coef(lm(r_vec ~ .x))[2])
p_value_perm <- (sum(abs(perm_stats) >= abs(obs_slope)) + 1) / (length(perm_stats) + 1)

cat("\n--- Permutation Test for Linear Trend ---\n")
cat("Observed slope:", obs_slope, "| Permutation p-value:", p_value_perm, "\n")

# ------------------------------------------------------------------------------
# Figure 1(g) - Linear trend of Group's CIs Correlation across intensity levels
# ------------------------------------------------------------------------------
r_conwithcat <- data.frame(level = bin_index, cor = r_vec)

p_cor <- ggplot(r_conwithcat, aes(x = level, y = cor)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = "#1F77B4", linewidth = 1.2) + # Use blue fit line for distinction
  labs(x = "Emotion intensity level", y = "Group's CIs Correlation") + 
  scale_x_continuous(breaks = sort(unique(r_conwithcat$level))) +
  theme_psychometric(base_size = 16) +
  theme(plot.margin = margin(10, 10, 10, 10))

plot(p_cor)
# Save Figure 1(g)
ggsave(file.path(results_dir, "group_cis_correlation.jpg"), p_cor, width = 4.5, height = 4, dpi = 300)


# ==============================================================================
# 7. Individual-Level Robustness (Supplementary S2: Jackknife & LME)
# ==============================================================================
cat("\n--- Computing Jackknife Pseudo-values ---\n")

# r_vec (overall group correlation per level) is computed in Section 6.
# We calculate Pseudo-values: Pseudo = N * Group_Cor - (N-1) * LOO_Cor
n_sub <- length(unique(cor_res_loo$participant))

cor_pseudo <- cor_res_loo %>%
  rowwise() %>%
  mutate(
    group_cor = r_vec[level], 
    pseudo_cor = n_sub * group_cor - (n_sub - 1) * loo_corWithCat
  ) %>%
  ungroup()

cat("\n--- Linear Mixed-Effects Model (LME) on Jackknife Pseudo-values ---\n")
# LME uses the denoised pseudo-values which act as independent individual contributions
model_full <- lmer(pseudo_cor ~ level + (level | participant), data = cor_pseudo, REML = FALSE)
print(summary(model_full))

# ------------------------------------------------------------------------------
# Figure S1 - Jackknife-estimated individual correlations plot
# ------------------------------------------------------------------------------
cat("\n--- Generating Jackknife-estimated Plot (Fig S1) ---\n")

# Calculate Jackknife means and standard errors per level using the Pseudo-values
jk_summary <- cor_pseudo %>%
  group_by(level) %>%
  summarise(
    jk_mean = mean(pseudo_cor),
    jk_se   = sd(pseudo_cor) / sqrt(n()), # Standard error of pseudo-values
    .groups = "drop"
  ) %>%
  mutate(
    ci_lower = jk_mean - jk_se,
    ci_upper = jk_mean +  jk_se
  )

p_jk <- ggplot(jk_summary, aes(x = level, y = jk_mean)) +
  geom_line(color = "black", linewidth = 0.8) +
  geom_point(color = "black", size = 3) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2, linewidth = 0.8, color = "black") +
  labs(x = "Emotion intensity level", y = "Jackknife CI Correlation") +
  scale_x_continuous(breaks = 1:6, expand = expansion(mult = 0.05)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2),
                     limits = c(0.2, 0.8),
                     expand = expansion(mult = 0.05))+
  theme_psychometric(base_size = 16) +
  theme(plot.margin = margin(10, 10, 10, 10))

plot(p_jk)
# Save Figure S1
ggsave(file.path(results_dir, "Fig_S1_jackknife.jpg"), p_jk, width = 4.5, height = 4, dpi = 300)
cat("Figure S1 (Jackknife plot) exported successfully.\n")
