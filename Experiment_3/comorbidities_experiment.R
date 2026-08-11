library(mlr3)
library(ggplot2)
library(data.table)
library(mlr3pipelines)
library(mlr3learners)
library(batchtools)
library(mlr3batchmark)

# 1. Load the data
data_dt = fread("dataset.csv")

# Remove unnecessary columns
cols_supp <- c(2, (ncol(data_dt) - 1))
set.seed(42)
data_for_pred <- data_dt[, -..cols_supp]

# 2. Data type conversion function for mlr3
convert_for_mlr3 <- function(dt) {
  colonnes_texte <- names(dt)[sapply(dt, is.character)]
  if (length(colonnes_texte) > 0) {
    dt[, (colonnes_texte) := lapply(.SD, as.factor), .SDcols = colonnes_texte]
  }
  dt[, oym := as.factor(oym)]
  dt[, patient_id := as.character(patient_id)]
  return(dt)
}
data_for_pred <- convert_for_mlr3(data_for_pred)

# 3. Definition of the 2 selected models
# A. Base GLMNET
lrn_cv_glmnet <- as_learner(po("encode") %>>% mlr3learners::LearnerClassifCVGlmnet$new())
lrn_cv_glmnet$id <- "cv_glmnet_base"

# B. Random Forest (Ranger) with feature importance extraction
lrn_rf <- mlr3learners::LearnerClassifRanger$new()
lrn_rf$param_set$values <- list(importance = "permutation")
lrn_rf$id <- "RF"

learners <- list(lrn_cv_glmnet, lrn_rf)

for(learner.i in seq_along(learners)){
  learners[[learner.i]]$predict_type <- "prob"
}

# 4. Create the classification task
task_survie <- as_task_classif(data_for_pred, target = "oym", id = "prediction_mortalite")
task_survie$set_col_roles("patient_id", roles = "group")
task_list <- list(task_survie)

# 5. Cross-validation setup (5 folds, grouped by patient_id)
kfold_cv <- rsmp("cv", folds = 5)

design_kfold <- benchmark_grid(
  tasks = task_list,
  learners = learners,
  resamplings = kfold_cv
)

# 6. Function for safe execution on the cluster
run_cluster_benchmark <- function(design, reg_name, minutes = 720, gigabytes = 32) {
  reg_dir <- paste0("~/scratch/project_test/", reg_name)
  batchtools::setDefaultRegistry(NULL)
  unlink(reg_dir, recursive = TRUE)
  
  reg <- batchtools::makeExperimentRegistry(reg_dir)
  mlr3batchmark::batchmark(design, reg = reg, store_models = TRUE) 
  
  not.done <- batchtools::getJobTable(reg = reg)[is.na(done)]
  
  # Submit jobs
  jobs_dt <- not.done
  jobs_dt$chunk <- 1
  batchtools::submitJobs(jobs_dt, resources = list(
    walltime = minutes * 60,
    memory = gigabytes * 1000,
    ncpus = 1,
    ntasks = 1,
    chunks.as.arrayjobs = TRUE
  ))
  
  batchtools::waitForJobs(reg = reg)
  
  bmr_final <- mlr3batchmark::reduceResultsBatchmark(store_backends = FALSE, reg = reg)
  return(bmr_final)
}

# 7. Launch the global training
bmr <- run_cluster_benchmark(design_kfold, "kfold_simple")


library(glmnet)

reg_dir <- "~/scratch/project_test/kfold_simple"
reg <- loadRegistry(reg_dir, writeable = FALSE)

cat("Loading benchmark results...\n")
bmr <- reduceResultsBatchmark(store_backends = TRUE, reg = reg)
cat("Loading complete!\n\n")


scores_dt <- bmr$score(msr("classif.auc"))
print("--- MEAN PERFORMANCE (AUC) ---")
print(scores_dt[, .(mean_auc = mean(classif.auc), sd_auc = sd(classif.auc)), by = learner_id])

# -------------------------------------------------------------------------

rr_glmnet_all <- bmr$aggregate()[learner_id == "cv_glmnet_base"]$resample_result[[1]]

glmnet_coeffs_list <- list()

for (i in seq_along(rr_glmnet_all$learners)) {
  learner_inst <- rr_glmnet_all$learners[[i]]
  
  # Safely retrieve the cv_glmnet model for each fold
  raw_fit <- NULL
  if (!is.null(learner_inst$state$model$classif.cv_glmnet$model)) {
    raw_fit <- learner_inst$state$model$classif.cv_glmnet$model
  } else if (!is.null(learner_inst$model$graph_model$pipeops$classif.cv_glmnet$learner_model$model)) {
    raw_fit <- learner_inst$model$graph_model$pipeops$classif.cv_glmnet$learner_model$model
  } else {
    raw_fit <- learner_inst$model
  }
  
  if (!is.null(raw_fit)) {
    coefs_raw <- predict(raw_fit, type = "coefficients", s = "lambda.min")
    if (is.list(coefs_raw)) coefs_raw <- coefs_raw[[2]]
    
    coeffs_mat <- as.matrix(coefs_raw)
    dt_fold <- data.table(
      variable = rownames(coeffs_mat),
      coefficient = as.numeric(coeffs_mat[, 1]),
      fold = paste0("Fold ", i)
    )
    glmnet_coeffs_list[[i]] <- dt_fold
  }
}

all_glmnet_coeffs <- rbindlist(glmnet_coeffs_list)
all_glmnet_coeffs <- all_glmnet_coeffs[variable != "(Intercept)"]

# Select the top 15 variables with the highest mean absolute magnitude
top_glm_global <- all_glmnet_coeffs[, .(mean_abs = mean(abs(coefficient))), by = variable][order(-mean_abs)][1:15]$variable
subset_glm_top <- all_glmnet_coeffs[variable %in% top_glm_global]

# Plot the stability of GLMNET coefficients across folds
p_glm_stability <- ggplot(subset_glm_top, aes(x = reorder(variable, abs(coefficient)), y = coefficient, color = fold)) +
  geom_point(size = 3, alpha = 0.85) +
  coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Stability of Top Coefficients (Base GLMNET)",
    subtitle = "Dispersion of log-odds coefficients across the 5 folds",
    x = "Variables",
    y = "Coefficient (Log-Odds)",
    color = "Fold"
  ) +
  theme_bw()

print(p_glm_stability)


rr_rf <- bmr$aggregate()[learner_id == "RF"]$resample_result[[1]]

importances_list <- list()
for (i in seq_along(rr_rf$learners)) {
  imp <- rr_rf$learners[[i]]$importance()
  dt_imp <- data.table(
    variable = names(imp),
    importance = as.numeric(imp),
    fold = paste0("Fold ", i)
  )
  importances_list[[i]] <- dt_imp
}
all_importances <- rbindlist(importances_list)

top_global <- all_importances[, .(mean_imp = mean(importance)), by = variable][order(-mean_imp)][1:15]$variable
subset_top <- all_importances[variable %in% top_global]

p_rf_stability <- ggplot(subset_top, aes(x = reorder(variable, importance), y = importance, color = fold)) +
  geom_point(size = 3, alpha = 0.85) +
  coord_flip() +
  labs(
    title = "Stability of Top Variables (Random Forest)",
    subtitle = "Permutation importance dispersion across the 5 folds",
    x = "Variables",
    y = "Permutation Importance",
    color = "Fold"
  ) +
  theme_bw()

print(p_rf_stability)