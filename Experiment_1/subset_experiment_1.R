library(mlr3)
library(ggplot2)

# Load data

library(data.table)

data_dt = fread("dataset.csv")

dim(data_dt)
names(data_dt)

# Delete columns not used or relevant for prediction
### 2nd columns : "visit_id" a unique number for all hospital visits
### ncol(data_dt) -1 (the column before the last one) : "CSO" the Clinical Worker

cols_supp <- c(2, (ncol(data_dt) - 1))

# Under-sampling to 1000-first rows only

data_for_pred <- data_dt[, -..cols_supp]

# --------------------------- Functions utils ---------------------------

# Factor supported by mlr3 tasks
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

#Implemented this function to not repeat the proportion calculi
calculate_proportion <- function(data, group_col) {
  print(data[, .(
    patients_count = .N, 
    proportion = round((.N / nrow(data)) * 100, 2)
  ), by = group_col][order(-proportion)])
}

#Submit Jobs to cluster nodes
submit_job_array <- function(jobs_dt, minutes=1, gigabytes=1){
  jobs_dt$chunk <- 1
  batchtools::submitJobs(jobs_dt, resources=list(
    walltime = minutes*60,#seconds
    memory = gigabytes*1000,#megabytes per cpu
    ncpus=1,  #>1 for multicore/parallel jobs.
    ntasks=1, #>1 for MPI jobs.
    chunks.as.arrayjobs=TRUE))
}
# Run the benchmark and get the results from the cluster
run_cluster_benchmark <- function(design, reg_name, minutes = 60, gigabytes = 16) {
  
  reg_dir <- paste0("~/scratch/project_test/", reg_name)
  batchtools::setDefaultRegistry(NULL)
  unlink(reg_dir, recursive = TRUE)
  
  reg <- batchtools::makeExperimentRegistry(reg_dir)
  
  mlr3batchmark::batchmark(design, reg = reg) 
  
  not.done <- batchtools::getJobTable(reg = reg)[is.na(done)]
  submit_job_array(not.done, minutes = minutes, gigabytes = gigabytes)
  
  batchtools::waitForJobs(reg = reg)
  
  erreurs <- batchtools::findErrors(reg = reg)
  expires <- batchtools::findExpired(reg = reg)
  if(nrow(erreurs) > 0 | nrow(expires) > 0) {
    warning(paste("!!! Jobs expired or errored in the register :", reg_name))
  }
  
  bmr_final <- mlr3batchmark::reduceResultsBatchmark(store_backends = FALSE, reg = reg)
  
  return(bmr_final)
}

# Subset Gender(M,F)

calculate_proportion(data_for_pred, "gender")
calculate_proportion(data_for_pred, "oym")

task_list = list()

for (g in c('F','M')) {
  data_gender_filter = data_for_pred[gender == g]
  task_list[[g]] = TaskClassif$new(
    id = g,
    backend = data_gender_filter,
    target = "oym"
  )$set_col_roles("patient_id", roles = "name")
  calculate_proportion(data_gender_filter, "oym")
}

# Learners
library(mlr3pipelines)

lrn_featureless <- mlr3::LearnerClassifFeatureless$new()$configure(id = "featureless")

lrn_cv_glmnet <- as_learner(po("encode") %>>% mlr3learners::LearnerClassifCVGlmnet$new())
lrn_cv_glmnet$id <- "cv_glmnet"

lrn_knn <- mlr3learners::LearnerClassifKKNN$new()
lrn_knn$id <- "KNN"
print(lrn_knn$param_set)

lrn_rf <- mlr3learners::LearnerClassifRanger$new()
lrn_rf$id <- "RF"

learners <- list(
  lrn_featureless,
  lrn_cv_glmnet,
  lrn_knn,
  lrn_rf
)

for(learner.i in seq_along(learners)){
  learners[[learner.i]]$predict_type <- "prob"
}

#Define Benchmark

kfold_cv <- rsmp("cv", folds = 5)

design_kfold <- benchmark_grid(
  tasks = task_list,
  learners = learners,
  resamplings = kfold_cv
)

#Training
library(batchtools)
library(mlr3batchmark)

train_kfold_cluster <- function(bmr_design) {
  print("Train Benchmark KFOLD_CV...")
  bmr_final <- run_cluster_benchmark(bmr_design, "kfold")
  return(bmr_final)
}

# Uncomment this to train the benckmark on cluster
bmr <- train_kfold_cluster(design_kfold)

# Uncomment this to train the benckmark on pc/1-node
#bmr <- bmr <- benchmark(design_kfold)

#Evaluation
metrics_list <- c("classif.auc", "classif.ce", "classif.tpr", "classif.fpr",
                  "classif.tnr", "classif.fnr")

test_measure <- mlr3::msrs(metrics_list)

scores <- bmr$score(test_measure)
tab_avg_score <- dcast(scores, 
                       task_id + learner_id ~ ., 
                       list(mean, sd),
                       value.var = metrics_list
)
tab_avg_score <- tab_avg_score[order(task_id, -classif.auc_mean)]

print(tab_avg_score)

#Graphic for AUC and Classification Error

scores[, let(percent_error=100*classif.ce)]
ggplot()+
  facet_grid(task_id ~ .)+
  geom_point(aes(
    percent_error, learner_id),
    data=scores)+
  scale_x_continuous(
    breaks=seq(0,100,by=10),
    limits=c(0,60))

ggplot()+
  facet_grid(task_id ~ .)+
  geom_point(aes(
    classif.auc, learner_id),
    data=scores)

# --------------------------- Functions ---------------------------

run_soak_experiment <- function(data, subset_col, group_col = "patient_id", target_col = "oym", learners_list, folds = 5, reg_name) {
  
  task_id <- paste0("Task_", subset_col)
  task <- TaskClassif$new(id = task_id, backend = data, target = target_col)
  task$col_roles$subset <- subset_col
  task$set_col_roles(group_col, roles = "group")
  
  SOAK <- mlr3resampling::ResamplingSameOtherSizesCV$new()
  SOAK$param_set$values$folds <- folds
  SOAK$instantiate(task)
  
  design_soak <- benchmark_grid(
    tasks = list(task),
    learners = learners_list,
    resamplings = SOAK
  )
  
  # Train the benckmark on cluster
  #bmr_soak <- run_cluster_benchmark(design_soak, reg_name)
  
  # Train the benckmark on pc/1-node
  bmr_soak <- benchmark(design_soak)
  
  score_obj <- mlr3resampling::score(bmr_soak, mlr3::msrs("classif.auc"))
  pval_obj  <- mlr3resampling::pvalue(score_obj)
  
  return(list(score = score_obj, pvalue = pval_obj, bmr = bmr_soak))
}

# Visualize AUC of the SOAK results
visualize_graphic <- function(score_to_plot, experience_title) {
  auc_graphic <- ggplot(score_to_plot, aes(x = classif.auc, y = train.subsets)) +
    geom_point(shape = 1, size = 2.5) +
    facet_grid(learner_id ~ test.subset, labeller = label_both) +
    labs(
      title = paste("AUC Score -", experience_title),
      x = "classif.auc",
      y = "Train subsets"
    ) +
    theme_bw()
  
  print(auc_graphic)
}

# --------------------------- Experience 1: Cross Gender ---------------------------
# Run Gender SOAK
res_gender <- run_soak_experiment(
  data = data_for_pred, 
  subset_col = "gender", 
  learners_list = learners,
  reg_name = "jobs_soak_gender"
)

visualize_graphic(res_gender$score, "Cross Gender")

plot(res_gender$pvalue)

# --------------------------- Experience 2: Cross Age ------------------------------
set.seed(42)
data_age <- data_dt[sample(.N, 1000), -..cols_supp]
data_age <- convert_for_mlr3(data_age)

data_age[, age_interval := cut(age_original, 
                               breaks = c(0, 65, 120), 
                               labels = c("age < 65", "65 <= age"),
                               include.lowest = TRUE)]

# Run age SOAK
res_age <- run_soak_experiment(
  data = data_age, 
  subset_col = "age_interval", 
  learners_list = learners,
  reg_name = "jobs_soak_age"
)

visualize_graphic(res_age$score, "Cross Age")

plot(res_age$pvalue)

