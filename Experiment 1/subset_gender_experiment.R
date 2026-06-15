library(mlr3)

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

data_for_pred <- data_dt[1:1000, -..cols_supp]

# Factor supported by mlr3 tasks

colonnes_texte <- names(data_for_pred)[sapply(data_for_pred, is.character)]

if (length(colonnes_texte) > 0) {
  data_for_pred[, (colonnes_texte) := lapply(.SD, as.factor), .SDcols = colonnes_texte]
}
data_for_pred[, oym := as.factor(oym)]
data_for_pred[, patient_id := as.character(patient_id)]

# Subset Gender(M,F)

#Implemented this function to not repeat 3 times the proportion calculi
calculate_proportion <- function(data, group_col) {
  print(data[, .(
    patients_count = .N, 
    proportion = round((.N / nrow(data)) * 100, 2)
  ), by = group_col][order(-proportion)])
}

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

print("Train Benchmark KFOLD_CV...")

bmr <- benchmark(design_kfold)

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

library(ggplot2)

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


# --------------------------- Experience 1 Cross Gender ----------------------------


# New Benchmarking SOAK strategy + grouping (patient_id)

SOAK <- mlr3resampling::ResamplingSameOtherSizesCV$new()
SOAK$param_set$values$folds <- 5

task_gender = TaskClassif$new(
  id = "Gender_task",
  backend = data_for_pred,
  target = "oym"
)
task_gender$col_roles$subset <- "gender"
task_gender$set_col_roles("patient_id", roles = "group")

SOAK$instantiate(task_gender)

#Define new Benchmark

design_soak <- benchmark_grid(
  tasks = list(task_gender),
  learners = learners,
  resamplings = SOAK
)

#Training

print("Train Benchmark SOAK...")

bmr_soak <- benchmark(design_soak)

#Evaluation

score_obj = mlr3resampling::score(bmr_soak, mlr3::msrs("classif.auc"))
plot(score_obj)
pval_obj = mlr3resampling::pvalue(score_obj)
plot(pval_obj)

visualize_graphic <- function(score_to_plot){
    auc_graphic <- ggplot(score_to_plot, aes(x = classif.auc, y = train.subsets))+
    geom_point(shape = 1, size = 2.5) +
    facet_grid(learner_id ~ test.subset) +
    labs(
      title = "AUC Score",
      x = "classif.auc",
      y = "Train subsets"
    ) +
    theme_bw()
  
  print(auc_graphic)
}

score_to_plot = score_obj

visualize_graphic(score_to_plot)
