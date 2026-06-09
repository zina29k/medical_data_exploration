library(mlr3)
library(mlr3extralearners)

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

for (g in c('F','M')) {
  data_gender_filter = data_for_pred[gender == g]
  assign(paste0("data_gender_", g),data_gender_filter)
  calculate_proportion(data_gender_filter, "oym")
}

#Tasks definition

task_gender_F <- TaskClassif$new(
  id = "Gender_F",
  backend = data_gender_F,
  target = "oym"
)

task_gender_M <- TaskClassif$new(
  id = "Gender_M",
  backend = data_gender_M,
  target = "oym"
)

# Learners

library(mlr3learners)
library(kknn)
library(glmnet)
library(randomForest)
library(mlr3pipelines)

lrn_featureless <- mlr3::LearnerClassifFeatureless$new()$configure(id = "featureless")

lrn_cv_glmnet <- as_learner(po("encode") %>>% mlr3learners::LearnerClassifCVGlmnet$new())
lrn_cv_glmnet$id <- "cv_glmnet"

lrn_knn <- mlr3learners::LearnerClassifKKNN$new()
lrn_knn$id <- "KNN"

lrn_rf <- lrn("classif.randomForest", id = "RandomForest", ntree = 100)

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
  tasks = list(task_gender_F,task_gender_M),
  learners = learners,
  resamplings = kfold_cv
)

#Training

print("Train Benchmark KFOLD_CV...")

bmr <- benchmark(design_kfold)

#Evaluation

test_measure <- mlr3::msrs(c('classif.auc', 'classif.ce', 'classif.tpr', 'classif.fpr','classif.tnr', 'classif.fnr'))

scores <- bmr$score(test_measure)

tab_avg_score <- scores[, .(
  Mean_AUC = mean(classif.auc),
  Mean_Error = mean(classif.ce),
  Mean_TPR = mean(classif.tpr, na.rm = TRUE), 
  Mean_FPR = mean(classif.fpr, na.rm = TRUE),
  Mean_TNR = mean(classif.tnr, na.rm = TRUE), 
  Mean_FNR = mean(classif.fnr, na.rm = TRUE)
), by = .(task_id, learner_id)]

tab_avg_score<- tab_avg_score[order(task_id, -Mean_AUC)]

print(tab_avg_score)

#Graphic for AUC and Classification Error

library(ggplot2)

library(animint2)
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

score_dt <- bmr_soak$score(mlr3::msrs('classif.auc'))
info_iterations <- SOAK$instance$iteration.dt
score_to_plot <- merge(score_dt, info_iterations, by = "iteration")

auc_graphic <- ggplot(score_to_plot, aes(x = classif.auc, y = train.subsets)) +
  geom_point(shape = 1, size = 2.5) +
  facet_grid(learner_id ~ test.subset) +
  labs(
    title = "AUC Score",
    x = "classif.auc",
    y = "Train subsets"
  ) +
  theme_bw()

print(auc_graphic)


