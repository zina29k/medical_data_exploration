library(mlr3)
library(mlr3extralearners)

# Load data

library(data.table)

data_dt = fread("dataset.csv")

dim(data_dt)
names(data_dt)

### Supp columns not used for prediction

cols_supp <- c(1:2, (ncol(data_dt) - 1))

# Under-sampling to 1000-first rows only

data_for_pred <- data_dt[1:1000, -..cols_supp]

# Factor supported by mlr3 tasks

colonnes_texte <- names(data_for_pred)[sapply(data_for_pred, is.character)]

if (length(colonnes_texte) > 0) {
  data_for_pred[, (colonnes_texte) := lapply(.SD, as.factor), .SDcols = colonnes_texte]
}
data_for_pred[, oym := as.factor(oym)]


# Subset Gender(M,F)

#Implemented this function to not repeat 3 times the proportion calculi
calculate_proportion <- function(data, group_col) {
  data[, .(
    patients_count = .N, 
    proportion = round((.N / nrow(data)) * 100, 2)
  ), by = group_col][order(-proportion)]
}

calculate_proportion(data_for_pred, data_for_pred$gender)

data_gender_F <- data_for_pred[gender == 'F']

calculate_proportion(data_gender_F, data_gender_F$oym)

data_gender_M <- data_for_pred[gender == 'M']

calculate_proportion(data_gender_M, data_gender_M$oym)

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

design <- benchmark_grid(
  tasks = list(task_gender_F, task_gender_M),
  learners = learners,
  resamplings = kfold_cv
)

#Training

print("Train Benchmark...")

bmr <- benchmark(design)

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

#Graphic for AUC

library(ggplot2)

tab_auc <- tab_avg_score[, .(task_id, learner_id, Mean_AUC)]

ggplot(tab_auc, aes(x = learner_id, y = Mean_AUC, fill = task_id)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Performance AUC per Model",
    x = "Algorithms",
    y = "Score AUC",
    fill = "Group"
  ) +
  theme_minimal()
