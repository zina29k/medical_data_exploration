# Load data

library(data.table)

data_dt = fread("dataset.csv")

### Supp columns not used for prediction

cols_supp <- c(1:2, (ncol(data_dt) - 1):ncol(data_dt))

data_for_pred <- data_dt[, -..cols_supp]

# Choose new target

names(data_for_pred)

# Readmission aux urgences

print("(is_urg_readm)")
data_for_pred[, .(
  patients = .N, 
  proportion = round((.N / nrow(data_dt)) * 100, 2)
), by = is_urg_readm][order(-proportion)]

# Transfert en Soins Intensifs

print("(is_icu_start_ho)")
data_for_pred[, .(
  patients = .N, 
  proportion = round((.N / nrow(data_dt)) * 100, 2)
), by = is_icu_start_ho][order(-proportion)]

# Test first model for these 2 problems

library(mlr3)
library(mlr3extralearners)
library(randomForest)

# Define tasks

colonnes_texte <- names(data_for_pred)[sapply(data_for_pred, is.character)]

if (length(colonnes_texte) > 0) {
  data_for_pred[, (colonnes_texte) := lapply(.SD, as.factor), .SDcols = colonnes_texte]
}

task_urg <- TaskClassif$new(
  id = "Urgent Readmission Problem",
  backend = data_for_pred[, is_urg_readm := as.factor(is_urg_readm)],
  target = "is_urg_readm"
)

task_icu <- TaskClassif$new(
  id = "Intensive Care Unit Problem",
  backend = data_for_pred[, is_icu_start_ho := as.factor(is_icu_start_ho)],
  target = "is_icu_start_ho"
)

#Split train/test

  ## Proportion is maintained for each class during the split
task_urg$col_roles$stratum <- "is_urg_readm"
task_icu$col_roles$stratum <- "is_icu_start_ho"

  ## Split 80/20
splits_urg <- partition(task_urg, ratio = 0.80)
splits_icu <- partition(task_icu, ratio = 0.80)


#RF model train and prediction

model = lrn("classif.randomForest")

model$train(task_urg, row_ids = splits_urg$train)

predictions_urg <- model$predict(task_urg, row_ids = splits_urg$test)

#Metrics

print("Accuracy :")
score_acc <- predictions_urg$score(msr("classif.acc"))
print("Confusion matrix :")
print(predictions_urg$confusion)

