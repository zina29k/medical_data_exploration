# Load data

library(data.table)

data_dt = fread("dataset.csv")

### Supp non-used col for prediction

cols_supp <- c(1:2, (ncol(data_dt) - 1):ncol(data_dt))

data_for_pred <- data_dt[, -..cols_supp]

# Choose new target

names(data_for_pred)

# Readmission aux urgences

print("(is_urg_readm)")
data_for_pred[, .(
  Patients = .N, 
  Pourcentage = round((.N / nrow(data_dt)) * 100, 2)
), by = is_urg_readm][order(-Pourcentage)]

# Transfert en Soins Intensifs

print("(is_icu_start_ho)")
data_for_pred[, .(
  Patients = .N, 
  Pourcentage = round((.N / nrow(data_dt)) * 100, 2)
), by = is_icu_start_ho][order(-Pourcentage)]

