# medical_data_exploration

### Experiment 1 : Subset Gender

#### Step 1 : Evaluate the baseline performance of each model on each gender dataset independently

- Undersampling data for faster results 1:1000

- Divide dataset in 2 Gender (M,F) (good proportion 50.5 \# 49.5)

- Tasks (mlr3) for the 2 datasets -\> target "oym" one year mortality

- Learners : featureless, knn, glmnet, randomForest

- Resamplings : kfoldcv (5-fold)

- Create the grid_benchmark(tasks,learners,Resamplings) -\> Train

- Evaluation of the preformance of each model: mesures(AUC, CE, TPR, FPR, FNR, TNR)

- Diagrams on AUC for a better visualization of the results

#### Step 2 : Evaluate if a model trained exclusively on one gender generalizes well and maintains its performance when tested on the opposite gender

### Experiment 2 : Subset Age

### Presentation of the dataset : Synthetic data based on real data from CHUS Sherbrooke

data source link : <https://zenodo.org/records/12954673>

- 248 columns (minus first 2 and last 2) -\> 244 used for prediction
- 248 485 rows (visits) ≃ 123 646 pats
