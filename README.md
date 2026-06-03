# medical_data_exploration

### Goal

#### Step 1 :

- Change the target (find imbalanced axes)
- Binary classification only for now
- Train RF like in the paper for reference
- Produce a visual on what feature/sub-groups of feature are relevant or not for the new target

#### Step 2 :

- Choose a few models to solve the new problem (torch?)
- Evaluate their performances

#### Step 3 :

- Visualize the performance of each model (curves ROC-AUC or confusion matrix or classification report)
- Write an interpretation to each
- Read related paper or similar project to compare
- Conclusion

### Presentation of the dataset : Synthetic data based on real data from CHUS Sherbrooke

data source link : <https://zenodo.org/records/12954673>

- 248 columns (minus first 2 and last 2) -\> 244 used for prediction
- 248 485 rows (visits) ≃ 123 646 patients

### Reasearch on RF classifier beforehand

link : <https://journals.mesopotamian.press/index.php/BJML/article/view/417>

Authors : Hasan Ahmed Salman ,Ali Kalakech ,Amani Steiti

### Implementation

- Load data and create mlr3 tasks - Split 80/20 train/test (keeping proportion for each class)
- 2 First choosen target : Urgent Readmission , Intensive Care Unit Start (Binary Variables)
- Choice based on how intuitively relevant is the problem
- First Learner : RandomForest without changing any hyperparameters
