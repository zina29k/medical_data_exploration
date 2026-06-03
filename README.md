# medical_data_exploration

### Goal

#### Step 1 :

- Change a target to find (find imbalanced axes)
- Binary classification only for now
- Train RF like in the paper for reference
- Produce a visual on what feature/sub-groups of feature are relevant or not for the new target

#### Step 2 :

- Chose a few models(linear + regularization or non-linear (torch maybe)) to solve the new problem
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
