# medical_data_exploration

### Goal

#### Step 1 :

- Find a good way to divide the features in sub-group
- Produce curves to have a visual on what sub-group is relevant or not
- Find some balanced/imbalanced sub-group to work on and compare

#### Step 2 :

- Chose a few models(linear + regularization or non-linear (torch maybe)) to train on each sub-groups
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
