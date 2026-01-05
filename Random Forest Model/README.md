# Random Forest Classifier

## Model Overview
Random Forest Classifier implementation for loan default prediction with ensemble learning, optimized for maximum recall on the minority class (defaulters).

---

## Training Configuration
**Key Parameters**:
- bootstrap: True (enables bagging)
- oob_score: True (Out-of-Bag validation)
- n_jobs: -1 (uses all CPU cores for parallel training)
- random_state: 42
- class_weight: 'balanced_subsample' (handles class imbalance per bootstrap sample)


## Data Preprocessing
- **Scaling** : Not required for Random Forest (tree-based model)
- Training on raw, unscaled data
- Random Forest is invariant to feature scaling
- **Features Used**: 35 features (all except loan_status)
- **Train-Test Split**: 80-20 ratio (random_state=42)

## Class Distribution:
- Class 0 (Default): 39,887 samples
- Class 1 (Paid): 196,959 samples

## Hyperparameter Optimization

- **Method**: RandomizedSearchCV
- **n_iter**: 15 combinations tested
- **cv**: 3-fold cross-validation
- **Optimization metric**: Recall for class 0 (defaulters)

## Search Space
```
'n_estimators': [100, 200, 300, 400, 500],
'max_depth': [5, 10, 15, 20],
'min_samples_split': [5, 25, 45, 65, 85, 105, 125, 145, 165, 185],
'min_samples_leaf': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
'max_features': ['sqrt', 'log2']
```
## Test Set Performance:
```
              precision    recall  f1-score   support
           0       0.34      0.71      0.46      7951
           1       0.93      0.73      0.81     39419
    accuracy                           0.72     47370
   macro avg       0.64      0.72      0.64     47370
weighted avg       0.83      0.72      0.76     47370
```

## Key Metrics:

- Recall for Defaulters (Class 0): 71% - Identifies 7 out of 10 potential defaulters
- Precision for Non-Defaulters (Class 1): 93% - High confidence in approved loans
- Overall Accuracy: 72%
- OOB Score: 84.76% - Strong internal validation

## Key Techniques Used:

- Bootstrap Aggregating (Bagging) - Each tree trained on random sample
- balanced_subsample - Balances classes within each bootstrap sample
- Random Feature Selection - Each split considers sqrt(35) ≈ 6 features

## Technical Details
**Bootstrap and OOB Score**
- **Bootstrap Sampling**:
  - Each tree trained on random sample with replacement
  - Left-out samples used for OOB validation

- **OOB Score (84.76%)**:
  - Unbiased estimate of test performance
  - Higher than test accuracy (72%) indicates good generalization
  - No overfitting detected

## How to use model












