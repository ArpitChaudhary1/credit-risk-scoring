# XGBoost Classifier
XGBoost (eXtreme Gradient Boosting) Classifier implementation for loan default prediction with gradient boosting, optimized for maximum recall on the minority class (defaulters).

## Training Configuration
### Model Type
XGBoost Classifier with gradient boosting\

**Key Parameters**
- n_estimators
- learning_rate
- max_depth
- min_child_weight
- gamma (minimum loss reduction for split)
- subsample (fraction of samples per tree)
- colsample_bytree (fraction of features per tree)
- scale_pos_weight (balances class weight)
- random_state 
- eval_metric: 'logloss'


## Data Preprocessing
### Label Reversal Strategy
XGBoost defaults to treating class 1 as the positive class and automatically increases its weight via scale_pos_weight. This caused the model to overfit on non-defaulters (class 1) and fail to detect defaulters (class 0).

### Solution Implemented:
```
pythony_train_rev = 1 - y_train  # Class 0 becomes 1, Class 1 becomes 0
y_test_rev = 1 - y_test
```
After reversal, defaulters become the positive class (1), allowing scale_pos_weight=5 to correctly prioritize default detection.
### Scaling
Not required for XGBoost (tree-based model)

- Training on raw, unscaled data
- XGBoost is invariant to feature scaling

### Features Used
35 features (all except loan_status)
### Train-Test Split
80-20 ratio (random_state=42)

### Class Distribution (Original Labels)

- Class 0 (Default): 39,887 samples
- Class 1 (Paid): 196,959 samples
- Imbalance ratio: ~1:5

## Hyperparameter Optimization
**Method:** RandomizedSearchCV

- n_iter: 15 combinations tested
- cv: 3-fold cross-validation
- Optimization metric: Recall for defaulters (positive class after label reversal)

## Search Space
```
{
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 8, 10],
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [5]
}
```
## Model Performance

### Test Set Performance (After Label Reversal)
```
              precision    recall  f1-score   support
           0       0.92      0.72      0.81     39419
           1       0.34      0.71      0.46      7951
    accuracy                           0.71     47370
   macro avg       0.63      0.71      0.63     47370
weighted avg       0.83      0.71      0.75     47370
```
Note: After reversal, class 0 = paid loans, class 1 = defaulters
### Key Metrics

- Recall for Defaulters (Class 1): 71% - Identifies 7 out of 10 potential defaulters
- Precision for Non-Defaulters (Class 0): 92% - High confidence in approved loans
- Overall Accuracy: 71%
- Balanced Performance: 71% recall on both classes

## Regularization Strategy
- gamma: Requires minimum loss reduction for new splits (prevents weak splits)
- min_child_weight: Each leaf must have at least 'n' samples (prevents overfitting)
- learning_rate: Small steps prevent overfitting (requires more trees but more stable)

## How to use the model
1. Load the model
```
import joblib

# Load the saved model
model = joblib.load('Randomforest.pkl')
```
2. Prepare the new data (Example of one applicant)
```
new_applicant = pd.DataFrame({
    'income': [50000],
    'credit_score': [650],
    'loan_amount': [15000],
    'debt_to_income': [0.25],
    # ... add all other features here
})
```
3. Get the Prediction (0 or 1)
```
prediction = model.predict(new_applicant)
```
Print the Prediction
```
print(f"Prediction: {'Defaulter' if prediction[0] == 0 else 'Safe'}")
```












