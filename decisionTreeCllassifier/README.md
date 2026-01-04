# Decision Tree Classifier

## Model Overview
Decision Tree Classifier implementation for loan default prediction with two training approaches.

## Training Configurations
- Configuration 1: F1-Score Optimized
- Objective: Balanced performance across both classes

## Hyperparameter Search Space:
```
{
    'criterion': ['entropy', 'gini'],
    'max_depth': [0, 5, 10, 20],
    'min_samples_leaf': range(2, 100, 10),
    'min_samples_split': range(1, 50, 5),
    'class_weight': [None, 'balanced']
}
```

**Best Parameters:**
- criterion: 'entropy'
- max_depth: 10
- min_samples_leaf: 82
- min_samples_split: 46
- class_weight: None

**Performance:**
```
              precision    recall  f1-score   support
           0       0.58      0.25      0.35      7951
           1       0.86      0.96      0.91     39419
    accuracy                           0.84     47370
```

**Confusion Matrix:**
```
[[ 2010  5941]
 [ 1432 37987]]
```
**Issue**: Low recall (25%) for defaulters - misses 75% of actual defaults.


# Configuration 2: Recall-Optimized for Defaulters ⭐ FINAL MODEL

**Objective**: Maximize detection of loan defaulters (Class 0)

## Key Changes:
- Custom scoring: recall_score(pos_label=0)
- Forced class_weight='balanced'
- Training on non-scaled data
- Narrower hyperparameter ranges for shallow trees

## Hyperparameter Search Space:
```
{
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, 20],
    'min_samples_leaf': range(5, 100, 10),
    'min_samples_split': range(5, 200, 20),
    'class_weight': ['balanced']
}
```

**Best Parameters:**
- criterion: 'entropy'
- max_depth: 3
- min_samples_leaf: 65
- min_samples_split: 65
- class_weight: 'balanced'

**Performance:**
```
              precision    recall  f1-score   support
           0       0.31      0.76      0.44      7951
           1       0.93      0.65      0.77     39419
    accuracy                           0.67     47370
```


## Key Metrics:

- Recall for Class 0 (Defaulters): 76% ✓
- Successfully identifies 3 out of 4 potential defaulters
- Precision for Class 1: 93% - high confidence in loan approvals


## Data Preprocessing
- Features Used: 36 features (all from cleaned dataset)
- No feature scaling applied (tree-based model)
- Train-test split: 80-20
- Random state: 42

## Training Process
- Optimization Method: RandomizedSearchCV
- n_iter: 50 (Config 1), 30 (Config 2)
- cv: 5-fold cross-validation
- verbose: 2 (Config 1), 1 (Config 2)

## How to use the model
**1. Load the model**
```
import joblib

# Load the saved model
model = joblib.load('loan_decision_tree.pkl')
```
**2. Prepare the new data (Example of one applicant)**
```
new_applicant = pd.DataFrame({
    'income': [50000],
    'credit_score': [650],
    'loan_amount': [15000],
    'debt_to_income': [0.25],
    # ... add all other features here
})
```
**3. Get the Prediction (0 or 1)**
```
prediction = model.predict(new_applicant)
```
**Print the Prediction**
```
print(f"Prediction: {'Defaulter' if prediction[0] == 0 else 'Safe'}")
```














