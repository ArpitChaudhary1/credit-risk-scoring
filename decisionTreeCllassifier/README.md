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
