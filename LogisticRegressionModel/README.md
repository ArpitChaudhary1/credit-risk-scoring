# Logistic Regression Model 
Logistic Regression implementation for loan default prediction with regularization optimization using L2 penalty (Ridge regularization).

---

## Training Configuration
**Model Type** : Logistic Regression with balanced class weights

## Key Parameters:
**max_iter**: 1000 (initial), 100 (final optimized)
**class_weight**: 'balanced'
C: 1/951 (optimized through grid search)

**Features Used**: 35 features (all except loan_status)
**Train-Test Split**: 80-20 ratio (random_state=42)

## Class Distribution:
- Class 0 (Defaulters): 39,887 samples
- Class 1 (Paid): 196,959 samples

## Hyperparameter Optimization
**Method**: Custom grid search over regularization parameter
**Classification Report:**
```
              precision    recall  f1-score   support
           0       0.36      0.69      0.47      7951
           1       0.92      0.75      0.83     39419
    accuracy                           0.74     47370
   macro avg       0.64      0.72      0.65     47370
weighted avg       0.83      0.74      0.77     47370
```
## Key Metrics:
- Recall for Defaulters (Class 0): 69% - Identifies 7 out of 10 potential defaulters
- Precision for Non-Defaulters (Class 1): 92% - High confidence in approved loans
- Overall Accuracy: 74%
- Balanced class handling through class_weight parameter

## How to use
1. Load the Model
```
import joblib
model = joblib.load('LogisticModel.pkl')
predictions = model.predict(scaled_data)  # Remember to scale
```
2. Critical Requirements
⚠️ Data Must Be Scaled:
Logistic Regression requires feature scaling. Always apply StandardScaler before prediction:
```
from sklearn.preprocessing import StandardScaler

# Load scaler (you need to save this separately)
scaler = StandardScaler()
scaler.fit(training_data)  # Fit on training data once

# Scale new data before prediction
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
```


##Use Case Recommendations
**Best For**:
- Baseline model comparison
- Applications requiring model interpretability
- Fast predictions in production
- When probabilistic outputs are needed
- Balanced accuracy and recall requirements

**Not Ideal For**:
Maximum default detection (use Decision Tree recall-optimized)
Complex non-linear patterns
When recall > 70% is critical
