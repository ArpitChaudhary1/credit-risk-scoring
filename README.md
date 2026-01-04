# Loan Approval System

## 🎯Mission of the Project
The objective of the project is to construct a reliable machine learning system for the correct prediction of loan defaulters. In the banking sector, the cost of missing a potential defaulter is far greater than the cost of accidentally rejecting a safe applicant. It compares four different machine learning architectures to find the optimal balance between risk detection (Recall) and operational efficiency (Accuracy/Precision).

---

## 🏦 The Business Problem
It is a highly biased dataset, with only 17% of the applicants being defaulters (Class 0).

- The Goal: Maximize the detection of Class 0 (Defaulters).
- The Metric: Our primary focus in Recall is on Class 0 to ensure the bank maintains a minimized loss metric.

---

## 📊 Dataset Overview
- Total Records: 2,36,846
- Target Variable: loan_status
- Class 1 (Safe): 1,96,959 (83%)
- Class 0 (Defaulter): 39,887 (17%)
- Features: 36 variables including credit scores, income, loan amounts, and debt-to-income ratios.

---

## 🛠️ Model Performance Summary
Each model was tuned using RandomizedSearchCV with a focus on maximizing the recall of the minority class.

```bash
Model            Class 0 Recall     Class 0 Precision         Accuracy        About
Decision Tree         0.76               0.31                   0.67         Highest sensitivity to risk.
XGBoost               0.71               0.34                   0.71         Optimized via weighted log-loss.
Random Forest         0.71               0.34                   0.72         Most stable/robust ensemble.
Logistic Regression   0.69               0.36                   0.74         Baseline linear interpretation.
```

---

## 📂 Repository Structure

```bash
.
├── LogisticRegressionModel                   
│   ├── LogisticModel.pkl
│   ├── model_traning.ipynb
│   └── README.md
|
├── Random Forest Model
│   ├── Randomforest.pkl
│   ├── randomForest.ipynb
│   └── README.md
│   
├── XGBoost Classifier
│   ├── XGBoost.ipynb
│   ├── xgb_model.pkl
│   └── README.md
|
├── decisionTreeCllassifier
│   ├── DTC.ipynb
│   ├── loan_decision_tree.pkl
│   └── README.md
|
├── main.py                        
├── lending_club_cleaned_v1.csv        
├── requirements.txt             
└── README.md                    

```

---

## ⚙️ Technical Highlights
- **Imbalance Management**: Leveraged class_weight='balanced', scale_pos_weight, and custom scoring functions (make_scorer).
- **Hyperparameter Tuning**: Conducted extensive searches via RandomizedSearchCV to optimize depth, leaf size, and ensemble power.
- **Validation**: All the models have been validated with a n-fold cross validation on 230k+ rows for generalization.









