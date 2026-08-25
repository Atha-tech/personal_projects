# personal_projects: Classification Model Comparison: Preprocessing, Cross-Validation & Ensemble Methods

A hands-on machine learning project exploring the full model-building pipeline — from data cleaning through to ensemble methods and hyperparameter tuning — applied to an employee performance dataset alongside benchmark datasets (Iris, Wine).

## Overview

This project investigates which modelling approach best predicts employee performance from features such as age, salary, department, and experience level. It walks through the complete supervised learning workflow: cleaning and encoding raw data, building a leakage-safe preprocessing pipeline, validating models robustly, tuning hyperparameters, and comparing ensemble techniques.

## What's included

**Data preprocessing**
- Missing value imputation (mean for numerical features, mode for categorical)
- Categorical encoding (ordinal, one-hot, and label encoding depending on variable type)
- A reusable `preprocess_data()` pipeline function that fits transformations only on training data to prevent data leakage

**Model validation**
- K-Fold and Stratified K-Fold cross-validation, compared to show the effect of class-balance-aware splitting
- A reusable `run_cross_validation()` function supporting multiple datasets (Iris, Wine, Employee) and multiple classifiers (Logistic Regression, SVM, KNN)

**Model evaluation**
- A `detailed_evaluation()` function reporting accuracy, precision, recall, F1, and a confusion matrix visualization for any trained classifier

**Hyperparameter tuning**
- `GridSearchCV` applied to a Random Forest classifier, with before/after performance comparison

**Ensemble methods**
- Voting Classifier (combining Logistic Regression, SVM, and Decision Tree)
- Bagging Classifier, evaluated across different numbers of base estimators
- A comparison of Random Forest, AdaBoost, and Gradient Boosting using cross-validation

## Tools used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Key takeaway

Stratified K-Fold cross-validation gave more consistent fold-level accuracy than standard K-Fold on the imbalanced employee dataset, and ensemble methods (particularly tuned Random Forest) outperformed single classifiers on this task.

---
*This project was developed as part of my studies in Mathematical Sciences at the Cape Peninsula University of Technology, and has been adapted here to showcase applied machine learning workflow design.*
