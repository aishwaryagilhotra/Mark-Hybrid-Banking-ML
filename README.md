# Mark-AI_Financial_Advisor
Python | Scikit-Learn | XGBoost | LightGBM | Pandas | Matplotlib

Hybrid Machine Learning Pipeline with Regression and Classification
This project implements a hybrid ML framework that:
- Classifies customers based on financial risk
- Predicts portfolio allocation/interest rate
- Integrates macroeconomic indicators (Repo, CRR, SLR)
- Applies cross-validation, threshold tuning, and advanced evaluation metrics
- The system combines Random Forest, XGBoost, and LightGBM into a unified decision pipeline.

# Dataset used
1. Monetary Policies from RBI DBIE
2. Bank Customer Data from Kaggle
3. Bank-Interest Rates from Kaggle

# Architecture

<img width="740" height="2072" alt="Blank diagram" src="https://github.com/user-attachments/assets/8ca985f3-f5fc-4f4e-996c-dcb2ea0d6cf6" />

# Project Structure 

MARK-Hybrid-Banking-ML/
│
├── data/
│   ├── BankCustomerData.csv
│   ├── Banks-Interest-Rates.csv
│   ├── monetary policies.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── classification_models.py
│   ├── regression_models.py
│   ├── hybrid_pipeline.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│
├── requirements.txt
└── README.md

# Key Features 

Hybrid ML architecture (Classification + Regression)
- Cross-validation (Stratified K-Fold)
- ROC-AUC & PR-AUC optimization
- Threshold tuning
- Class imbalance handling
- Macro-aware modeling

# Evaluation Metrics
Classification:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Regression:
- RMSE
- MAE
- R² Score
- Cross-validated RMSE

# Best Performing Models

Classification: LightGBM (Highest ROC-AUC)
Regression: XGBoost (Lowest RMSE, Highest R²)

# Future Improvements

- Real-time economic data API integration
- Explainable AI (SHAP values)
- Web-based deployment
- Auto model retraining
