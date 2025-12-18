# Fraud Detection System using Machine Learning

## Overview
This project implements a machine learning-based fraud detection system using XGBoost to identify fraudulent transactions from highly imbalanced financial data.

## Features
- Handles class imbalance using scale_pos_weight
- Uses XGBoost for high-performance classification
- Evaluates model using ROC-AUC, confusion matrix, and precision/recall

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost

## Dataset
Credit Card Fraud Detection Dataset (Kaggle)

## Results
- High recall on fraudulent transactions
- ROC-AUC score optimized for imbalanced data

## How to Run
```bash
pip install -r requirements.txt
python fraud_detection.py

