# Customer Churn Prediction (PyTorch)

##  Project Overview
A Deep Learning pipeline built with PyTorch to predict customer churn for a telecommunications company. 

##  Architecture & Pipeline
- **Data Preprocessing:** One-Hot Encoding for categorical features, Standard Scaling for numerical features.
- **Data Splits:** 70% Train, 15% Validation, 15% Test.
- **Model:** Multi-Layer Perceptron (30 → 64 → 32 → 1) with Batch Normalization and Dropout (0.2).
- **Loss & Optimizer:** `BCEWithLogitsLoss` and `Adam`.
- **Training:** 50 Epochs with Best-Model Checkpointing based on Validation Loss.

##  Final Results
- **Test Accuracy:** 79.00%
- **Test Loss:** 0.4210
- The model successfully loads from a saved state dict (`best_churn_model.pt`) and performs inference on unseen data.
