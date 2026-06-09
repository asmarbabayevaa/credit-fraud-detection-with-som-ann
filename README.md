# Credit Fraud Detection with SOM + ANN

A hybrid unsupervised-supervised machine learning pipeline that detects potentially fraudulent credit applications using a Self-Organizing Map (SOM) for anomaly detection, followed by an Artificial Neural Network (ANN) for classification.

---

## Overview

This project combines two models in a sequential pipeline:

1. **SOM (Self-Organizing Map)** — unsupervised model that clusters customers and identifies suspicious/anomalous nodes on the map
2. **ANN (Artificial Neural Network)** — supervised binary classifier trained on SOM-generated fraud labels

Both models are tuned automatically using **Optuna** hyperparameter optimization.

---

## Pipeline

```
Raw Data
   ↓
EDA + Missing Value Imputation + Outlier Capping
   ↓
MinMaxScaler → SOM Training (Optuna tuned)
   ↓
Identify Suspicious Nodes (distance ≥ 0.9)
   ↓
Label Fraud Candidates → is_fraud = 1
   ↓
StandardScaler → ANN Training (Optuna tuned)
   ↓
Evaluate with Gini Coefficient
   ↓
Predict on New Deployment Data
```

---

## Model Details

### SOM Hyperparameters (Optuna tuned)

| Parameter | Search Space |
|-----------|-------------|
| Grid size (x, y) | 10 to 20 |
| Sigma | 0.1 to 2.0 |
| Learning rate | 0.01 to 1.0 |
| Iterations | 50 to 500 |

Evaluated using **Silhouette Score** (maximize)

### ANN Architecture & Hyperparameters (Optuna tuned)

| Parameter | Search Space |
|-----------|-------------|
| Units Layer 1 | 6 to 32 |
| Units Layer 2 | 6 to 32 |
| Optimizer | Adam, SGD, RMSprop, Adagrad |
| Learning rate | 1e-5 to 1e-2 |
| Epochs | 10 to 50 |
| Batch size | 16 to 64 |

Evaluated using **AUC score** (maximize)

---

## Evaluation Metric

The final model is evaluated using the **Gini Coefficient**, a standard metric in the credit risk industry:

```
Gini = 2 × AUC − 1
```

| Gini Range | Interpretation |
|------------|---------------|
| < 0.2 | Weak model |
| 0.2 – 0.4 | Acceptable |
| 0.4 – 0.6 | Good |
| > 0.6 | Excellent |

