# Network Failure Prediction using Logistic Regression

## Overview
This project implements a **machine learning model** to predict network failures based on key performance metrics.

It trains a **Logistic Regression model** that classifies whether a network state is *Normal* or *Failure*.


##  Objectives
- Simulate network performance data  
- Train a classification model  
- Predict potential network failures  
- Evaluate model performance  

##  Dataset
Synthetic dataset with the following features:

- `packet_loss` (%)
- `latency` (ms)
- `throughput` (Mbps)
- `cpu_usage` (%)
- `memory_usage` (%)

###  Target Variable
- `failure`:
  - `0` → Normal  
  - `1` → Failure  

Failure is defined as:
- Packet loss > 5% OR  
- Latency > 200 ms OR  
- CPU usage > 85%  

---

##  Technologies Used
- Language: Python  
- Libraries:
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib  
  - seaborn  
  - joblib  

---

##  Workflow

### 1. Data Generation
- Create synthetic network dataset  
- Save as `network_data.csv`  

### 2. Data Preparation
- Split dataset into:
  - Training set (80%)  
  - Testing set (20%)  

### 3. Model Training
- Train **Logistic Regression** model  
- Analyze feature importance (coefficients)  

### 4. Prediction
- Predict:
  - Failure labels (0 / 1)  
  - Failure probabilities  

### 5. Evaluation
- Confusion matrix  
- Prediction vs actual comparison  

### 6. Model Saving
- Export trained model as:
  ```bash
  network_model.pkl
