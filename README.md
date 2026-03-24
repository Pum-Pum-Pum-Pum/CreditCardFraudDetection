# Credit Card Fraud Detection

Production-style machine learning project for fraud detection using the anonymized European cardholder transactions dataset.

## 1. Project Objective

Build an end-to-end fraud detection workflow that goes beyond notebook experimentation and covers:

- exploratory data analysis (EDA)
- data preparation
- baseline modeling
- hyperparameter tuning
- cost-based threshold tuning
- final evaluation and error analysis
- model packaging
- API serving
- inference logging
- drift monitoring

This project is designed with an interview-ready and production-minded workflow.

---

## 2. Dataset Summary

- Rows: **284,807**
- Columns: **31**
- Fraud cases: **492**
- Fraud rate: **0.1727%**

Columns:
- `Time`
- `V1` to `V28` (PCA-transformed anonymized features)
- `Amount`
- `Class` (`1` = fraud, `0` = non-fraud)

Key challenge:
- This is an **extremely imbalanced classification problem**, so accuracy is not a useful primary metric.

---

## 3. Project Structure

```text
.
├── data/
│   └── raw/
│       └── creditcard.csv
├── artifacts/
│   ├── splits/
│   └── model/
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   └── prepare_data.py
│   ├── evaluation/
│   │   ├── error_analysis.py
│   │   └── drift_monitor.py
│   └── models/
│       ├── train_baseline.py
│       ├── tune_models.py
│       ├── cost_based_threshold.py
│       ├── advanced_models.py
│       ├── xgboost_cost_threshold.py
│       ├── package_model.py
│       └── predict.py
└── README.md
```

---

## 4. Workflow Summary

### Step 1 — EDA
- inspected schema, distributions, imbalance, duplicates, outliers, correlations, and statistical tests
- found extreme imbalance and duplicate rows that required careful handling

### Step 2 — Data Preparation
- removed exact duplicates before splitting
- used stratified train/validation/test split
- preserved train/validation/test separation for honest selection

### Step 3 — Baseline Models
- Logistic Regression
- Random Forest

### Step 4 — Hyperparameter Tuning
- tuned candidate models using `GridSearchCV`
- used **average precision / PR-AUC** as the scoring metric

### Step 5 — Cost-Based Threshold Tuning
- used explicit business cost assumptions
- tested thresholds on validation instead of using default `0.5`

### Step 6 — Final Evaluation and Error Analysis
- analyzed false positives and false negatives
- inspected model failure behavior rather than only reporting metrics

### Step 7 — Advanced Model Comparison
- compared Random Forest, AdaBoost, and XGBoost
- selected XGBoost as the strongest advanced candidate

### Step 8 — XGBoost Cost-Based Threshold Tuning
- tuned threshold for XGBoost specifically
- selected deployment threshold under business assumption `FN:FP = 5:1`

### Step 9 — Model Packaging
- saved final model artifact
- saved metadata with threshold and cost assumptions

### Step 10 — API Serving
- served model using FastAPI
- added inference logging

### Step 11 — Drift Monitoring
- added simple reference-vs-current drift monitoring using feature mean shift

---

## 5. Final Model Choice

Final selected model:
- **XGBoost**

Why it was selected:
- stronger validation PR-AUC than Random Forest and AdaBoost
- lower business cost after threshold tuning
- better precision-recall balance at the chosen operating point

Deployment threshold:
- **0.17**

Business assumption used for threshold tuning:
- `FP_COST = 1`
- `FN_COST = 5`

Interpretation:
- missing fraud is considered 5x more expensive than falsely flagging a legitimate transaction

---

## 6. Final Test Performance

Random Forest test result at chosen cost-based threshold:
- precision ≈ **0.8209**
- recall ≈ **0.7746**
- f1 ≈ **0.7971**

XGBoost test result at chosen cost-based threshold:
- precision ≈ **0.8615**
- recall ≈ **0.7887**
- f1 ≈ **0.8235**
- lower total business cost than Random Forest under the selected cost ratio

Why PR-AUC matters here:
- the fraud class is extremely rare
- PR-AUC is more informative than accuracy for ranking minority-class performance

---

## 7. How to Run

### 7.1 Prepare data
```bash
python src/data/prepare_data.py
```

### 7.2 Train baseline models
```bash
python src/models/train_baseline.py
```

### 7.3 Tune baseline models
```bash
python src/models/tune_models.py
```

### 7.4 Compare advanced models
```bash
python src/models/advanced_models.py
```

### 7.5 Cost-based threshold tuning for XGBoost
```bash
python src/models/xgboost_cost_threshold.py
```

### 7.6 Final evaluation and error analysis
```bash
python src/evaluation/error_analysis.py
```

### 7.7 Package final model
```bash
python src/models/package_model.py
```

### 7.8 Local inference from CSV
```bash
python src/models/predict.py
```

### 7.9 Run the API
```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### 7.10 Run drift monitoring
```bash
python src/evaluation/drift_monitor.py
```

---

## 8. API Endpoints

### Health check
`GET /health`

Returns:
- API status
- model name
- active threshold

### Prediction
`POST /predict`

Returns:
- `fraud_probability`
- `predicted_class`
- `threshold`

Each inference is logged to:
- `artifacts/model/inference_log.jsonl`

---

## 9. Monitoring Notes

Current monitoring includes:
- inference event logging
- simple drift report based on feature mean shifts

Recommended future improvements:
- score-distribution monitoring
- label-based performance monitoring once delayed labels arrive
- PSI / KS / distribution-shape drift checks
- automated alerts for feature drift or score drift thresholds
- scheduled retraining / recalibration policy

---

## 10. Limitations

- Features `V1`–`V28` are PCA-transformed and anonymized, so interpretability is limited.
- Drift monitor is intentionally simple and should be extended for production-grade monitoring.
- Threshold tuning depends on chosen business costs; changing business assumptions requires retuning.
- Fraud labels in real production often arrive late, so true live monitoring is harder than offline evaluation.

---

## 11. Key Production Lessons

- Accuracy is misleading in extreme class imbalance.
- PR-AUC is more useful than accuracy or even ROC-AUC for rare-event ranking.
- Threshold should not default to `0.5`; it must match business cost.
- Model selection is not enough — operating policy matters.
- Saving model metadata is essential for reproducibility.
- Local inference scripts are not the same as production APIs.
- Monitoring must continue after deployment.

---
