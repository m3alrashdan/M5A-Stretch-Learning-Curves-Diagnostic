#  Learning Curves Diagnostic – Telecom Churn Prediction

##  Project Overview

This project is part of the **Honors Track** (Module 5, Week A – Stretch Assignment).
It uses `sklearn.model_selection.learning_curve` to diagnose whether a logistic regression model suffers from:

* **High Bias (Underfitting)**
* **High Variance (Overfitting)**

We use a telecom churn dataset (≈1500 customers, 15 features after encoding).
The churn rate is **~16.3%**, which introduces class imbalance.

To handle this:

* We use `f1_macro` as the evaluation metric
* We apply `class_weight="balanced"` in the model

---

##  Repository Structure


M5A-Stretch-Learning-Curves-Diagnostic/
│
├── learning_curves.py
├── telecom_churn.csv
├── output/
│   ├── learning_curves_original.png
│   ├── learning_curves_overlay.png
│   └── validation_curves_all_models.png
├── README.md
└── requirements.txt
```

---

##  Setup & Requirements

### 1. Python Environment

* Python 3.8+
* Recommended: virtual environment

---

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

Or create a `requirements.txt`:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

Then run:

```bash
pip install -r requirements.txt
```

---

### 3. File Placement

* Place `telecom_churn.csv` in the same directory as `learning_curves.py`
* The script will automatically create the `output/` folder

---

##  How to Run

Run the script from terminal:

```bash
python learning_curves.py
```

---

##  Expected Output (Example)

```
============================================================
Dataset summary (REAL telecom churn data)
============================================================
Rows: 1500, Features: 15
Churn rate: 16.3% → class imbalance present

Computing learning curves for all models...
LR C=0.01 ... final train=0.557, val=0.532, gap=0.025
LR C=1.0  ... final train=0.554, val=0.537, gap=0.018
LR C=10   ... final train=0.555, val=0.537, gap=0.018
LR C=100  ... final train=0.555, val=0.537, gap=0.018
Dummy (most_frequent) ... final train=0.456, val=0.456, gap=0.000
Dummy (stratified)    ... final train=0.507, val=0.505, gap=0.001

Saved: output/learning_curves_original.png
Saved: output/learning_curves_overlay.png
Saved: output/validation_curves_all_models.png
```

---

##  Models Evaluated

### Logistic Regression

* C = 0.01 → Strong regularisation (higher bias)
* C = 1.0 → Default
* C = 10
* C = 100 → Weak regularisation (higher variance)

### Dummy Models (Baselines)

* **most_frequent** → Always predicts majority class
* **stratified** → Random predictions based on class distribution

---

## Results Summary

| Model                 | Train F1 | Validation F1 | Gap   |
| --------------------- | -------- | ------------- | ----- |
| LR (C=0.01)           | 0.557    | 0.532         | 0.025 |
| LR (C=1.0)            | 0.554    | 0.537         | 0.018 |
| LR (C=10)             | 0.555    | 0.537         | 0.018 |
| LR (C=100)            | 0.555    | 0.537         | 0.018 |
| Dummy (most_frequent) | 0.456    | 0.456         | 0.000 |
| Dummy (stratified)    | 0.507    | 0.505         | 0.001 |

---

##  Key Observations

* Logistic regression models outperform dummy baselines
* Overall performance is **moderate (F1 ≈ 0.53–0.56)**
* Train vs validation gap is **very small (≤ 0.025)**
* Increasing dataset size does **not significantly improve performance**
* Changing regularisation (`C`) has **minimal impact**

---

## Final Diagnosis

 The model is suffering from **High Bias (Underfitting)**
