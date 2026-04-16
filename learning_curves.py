"""
Stretch 5A-S2: Learning Curves Diagnostic
Module 5 Week A — Honors Track

Diagnoses bias vs variance for logistic regression on REAL telecom churn dataset
using sklearn.model_selection.learning_curve.
Includes dummy baselines and multiple visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. Load and preprocess REAL telecom churn dataset
# ============================================================
df = pd.read_csv("Data/telecom_churn.csv")  # تأكد من وجود الملف في المسار الصحيح
df = df.drop(columns=["customer_id"])

categorical_cols = ["gender", "contract_type", "internet_service",
                    "payment_method", "has_partner", "has_dependents"]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X_raw = df_encoded.drop(columns=["churned"]).values
y = df_encoded["churned"].values

print("=" * 60)
print("Dataset summary (REAL telecom churn data)")
print("=" * 60)
print(f"  Rows: {X_raw.shape[0]}, Features: {X_raw.shape[1]}")
churn_rate = y.mean() * 100
print(f"  Churn rate: {churn_rate:.1f}%  →  class imbalance present")

# ============================================================
# 2. Model definitions
# ============================================================
def make_lr_pipeline(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight="balanced",
                                   solver="lbfgs", max_iter=1000, random_state=42))
    ])

def make_dummy_pipeline(strategy):
    return Pipeline([
        ("clf", DummyClassifier(strategy=strategy, random_state=42))
    ])  # no scaler needed

models = {
    "LR C=0.01": make_lr_pipeline(0.01),
    "LR C=1.0":  make_lr_pipeline(1.0),
    "LR C=10":   make_lr_pipeline(10.0),
    "LR C=100":  make_lr_pipeline(100.0),
    "Dummy (most_frequent)": make_dummy_pipeline("most_frequent"),
    "Dummy (stratified)":    make_dummy_pipeline("stratified"),
}

# ============================================================
# 3. Learning curve parameters
# ============================================================
SCORING = "f1_macro"
TRAIN_SIZES = np.linspace(0.10, 1.0, 10)   # 10 points
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nComputing learning curves for all models...")
results = {}
for name, model in models.items():
    print(f"  {name} ...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_raw, y=y,
        train_sizes=TRAIN_SIZES,
        cv=CV,
        scoring=SCORING,
        n_jobs=-1,
    )
    results[name] = {
        "sizes": train_sizes_abs,
        "train_mean": train_scores.mean(axis=1),
        "train_std":  train_scores.std(axis=1),
        "val_mean":   val_scores.mean(axis=1),
        "val_std":    val_scores.std(axis=1),
    }
    final_gap = results[name]["train_mean"][-1] - results[name]["val_mean"][-1]
    print(f"    final train={results[name]['train_mean'][-1]:.3f}, "
          f"val={results[name]['val_mean'][-1]:.3f}, gap={final_gap:.3f}")

# ============================================================
# 4. Plot 1: Original three-panel figure (LR C=0.01, 1.0, 100)
# ============================================================
selected = ["LR C=0.01", "LR C=1.0", "LR C=100"]
colors = {"LR C=0.01": ("#2196F3", "#90CAF9"),
          "LR C=1.0":  ("#4CAF50", "#A5D6A7"),
          "LR C=100":  ("#F44336", "#EF9A9A")}

fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("#0F1117")
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

for ax_idx, name in enumerate(selected):
    r = results[name]
    ax = fig.add_subplot(gs[ax_idx])
    ax.set_facecolor("#1A1D27")
    c_solid, c_band = colors[name]

    ax.plot(r["sizes"], r["train_mean"], color=c_solid, linewidth=2.2, label="Training", zorder=3)
    ax.fill_between(r["sizes"], r["train_mean"]-r["train_std"], r["train_mean"]+r["train_std"],
                    alpha=0.18, color=c_solid)
    ax.plot(r["sizes"], r["val_mean"], color=c_solid, linewidth=2.2, linestyle="--", label="Validation", zorder=3)
    ax.fill_between(r["sizes"], r["val_mean"]-r["val_std"], r["val_mean"]+r["val_std"],
                    alpha=0.30, color=c_band)

    final_train = r["train_mean"][-1]
    final_val = r["val_mean"][-1]
    gap = final_train - final_val
    ax.annotate(f"gap = {gap:.3f}", xy=(r["sizes"][-1], (final_train+final_val)/2),
                xytext=(-80,0), textcoords="offset points", fontsize=8, color="#FFD54F",
                arrowprops=dict(arrowstyle="->", color="#FFD54F", lw=1.2), va="center")

    ax.set_title(name, color="white", fontsize=10, pad=8)
    ax.set_xlabel("Training set size", color="#AAAAAA", fontsize=9)
    ax.set_ylabel("F1-macro score", color="#AAAAAA", fontsize=9)
    ax.tick_params(colors="#AAAAAA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, color="#2A2D3A", linewidth=0.6, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.25, labelcolor="white", facecolor="#1A1D27", edgecolor="#444444")

fig.suptitle("Learning Curves — Logistic Regression (C=0.01, 1, 100)\nScoring: F1-macro | CV: StratifiedKFold(5)",
             color="white", fontsize=12, y=1.01)
plt.savefig("output/learning_curves_original.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\nSaved: output/learning_curves_original.png")

# ============================================================
# 5. Plot 2: Overlay of train & validation curves for ALL models
# ============================================================
plt.figure(figsize=(10, 6))
plt.style.use('default')
# define a color cycle
model_colors = {name: plt.cm.tab10(i) for i, name in enumerate(results.keys())}

for name, r in results.items():
    color = model_colors[name]
    # training (solid)
    plt.plot(r["sizes"], r["train_mean"], color=color, linestyle='-', linewidth=1.5,
             label=f"{name} (train)")
    plt.fill_between(r["sizes"], r["train_mean"]-r["train_std"], r["train_mean"]+r["train_std"],
                     alpha=0.1, color=color)
    # validation (dashed)
    plt.plot(r["sizes"], r["val_mean"], color=color, linestyle='--', linewidth=1.5,
             label=f"{name} (val)")
    plt.fill_between(r["sizes"], r["val_mean"]-r["val_std"], r["val_mean"]+r["val_std"],
                     alpha=0.2, color=color)

plt.xlabel("Training set size", fontsize=11)
plt.ylabel("F1-macro score", fontsize=11)
plt.title("Learning Curves Overlay (All Models) — Train (solid) / Val (dashed)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("output/learning_curves_overlay.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: output/learning_curves_overlay.png")

# ============================================================
# 6. Plot 3: Validation curves only (all models) - similar to second screenshot
# ============================================================
plt.figure(figsize=(10, 6))
for name, r in results.items():
    color = model_colors[name]
    plt.plot(r["sizes"], r["val_mean"], color=color, linewidth=2,
             label=f"{name} (validation)", linestyle='-')
    plt.fill_between(r["sizes"], r["val_mean"]-r["val_std"], r["val_mean"]+r["val_std"],
                     alpha=0.2, color=color)

plt.xlabel("Training set size", fontsize=11)
plt.ylabel("F1-macro score", fontsize=11)
plt.title("Validation Learning Curves for All Models", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("output/validation_curves_all_models.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: output/validation_curves_all_models.png")

print("\nAll plots saved successfully in 'output/' directory.")