"""
train_model.py
--------------
Trains a Gradient Boosting classifier to predict hospital supply stockouts
14 days in advance. Evaluates on a held-out temporal test split and
saves the trained model, feature importances, and evaluation metrics.
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES = [
    "closing_stock",
    "days_of_supply",
    "avg_consumption_4w",
    "avg_received_4w",
    "stockout_rate_4w",
    "below_safety_rate_4w",
    "avg_supplier_reliability",
    "avg_delay_days",
    "fulfillment_rate",
    "facility_beds",
    "week_of_year",
    "critical_item",
]

TARGET = "stockout_next_14d"
TEST_CUTOFF = "2024-07-01"   # ~75% train / 25% test temporal split

GB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "min_samples_leaf": 10,
    "random_state":     42,
}

# Decision threshold tuned for high recall on stockout events
# (clinical operations prioritize not missing a stockout)
DECISION_THRESHOLD = 0.30

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading ML dataset...")
df = pd.read_csv("data/ml_dataset.csv", parse_dates=["snapshot_date"])

# Encode categorical
le_fac  = LabelEncoder()
le_prod = LabelEncoder()
df["facility_id_enc"] = le_fac.fit_transform(df["facility_id"])
df["product_id_enc"]  = le_prod.fit_transform(df["product_id"])

# Fill any NaN supplier stats with column medians
for col in ["avg_supplier_reliability", "avg_delay_days", "fulfillment_rate"]:
    df[col] = df[col].fillna(df[col].median())

df["critical_item"] = df["critical_item"].astype(int)

# ── Temporal train/test split ─────────────────────────────────────────────────
train_df = df[df["snapshot_date"] <  TEST_CUTOFF]
test_df  = df[df["snapshot_date"] >= TEST_CUTOFF]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]

print(f"  Train: {len(X_train):,} samples | {y_train.mean():.3f} positive rate")
print(f"  Test:  {len(X_test):,}  samples | {y_test.mean():.3f} positive rate")


# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining Gradient Boosting classifier...")
# Compute sample weights to handle class imbalance (~5% positive rate)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
sample_weights = np.where(y_train == 1, pos_weight, 1.0)

model = GradientBoostingClassifier(**GB_PARAMS)
model.fit(X_train, y_train, sample_weight=sample_weights)
print("  Training complete.")


# ── Evaluate ──────────────────────────────────────────────────────────────────
y_prob      = model.predict_proba(X_test)[:, 1]
y_pred      = (y_prob >= DECISION_THRESHOLD).astype(int)   # tuned threshold

roc_auc     = roc_auc_score(y_test, y_prob)
avg_prec    = average_precision_score(y_test, y_prob)
report      = classification_report(y_test, y_pred, output_dict=True)
cm          = confusion_matrix(y_test, y_pred)

print(f"\n── Evaluation Results ───────────────────────────────")
print(f"  ROC-AUC:           {roc_auc:.4f}")
print(f"  Avg Precision:     {avg_prec:.4f}")
print(f"  Precision (pos):   {report['1']['precision']:.4f}")
print(f"  Recall (pos):      {report['1']['recall']:.4f}")
print(f"  F1 (pos):          {report['1']['f1-score']:.4f}")
print(f"  Accuracy:          {report['accuracy']:.4f}")
print(f"\nConfusion Matrix:\n{cm}")


# ── Save metrics ──────────────────────────────────────────────────────────────
metrics = {
    "roc_auc":            round(roc_auc, 4),
    "average_precision":  round(avg_prec, 4),
    "precision_pos":      round(report["1"]["precision"], 4),
    "recall_pos":         round(report["1"]["recall"], 4),
    "f1_pos":             round(report["1"]["f1-score"], 4),
    "accuracy":           round(report["accuracy"], 4),
    "train_samples":      int(len(X_train)),
    "test_samples":       int(len(X_test)),
    "test_positive_rate": round(float(y_test.mean()), 4),
    "test_cutoff":        TEST_CUTOFF,
    "confusion_matrix":   cm.tolist(),
    "decision_threshold": DECISION_THRESHOLD,
    "features":           FEATURES,
}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nMetrics saved to outputs/metrics.json")


# ── Save model ────────────────────────────────────────────────────────────────
with open("models/stockout_model.pkl", "wb") as f:
    pickle.dump({"model": model, "label_encoders": {"facility": le_fac, "product": le_prod}}, f)
print("Model saved to models/stockout_model.pkl")


# ── Feature importance plot ───────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#c0392b" if imp > importances.median() else "#2980b9" for imp in importances]
importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
ax.set_title("Gradient Boosting — Feature Importances\nStockout Prediction (14-Day Horizon)", fontsize=13, fontweight="bold")
ax.axvline(importances.median(), color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Median importance")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Feature importance plot saved.")


# ── ROC curve ────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, color="#c0392b", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
axes[0].set_xlabel("False Positive Rate", fontsize=11)
axes[0].set_ylabel("True Positive Rate", fontsize=11)
axes[0].set_title("ROC Curve", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].spines[["top", "right"]].set_visible(False)

prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(rec, prec, color="#2980b9", lw=2, label=f"PR curve (AP = {avg_prec:.3f})")
axes[1].axhline(y_test.mean(), color="gray", linestyle="--", lw=1, alpha=0.7, label="No-skill baseline")
axes[1].set_xlabel("Recall", fontsize=11)
axes[1].set_ylabel("Precision", fontsize=11)
axes[1].set_title("Precision–Recall Curve", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].spines[["top", "right"]].set_visible(False)

plt.suptitle("Model Evaluation — 14-Day Stockout Prediction", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("ROC/PR curves saved.")


# ── Confusion matrix plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=ax,
    xticklabels=["No Stockout", "Stockout"],
    yticklabels=["No Stockout", "Stockout"],
    linewidths=0.5,
)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Confusion matrix saved.")


# ── Stockout rate by facility ─────────────────────────────────────────────────
test_df = test_df.copy()
test_df["predicted_prob"] = y_prob

fac_summary = (
    test_df.groupby("facility_id")
    .agg(
        actual_stockout_rate=("stockout_next_14d", "mean"),
        predicted_risk_score=("predicted_prob", "mean"),
        n_samples=("stockout_next_14d", "count"),
    )
    .reset_index()
    .sort_values("actual_stockout_rate", ascending=False)
)
fac_summary.to_csv("outputs/facility_risk_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(fac_summary))
w = 0.35
ax.bar(x - w/2, fac_summary["actual_stockout_rate"],  w, label="Actual Stockout Rate",   color="#c0392b", alpha=0.85)
ax.bar(x + w/2, fac_summary["predicted_risk_score"],  w, label="Mean Predicted Risk",     color="#2980b9", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(fac_summary["facility_id"], rotation=0)
ax.set_ylabel("Rate / Score", fontsize=11)
ax.set_title("Per-Facility Stockout Rate vs. Mean Predicted Risk Score\n(Test Period)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/facility_risk_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Facility risk comparison plot saved.")


# ── Stockout rate by product category ────────────────────────────────────────
inv = pd.read_csv("data/inventory_snapshots.csv", parse_dates=["snapshot_date"])
orders = pd.read_csv("data/orders.csv")
prod_cat = orders[["product_id","category","critical_item"]].drop_duplicates()
inv = inv.merge(prod_cat, on=["product_id","critical_item"], how="left")
cat_stockout = inv.groupby("category")["stockout"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
colors_cat = ["#c0392b" if v > cat_stockout.median() else "#2980b9" for v in cat_stockout]
cat_stockout.plot(kind="bar", ax=ax, color=colors_cat, edgecolor="white")
ax.set_xlabel("Product Category", fontsize=11)
ax.set_ylabel("Stockout Frequency", fontsize=11)
ax.set_title("Stockout Frequency by Product Category", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/stockout_by_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("Category stockout plot saved.")


print("\n── All outputs generated successfully. ──────────────────")
