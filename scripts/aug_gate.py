"""
Augmented §0 gate check — compares A's basic feature set vs expanded one.

The dataset analysis showed slot geometry (slot_w, slot_h) and (weekday, hour)
joint cells carry meaningful signal. This script verifies whether expanding
A's basic feature set pushes the held-out AUC above 0.65 (= strong baseline).

Usage:
  cd /d "D:\\Research Methodology\\DAAWBC\\dynamic_ad_allocation"
  python scripts\\aug_gate.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data_2" / "shared_auction_log_v4_dense.txt"

print(f"Loading: {FILE}")
df = pd.read_csv(FILE, sep="\t")
print(f"Rows: {len(df):,}, columns: {len(df.columns)}")
print(f"Click rate: {df['click'].mean()*100:.2f}%")
print()

# Discretize market price into quintiles (consistent with A's spec)
df["mp_tier"] = pd.qcut(df["market_price"], 5, labels=False, duplicates="drop")

# Define feature sets
FEATURE_SETS = {
    "BASIC (A's spec)": [
        "mp_tier", "slotvisibility", "slotformat", "device_type", "hour"
    ],
    "EXPANDED (+ weekday + slot geometry)": [
        "mp_tier", "slotvisibility", "slotformat", "device_type",
        "hour", "weekday", "slot_w", "slot_h"
    ],
    "EXPANDED + usertag_count + region": [
        "mp_tier", "slotvisibility", "slotformat", "device_type",
        "hour", "weekday", "slot_w", "slot_h", "usertag_count", "region"
    ],
}

# Train/eval split per A's spec: fit on weekday {3,4}, eval on weekday 5
y = df["click"].values
train_idx = df["weekday"].isin([3, 4]).values
test_idx = df["weekday"].eq(5).values

print(f"Train rows: {train_idx.sum():,}  (weekday 3,4)")
print(f"Test rows:  {test_idx.sum():,}  (weekday 5)")
print(f"Test clicks: {y[test_idx].sum():,}")
print()

print("=" * 78)
print(f"{'Feature set':<45} {'#Features':>10} {'AUC':>10}")
print("=" * 78)

results = {}
for name, feats in FEATURE_SETS.items():
    # Filter to available features
    feats_present = [f for f in feats if f in df.columns]
    missing = set(feats) - set(feats_present)
    if missing:
        print(f"  Warning: missing features {missing} — using available subset")

    X = pd.get_dummies(df[feats_present].astype("category"), drop_first=False)
    n_features = X.shape[1]

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])

    p_test = model.predict_proba(X[test_idx])[:, 1]
    auc = roc_auc_score(y[test_idx], p_test)
    results[name] = auc

    print(f"{name:<45} {n_features:>10} {auc:>10.4f}")

print("=" * 78)

# Decision logic
best = max(results.values())
basic = results["BASIC (A's spec)"]
expanded = results["EXPANDED (+ weekday + slot geometry)"]

print(f"\nGain from expansion: +{expanded - basic:.4f} AUC points")
print()

if best >= 0.70:
    verdict = "STRONG — proceed with src5 build, use the best-performing feature set"
elif best >= 0.65:
    verdict = "GOOD — proceed with src5 build, expected modest RL improvement"
elif best >= 0.60:
    verdict = "MARGINAL — per A's spec, send these AUCs before build"
else:
    verdict = "INSUFFICIENT — do not build src5 on this file"

print(f"Best AUC: {best:.4f}")
print(f"Verdict: {verdict}")