"""
build_pctr_enhanced.py
Adds LR pCTR to enhanced log files (13 features)
Run from: dynamic_ad_allocation/src2/
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output")
ADV_IDS  = ["1458", "2259", "2821", "2997", "3358"]

# All 13 features — enhanced dataset
CAT_FEATURES = ["siteid"]
NUM_FEATURES = [
    "weekday", "hour", "slot_w", "slot_h",
    "region", "slotvisibility", "slotformat",
    "device_type", "usertag_count"
]


def build_pctr_for_advertiser(adv_id):
    print(f"\n{'='*50}")
    print(f" Processing advertiser {adv_id}")
    print(f"{'='*50}")

    IN_FILE  = BASE_DIR / adv_id / "enhanced" / "final_sample_log_enhanced.txt"
    OUT_FILE = BASE_DIR / adv_id / "enhanced" / "final_sample_log_with_pctr.txt"
    MDL_FILE = BASE_DIR / adv_id / "enhanced" / "pctr_model_enhanced.joblib"

    # ======================================================
    # LOAD
    # ======================================================
    print(f"Loading {IN_FILE.name}...")
    df = pd.read_csv(IN_FILE, sep="\t")
    print(f"  Rows     : {len(df):,}")
    print(f"  CTR      : {df['click'].mean()*100:.4f}%")
    print(f"  Clicks   : {df['click'].sum():,}")

    # ======================================================
    # FEATURES
    # ======================================================
    X = df[CAT_FEATURES + NUM_FEATURES].copy()
    y = df["click"].astype(int)

    # Temporal split 75/25 — no data leakage
    m       = int(0.75 * len(df))
    X_train = X.iloc[:m]
    y_train = y.iloc[:m]

    print(f"  Train    : {len(X_train):,} rows | Clicks: {y_train.sum():,}")
    print(f"  Test     : {len(X) - m:,} rows  | Clicks: {y.iloc[m:].sum():,}")

    # ======================================================
    # PREPROCESSOR — sparse=True to avoid memory error
    # ======================================================
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,    # ← KEEP SPARSE to avoid 21GB RAM usage
            max_categories=200,    # ← limit siteid categories to top 200
        ), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    # ======================================================
    # LOGISTIC REGRESSION
    # ======================================================
    clf = LogisticRegression(
        solver       = "saga",
        max_iter     = 5000,
        n_jobs       = -1,
        class_weight = "balanced",
        tol          = 1e-2,
        C            = 0.1,
        verbose      = 0,
        random_state = 42,
    )

    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    print("  Training LR pCTR model...")
    pipe.fit(X_train, y_train)

    # ======================================================
    # PREDICT pCTR FOR ALL ROWS
    # ======================================================
    print("  Predicting pCTR for all rows...")
    df["pctr"] = pipe.predict_proba(X)[:, 1]

    # ======================================================
    # STATS
    # ======================================================
    mean_pctr = df["pctr"].mean()
    real_ctr  = df["click"].mean()
    ratio     = mean_pctr / real_ctr if real_ctr > 0 else 0

    print(f"\n  pCTR stats:")
    print(f"    mean : {mean_pctr:.6f}")
    print(f"    std  : {df['pctr'].std():.6f}")
    print(f"    min  : {df['pctr'].min():.6f}")
    print(f"    max  : {df['pctr'].max():.6f}")
    print(f"    Real CTR  : {real_ctr:.6f}")
    print(f"    Ratio     : {ratio:.1f}x "
          f"{'⚠️ high' if ratio > 50 else '✅ ok'}")

    # ======================================================
    # SAVE
    # ======================================================
    df.to_csv(OUT_FILE, sep="\t", index=False)
    joblib.dump(pipe, MDL_FILE)

    print(f"\n✅ Saved: {OUT_FILE.name}")
    print(f"✅ Model: {MDL_FILE.name}")
    return OUT_FILE


def main():
    print("="*55)
    print(" Building enhanced pCTR files (LR, 13 features)")
    print("="*55)

    for adv in ADV_IDS:
        build_pctr_for_advertiser(adv)

    print("\n" + "="*55)
    print(" 🎉 All enhanced pCTR files created!")
    print("="*55)
    print("\nOutput files:")
    for adv in ADV_IDS:
        path = BASE_DIR / adv / "enhanced" / "final_sample_log_with_pctr.txt"
        size = path.stat().st_size / 1e6 if path.exists() else 0
        print(f"  {adv}: {size:.1f} MB")


if __name__ == "__main__":
    main()