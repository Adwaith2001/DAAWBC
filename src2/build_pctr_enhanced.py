"""
build_pctr_enhanced.py
Adds LR pCTR to enhanced log files
Run from: dynamic_ad_allocation/src2/

Current: Building pCTR for 3476 with C=0.5
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

ADV_IDS = ["3476"]

CAT_FEATURES = ["siteid"]
NUM_FEATURES = [
    "weekday", "hour", "slot_w", "slot_h",
    "region", "slotvisibility", "slotformat",
    "device_type", "usertag_count"
]

# ← Tuning this value
C_VALUE = 0.5


def build_pctr_for_advertiser(adv_id):
    print(f"\n{'='*50}")
    print(f" Processing advertiser {adv_id} | C={C_VALUE}")
    print(f"{'='*50}")

    IN_FILE  = BASE_DIR / adv_id / "enhanced" / "final_sample_log_enhanced.txt"
    OUT_FILE = BASE_DIR / adv_id / "enhanced" / "final_sample_log_with_pctr.txt"
    MDL_FILE = BASE_DIR / adv_id / "enhanced" / "pctr_model_enhanced.joblib"

    print(f"Loading {IN_FILE.name}...")
    df = pd.read_csv(IN_FILE, sep="\t")
    print(f"  Rows     : {len(df):,}")
    print(f"  CTR      : {df['click'].mean()*100:.4f}%")
    print(f"  Clicks   : {df['click'].sum():,}")

    X = df[CAT_FEATURES + NUM_FEATURES].copy()
    y = df["click"].astype(int)

    m       = int(0.75 * len(df))
    X_train = X.iloc[:m]
    y_train = y.iloc[:m]

    print(f"  Train    : {len(X_train):,} rows | Clicks: {y_train.sum():,}")
    print(f"  Test     : {len(X) - m:,} rows  | Clicks: {y.iloc[m:].sum():,}")

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(
            handle_unknown = "ignore",
            sparse_output  = True,
            max_categories = 200,
        ), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    clf = LogisticRegression(
        solver       = "saga",
        max_iter     = 5000,
        n_jobs       = -1,
        class_weight = "balanced",
        tol          = 1e-2,
        C            = C_VALUE,
        verbose      = 0,
        random_state = 42,
    )

    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    print(f"  Training LR pCTR model (C={C_VALUE})...")
    pipe.fit(X_train, y_train)

    print("  Predicting pCTR for all rows...")
    df["pctr"] = pipe.predict_proba(X)[:, 1]

    mean_pctr = df["pctr"].mean()
    real_ctr  = df["click"].mean()
    ratio     = mean_pctr / real_ctr if real_ctr > 0 else 0

    print(f"\n  pCTR stats (C={C_VALUE}):")
    print(f"    mean : {mean_pctr:.6f}")
    print(f"    std  : {df['pctr'].std():.6f}")
    print(f"    min  : {df['pctr'].min():.6f}")
    print(f"    max  : {df['pctr'].max():.6f}")
    print(f"    Real CTR  : {real_ctr:.6f}")
    print(f"    Ratio     : {ratio:.1f}x")

    print(f"\n  Target range from other advertisers:")
    print(f"    1458: mean=0.273 (309x)")
    print(f"    2259: mean=0.423 (1347x)")
    print(f"    3386: mean=0.274 (301x)")
    print(f"    2997: mean=0.347 (101x)")
    print(f"    3476: mean={mean_pctr:.6f} ({ratio:.1f}x) ← current")

    # Decision guide
    if mean_pctr == 0.0:
        print(f"\n  ⚠️  All zeros! → Try C=0.8")
    elif ratio < 10:
        print(f"\n  ⚠️  Too low! → Try C=0.8")
    elif ratio > 2000:
        print(f"\n  ⚠️  Too high! → Try C=0.3")
    else:
        print(f"\n  ✅ Good match! Ready for shared dataset.")

    df.to_csv(OUT_FILE, sep="\t", index=False)
    joblib.dump(pipe, MDL_FILE)

    print(f"\n✅ Saved: {OUT_FILE.name}")
    print(f"✅ Model: {MDL_FILE.name}")

    return mean_pctr, ratio


def main():
    print("=" * 55)
    print(f" Building pCTR for 3476 (C={C_VALUE})")
    print(" Replacing 3358 in shared environment")
    print("=" * 55)

    for adv in ADV_IDS:
        mean_pctr, ratio = build_pctr_for_advertiser(adv)

    print("\n" + "=" * 55)
    print(" 🎉 Done!")
    print("=" * 55)

    print(f"\n C={C_VALUE} gave mean={mean_pctr:.6f} ratio={ratio:.1f}x")
    print("\n Tuning guide:")
    print(f"   C=0.01  → 0.000   (zeros)")
    print(f"   C=0.1   → 0.000   (too low)")
    print(f"   C=0.5   → {mean_pctr:.3f}   (current)")
    print(f"   C=1.0   → 0.991   (too high)")
    print(f"\n Target: mean between 0.01 and 0.50")


if __name__ == "__main__":
    main()