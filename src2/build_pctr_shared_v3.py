"""
build_pctr_shared_v3.py

pCTR v3: EXACT Phase 1 approach applied to 3427 data

Key fix:
  Phase 1 worked because:
    train on ALL data → predict on SAME ALL data
    Model memorizes siteid patterns → continuous pCTR ✅

  Our v1/v2/v3 failed because:
    train/test split → unseen siteids → OHE zeros → binary ❌

  Fix: NO train/test split. Train on all 3427, predict on all 3427.
  Same approach as original build_pctr_cache_lr.py ✅

Output: shared_auction_log_v3.txt (v1/v2 UNTOUCHED)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ======================================================
# PATHS
# ======================================================
DATA_3427  = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/"
    "filtered_output/3427/final_sample_log.txt"
)
SHARED_LOG = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/shared_auction_log.txt"
)
OUT_LOG    = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/shared_auction_log_v3.txt"
)
MODEL_DIR  = Path(
    "D:/Research Methodology/DAAWBC/dynamic_ad_allocation/ipinyou_v2"
)
MODEL_DIR.mkdir(exist_ok=True)

ADVERTISERS  = ["1458", "2259", "3386", "2997", "3476"]

# EXACT same features as Phase 1 (5 features)
CAT_FEATURES = ["siteid"]
NUM_FEATURES = ["weekday", "hour", "slot_w", "slot_h"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def build_pipeline():
    """Exact Phase 1 pipeline — no C specified (default=1.0)"""
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    clf = LogisticRegression(
        solver="saga",
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        tol=1e-3,
        verbose=0,
    )
    return Pipeline([("pre", preprocessor), ("clf", clf)])


def main():
    print("=" * 65)
    print(" build_pctr_shared_v3.py")
    print(" EXACT Phase 1 approach on 3427 data")
    print(" Train ALL → Predict SAME (no train/test split)")
    print(" Output: shared_auction_log_v3.txt")
    print("=" * 65)

    # ======================================================
    # LOAD 3427 DATA
    # ======================================================
    print(f"\nLoading 3427 data...")
    df = pd.read_csv(DATA_3427, sep="\t")

    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[ALL_FEATURES].copy()
    y = df["click"].astype(int)

    print(f"  Rows   : {len(df):,}")
    print(f"  Clicks : {y.sum()} ({y.mean()*100:.4f}%)")

    # ======================================================
    # TRAIN ON ALL DATA (Phase 1 approach)
    # NO train/test split!
    # ======================================================
    print(f"\nFitting pipeline on ALL 3427 data...")
    print(f"  (same approach as Phase 1 build_pctr_cache_lr.py)")
    pipe = build_pipeline()
    pipe.fit(X, y)

    # Predict on SAME training data
    pctr_train = pipe.predict_proba(X)[:, 1]

    zeros  = (pctr_train < 0.001).sum() / len(pctr_train) * 100
    ones   = (pctr_train > 0.999).sum() / len(pctr_train) * 100
    binary = zeros + ones

    print(f"\n  pCTR distribution on 3427 training data:")
    print(f"    Mean   : {pctr_train.mean():.6f}")
    print(f"    Std    : {pctr_train.std():.6f}")
    print(f"    Min    : {pctr_train.min():.6f}")
    print(f"    25%    : {np.percentile(pctr_train, 25):.6f}")
    print(f"    Median : {np.median(pctr_train):.6f}")
    print(f"    75%    : {np.percentile(pctr_train, 75):.6f}")
    print(f"    Max    : {pctr_train.max():.6f}")
    print(f"    %binary: {binary:.1f}%  ", end="")

    if binary < 20:
        print("← Continuous ✅ SAME AS PHASE 1!")
    elif binary < 60:
        print("← Partially continuous ⚠️")
    else:
        print("← Binary ❌")

    # Save model
    model_path = MODEL_DIR / "pctr_v3_3427.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n  Model saved: {model_path.name}")

    # ======================================================
    # NOW APPLY TO SHARED AUCTION LOG
    # Note: shared log IS 3427 data (same siteids)
    # So model has seen these siteids during training ✅
    # ======================================================
    print(f"\nLoading shared auction log...")
    shared = pd.read_csv(SHARED_LOG, sep="\t")
    print(f"  Rows: {len(shared):,}")

    for col in ALL_FEATURES:
        if col not in shared.columns:
            shared[col] = 0

    X_shared = shared[ALL_FEATURES].copy()

    # Apply model — same siteids as training data ✅
    print(f"\nApplying pCTR v3 to shared stream...")
    pctr_shared = pipe.predict_proba(X_shared)[:, 1]

    zeros_s  = (pctr_shared < 0.001).sum() / len(pctr_shared) * 100
    ones_s   = (pctr_shared > 0.999).sum() / len(pctr_shared) * 100
    binary_s = zeros_s + ones_s

    print(f"  pCTR on shared stream:")
    print(f"    Mean   : {pctr_shared.mean():.6f}")
    print(f"    Std    : {pctr_shared.std():.6f}")
    print(f"    Min    : {pctr_shared.min():.6f}")
    print(f"    Median : {np.median(pctr_shared):.6f}")
    print(f"    Max    : {pctr_shared.max():.6f}")
    print(f"    %binary: {binary_s:.1f}%  ", end="")

    if binary_s < 20:
        print("← Continuous ✅")
    elif binary_s < 60:
        print("← Partially continuous ⚠️")
    else:
        print("← Binary ❌")

    # Assign same pCTR to all agents
    for adv in ADVERTISERS:
        col = f"pctr_{adv}"
        if col in shared.columns:
            del shared[col]
        shared[col] = pctr_shared

    # ======================================================
    # SAVE
    # ======================================================
    print(f"\nSaving shared_auction_log_v3.txt...")
    shared.to_csv(OUT_LOG, sep="\t", index=False)
    size_mb = OUT_LOG.stat().st_size / 1024 / 1024
    print(f"  Saved: {OUT_LOG.name} ({size_mb:.1f} MB) ✅")
    print(f"  V1 untouched ✅ | V2 untouched ✅")

    # ======================================================
    # VERDICT
    # ======================================================
    print(f"\n{'='*65}")
    if binary_s < 20:
        print(f" ✅ SUCCESS! pCTR is continuous!")
        print(f"    Same approach as Phase 1 worked!")
        print(f"    Update train_shared.py:")
        print(f"      SHARED_DATA = 'shared_auction_log_v3.txt'")
        print(f"    Then retrain AC + MAPPO Shared (~6 hours)")
    elif binary_s < 60:
        print(f" ⚠️  Partially continuous ({100-binary_s:.0f}% in range)")
        print(f"    May still help RL threshold decisions")
        print(f"    Consider retraining")
    else:
        print(f" ❌ Still binary. Cannot be fixed with LR.")
        print(f"    Keep v1 framework as-is.")
        print(f"    pCTR limitation acknowledged in paper.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()