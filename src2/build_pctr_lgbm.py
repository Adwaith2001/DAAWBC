"""
build_pctr_lgbm.py
Phase 2 - Train LightGBM pCTR model with temporal split + regularization

Run from src2/:
    python build_pctr_lgbm.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb

# ======================================================
# PATHS
# ======================================================
ROOT     = Path(__file__).resolve().parents[1]
DATA_IN  = Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output")
DATA_OUT = ROOT / "data" / "ipinyou_v2"

ADV_IDS  = ["1458", "2259", "2821", "2997", "3358"]

FEATURES    = ["weekday", "hour", "slot_w", "slot_h", "siteid_enc"]
TARGET      = "click"
TRAIN_RATIO = 0.75

# ======================================================
# LIGHTGBM PARAMS — regularized to fix overfitting
# ======================================================
PARAMS = {
    "objective"        : "binary",
    "metric"           : "auc",
    "boosting_type"    : "gbdt",
    "num_leaves"       : 31,      # reduced from 63
    "max_depth"        : 6,       # limit tree depth
    "learning_rate"    : 0.05,
    "feature_fraction" : 0.8,
    "bagging_fraction" : 0.8,
    "bagging_freq"     : 5,
    "min_child_samples": 100,     # increased from 20
    "reg_alpha"        : 0.1,     # L1 regularization
    "reg_lambda"       : 1.0,     # L2 regularization
    "verbosity"        : -1,
    "random_state"     : 42,
}


def build_pctr_for_advertiser(adv_id: str):

    print(f"\n{'='*60}")
    print(f" Processing Advertiser {adv_id}")
    print(f"{'='*60}")

    # Paths
    in_file   = DATA_IN  / adv_id / "final_sample_log.txt"
    out_dir   = DATA_OUT / adv_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file  = out_dir / "final_sample_log_with_pctr.txt"
    model_out = out_dir / "pctr_lgbm_model.joblib"

    # Load
    df = pd.read_csv(in_file, sep="\t")
    print(f"✅ Loaded {len(df):,} rows | CTR: {df[TARGET].mean()*100:.4f}%")

    # Binarize clicks
    df[TARGET] = (df[TARGET] > 0).astype(int)

    # Encode siteid
    le = LabelEncoder()
    df["siteid_enc"] = le.fit_transform(df["siteid"].astype(str))

    # Temporal split
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()
    print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df[TARGET].values.astype(np.int32)
    X_test  = test_df[FEATURES].values.astype(np.float32)
    y_test  = test_df[TARGET].values.astype(np.int32)

    # Train
    print("   Training LightGBM (regularized)...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test,  label=y_test, reference=dtrain)

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100),
    ]

    booster = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )

    # Evaluate
    train_pred   = booster.predict(X_train)
    test_pred    = booster.predict(X_test)
    train_auc    = roc_auc_score(y_train, train_pred)
    test_auc     = roc_auc_score(y_test,  test_pred)
    test_logloss = log_loss(y_test, test_pred)
    overfit_gap  = train_auc - test_auc

    print(f"\n   📊 Train AUC   : {train_auc:.4f}")
    print(f"   📊 Test  AUC   : {test_auc:.4f}")
    print(f"   📊 Overfit gap : {overfit_gap:.4f} {'✅ OK' if overfit_gap < 0.05 else '⚠️ Still overfitting'}")
    print(f"   📊 Test LogLoss: {test_logloss:.4f}")

    # Compare with old LR
    lr_file = DATA_IN / adv_id / "final_sample_log_with_pctr.txt"
    if lr_file.exists():
        lr_df   = pd.read_csv(lr_file, sep="\t")
        lr_test = lr_df.iloc[split_idx:]
        if "pctr" in lr_test.columns:
            lr_auc = roc_auc_score(y_test, lr_test["pctr"].values)
            better = "✅ LGB better" if test_auc > lr_auc else "⚠️  LR better"
            print(f"   📊 LR AUC      : {lr_auc:.4f}  ({better})")

    # Predict pCTR for full dataset
    X_full     = df[FEATURES].values.astype(np.float32)
    df["pctr"] = booster.predict(X_full)

    # Drop helper column
    df = df.drop(columns=["siteid_enc"])

    # Save
    df.to_csv(out_file, sep="\t", index=False)
    joblib.dump(booster, model_out)

    print(f"\n   ✅ Saved: {out_file}")
    print(f"   ✅ Model: {model_out}")
    print(f"   📈 pCTR mean : {df['pctr'].mean():.4f}")
    print(f"   📈 pCTR max  : {df['pctr'].max():.4f}")

    return {
        "advertiser" : adv_id,
        "train_auc"  : round(train_auc, 4),
        "test_auc"   : round(test_auc, 4),
        "gap"        : round(overfit_gap, 4),
        "logloss"    : round(test_logloss, 4),
    }


if __name__ == "__main__":

    results = []
    for adv in ADV_IDS:
        result = build_pctr_for_advertiser(adv)
        results.append(result)

    print(f"\n{'='*60}")
    print(" FINAL SUMMARY")
    print(f"{'='*60}")
    df_summary = pd.DataFrame(results).set_index("advertiser")
    print(df_summary.to_string())

    avg_gap = df_summary["gap"].mean()
    print(f"\nAverage overfit gap: {avg_gap:.4f} {'✅ Good' if avg_gap < 0.05 else '⚠️ Consider more regularization'}")
    print(f"\n🎉 Done! Output: {DATA_OUT}")