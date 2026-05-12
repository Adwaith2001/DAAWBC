"""
build_pctr_lgbm_pooled.py  (PATCHED v2.1)

Pooled multi-task pCTR using LightGBM.

Patches over the previous version (which produced 93% binary output):
  FIX 1: Drop scale_pos_weight=1113 → was saturating predictions near 1.0
  FIX 2: One-hot encode advertiser_id (instead of categorical) → forces
         full granularity across all 5 advertisers (previous version
         collapsed them into 2 buckets)
  FIX 3: Fit isotonic calibration on a held-out slice → produces
         well-spread probability outputs instead of saturated logits
  FIX 4: No silent zero-fill of missing enhanced features → either use
         them properly or drop them, never fake them as zero columns
  FIX 5 (v2.1): Defensive binarization of 'click' column → handles rows
                where click is not strictly 0/1 (caused stratify error)

Output: shared_auction_log_v4.txt
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not found! Install: pip install lightgbm")

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ======================================================
# PATHS
# ======================================================
BASE_DIR   = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output"
)
SHARED_LOG = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/shared_auction_log.txt"
)
OUT_LOG    = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/shared_auction_log_v4.txt"
)
MODEL_DIR  = Path(
    "D:/Research Methodology/DAAWBC/dynamic_ad_allocation/ipinyou_v2"
)
MODEL_DIR.mkdir(exist_ok=True)

ADVERTISERS = ["1458", "2259", "3386", "2997", "3476"]

# Base features MUST exist in all advertiser logs
BASE_FEATURES = ["weekday", "hour", "slot_w", "slot_h"]

# Enhanced features used IF AND ONLY IF present in ALL advertiser logs
# (No silent zero-fill: missing = drop from feature set)
CANDIDATE_ENHANCED = [
    "region", "device_type", "usertag_count",
    "slotvisibility", "slotformat",
]


# ======================================================
# HELPERS
# ======================================================
def detect_available_enhanced(verbose: bool = True) -> list:
    """Return enhanced features present in EVERY advertiser's data file."""
    available = []
    for feat in CANDIDATE_ENHANCED:
        in_all = True
        for adv in ADVERTISERS:
            path = BASE_DIR / adv / "final_sample_log.txt"
            header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
            if feat not in header:
                in_all = False
                break
        if in_all:
            available.append(feat)
        elif verbose:
            print(f"    [skip] {feat:<15} not present in all advertisers")

    # Also require non-trivial variance (don't include columns that exist
    # but are constant/zero for an advertiser)
    keep = []
    for feat in available:
        keep_this = True
        for adv in ADVERTISERS:
            path = BASE_DIR / adv / "final_sample_log.txt"
            col_sample = pd.read_csv(
                path, sep="\t", usecols=[feat], nrows=50000
            )[feat]
            if col_sample.nunique() < 2:
                if verbose:
                    print(f"    [skip] {feat:<15} is constant in {adv}")
                keep_this = False
                break
        if keep_this:
            keep.append(feat)

    return keep


def load_advertiser_data(adv_id: str, feature_cols: list) -> pd.DataFrame:
    """Load one advertiser's data. NO silent zero-fill."""
    path = BASE_DIR / adv_id / "final_sample_log.txt"
    df = pd.read_csv(path, sep="\t")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Advertiser {adv_id} missing features {missing}. "
            f"Fix data pipeline or remove from feature list."
        )
    if "click" not in df.columns:
        raise ValueError(f"Advertiser {adv_id} has no 'click' column.")

    df["advertiser_id"] = adv_id
    return df


def make_input_matrix(df: pd.DataFrame, feature_cols: list,
                      adv_columns: list, set_adv: str | None = None):
    """
    Build a feature matrix with one-hot advertiser columns.
    If set_adv is given, force the one-hot to that advertiser for inference.
    Otherwise, expects df['advertiser_id'] to already be present.
    """
    X = df[feature_cols].copy().reset_index(drop=True)
    for col in adv_columns:
        X[col] = 0
    if set_adv is not None:
        target_col = f"adv_{set_adv}"
        if target_col in adv_columns:
            X[target_col] = 1
    else:
        for col in adv_columns:
            adv_value = col.replace("adv_", "")
            X[col] = (df["advertiser_id"].astype(str) == adv_value).astype(int).values
    return X


# ======================================================
# MAIN
# ======================================================
def main():
    if not HAS_LGB:
        return

    print("=" * 70)
    print(" build_pctr_lgbm_pooled.py  (PATCHED v2.1)")
    print(" Fixes: no scale_pos_weight | one-hot advertiser | isotonic cal")
    print("        + defensive click binarization")
    print("=" * 70)

    # ------------------------------------------------------
    # FEATURE DETECTION
    # ------------------------------------------------------
    print("\nDetecting available enhanced features across all advertisers...")
    available_enhanced = detect_available_enhanced(verbose=True)
    feature_cols = BASE_FEATURES + available_enhanced

    print(f"\n  Base features    : {BASE_FEATURES}")
    print(f"  Enhanced features: {available_enhanced if available_enhanced else '(none usable)'}")
    print(f"  Final feature set: {feature_cols} + one-hot advertiser_id (5 cols)")

    # ------------------------------------------------------
    # LOAD AND POOL
    # ------------------------------------------------------
    print(f"\nLoading and pooling all 5 advertisers...")
    dfs = []
    for adv in ADVERTISERS:
        df = load_advertiser_data(adv, feature_cols)
        click_uniques = sorted(df["click"].fillna(0).astype(int).unique().tolist())
        n_clicks = int((df["click"].fillna(0).astype(int) > 0).sum())
        print(f"  {adv}: {len(df):>9,} rows | "
              f"{n_clicks:>5} click-rows | unique click values: {click_uniques}")
        dfs.append(df)

    pooled = pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------
    # FIX 5: DEFENSIVE BINARIZATION OF CLICK COLUMN
    # ------------------------------------------------------
    clicks_raw = pooled["click"].fillna(0).astype(int)
    unique_clicks_pooled = sorted(clicks_raw.unique().tolist())
    print(f"\n  Pooled 'click' unique values: {unique_clicks_pooled}")
    if not set(unique_clicks_pooled).issubset({0, 1}):
        weird = [v for v in unique_clicks_pooled if v not in (0, 1)]
        weird_count = int(clicks_raw.isin(weird).sum())
        print(f"  ⚠️  Non-binary values found: {weird} ({weird_count} rows)")
        print(f"      Treating any click > 0 as positive.")

    y = (clicks_raw > 0).astype(int).values
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"  After binarization: pos={n_pos:,} | neg={n_neg:,} | "
          f"pos_rate={n_pos/len(y):.6f}")

    # ------------------------------------------------------
    # FIX 2: ONE-HOT ENCODE ADVERTISER_ID
    # ------------------------------------------------------
    adv_dummies = pd.get_dummies(
        pooled["advertiser_id"], prefix="adv"
    ).astype(int).reset_index(drop=True)
    adv_columns = adv_dummies.columns.tolist()

    X = pd.concat(
        [pooled[feature_cols].reset_index(drop=True), adv_dummies],
        axis=1
    )

    train_columns = X.columns.tolist()

    # ------------------------------------------------------
    # SPLIT: train-fit (60%) | calib (20%) | test (20%)
    # ------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_trainval, y_trainval, test_size=0.25,  # 0.25 * 0.80 = 0.20
        random_state=42, stratify=y_trainval
    )

    print(f"\n  Splits:")
    print(f"    Fit   : {len(X_fit):>9,} | Pos={int(y_fit.sum()):>5}")
    print(f"    Calib : {len(X_calib):>9,} | Pos={int(y_calib.sum()):>5}")
    print(f"    Test  : {len(X_test):>9,} | Pos={int(y_test.sum()):>5}")

    # ------------------------------------------------------
    # FIX 1: TRAIN LGBM WITHOUT scale_pos_weight
    # (Natural log-loss handles class imbalance; calibration fixes the scale)
    # ------------------------------------------------------
    print(f"\nTraining LGBMClassifier (no scale_pos_weight, no is_unbalance)...")

    base_clf = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.0,
        reg_lambda=0.0,
        verbose=-1,
        n_jobs=-1,
    )

    base_clf.fit(
        X_fit, y_fit,
        eval_set=[(X_calib, y_calib)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    # Diagnostics on raw (uncalibrated) output
    raw_test = base_clf.predict_proba(X_test)[:, 1]
    raw_auc = roc_auc_score(y_test, raw_test)
    print(f"\n  Raw (uncalibrated) test set:")
    print(f"    AUC    : {raw_auc:.4f}")
    print(f"    Mean   : {raw_test.mean():.6f}  (base rate: {y_test.mean():.6f})")
    print(f"    Range  : [{raw_test.min():.6f}, {raw_test.max():.6f}]")
    print(f"    Std    : {raw_test.std():.6f}")

    # Feature importances
    print(f"\n  Top feature importances (gain):")
    importances = sorted(
        zip(train_columns, base_clf.booster_.feature_importance(importance_type="gain")),
        key=lambda kv: -kv[1],
    )
    for feat, imp in importances[:12]:
        print(f"    {feat:<20}: {imp:>12.1f}")

    # ------------------------------------------------------
    # FIX 3: ISOTONIC CALIBRATION ON HELD-OUT SLICE
    # ------------------------------------------------------
    print(f"\nFitting isotonic calibration on the calib slice...")
    raw_calib = base_clf.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_calib, y_calib)

    cal_test = iso.transform(raw_test)
    cal_auc = roc_auc_score(y_test, cal_test)  # should equal raw_auc (isotonic is monotonic)

    zeros = (cal_test < 0.001).sum() / len(cal_test) * 100
    ones  = (cal_test > 0.999).sum() / len(cal_test) * 100
    binary_pct = zeros + ones

    print(f"\n  Calibrated test set pCTR distribution:")
    print(f"    AUC      : {cal_auc:.4f}  (raw was {raw_auc:.4f})")
    print(f"    Mean     : {cal_test.mean():.6f}")
    print(f"    Std      : {cal_test.std():.6f}")
    print(f"    Min      : {cal_test.min():.6f}")
    print(f"    25%      : {np.percentile(cal_test, 25):.6f}")
    print(f"    Median   : {np.median(cal_test):.6f}")
    print(f"    75%      : {np.percentile(cal_test, 75):.6f}")
    print(f"    Max      : {cal_test.max():.6f}")
    print(f"    %binary  : {binary_pct:.1f}%  ", end="")
    if binary_pct < 20:
        print("← Continuous ✅")
    elif binary_pct < 60:
        print("← Partially continuous ⚠️")
    else:
        print("← Still binary ❌")

    # ------------------------------------------------------
    # SAVE MODEL + CALIBRATOR
    # ------------------------------------------------------
    model_path = MODEL_DIR / "pctr_lgbm_pooled.joblib"
    joblib.dump(
        {
            "model": base_clf,
            "calibrator": iso,
            "feature_cols": feature_cols,
            "adv_columns": adv_columns,
            "train_columns": train_columns,
            "raw_auc": float(raw_auc),
            "calibrated_auc": float(cal_auc),
        },
        model_path,
    )
    print(f"\n  Saved: {model_path.name}")

    # ------------------------------------------------------
    # APPLY TO SHARED STREAM PER AGENT
    # ------------------------------------------------------
    print(f"\nLoading shared auction log...")
    shared = pd.read_csv(SHARED_LOG, sep="\t")
    print(f"  Rows: {len(shared):,}")

    missing_in_shared = [c for c in feature_cols if c not in shared.columns]
    if missing_in_shared:
        raise ValueError(
            f"shared_auction_log.txt missing features {missing_in_shared}. "
            f"Either regenerate the shared log to include them, or remove from "
            f"CANDIDATE_ENHANCED so they are dropped from training too."
        )

    print(f"\nApplying calibrated pooled pCTR per agent...\n")
    header = (
        f"{'Adv':<6} {'Mean':>10} {'Std':>10} "
        f"{'Min':>10} {'Max':>10} {'%binary':>9}  Status"
    )
    print(header)
    print("-" * len(header))

    per_agent_binary = {}
    for adv in ADVERTISERS:
        X_inf = make_input_matrix(
            shared, feature_cols, adv_columns, set_adv=adv
        )
        X_inf = X_inf[train_columns]  # enforce column order

        raw = base_clf.predict_proba(X_inf)[:, 1]
        pctr = iso.transform(raw)

        col_name = f"pctr_{adv}"
        if col_name in shared.columns:
            del shared[col_name]
        shared[col_name] = pctr

        z = (pctr < 0.001).sum() / len(pctr) * 100
        o = (pctr > 0.999).sum() / len(pctr) * 100
        b = z + o
        per_agent_binary[adv] = b
        flag = "✅" if b < 20 else "⚠️" if b < 60 else "❌"

        print(
            f"{adv:<6} {pctr.mean():>10.6f} {pctr.std():>10.6f} "
            f"{pctr.min():>10.6f} {pctr.max():>10.6f} "
            f"{b:>8.1f}%  {flag}"
        )

    # ------------------------------------------------------
    # SAVE V4 LOG
    # ------------------------------------------------------
    print(f"\nSaving shared_auction_log_v4.txt...")
    shared.to_csv(OUT_LOG, sep="\t", index=False)
    size_mb = OUT_LOG.stat().st_size / 1024 / 1024
    print(f"  Saved: {OUT_LOG.name} ({size_mb:.1f} MB) ✅")

    # ------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------
    all_pctrs = np.concatenate(
        [shared[f"pctr_{adv}"].values for adv in ADVERTISERS]
    )
    overall_binary = (
        (all_pctrs < 0.001).sum() + (all_pctrs > 0.999).sum()
    ) / len(all_pctrs) * 100

    # Are the agents getting actually-different predictions per impression?
    cols = [f"pctr_{adv}" for adv in ADVERTISERS]
    per_row_std = shared[cols].std(axis=1)
    pct_rows_differentiated = (per_row_std > 1e-6).sum() / len(shared) * 100

    print(f"\n{'='*70}")
    print(f" Overall %binary across all agents: {overall_binary:.1f}%")
    print(f" Rows where agents got different pCTR: {pct_rows_differentiated:.1f}%")
    print(f"{'='*70}")
    if overall_binary < 20 and pct_rows_differentiated > 50:
        print(" ✅ POOLED pCTR IS CONTINUOUS AND DIFFERENTIATED")
        print("    Each agent gets a meaningfully different pCTR per impression.")
        print("    → Point SHARED_DATA in your training scripts to")
        print("      shared_auction_log_v4.txt and retrain AC + MAPPO.")
    elif overall_binary < 60:
        print(" ⚠️  Partially continuous — significant improvement over v1.")
        print("    Worth retraining AC + MAPPO to see if RL behavior changes.")
    else:
        print(" ❌ Still binary.")
        print("    Likely cause: features in shared_auction_log.txt lack")
        print("    discriminative power. Try adding usertag features via")
        print("    transform_enhanced_logs.py and rerun.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()