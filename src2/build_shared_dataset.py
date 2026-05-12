"""
build_shared_dataset.py

Creates a neutral shared auction stream using advertiser 3427's log.
Applies each of the 5 agents' pCTR models to every impression.
All agents compete on the SAME impression — true competition!

Run from: dynamic_ad_allocation/src2/

Output: shared_auction_log.txt
Columns: all features + pctr_1458 + pctr_2259 + pctr_3386 + pctr_2997 + pctr_3476
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ======================================================
# PATHS
# ======================================================
BASE_DIR   = Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output")
OUT_DIR    = Path("D:/dataset/ipinyou-project/make-ipinyou-data")

# Neutral master stream — advertiser NOT in our agent pool
MASTER_ADV = "3427"

# Our 5 competing agents (3476 replaces 3358)
AGENT_ADVS = ["1458", "2259", "3386", "2997", "3476"]

# Features used by pCTR models
CAT_FEATURES = ["siteid"]
NUM_FEATURES = [
    "weekday", "hour", "slot_w", "slot_h",
    "region", "slotvisibility", "slotformat",
    "device_type", "usertag_count"
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def main():
    print("=" * 60)
    print(" Building Shared Auction Dataset")
    print(f" Master stream : Advertiser {MASTER_ADV}")
    print(f" Agent pool    : {AGENT_ADVS}")
    print("=" * 60)

    # ======================================================
    # STEP 1 — Load master stream (3427)
    # ======================================================
    print(f"\n[1/3] Loading master stream ({MASTER_ADV})...")

    master_path = BASE_DIR / MASTER_ADV / "enhanced" / \
                  "final_sample_log_enhanced.txt"

    df = pd.read_csv(master_path, sep="\t")

    print(f"  Rows        : {len(df):,}")
    print(f"  CTR         : {df['click'].mean()*100:.4f}%")
    print(f"  Avg Price   : {df['market_price'].mean():.1f}")
    print(f"  Columns     : {df.columns.tolist()}")

    # Clean missing values
    df.dropna(subset=["market_price", "slot_w", "slot_h"],
              inplace=True)
    df["siteid"] = df["siteid"].fillna("UNKNOWN_SITE")

    print(f"  After clean : {len(df):,} rows")

    # ======================================================
    # STEP 2 — Apply each agent's pCTR model
    # ======================================================
    print(f"\n[2/3] Applying pCTR models to master stream...")

    X = df[ALL_FEATURES].copy()

    for adv in AGENT_ADVS:
        model_path = BASE_DIR / adv / "enhanced" / \
                     "pctr_model_enhanced.joblib"

        if not model_path.exists():
            print(f"  ❌ Model not found for {adv}: {model_path}")
            continue

        print(f"  Loading model for {adv}...")
        pipe = joblib.load(model_path)

        print(f"  Predicting pctr_{adv}...")
        df[f"pctr_{adv}"] = pipe.predict_proba(X)[:, 1]

        mean_pctr = df[f"pctr_{adv}"].mean()
        print(f"  pctr_{adv}: mean={mean_pctr:.4f} ✅")

    # ======================================================
    # STEP 3 — Save shared dataset
    # ======================================================
    print(f"\n[3/3] Saving shared auction log...")

    out_file = OUT_DIR / "shared_auction_log.txt"
    df.to_csv(out_file, sep="\t", index=False)

    size_mb = out_file.stat().st_size / 1e6
    print(f"✅ Saved: {out_file}")
    print(f"   Rows   : {len(df):,}")
    print(f"   Size   : {size_mb:.1f} MB")

    # ======================================================
    # SUMMARY
    # ======================================================
    print("\n" + "=" * 60)
    print(" SHARED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Master stream  : {MASTER_ADV} (neutral)")
    print(f"  Impressions    : {len(df):,}")
    print(f"  Market prices  : mean={df['market_price'].mean():.1f} "
          f"min={df['market_price'].min():.0f} "
          f"max={df['market_price'].max():.0f}")
    print(f"\n  pCTR per agent:")

    for adv in AGENT_ADVS:
        col = f"pctr_{adv}"
        if col in df.columns:
            print(f"    {adv}: mean={df[col].mean():.4f} "
                  f"std={df[col].std():.4f}")

    print(f"\n  Columns: {df.columns.tolist()}")
    print("\n✅ Ready for multi_environment_shared.py!")


if __name__ == "__main__":
    main()