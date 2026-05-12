"""
check_pctr_v1.py
Check if Phase 1 pCTR was binary or continuous.
Loads the existing final_sample_log_with_pctr.txt for advertiser 1458.
Run from: dynamic_ad_allocation/src2/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Phase 1 data files with pCTR already computed
FILES = {
    "1458": Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output/1458/final_sample_log_with_pctr.txt"),
    "2259": Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output/2259/final_sample_log_with_pctr.txt"),
    "3386": Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output/3386/final_sample_log_with_pctr.txt"),
    "2997": Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output/2997/final_sample_log_with_pctr.txt"),
    "3476": Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output/3476/final_sample_log_with_pctr.txt"),
}

print("=" * 65)
print(" Phase 1 pCTR Distribution Check")
print(" Loading existing final_sample_log_with_pctr.txt files")
print("=" * 65)

for adv, path in FILES.items():
    if not path.exists():
        print(f"\n  {adv}: FILE NOT FOUND — {path}")
        continue

    df = pd.read_csv(path, sep="\t")

    if "pctr" not in df.columns:
        print(f"\n  {adv}: NO pctr COLUMN FOUND")
        print(f"    Columns: {df.columns.tolist()}")
        continue

    p = df["pctr"]
    zeros  = (p == 0.0).sum() / len(p) * 100
    ones   = (p == 1.0).sum() / len(p) * 100
    lo     = (p < 0.001).sum() / len(p) * 100
    hi     = (p > 0.999).sum() / len(p) * 100
    binary = lo + hi

    print(f"\n  Advertiser {adv}:")
    print(f"    Rows     : {len(df):,}")
    print(f"    Mean pCTR: {p.mean():.6f}")
    print(f"    Std      : {p.std():.6f}")
    print(f"    Min      : {p.min():.6f}")
    print(f"    25%      : {p.quantile(0.25):.6f}")
    print(f"    Median   : {p.median():.6f}")
    print(f"    75%      : {p.quantile(0.75):.6f}")
    print(f"    Max      : {p.max():.6f}")
    print(f"    % exact 0: {zeros:.1f}%")
    print(f"    % exact 1: {ones:.1f}%")
    print(f"    % binary (<0.001 or >0.999): {binary:.1f}%")

    # Sample some values
    unique_vals = p.unique()
    print(f"    Unique values (sample): {sorted(unique_vals)[:10]}")

    if binary < 20:
        print(f"    → CONTINUOUS ✅ Phase 1 pCTR worked!")
    elif binary < 60:
        print(f"    → PARTIALLY CONTINUOUS ⚠️")
    else:
        print(f"    → BINARY ❌ Phase 1 also had binary pCTR")
