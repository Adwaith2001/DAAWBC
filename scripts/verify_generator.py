"""
verify_generator.py
===================

Determines which densification generator produced your
shared_auction_log_v4_dense.txt file.

The two generators leave very different fingerprints:

  - OLD generator (Bernoulli on max pCTR):
      Clicks are generated as a direct function of pCTR.
      So pCTR strongly predicts click.
      EXPECTED: Spearman correlation between pctr_raw_{adv} and click
                will be POSITIVE and LARGE for at least one advertiser
                (typically 0.2 to 0.6).
      EXPECTED: held-out AUC of pctr_raw_{adv} predicting click
                will be HIGH (typically 0.65 to 0.85).

  - COEFFICIENT-BASED generator (src7, logistic on context features):
      Clicks are generated from contextual features. pCTR is decoupled
      by design and the verify_noncircular gate enforces |rho| < 0.05.
      EXPECTED: Spearman correlation between pctr_raw_{adv} and click
                will be NEAR ZERO (|rho| < 0.05) for ALL advertisers.
      EXPECTED: pctr_raw_{adv} AUC will be NEAR 0.50.

Usage:
  python verify_generator.py
  python verify_generator.py --data path/to/your/densified_file.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


ADV_IDS = ["1458", "2259", "3386", "2997", "3476"]


def verify_generator(data_path: Path) -> str:
    """Returns 'OLD' or 'COEFFICIENT' or 'AMBIGUOUS' based on data fingerprints."""

    print("=" * 72)
    print("DENSIFIER GENERATOR VERIFICATION")
    print("=" * 72)
    print(f"File: {data_path}")

    if not data_path.exists():
        print(f"ERROR: file not found.")
        sys.exit(1)

    # Load
    print("\nLoading...")
    df = pd.read_csv(data_path, sep="\t")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Sanity checks
    if "click" not in df.columns:
        print("ERROR: no 'click' column in file.")
        sys.exit(1)

    missing = [f"pctr_raw_{a}" for a in ADV_IDS
               if f"pctr_raw_{a}" not in df.columns]
    if missing:
        print(f"ERROR: missing pctr columns: {missing}")
        sys.exit(1)

    ctr = df["click"].mean()
    print(f"  CTR realized: {ctr * 100:.2f}%")

    # ------------------------------------------------------------------
    # TEST 1: Spearman correlation of pctr_raw_{adv} vs click
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("TEST 1: Spearman correlation between pctr_raw and click")
    print("-" * 72)
    print("  OLD generator        --> expect |rho| LARGE (typically 0.2 to 0.6)")
    print("  COEFFICIENT generator --> expect |rho| NEAR ZERO (< 0.05)")
    print()

    correlations = []
    for adv in ADV_IDS:
        col = f"pctr_raw_{adv}"
        rho = spearmanr(df[col], df["click"]).correlation
        correlations.append(rho)
        print(f"    pctr_raw_{adv} vs click:  rho = {rho:+.4f}")

    max_abs_rho = max(abs(r) for r in correlations)
    print(f"\n    Maximum |rho| across advertisers: {max_abs_rho:.4f}")

    # ------------------------------------------------------------------
    # TEST 2: Held-out AUC of pctr_raw predicting click on weekday 5
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("TEST 2: Can pctr_raw predict click on held-out weekday 5?")
    print("-" * 72)
    print("  OLD generator        --> expect AUC HIGH (typically 0.65 to 0.85)")
    print("  COEFFICIENT generator --> expect AUC NEAR 0.50")
    print()

    eval_mask = df["weekday"].eq(5).values
    if eval_mask.sum() == 0:
        print("  No weekday-5 rows; using all rows for this check.")
        eval_mask = np.ones(len(df), dtype=bool)

    aucs = []
    for adv in ADV_IDS:
        col = f"pctr_raw_{adv}"
        try:
            auc = roc_auc_score(df.loc[eval_mask, "click"],
                                df.loc[eval_mask, col])
            aucs.append(auc)
            print(f"    pctr_raw_{adv} AUC (weekday 5): {auc:.4f}")
        except Exception as e:
            print(f"    pctr_raw_{adv} AUC failed: {e}")
            aucs.append(0.5)

    max_dev_from_half = max(abs(a - 0.5) for a in aucs)
    print(f"\n    Maximum |AUC - 0.50|: {max_dev_from_half:.4f}")

    # ------------------------------------------------------------------
    # TEST 3: AUC of max(pctr_raw) -- the actual densifier signal
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("TEST 3: max(pctr_raw across 5 advertisers) AUC")
    print("-" * 72)
    print("  This is the exact signal the OLD generator used to generate clicks.")
    print("  OLD generator        --> expect AUC VERY HIGH (typically 0.80+)")
    print("  COEFFICIENT generator --> expect AUC NEAR 0.50")
    print()

    max_pctr = df[[f"pctr_raw_{a}" for a in ADV_IDS]].max(axis=1)
    try:
        max_auc = roc_auc_score(df.loc[eval_mask, "click"],
                                max_pctr[eval_mask])
        print(f"    max(pctr_raw) AUC (weekday 5): {max_auc:.4f}")
    except Exception as e:
        print(f"    max(pctr_raw) AUC failed: {e}")
        max_auc = 0.5

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    # Decision rules based on the fingerprints
    is_old = (max_abs_rho > 0.10) and (max_auc > 0.65)
    is_coeff = (max_abs_rho < 0.10) and (max_auc < 0.60)

    print(f"\n  Test 1: Max |Spearman rho| = {max_abs_rho:.4f}")
    print(f"  Test 2: Max |AUC - 0.50| of individual pctr = {max_dev_from_half:.4f}")
    print(f"  Test 3: AUC of max(pctr_raw) = {max_auc:.4f}")

    print()
    if is_old:
        verdict = "OLD"
        print("  >> VERDICT: OLD generator (Bernoulli on max pCTR).")
        print("  >> Your src5/src6 thesis data was produced by this generator.")
        print("  >> The thesis description (densification using max scaled pCTR")
        print("     to generate Bernoulli click draws) is CORRECT.")
        print("  >> No methodological change to the thesis is needed.")
    elif is_coeff:
        verdict = "COEFFICIENT"
        print("  >> VERDICT: COEFFICIENT-BASED generator (src7-style).")
        print("  >> Clicks were generated from contextual features, NOT pCTR.")
        print("  >> If your src5/src6 thesis numbers came from THIS file, then")
        print("     the thesis densification description needs revision.")
    else:
        verdict = "AMBIGUOUS"
        print("  >> VERDICT: AMBIGUOUS.")
        print("  >> The fingerprints don't clearly match either generator.")
        print("  >> Investigate: paste this output and decide manually.")

    print("\n" + "=" * 72)
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data_2/shared_auction_log_v4_dense.txt",
        help="Path to the densified data file used by src5/src6 training.",
    )
    args = parser.parse_args()

    if not HAVE_SKLEARN:
        print("WARNING: sklearn not installed. Test 2 and Test 3 will be skipped.")
        print("         Run: pip install scikit-learn")
        sys.exit(1)

    verify_generator(Path(args.data).resolve())


if __name__ == "__main__":
    main()
