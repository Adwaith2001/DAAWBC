"""
Generate dense-CTR variant of shared_auction_log_v4.txt  [FIXED]
================================================================

WHY THIS VERSION EXISTS
-----------------------
The previous generator drew clicks as Bernoulli(max pctr_raw across the 5
advertisers). That is CIRCULAR: the click label became a direct function
of pctr, so any "pctr predicts clicks" result was manufactured, and an RL
policy could only rediscover linear-pCTR. Nothing new was learnable.

THIS version generates clicks from a STATED, PRE-COMMITTED logistic
function of REAL auction features (market price, time-of-day, weekday,
slot geometry, visibility, format) and explicitly NOT from pctr. This is
the standard declared-synthetic-environment methodology used in the
RL-for-RTB literature (e.g. synthetic-environment bandit/RL papers). It
makes the click signal:
  - dense enough for on-policy actor-critic to receive gradient
    (target CTR configurable, default 15%),
  - genuinely dependent on observable auction context (so a learned
    policy can capture structure a pctr-proportional rule cannot),
  - NON-circular (pctr is not used to generate clicks).

HONESTY / VALIDITY CONTRACT
---------------------------
1. This is a SEMI-SYNTHETIC, DENSIFIED variant. Real iPinYou CTR is
   ~0.075%. This file's ~15% CTR is synthetic and MUST be declared as
   such in the report. Absolute click/reward magnitudes do NOT transfer
   to real RTB; only relative learning behaviour does.
2. The click-model coefficients below are FIXED and PRE-COMMITTED. They
   must be copied into the report's methods section BEFORE any src5
   training run, and NEVER tuned in response to agent performance.
   Doing so would make the experiment circular and invalid.
3. Each coefficient's sign/magnitude is grounded in a pattern the REAL
   iPinYou analysis actually observed (clicked impressions clear above
   median price; hour-of-day matters more than weekday; larger/visible
   slots convert better). This is a synthetic world whose structure
   echoes the real one, with magnitudes set so the learning problem is
   non-trivial — not arbitrary numbers, and not reverse-engineered.

The pctr_raw_{adv} columns are still linearly rescaled (ranking
preserved) so the auction's bid scale is sensible; this is unchanged and
is independent of the click model.

Usage:
  python generate_dense_dataset.py
  python generate_dense_dataset.py --target-ctr 0.15
  python generate_dense_dataset.py --verify-only      # re-run checks on an
                                                      # already-generated file
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


# ======================================================================
# PRE-COMMITTED CLICK-MODEL COEFFICIENTS  (freeze before any src5 run)
# ----------------------------------------------------------------------
# logit(click) = B0
#   + B_MP   * z(market_price)        higher price -> modestly more clicks
#   + B_MP2  * z(market_price)^2      concave: extreme price tail tapers
#   + B_HSIN * sin(2pi*hour/24)       smooth time-of-day effect
#   + B_HCOS * cos(2pi*hour/24)       (cyclical, so hr23~hr0)
#   + B_WDAY * (weekday - mean)       weak day-of-week (real data: small)
#   + B_AREA * z(slot_w*slot_h)       bigger slots convert better
#   + B_VIS1 * 1[slotvisibility==1]   visibility==1 bump (real: separates)
#   + B_FMT5 * 1[slotformat==5]       slotformat==5 bump (real: separates)
# B0 is auto-calibrated by bisection so realized mean CTR ~= target_ctr.
# pctr is deliberately ABSENT from this model (non-circularity).
# ======================================================================
COEFFS = {
    "B_MP":   0.45,
    "B_MP2": -0.20,
    "B_HSIN": 0.35,
    "B_HCOS": -0.25,
    "B_WDAY": 0.08,
    "B_AREA": 0.30,
    "B_VIS1": 0.25,
    "B_FMT5": 0.40,
}
ADV_IDS = ["1458", "2259", "3386", "2997", "3476"]


# ----------------------------------------------------------------------
def build_logit(df):
    """Construct the pre-committed feature logit (pctr NOT used)."""
    n = len(df)

    def col(name, default=0.0):
        return (df[name].astype(float).values
                if name in df.columns else np.full(n, default))

    mp = col("market_price")
    mp_z = (mp - np.nanmean(mp)) / (np.nanstd(mp) + 1e-9)

    hour = col("hour")
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    wday = col("weekday")
    wday_c = wday - np.nanmean(wday)

    area = col("slot_w") * col("slot_h")
    area_z = (area - area.mean()) / (area.std() + 1e-9)

    vis1 = (col("slotvisibility") == 1).astype(float)
    fmt5 = (col("slotformat") == 5).astype(float)

    logit = (
        COEFFS["B_MP"]   * mp_z
        + COEFFS["B_MP2"]  * mp_z**2
        + COEFFS["B_HSIN"] * hour_sin
        + COEFFS["B_HCOS"] * hour_cos
        + COEFFS["B_WDAY"] * wday_c
        + COEFFS["B_AREA"] * area_z
        + COEFFS["B_VIS1"] * vis1
        + COEFFS["B_FMT5"] * fmt5
    )
    return logit


def calibrate_intercept(logit, target_ctr):
    """Bisection on an additive intercept so mean(sigmoid)=target_ctr."""
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if expit(logit + mid).mean() > target_ctr:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def verify_noncircular(df, target_ctr):
    """Post-generation gate. Prints PASS/FAIL on the three conditions
    that determine whether src5 is a valid experiment."""
    from scipy.stats import spearmanr
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        have_sklearn = True
    except Exception:
        have_sklearn = False

    print("\n" + "=" * 72)
    print("POST-GENERATION VERIFICATION (src5 validity gate)")
    print("=" * 72)
    ctr = df["click"].mean()
    print(f"CTR realized: {ctr*100:.2f}%  (target {target_ctr*100:.1f}%)")

    # 1. Non-circularity: pctr must NOT predict the new clicks
    print("\n[1] Non-circularity  (want |rho| < 0.05 for all advertisers):")
    max_abs_rho = 0.0
    for a in ADV_IDS:
        c = f"pctr_raw_{a}"
        if c in df.columns:
            rho = spearmanr(df[c], df["click"]).correlation
            max_abs_rho = max(max_abs_rho, abs(rho))
            flag = "ok" if abs(rho) < 0.05 else "HIGH (circular!)"
            print(f"    pctr_raw_{a} vs click  rho = {rho:+.4f}   {flag}")

    cond1 = max_abs_rho < 0.05

    cond2 = cond3 = True
    if have_sklearn:
        d = df.copy()
        d["mp_tier"] = pd.qcut(d["market_price"], 5, labels=False,
                               duplicates="drop")
        feat = ["mp_tier", "slotvisibility", "slotformat", "hour", "weekday"]
        feat = [f for f in feat if f in d.columns]
        X = pd.get_dummies(d[feat].astype("category"))
        y = d["click"].values
        tr = d["weekday"].isin([3, 4]).values
        te = d["weekday"].eq(5).values
        if tr.sum() > 0 and te.sum() > 0 and y[te].sum() > 0:
            m = LogisticRegression(max_iter=300,
                                   class_weight="balanced").fit(X[tr], y[tr])
            auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
            cond2 = auc >= 0.60
            print(f"\n[2] Features drive clicks  (want AUC >= 0.60):")
            print(f"    held-out feature-model AUC = {auc:.3f}"
                  f"   {'ok' if cond2 else 'TOO LOW'}")

            print(f"\n[3] pctr is now weak  (want AUC ~ 0.50):")
            worst_dev = 0.0
            for a in ADV_IDS:
                c = f"pctr_raw_{a}"
                if c in df.columns:
                    pa = roc_auc_score(y[te], df.loc[te, c])
                    worst_dev = max(worst_dev, abs(pa - 0.5))
                    print(f"    pctr_raw_{a} held-out AUC = {pa:.3f}")
            cond3 = worst_dev < 0.10
    else:
        print("\n[2/3] sklearn not available — run the standalone "
              "verification script for AUC checks.")

    print("\n" + "-" * 72)
    verdict = cond1 and cond2 and cond3
    print(f"VERDICT: {'PASS - src5 may be built' if verdict else 'FAIL - do NOT build src5 on this file'}")
    if not verdict:
        if not cond1:
            print("  - pctr still predicts clicks: click model is still "
                  "circular; check that pctr is not leaking into build_logit.")
        if not cond2:
            print("  - features do not separate clicks on held-out data: "
                  "signal too weak; coefficients may need to be (openly, "
                  "pre-commit) strengthened BEFORE any src5 run, not after.")
        if not cond3:
            print("  - pctr unexpectedly strong: investigate before building.")
    print("=" * 72)
    return verdict


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data_2/shared_auction_log_v4.txt")
    p.add_argument("--output",
                   default="data_2/shared_auction_log_v4_dense.txt")
    p.add_argument("--target-ctr", type=float, default=0.15,
                   help="Target mean CTR for the synthetic click model")
    p.add_argument("--pctr-target", type=float, default=0.15,
                   help="Target mean for rescaled pctr_raw (bid scale only; "
                        "does NOT affect clicks)")
    p.add_argument("--pctr-cap", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify-only", action="store_true",
                   help="Skip generation; just re-run the validity gate on "
                        "an existing --output file.")
    args = p.parse_args()

    out_path = Path(args.output).resolve()

    if args.verify_only:
        if not out_path.exists():
            print(f"ERROR: {out_path} not found.")
            sys.exit(1)
        df = pd.read_csv(out_path, sep="\t")
        ok = verify_noncircular(df, args.target_ctr)
        sys.exit(0 if ok else 2)

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}  (cwd={os.getcwd()})")
        sys.exit(1)

    print(f"Input  : {in_path}")
    print(f"Output : {out_path}")
    print(f"Target CTR (synthetic click model) : {args.target_ctr}")
    print(f"pctr rescale target (bid scale)    : {args.pctr_target}")
    print(f"Seed                               : {args.seed}")
    print()

    df = pd.read_csv(in_path, sep="\t")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} cols")

    missing = [f"pctr_raw_{a}" for a in ADV_IDS
               if f"pctr_raw_{a}" not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    if "click" not in df.columns:
        print("ERROR: no 'click' column in input.")
        sys.exit(1)

    orig_clicks = int(df["click"].sum())
    print(f"Original clicks: {orig_clicks:,} "
          f"(CTR {orig_clicks/len(df)*100:.4f}%)")

    # --- (a) rescale pctr_raw for bid realism ONLY (clicks不 use this) ---
    print("\nRescaling pctr_raw columns (bid scale only; not used for "
          "clicks):")
    for a in ADV_IDS:
        c = f"pctr_raw_{a}"
        cur = df[c].mean()
        scale = args.pctr_target / cur if cur > 0 else 1.0
        df[c] = (df[c] * scale).clip(0.0, args.pctr_cap)
        print(f"  {a}: x{scale:9.1f} -> mean {df[c].mean():.4f}")

    # --- (b) generate clicks from the PRE-COMMITTED feature logit ---
    print("\nGenerating clicks from pre-committed logistic feature model")
    print("  features: z(price), z(price)^2, hour(cyc), weekday, "
          "z(slot_area), vis==1, fmt==5")
    print("  pctr is NOT an input to the click model (non-circular).")
    rng = np.random.default_rng(args.seed)
    logit = build_logit(df)
    b0 = calibrate_intercept(logit, args.target_ctr)
    p_click = expit(logit + b0)
    df["click"] = rng.binomial(1, p_click).astype(np.int64)

    nc = int(df["click"].sum())
    print(f"  calibrated intercept B0 = {b0:+.4f}")
    print(f"  realized clicks: {nc:,}  CTR = {nc/len(df)*100:.2f}%")
    print(f"  coefficients (FROZEN, copy into report methods):")
    for k, v in COEFFS.items():
        print(f"    {k:7s} = {v:+.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved: {out_path}  ({len(df):,} rows, {len(df.columns)} cols)")

    # --- (c) mandatory validity gate ---
    ok = verify_noncircular(df, args.target_ctr)
    if not ok:
        print("\n*** DO NOT BUILD src5 ON THIS FILE until the gate PASSes. "
              "***")
        sys.exit(2)
    print("\nGate PASSED. src5 may be built on this file. Copy the FROZEN "
          "coefficients above into the report BEFORE any training run.")


if __name__ == "__main__":
    main()
