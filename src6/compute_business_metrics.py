"""
src6/compute_business_metrics.py
=================================
Business-oriented metrics for reviewer comment R1-4.

Reports metrics in native iPinYou budget units and as ratios relative to
the strongest stateless baseline (Lin (pCTR)). No currency assumption is
introduced. The ordering between methods is preserved regardless of any
downstream $/click or $/impression conversion the reader might apply.

Reads outputs/eval_all/eval_all_per_seed.csv and computes:
  - Absolute clicks per episode           (native)
  - Absolute cost per episode             (budget units)
  - Absolute cost-per-click               (budget units)
  - Budget pacing efficiency              (spent / total budget, percent)
  - Revenue per unit spent                (clicks / cost * 1000 for readability)
  - ROI ratio relative to Lin             (see formula below)
  - Auction win rate IF mean_wins column exists, else N/A

ROI ratio definition:
  For each method M we compute
     efficiency(M) = clicks(M) / cost(M)
  which is dimensionless (clicks per budget unit spent). Then
     ROI_ratio(M) = efficiency(M) / efficiency(Lin)
  A value of 1.0 means "same clicks per unit spent as Lin".
  A value of 1.65 means "65% more clicks per unit spent than Lin".
  This is the currency-free version of return on investment.

USAGE:
    python src6/compute_business_metrics.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_CSV_DIR = PROJECT_ROOT / "outputs" / "eval_all"
DEFAULT_OUT_DIR = HERE / "plots"

DISPLAY = {
    "rand_baseline":  "Rand",
    "const_baseline": "Const",
    "pctr_baseline":  "Lin (pCTR)",
    "src5_ac":        "AC",
    "src6_mappo":     "MAPPO",
}
ORDER = ["rand_baseline", "const_baseline", "pctr_baseline",
         "src5_ac", "src6_mappo"]
REFERENCE_POLICY = "pctr_baseline"  # Lin is the reference for ratios

BUDGET_PER_AGENT        = 50_000
N_AGENTS                = 5
TOTAL_BUDGET_PER_EP     = BUDGET_PER_AGENT * N_AGENTS   # 250,000
IMPRESSIONS_PER_EPISODE = 125_000


def compute_metrics(per_seed_df):
    rows = []
    has_wins = "mean_wins" in per_seed_df.columns

    # First pass: gather per-policy averages
    policy_stats = {}
    for policy in ORDER:
        sub = per_seed_df[per_seed_df["policy"] == policy]
        if sub.empty:
            continue
        clicks_arr = sub["mean_clicks"].values
        cost_arr   = sub["mean_cost"].values
        policy_stats[policy] = {
            "clicks_mean": float(np.mean(clicks_arr)),
            "cost_mean": float(np.mean(cost_arr)),
            "wins_mean": (float(np.mean(sub["mean_wins"].values))
                          if has_wins else None),
        }

    # Reference for ratios
    if REFERENCE_POLICY not in policy_stats:
        raise SystemExit(f"Reference policy '{REFERENCE_POLICY}' not in CSV")
    ref = policy_stats[REFERENCE_POLICY]
    ref_efficiency = ref["clicks_mean"] / max(ref["cost_mean"], 1e-9)

    # Second pass: compute derived metrics
    for policy, stats in policy_stats.items():
        clicks = stats["clicks_mean"]
        cost = stats["cost_mean"]

        # Auction efficiency (clicks per budget unit) and ratio-vs-Lin
        efficiency = clicks / max(cost, 1e-9)
        roi_ratio = efficiency / max(ref_efficiency, 1e-9)

        # Pacing = cost / total budget
        pacing_pct = cost / TOTAL_BUDGET_PER_EP * 100

        # CPC
        cpc = cost / max(clicks, 1e-9)
        cpc_ratio_vs_lin = cpc / max(ref["cost_mean"] / ref["clicks_mean"], 1e-9)

        # Win rate
        if stats["wins_mean"] is not None:
            win_rate = stats["wins_mean"] / IMPRESSIONS_PER_EPISODE * 100
        else:
            win_rate = None

        rows.append({
            "policy": DISPLAY[policy],
            "policy_key": policy,
            "clicks_mean": clicks,
            "cost_mean": cost,
            "cpc": cpc,
            "cpc_ratio_vs_lin": cpc_ratio_vs_lin,
            "pacing_pct": pacing_pct,
            "efficiency_clicks_per_1k_units": efficiency * 1000,
            "roi_ratio_vs_lin": roi_ratio,
            "roi_gain_pct_vs_lin": (roi_ratio - 1.0) * 100,
            "win_rate_pct": win_rate,
        })

    return pd.DataFrame(rows), has_wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    a = ap.parse_args()

    csv_dir = Path(a.csv_dir).resolve()
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_path = csv_dir / "eval_all_per_seed.csv"
    if not per_seed_path.exists():
        print(f"ERROR: {per_seed_path} not found")
        raise SystemExit(1)

    per_seed_df = pd.read_csv(per_seed_path)

    print("=" * 78)
    print("DAAWBC Business Metrics (Reviewer 1, point 4)")
    print("=" * 78)
    print(f"CSV source        : {per_seed_path}")
    print(f"Reference policy  : Lin (pCTR) (ratios normalised to this)")
    print(f"Total budget/ep   : {TOTAL_BUDGET_PER_EP:,} units "
          f"({N_AGENTS} agents x {BUDGET_PER_AGENT:,})")
    print("Currency          : native budget units (no $ conversion)\n")

    metrics, has_wins = compute_metrics(per_seed_df)

    # ---- Print human-readable table ----
    win_hdr = "Win rate" if has_wins else "Win rate*"
    print(f"{'Method':<12} {'Clicks':>8} {'Cost':>10} {'CPC':>7} "
          f"{'CPC/Lin':>8} {win_hdr:>10} {'Pacing':>8} {'ROI/Lin':>10}")
    print("-" * 82)
    for _, row in metrics.iterrows():
        win_str = f"{row['win_rate_pct']:>9.2f}%" if row['win_rate_pct'] is not None else "      N/A"
        roi_str = f"{row['roi_ratio_vs_lin']:.2f}x"
        gain_str = f"({row['roi_gain_pct_vs_lin']:+.1f}%)"
        print(f"{row['policy']:<12} {row['clicks_mean']:>8.1f} "
              f"{row['cost_mean']:>10,.0f} {row['cpc']:>7.1f} "
              f"{row['cpc_ratio_vs_lin']:>7.2f}x "
              f"{win_str} "
              f"{row['pacing_pct']:>7.1f}% {roi_str:>6} {gain_str:>10}")

    if not has_wins:
        print("\n* Win rate: not computed. Your eval_all_per_seed.csv does not")
        print("  include a mean_wins column. One line in evaluate_all.py would")
        print("  compute it, or it can be omitted with justification.")

    print("\nInterpretation:")
    print(f"  Pacing        : all methods should sit near 99-100% of total budget")
    print(f"  CPC/Lin ratio : <1.00x means cheaper per click than Lin")
    print(f"  ROI/Lin ratio : >1.00x means more clicks per unit spent than Lin")

    # Save CSV
    csv_path = out_dir / "business_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    print(f"\nCSV saved  : {csv_path}")

    # ---- Save LaTeX table ----
    tex_lines = [
        "% Auto-generated by src6/compute_business_metrics.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Business-oriented evaluation metrics on the iPinYou "
        "held-out split. Pacing efficiency is the fraction of the total "
        f"{TOTAL_BUDGET_PER_EP:,}-unit episode budget that was actually spent. "
        "CPC and ROI ratios are reported relative to the strongest stateless "
        "baseline (Lin (pCTR)), so no external currency assumption is required. "
        "ROI ratio $>1.00$ means the method delivers more clicks per unit spent "
        "than Lin. Conversion rate is not computed because iPinYou records "
        "impression-level clicks but not post-click conversion outcomes.}",
        "\\label{tab:business}",
        "\\small",
    ]

    if has_wins:
        tex_lines += [
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Method & Win rate & Pacing & CPC (units) & CPC ratio & ROI ratio \\\\",
            " & (\\%) & (\\%) & & (vs Lin) & (vs Lin) \\\\",
        ]
    else:
        tex_lines += [
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Method & Pacing & CPC (units) & CPC ratio & ROI ratio \\\\",
            " & (\\%) & & (vs Lin) & (vs Lin) \\\\",
        ]
    tex_lines.append("\\midrule")

    for _, row in metrics.iterrows():
        name = row["policy"]
        if name == "AC":
            name = "\\textbf{AC (proposed)}"
        if has_wins:
            tex_lines.append(
                f"{name} & {row['win_rate_pct']:.2f} & {row['pacing_pct']:.1f} & "
                f"{row['cpc']:.0f} & {row['cpc_ratio_vs_lin']:.2f}$\\times$ & "
                f"{row['roi_ratio_vs_lin']:.2f}$\\times$ \\\\"
            )
        else:
            tex_lines.append(
                f"{name} & {row['pacing_pct']:.1f} & "
                f"{row['cpc']:.0f} & {row['cpc_ratio_vs_lin']:.2f}$\\times$ & "
                f"{row['roi_ratio_vs_lin']:.2f}$\\times$ \\\\"
            )
    tex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    tex_path = out_dir / "business_metrics.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
    print(f"LaTeX saved: {tex_path}")

    # ---- Save prose notes ----
    notes_path = out_dir / "business_metrics_notes.txt"
    with open(notes_path, "w") as f:
        f.write(
            "Business Metrics: Notes for the paper\n"
            "======================================\n\n"
            "1. All metrics are reported in native iPinYou budget units. No\n"
            "   currency conversion is applied; no revenue assumption is made.\n\n"
            "2. ROI is reported as a ratio relative to the Linear-pCTR baseline\n"
            "   (Lin), computed as:\n\n"
            "      ROI_ratio(M) = (clicks(M) / cost(M))  divided by\n"
            "                    (clicks(Lin) / cost(Lin))\n\n"
            "   A value of 1.00x means the method achieves exactly the same\n"
            "   clicks-per-unit-spent as Lin. A value of 1.65x means it achieves\n"
            "   65% more clicks per unit spent than Lin. This is currency-free\n"
            "   and reader-friendly.\n\n"
            "3. Conversion rate is not reported. The iPinYou dataset provides\n"
            "   impression-level click labels but not post-click conversion\n"
            "   outcomes for the five advertisers evaluated. Extending to a\n"
            "   conversion-labelled dataset (Criteo Attribution Modeling for\n"
            "   Bidding, Diemert et al. 2017) is noted as future work.\n\n"
            "4. Auction win rate: not currently logged in eval_all_per_seed.csv.\n"
            "   To include it, evaluate_all.py can be extended to record\n"
            "   per-agent wins during evaluation. Alternatively, the metric can\n"
            "   be omitted with a brief justification that pacing efficiency\n"
            "   near 100% for every method makes win rate a redundant summary.\n"
        )
    print(f"Notes saved: {notes_path}")


if __name__ == "__main__":
    main()
