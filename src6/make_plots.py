"""
src6/make_plots.py
==================
Generate the four result figures for the DAAWBC project from the locked
evaluation CSVs produced by src6/evaluate_all.py.

This script mirrors the path logic of evaluate_all.py: it anchors itself to
the project root using its own file location, so it works no matter which
directory you launch it from.

DEFAULT PATHS (no flags needed):
    reads  CSVs from   <project_root>/outputs/eval_all/
    writes plots to    <project_root>/src6/plots/

The four input CSVs (written by evaluate_all.py):
    eval_all_aggregate.csv
    eval_all_per_seed.csv
    eval_all_per_seed_table.csv
    eval_all_paired.csv

The four output figures (PNG + PDF each):
    01_headline_bars
    02_per_episode_trajs
    03_cpc_comparison
    04_paired_dotplot

USAGE (from anywhere, e.g. the project root):
    python src6/make_plots.py
    python -m src6.make_plots
    python src6/make_plots.py --plot 1,4
    python src6/make_plots.py --csv-dir some/other/dir --out-dir some/plots

Requires: matplotlib, pandas, numpy
    pip install matplotlib pandas numpy
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Path anchoring, identical convention to evaluate_all.py
#   HERE         = .../dynamic_ad_allocation/src6
#   PROJECT_ROOT = .../dynamic_ad_allocation
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_CSV_DIR = PROJECT_ROOT / "outputs" / "eval_all"
DEFAULT_OUT_DIR = HERE / "plots"            # -> src6/plots/

# ----------------------------------------------------------------------
# Thesis style: sans-serif, white background, print-friendly, >=9pt fonts
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

DISPLAY = {
    "rand_baseline": "Rand",
    "const_baseline": "Const",
    "pctr_baseline": "Lin (pCTR)",
    "src5_ac": "src5 AC",
    "src6_mappo": "src6 MAPPO",
}
COLOR = {
    "rand_baseline":  "#bdbdbd",
    "const_baseline": "#878787",
    "pctr_baseline":  "#525252",
    "src5_ac":        "#08519c",
    "src6_mappo":     "#cb181d",
}
ORDER = ["rand_baseline", "const_baseline", "pctr_baseline", "src5_ac", "src6_mappo"]


def _save(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    print(f"  wrote {stem}.png and {stem}.pdf")


# ======================================================================
# Plot 1 - headline bars
# ======================================================================
def plot_headline(agg, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    means, sds, colors, labels, keys = [], [], [], [], []
    for p in ORDER:
        row = agg[agg["policy"] == p]
        if row.empty:
            continue
        means.append(float(row["across_seed_mean_clicks"].values[0]))
        sds.append(float(row["across_seed_sd_of_means"].values[0]))
        colors.append(COLOR[p]); labels.append(DISPLAY[p]); keys.append(p)

    xs = np.arange(len(means))
    ax.bar(xs, means, yerr=sds, color=colors, capsize=4,
           error_kw={"linewidth": 1.0, "ecolor": "#404040"},
           edgecolor="black", linewidth=0.5)

    lin = float(agg[agg["policy"] == "pctr_baseline"]["across_seed_mean_clicks"].values[0])
    for i, (k, m, sd) in enumerate(zip(keys, means, sds)):
        if k in ("src5_ac", "src6_mappo"):
            lift = (m - lin) / lin * 100
            ax.text(xs[i], m + sd + 30, f"+{lift:.1f}%\nvs Lin",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222")

    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("Clicks per episode\n(mean across 5 seeds x 10 episodes)")
    ax.set_title("Held-out evaluation: clicks by policy")
    ax.set_ylim(0, max(means) + max(sds) + 150)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
    _save(fig, out_dir, "01_headline_bars"); plt.close(fig)


# ======================================================================
# Plot 2 - per-episode trajectories, 2x3 grid (5 seeds + legend)
# ======================================================================
def plot_trajectories(table, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.8), sharey=True)
    axes = axes.flatten()
    data = table[table["seed"].astype(str).str.match(r"^\d+$")
                 & table["ep"].astype(str).str.match(r"^\d+$")].copy()
    data["seed"] = data["seed"].astype(int)
    data["ep"] = data["ep"].astype(int)

    seeds = sorted(data["seed"].unique())
    for i, seed in enumerate(seeds):
        if i >= len(axes):
            break
        ax = axes[i]
        sub = data[data["seed"] == seed].sort_values("ep")
        for p in ORDER:
            if p in sub.columns:
                ax.plot(sub["ep"].values, sub[p].values, color=COLOR[p],
                        marker="o", markersize=3, label=DISPLAY[p])
        ax.set_title(f"Seed {seed}", fontsize=10)
        ax.set_xticks(range(0, 10, 2))
        ax.grid(alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
        if i % 3 == 0:
            ax.set_ylabel("Clicks")
        if i >= 3:
            ax.set_xlabel("Episode")

    # use any leftover panel for the legend
    legend_ax = axes[len(seeds)] if len(seeds) < len(axes) else axes[-1]
    legend_ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", fontsize=10,
                     title="Policy", title_fontsize=10)
    # hide any remaining empty panels
    for j in range(len(seeds), len(axes)):
        if axes[j] is not legend_ax:
            axes[j].axis("off")

    fig.suptitle("Clicks per episode, all seeds", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_dir, "02_per_episode_trajs"); plt.close(fig)


# ======================================================================
# Plot 3 - cost-per-click bars
# ======================================================================
def plot_cpc(per_seed, out_dir):
    df = per_seed.copy()
    df["cpc"] = df["mean_cost"] / df["mean_clicks"]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    means, sds, colors, labels = [], [], [], []
    for p in ORDER:
        sub = df[df["policy"] == p]
        if sub.empty:
            continue
        v = sub["cpc"].values
        means.append(float(np.mean(v)))
        sds.append(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        colors.append(COLOR[p]); labels.append(DISPLAY[p])

    xs = np.arange(len(means))
    ax.bar(xs, means, yerr=sds, color=colors, capsize=4,
           error_kw={"linewidth": 1.0, "ecolor": "#404040"},
           edgecolor="black", linewidth=0.5)
    for i, (m, sd) in enumerate(zip(means, sds)):
        ax.text(xs[i], m + sd + max(means) * 0.02, f"{m:,.0f}",
                ha="center", va="bottom", fontsize=8.5, color="#222")

    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("Cost per click\n(budget units, lower is better)")
    ax.set_title("Bidding efficiency: cost per acquired click")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
    ax.set_ylim(0, max(means) + max(sds) + max(means) * 0.10)
    _save(fig, out_dir, "03_cpc_comparison"); plt.close(fig)


# ======================================================================
# Plot 4 - paired-difference dotplot (src5 AC - Lin)
# ======================================================================
def plot_paired(paired, per_seed, out_dir):
    ac = per_seed[per_seed["policy"] == "src5_ac"].sort_values("seed")
    lin = per_seed[per_seed["policy"] == "pctr_baseline"].sort_values("seed")
    seeds = ac["seed"].values
    diffs = ac["mean_clicks"].values - lin["mean_clicks"].values
    mean_d = float(np.mean(diffs))

    row = paired[paired["comparison"].str.contains("src5_ac vs pctr", na=False)]
    if not row.empty:
        t_val = float(row["paired_t"].values[0])
        p_lab = str(row["p_t_label"].values[0])
        npos = int(row["n_positive"].values[0])
        npair = int(row["n_paired_seeds"].values[0])
    else:
        t_val, p_lab, npos, npair = float("nan"), "p?", len(diffs), len(diffs)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.scatter(seeds, diffs, s=80, color=COLOR["src5_ac"],
               edgecolor="black", linewidth=0.6, zorder=3,
               label="Per-seed paired difference")
    ax.axhline(mean_d, color=COLOR["src5_ac"], linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Mean diff = +{mean_d:.1f} clicks")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)

    ax.text(0.98, 0.40,
            f"Paired t = {t_val:.1f}, df = {npair - 1}\n{p_lab}, sign test: {npos}/{npair} positive",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.4", linewidth=0.5))

    ax.set_xticks(seeds)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("src5 AC - Lin (clicks)")
    ax.set_title("Paired per-seed difference: src5 AC vs Lin")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30),
              fontsize=9, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
    ax.set_ylim(-50, max(diffs) + 60)
    _save(fig, out_dir, "04_paired_dotplot"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate DAAWBC result figures.")
    ap.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR),
                    help="folder holding the four eval_all_*.csv files")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="folder to write the plots into (default: src6/plots)")
    ap.add_argument("--plot", default="all", help="'all' or comma list e.g. 1,4")
    a = ap.parse_args()

    csv_dir = Path(a.csv_dir).resolve()
    out_dir = Path(a.out_dir).resolve()

    needed = ["eval_all_aggregate.csv", "eval_all_per_seed.csv",
              "eval_all_per_seed_table.csv", "eval_all_paired.csv"]
    missing = [f for f in needed if not (csv_dir / f).exists()]
    if missing:
        print(f"ERROR: could not find these CSV files in {csv_dir}:")
        for f in missing:
            print(f"   - {f}")
        print("\nPoint --csv-dir at the folder where evaluate_all.py wrote them,")
        print("e.g.  python src6/make_plots.py --csv-dir outputs/eval_all")
        sys.exit(1)

    agg = pd.read_csv(csv_dir / "eval_all_aggregate.csv")
    per_seed = pd.read_csv(csv_dir / "eval_all_per_seed.csv")
    table = pd.read_csv(csv_dir / "eval_all_per_seed_table.csv")
    paired = pd.read_csv(csv_dir / "eval_all_paired.csv")

    want = {1, 2, 3, 4} if a.plot == "all" else {int(x) for x in a.plot.split(",")}
    print(f"Reading CSVs from : {csv_dir}")
    print(f"Writing plots to  : {out_dir}")
    if 1 in want: plot_headline(agg, out_dir)
    if 2 in want: plot_trajectories(table, out_dir)
    if 3 in want: plot_cpc(per_seed, out_dir)
    if 4 in want: plot_paired(paired, per_seed, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
