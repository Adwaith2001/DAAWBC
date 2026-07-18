"""
src6/sensitivity_full.py
=========================
Hyperparameter sensitivity study for reviewer comment R1-3.

Sweeps three training-time hyperparameters using your actual training script:
  - Entropy coefficient (--entropy-beta)
  - Discount factor (--gamma)
  - Learning rate (--lr)

Each sweep launches src5/train_context_ac.py via subprocess with modified
CLI flags. Reads back the training CSV (context_ac_seed_{N}.csv) and reports
mean total_clicks over the last 20% of episodes as the sensitivity metric.

The densification target CTR sweep from sensitivity_densification.py is
independent and cheap; if you also want it in this table, run this after:
    python src6/sensitivity_densification.py

USAGE (overnight balanced run):
    python src6/sensitivity_full.py --sweep all --n-seeds 2 --n-episodes 25

Safety test (single hyperparameter, small):
    python src6/sensitivity_full.py --sweep entropy --n-seeds 1 --n-episodes 15

Individual sweep:
    python src6/sensitivity_full.py --sweep gamma --n-seeds 2 --n-episodes 25
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEFAULT_TRAINING_SCRIPT = PROJECT_ROOT / "src5" / "train_context_ac.py"
DEFAULT_DATA_FILE       = PROJECT_ROOT / "data_2" / "shared_auction_log_v4_dense.txt"
DEFAULT_OUT_DIR         = HERE / "plots"
DEFAULT_SWEEP_OUT_ROOT  = HERE / "sensitivity_runs"


# Paper's chosen values
PAPER_VALUES = {
    "entropy_beta": 0.03,
    "gamma":        0.99,
    "lr":           3e-4,
}

# Sweep grids (3 values each, centered on paper value)
SWEEP_GRIDS = {
    "entropy_beta": [0.01, 0.03, 0.10],
    "gamma":        [0.95, 0.99, 0.999],
    "lr":           [1e-4, 3e-4, 1e-3],
}

# Map internal names to actual train_context_ac.py CLI flags
FLAG_MAP = {
    "entropy_beta": "--entropy-beta",
    "gamma":        "--gamma",
    "lr":           "--lr",
}


# ================================================================
# One training sweep
# ================================================================
def run_one_training(
    hyperparameter: str,
    value: float,
    seed: int,
    n_episodes: int,
    training_script: Path,
    data_file: Path,
    sweep_out_root: Path,
) -> dict | None:
    """Launch train_context_ac.py with modified hyperparameter, one seed."""
    run_id = f"{hyperparameter}_{value:g}_seed{seed}".replace(".", "p")
    output_dir = sweep_out_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    flag = FLAG_MAP[hyperparameter]
    cmd = [
        sys.executable, str(training_script),
        "--data", str(data_file),
        "--episodes", str(n_episodes),
        "--seeds", str(seed),
        flag, str(value),
        "--output-dir", str(output_dir),
    ]

    print(f"    seed {seed}: launching {' '.join(cmd[-6:])}")
    t0 = time.time()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=6 * 3600)  # 6h cap per single training
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"    seed {seed}: NONZERO RETURN {proc.returncode}")
            print(f"    stderr tail: ...{proc.stderr[-400:]}")
            return {
                "seed": seed, "value": value, "hyperparameter": hyperparameter,
                "mean_final_clicks": None, "wall_time_sec": elapsed,
                "returncode": proc.returncode,
                "error": proc.stderr[-400:],
            }

        # Read back the training CSV
        csv_path = output_dir / f"context_ac_seed_{seed}.csv"
        if not csv_path.exists():
            print(f"    seed {seed}: NO CSV FOUND at {csv_path}")
            return {
                "seed": seed, "value": value, "hyperparameter": hyperparameter,
                "mean_final_clicks": None, "wall_time_sec": elapsed,
                "error": "training CSV missing",
            }

        df = pd.read_csv(csv_path)
        # Take mean of last 20% of episodes as the "converged" metric
        n_last = max(1, int(0.20 * len(df)))
        last_clicks = df["total_clicks"].tail(n_last).values
        mean_final = float(np.mean(last_clicks))

        print(f"    seed {seed}: done in {elapsed/60:.1f} min, "
              f"final clicks = {mean_final:.1f} "
              f"(mean of last {n_last} eps)")

        return {
            "seed": seed, "value": value, "hyperparameter": hyperparameter,
            "mean_final_clicks": mean_final,
            "n_last_episodes_averaged": n_last,
            "n_episodes_total": len(df),
            "wall_time_sec": elapsed,
            "returncode": 0,
        }

    except subprocess.TimeoutExpired:
        print(f"    seed {seed}: TIMEOUT after 6 hours")
        return {
            "seed": seed, "value": value, "hyperparameter": hyperparameter,
            "mean_final_clicks": None,
            "wall_time_sec": 6 * 3600,
            "error": "timeout",
        }


def run_sweep(
    hyperparameter: str,
    values: list,
    n_seeds: int,
    n_episodes: int,
    training_script: Path,
    data_file: Path,
    sweep_out_root: Path,
) -> list:
    """Run all (value, seed) combinations for one hyperparameter."""
    print(f"\n{'=' * 70}")
    print(f"SWEEP: {hyperparameter}")
    print(f"  values : {values}")
    print(f"  seeds  : {list(range(n_seeds))}")
    print(f"  episodes/seed: {n_episodes}")
    print(f"  paper value  : {PAPER_VALUES.get(hyperparameter)}")
    print(f"  est. runs    : {len(values) * n_seeds}")
    print(f"  est. time    : {len(values) * n_seeds * n_episodes * 2 / 60:.1f} hours")
    print(f"{'=' * 70}")

    per_run = []
    for value in values:
        print(f"\n  --- {hyperparameter} = {value} ---")
        for seed in range(n_seeds):
            result = run_one_training(
                hyperparameter, value, seed, n_episodes,
                training_script, data_file, sweep_out_root,
            )
            if result:
                per_run.append(result)
            # Save intermediate results after every run
            save_intermediate(per_run, sweep_out_root / f"{hyperparameter}_progress.json")

    # Aggregate per value
    aggregated = []
    for value in values:
        rows = [r for r in per_run
                if r["value"] == value and r.get("mean_final_clicks") is not None]
        if not rows:
            aggregated.append({
                "hyperparameter": hyperparameter,
                "value": value,
                "mean_clicks": None,
                "sd_clicks": None,
                "n_seeds_completed": 0,
                "is_paper_value": abs(value - PAPER_VALUES.get(hyperparameter, -999)) < 1e-9,
            })
            continue
        clicks = [r["mean_final_clicks"] for r in rows]
        aggregated.append({
            "hyperparameter": hyperparameter,
            "value": value,
            "mean_clicks": float(np.mean(clicks)),
            "sd_clicks": float(np.std(clicks, ddof=1)) if len(clicks) > 1 else 0.0,
            "n_seeds_completed": len(clicks),
            "is_paper_value": abs(value - PAPER_VALUES.get(hyperparameter, -999)) < 1e-9,
        })

    return aggregated


def save_intermediate(per_run: list, path: Path):
    """Save progress after every training run so nothing's lost on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(per_run, f, indent=2, default=str)


# ================================================================
# LaTeX output
# ================================================================
DISPLAY_NAMES = {
    "entropy_beta": "Entropy $\\beta_{\\text{ent}}$",
    "gamma":        "Discount $\\gamma$",
    "lr":           "Learning rate",
}


def generate_latex(all_aggregated: list, out_path: Path):
    lines = [
        "% Auto-generated by src6/sensitivity_full.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Training-time hyperparameter sensitivity study. Each row "
        "reports the mean total-clicks per episode over the last 20\\% of "
        "training episodes, averaged across 2 seeds. Paper's chosen value "
        "is in bold. Reduced-cost protocol: 2 seeds, 25 episodes each, "
        "instead of the paper's 5 seeds x 80 episodes.}",
        "\\label{tab:sensitivity_full}",
        "\\small",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Hyperparameter & Value & Final clicks (mean $\\pm$ SD) \\\\",
        "\\midrule",
    ]

    grouped = {}
    for row in all_aggregated:
        grouped.setdefault(row["hyperparameter"], []).append(row)

    for hp, rows in grouped.items():
        name = DISPLAY_NAMES.get(hp, hp)
        for i, row in enumerate(rows):
            hp_col = name if i == 0 else ""
            val_str = f"{row['value']:g}"
            if row['mean_clicks'] is not None:
                res_str = f"{row['mean_clicks']:.1f} $\\pm$ {row['sd_clicks']:.1f}"
            else:
                res_str = "--"

            if row["is_paper_value"]:
                val_str = "\\textbf{" + val_str + "}"
                res_str = "\\textbf{" + res_str + "}"

            lines.append(f"{hp_col} & {val_str} & {res_str} \\\\")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines = lines[:-1]

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep",
                    choices=["all", "entropy", "gamma", "lr"],
                    default="all")
    ap.add_argument("--n-seeds", type=int, default=2,
                    help="Seeds per hyperparameter value (default: 2)")
    ap.add_argument("--n-episodes", type=int, default=25,
                    help="Episodes per training run (default: 25)")
    ap.add_argument("--training-script", default=str(DEFAULT_TRAINING_SCRIPT))
    ap.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="Where to write the aggregated JSON and .tex")
    ap.add_argument("--sweep-out-root", default=str(DEFAULT_SWEEP_OUT_ROOT),
                    help="Where to write per-run training outputs")
    a = ap.parse_args()

    training_script = Path(a.training_script).resolve()
    data_file       = Path(a.data_file).resolve()
    out_dir         = Path(a.out_dir).resolve()
    sweep_out_root  = Path(a.sweep_out_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_out_root.mkdir(parents=True, exist_ok=True)

    if not training_script.exists():
        print(f"ERROR: training script not found at {training_script}")
        raise SystemExit(1)
    if not data_file.exists():
        print(f"ERROR: data file not found at {data_file}")
        raise SystemExit(1)

    print("=" * 70)
    print("DAAWBC Full Sensitivity Study (Reviewer 1, point 3)")
    print("=" * 70)
    print(f"Sweep type      : {a.sweep}")
    print(f"Training script : {training_script}")
    print(f"Data file       : {data_file}")
    print(f"Seeds per value : {a.n_seeds}")
    print(f"Episodes/seed   : {a.n_episodes}")
    print(f"Sweep runs at   : {sweep_out_root}")
    print(f"Final outputs   : {out_dir}")

    if a.sweep == "all":
        to_run = ["entropy_beta", "gamma", "lr"]
    else:
        map_flag = {"entropy": "entropy_beta", "gamma": "gamma", "lr": "lr"}
        to_run = [map_flag[a.sweep]]

    # Estimate total time
    total_runs = len(to_run) * len(SWEEP_GRIDS[to_run[0]]) * a.n_seeds
    est_hours = total_runs * a.n_episodes * 2 / 60
    print(f"\nTotal runs      : {total_runs}")
    print(f"Estimated time  : {est_hours:.1f} hours")
    print()

    t0 = time.time()
    all_aggregated = []
    for hp in to_run:
        agg = run_sweep(
            hp, SWEEP_GRIDS[hp], a.n_seeds, a.n_episodes,
            training_script, data_file, sweep_out_root,
        )
        all_aggregated.extend(agg)

        # Save incrementally after each sweep
        json_path = out_dir / "sensitivity_full.json"
        with open(json_path, "w") as f:
            json.dump({
                "sweep": a.sweep,
                "n_seeds": a.n_seeds,
                "n_episodes": a.n_episodes,
                "paper_values": PAPER_VALUES,
                "results": all_aggregated,
            }, f, indent=2, default=str)
        print(f"\n  Intermediate JSON saved: {json_path}")

        tex_path = out_dir / "sensitivity_full.tex"
        generate_latex(all_aggregated, tex_path)
        print(f"  Intermediate LaTeX saved: {tex_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total wall time : {elapsed / 60:.1f} minutes")
    print(f"Aggregate JSON  : {out_dir / 'sensitivity_full.json'}")
    print(f"LaTeX table     : {out_dir / 'sensitivity_full.tex'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
