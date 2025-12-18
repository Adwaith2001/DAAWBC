import pandas as pd
from pathlib import Path
import numpy as np

# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"

SEEDS = [0, 1, 2, 3, 4]

# =========================
# LOAD LAST EPISODE PER SEED
# =========================
records = []

for seed in SEEDS:
    log_file = LOG_DIR / f"reinforce_training_seed_{seed}.csv"
    if not log_file.exists():
        raise FileNotFoundError(f"Missing log file: {log_file}")

    df = pd.read_csv(log_file)

    # Take last episode (final policy performance)
    last = df.iloc[-1]

    records.append({
        "seed": seed,
        "return": last["return"],
        "clicks": last["clicks"],
        "cost": last["cost"],
        "budget_left": last["budget_left"],
        "lambda": last["lambda"],
    })

results = pd.DataFrame(records)

# =========================
# AGGREGATE STATS
# =========================
summary = {
    "clicks_mean": results["clicks"].mean(),
    "clicks_std": results["clicks"].std(),
    "return_mean": results["return"].mean(),
    "return_std": results["return"].std(),
    "cost_mean": results["cost"].mean(),
    "budget_left_mean": results["budget_left"].mean(),
}

# =========================
# PRINT RESULTS
# =========================
print("\n==============================")
print(" MULTI-SEED REINFORCE SUMMARY ")
print("==============================\n")

print("Per-seed final performance:")
print(results.to_string(index=False))

print("\nAggregated (mean ± std):")
print(f"Clicks        : {summary['clicks_mean']:.2f} ± {summary['clicks_std']:.2f}")
print(f"Return        : {summary['return_mean']:.2f} ± {summary['return_std']:.2f}")
print(f"Cost (mean)   : {summary['cost_mean']:.2f}")
print(f"Budget left   : {summary['budget_left_mean']:.2f}")

print("\n✅ Aggregation complete")
