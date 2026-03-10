import pandas as pd
# Used for loading CSV logs and computing statistical aggregates

from pathlib import Path
# Provides robust, platform-independent path handling

import numpy as np
# Used implicitly for numerical stability and statistics (mean, std)


# =========================
# CONFIGURATION
# =========================

# Resolve project root directory (…/dynamic_ad_allocation)
ROOT = Path(__file__).resolve().parents[1]

# Directory where per-seed training logs are stored
LOG_DIR = ROOT / "logs"

# Random seeds used during REINFORCE training
# Each seed corresponds to an independent training run
SEEDS = [0, 1, 2, 3, 4]


# =========================
# LOAD FINAL EPISODE PER SEED
# =========================

# Container to store final performance metrics from each seed
records = []

for seed in SEEDS:

    # Construct path to the CSV log for this seed
    log_file = LOG_DIR / f"reinforce_training_seed_{seed}.csv"

    # Safety check: ensure log file exists
    if not log_file.exists():
        raise FileNotFoundError(f"Missing log file: {log_file}")

    # Load training log into DataFrame
    df = pd.read_csv(log_file)

    # ------------------------------------------------------------
    # Select final episode
    # ------------------------------------------------------------
    # The last row corresponds to the final learned policy
    # and represents post-convergence performance
    last = df.iloc[-1]

    # Store relevant metrics for this seed
    records.append({
        "seed": seed,                      # Random seed identifier
        "return": last["return"],          # Cumulative discounted reward
        "clicks": last["clicks"],          # Total clicks obtained
        "cost": last["cost"],              # Total cost incurred
        "budget_left": last["budget_left"],# Remaining unused budget
        "lambda": last["lambda"],          # Final Lagrangian penalty value
    })

# Convert collected records into a DataFrame
results = pd.DataFrame(records)


# =========================
# AGGREGATED STATISTICS
# =========================

# Compute mean and standard deviation across seeds
# This evaluates stability and robustness of the learned policy
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

# Per-seed performance table
print("Per-seed final performance:")
print(results.to_string(index=False))

# Aggregate statistics (mean ± standard deviation)
print("\nAggregated (mean ± std):")
print(f"Clicks        : {summary['clicks_mean']:.2f} ± {summary['clicks_std']:.2f}")
print(f"Return        : {summary['return_mean']:.2f} ± {summary['return_std']:.2f}")
print(f"Cost (mean)   : {summary['cost_mean']:.2f}")
print(f"Budget left   : {summary['budget_left_mean']:.2f}")

print("\n✅ Aggregation complete")
