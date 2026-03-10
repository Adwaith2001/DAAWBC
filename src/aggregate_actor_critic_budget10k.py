import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"

SEEDS = [0, 1, 2, 3, 4]

records = []

for seed in SEEDS:
    file = LOG_DIR / f"actor_critic_budget10k_seed_{seed}.csv"
    df = pd.read_csv(file)

    last = df.iloc[-1]

    records.append({
        "seed": seed,
        "clicks": last["clicks"],
        "return": last["return"],
        "cost": last["cost"],
        "lambda": last["lambda"],
    })

results = pd.DataFrame(records)

print("\n=== Actor–Critic (Budget = 10k) | Final Episode ===\n")
print(results)

print("\n=== Mean ± Std ===")
print(f"Clicks : {results['clicks'].mean():.2f} ± {results['clicks'].std():.2f}")
print(f"Return : {results['return'].mean():.2f} ± {results['return'].std():.2f}")
print(f"Cost   : {results['cost'].mean():.2f}")
print(f"Lambda : {results['lambda'].mean():.5f}")
