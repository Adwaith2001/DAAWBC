import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"

SEEDS = [0, 1, 2, 3, 4]

def load_final(prefix):
    rows = []
    for seed in SEEDS:
        df = pd.read_csv(LOG_DIR / f"{prefix}_seed_{seed}.csv")
        last = df.iloc[-1]
        rows.append(last["clicks"])
    return np.mean(rows), np.std(rows)

mean_300, std_300 = load_final("actor_critic_training")
mean_10k, std_10k = load_final("actor_critic_budget10k")

print("\n=== Budget Comparison (Actor–Critic) ===\n")
print("Budget = 300")
print(f"Clicks : {mean_300:.2f} ± {std_300:.2f}\n")

print("Budget = 10k")
print(f"Clicks : {mean_10k:.2f} ± {std_10k:.2f}")

