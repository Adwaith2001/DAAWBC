import random
import numpy as np
from pathlib import Path
import pandas as pd

from simulator.environment import RTBEnvironment

# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

BUDGET = 300.0
FIXED_BID = 50.0
LINEAR_ALPHA = 300.0

SEEDS = [0, 1, 2, 3, 4]

# =========================
# UTILS
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# =========================
# BASELINES
# =========================
def run_fixed_bid(seed):
    set_seed(seed)

    env = RTBEnvironment(
        data_path=str(DATA_FILE),
        budget=BUDGET,
        max_steps=10000,
    )

    env.reset()

    done = False
    while not done:
        _, _, done = env.step(FIXED_BID)

    return env.total_clicks, env.cost


def run_linear_pctr(seed):
    set_seed(seed)

    env = RTBEnvironment(
        data_path=str(DATA_FILE),
        budget=BUDGET,
        max_steps=10000,
    )

    env.reset()

    done = False
    while not done:
        row = env.df.iloc[env.ptr]
        bid = LINEAR_ALPHA * row["pctr"]
        _, _, done = env.step(bid)

    return env.total_clicks, env.cost


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("\n==============================")
    print(" MULTI-SEED BASELINE COMPARISON")
    print("==============================\n")

    fixed_results = []
    linear_results = []

    for seed in SEEDS:
        c, cost = run_fixed_bid(seed)
        fixed_results.append((c, cost))

        c, cost = run_linear_pctr(seed)
        linear_results.append((c, cost))

    fixed_clicks = np.array([x[0] for x in fixed_results])
    linear_clicks = np.array([x[0] for x in linear_results])

    print("Fixed Bid (mean ± std)")
    print(f"Clicks : {fixed_clicks.mean():.2f} ± {fixed_clicks.std():.2f}")

    print("\nLinear pCTR (mean ± std)")
    print(f"Clicks : {linear_clicks.mean():.2f} ± {linear_clicks.std():.2f}")
