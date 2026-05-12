import numpy as np
from simulator.environment import RTBEnvironment
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

SEEDS = [0, 1, 2, 3, 4]
ALPHA = 300.0
BUDGET = 300.0
MAX_STEPS = 10000


def run_one(seed):
    env = RTBEnvironment(
        data_path=str(DATA_FILE),
        budget=BUDGET,
        max_steps=MAX_STEPS,
    )

    env.reset()
    np.random.seed(seed)

    done = False
    while not done:
        row = env.df.iloc[env.ptr]
        bid = ALPHA * row["pctr"]
        _, _, done = env.step(bid)

    return env.total_clicks, env.cost


def main():
    clicks = []
    costs = []

    for seed in SEEDS:
        c, cost = run_one(seed)
        clicks.append(c)
        costs.append(cost)

    print("\n=== LINEAR pCTR (Multi-Seed Evaluation) ===")
    print(f"Clicks : {np.mean(clicks):.2f} ± {np.std(clicks):.2f}")
    print(f"Cost   : {np.mean(costs):.2f}")


if __name__ == "__main__":
    main()
