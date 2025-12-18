from pathlib import Path
from simulator.environment import RTBEnvironment

# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

BUDGET = 300.0
MAX_STEPS = 10000
ALPHA = 300.0   # linear pCTR scaling


def main():
    env = RTBEnvironment(
        data_path=str(DATA_FILE),
        budget=BUDGET,
        max_steps=MAX_STEPS,
        lambda_init=0.0,
    )

    state = env.reset()
    done = False

    while not done:
        pctr = state[2]   # state = [budget_ratio, time_ratio, pctr, market_price]
        bid = ALPHA * pctr
        state, reward, done = env.step(bid)

    print("=== Linear pCTR Evaluation ===")
    print(f"Alpha          : {ALPHA}")
    print(f"Total clicks   : {env.total_clicks}")
    print(f"Total cost     : {env.cost:.2f}")
    print(f"Budget spent   : {BUDGET - env.remaining_budget:.2f}")
    print(f"Budget left    : {env.remaining_budget:.2f}")
    print(f"CTR            : {env.total_clicks / env.steps:.6f}")


if __name__ == "__main__":
    main()
