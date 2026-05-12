import torch
import numpy as np
from pathlib import Path
from torch.distributions import Categorical

from simulator.environment import RTBEnvironment
from policy_network import PolicyNetwork

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

SEEDS = [0, 1, 2, 3, 4]

# ✅ FIXED HERE
MODEL_TEMPLATE = str(ROOT / "policy_reinforce_seed_{}.pt")

BID_VALUES = [0.0] + list(np.linspace(10, 150, 20))
NUM_ACTIONS = len(BID_VALUES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUDGET = 300.0
MAX_STEPS = 10000


def run_one(seed):
    env = RTBEnvironment(
        data_path=str(DATA_FILE),
        budget=BUDGET,
        max_steps=MAX_STEPS,
    )

    policy = PolicyNetwork(input_dim=4, num_actions=NUM_ACTIONS).to(DEVICE)
    policy.load_state_dict(
    torch.load(MODEL_TEMPLATE.format(seed), map_location=DEVICE, weights_only=True)
    )

    policy.eval()

    state = torch.tensor(env.reset(), dtype=torch.float32, device=DEVICE)
    done = False

    while not done:
        with torch.no_grad():
            logits = policy(state)
            action = Categorical(logits=logits).sample()

        bid = BID_VALUES[action.item()]
        next_state, _, done = env.step(bid)
        state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)

    return env.total_clicks, env.cost


def main():
    clicks = []
    costs = []

    for seed in SEEDS:
        c, cost = run_one(seed)
        clicks.append(c)
        costs.append(cost)

    print("\n=== REINFORCE (Multi-Seed Evaluation) ===")
    print(f"Clicks : {np.mean(clicks):.2f} ± {np.std(clicks):.2f}")
    print(f"Cost   : {np.mean(costs):.2f}")


if __name__ == "__main__":
    main()
