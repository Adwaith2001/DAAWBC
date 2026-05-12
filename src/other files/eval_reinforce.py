import torch
import numpy as np
from pathlib import Path
from torch.distributions import Categorical

from simulator.environment import RTBEnvironment
from policy_network import PolicyNetwork

# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"
MODEL_FILE = ROOT / "policy_reinforce.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BID_VALUES = [0.0] + list(np.linspace(10, 150, 20))
NUM_ACTIONS = len(BID_VALUES)

BUDGET = 300.0
MAX_STEPS = 10000


def main():
    env = RTBEnvironment(
        data_path=str(DATA_FILE),
        budget=BUDGET,
        max_steps=MAX_STEPS,
        lambda_init=0.0,
    )

    policy = PolicyNetwork(input_dim=4, num_actions=NUM_ACTIONS).to(DEVICE)
    policy.load_state_dict(
    torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True))

    policy.eval()

    state = torch.tensor(env.reset(), dtype=torch.float32, device=DEVICE)
    done = False

    while not done:
        with torch.no_grad():
            logits = policy(state)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()

        bid = BID_VALUES[action.item()]
        next_state, reward, done = env.step(bid)
        state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)

    print("=== REINFORCE Evaluation ===")
    print(f"Total clicks   : {env.total_clicks}")
    print(f"Total cost     : {env.cost:.2f}")
    print(f"Budget spent   : {BUDGET - env.remaining_budget:.2f}")
    print(f"Budget left    : {env.remaining_budget:.2f}")
    print(f"CTR            : {env.total_clicks / env.steps:.6f}")


if __name__ == "__main__":
    main()
