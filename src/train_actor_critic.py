import random
import csv
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from simulator.environment import RTBEnvironment
from policy_network import ActorCriticNetwork

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [0, 1, 2, 3, 4]
GAMMA = 0.99
LR = 1e-3
ENTROPY_BETA = 1e-3
CLIP_GRAD_NORM = 5.0

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

BID_VALUES = [0.0] + list(np.linspace(10, 150, 20))
NUM_ACTIONS = len(BID_VALUES)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def train_for_seed(seed):
    print(f"\n=== Actor–Critic | Seed {seed} ===")
    set_seed(seed)

    model = ActorCriticNetwork(input_dim=4, num_actions=NUM_ACTIONS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log_file = LOG_DIR / f"actor_critic_training_seed_{seed}.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "return", "clicks", "cost", "budget_left"]
        )

    for ep in range(1, 51):

        env = RTBEnvironment(
            data_path=str(DATA_FILE),
            budget=300.0,
            max_steps=10000,
            lambda_init=0.013,
        )

        state = to_tensor(env.reset())
        done = False
        ep_return = 0.0

        while not done:

            logits, value = model(state)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()

            bid = BID_VALUES[action.item()]
            next_state, reward, done = env.step(bid)

            next_state = to_tensor(next_state)

            # Bootstrap value
            with torch.no_grad():
                _, next_value = model(next_state)
                td_target = reward + GAMMA * next_value.squeeze()

            advantage = td_target - value.squeeze()

            # Losses
            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)
            entropy = dist.entropy()

            loss = actor_loss + critic_loss - ENTROPY_BETA * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

            state = next_state
            ep_return += reward

        print(
            f"Seed {seed} | Ep {ep:03d} | "
            f"Return={ep_return:.3f} | "
            f"Clicks={env.total_clicks} | "
            f"Cost={env.cost:.2f}"
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [ep, ep_return, env.total_clicks, env.cost, env.remaining_budget]
            )

    model_path = ROOT / f"policy_actor_critic_seed_{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved: {model_path.name}")


if __name__ == "__main__":
    for seed in SEEDS:
        train_for_seed(seed)
