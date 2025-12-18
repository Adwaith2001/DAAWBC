import random
import csv
import os
import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path

from simulator.environment import RTBEnvironment
from policy_network import PolicyNetwork

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [0, 1, 2, 3, 4]

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

BID_VALUES = [0.0] + list(np.linspace(10, 150, 20))
NUM_ACTIONS = len(BID_VALUES)

EPISODES = 50
GAMMA = 0.99
LR = 1e-3
MAX_STEPS_PER_EP = 10000

ENTROPY_BETA = 1e-3
CLIP_GRAD_NORM = 5.0
LAMBDA_LR = 1e-4

# =========================
# LOGGING
# =========================
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# =========================
# UTILS
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)

def discounted_returns(rewards, gamma):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32, device=DEVICE)

# =========================
# TRAIN
# =========================
def train_for_seed(seed):
    print(f"\n==============================")
    print(f" Training with SEED = {seed}")
    print(f"==============================")

    set_seed(seed)

    policy = PolicyNetwork(input_dim=4, num_actions=NUM_ACTIONS).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    log_file = LOG_DIR / f"reinforce_training_seed_{seed}.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "return", "clicks", "cost", "budget_left", "lambda"]
        )

    for ep in range(1, EPISODES + 1):
        env = RTBEnvironment(
            data_path=str(DATA_FILE),
            budget=300.0,
            max_steps=MAX_STEPS_PER_EP,
            lambda_init=0.013,
        )

        state = to_tensor(env.reset())
        log_probs, rewards, entropies = [], [], []
        done, t = False, 0

        while not done:
            logits = policy(state)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            bid = BID_VALUES[action.item()]
            next_state, reward, done = env.step(bid)

            rewards.append(float(reward))
            state = to_tensor(next_state)

            t += 1
            if t >= MAX_STEPS_PER_EP:
                break

        returns = discounted_returns(rewards, GAMMA)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = (
            -(torch.stack(log_probs) * returns).sum()
            - ENTROPY_BETA * torch.stack(entropies).sum()
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        ep_return = sum(rewards)

        overspend = env.cost - env.budget
        env.lambda_penalty += LAMBDA_LR * overspend

        print(
            f"Seed {seed} | Ep {ep:03d} | "
            f"return={ep_return:.3f} | "
            f"clicks={env.total_clicks} | "
            f"cost={env.cost:.2f}"
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ep,
                    ep_return,
                    env.total_clicks,
                    env.cost,
                    env.remaining_budget,
                    env.lambda_penalty,
                ]
            )

    model_path = ROOT / f"policy_reinforce_seed_{seed}.pt"
    torch.save(policy.state_dict(), model_path)
    print(f"✅ Saved policy: {model_path.name}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("Multi-seed REINFORCE training started")

    for seed in SEEDS:
        train_for_seed(seed)

    print("\n✅ All seeds finished")
