# ============================================================
# STANDARD LIBRARIES
# ============================================================
import random
import csv
from pathlib import Path

# ============================================================
# NUMERICAL / ML LIBRARIES
# ============================================================
import numpy as np
import torch
from torch.distributions import Categorical

# ============================================================
# PROJECT MODULES
# ============================================================
from simulator.environment import RTBEnvironment
from policy_network import ActorCriticNetwork

# ============================================================
# DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# TRAINING CONFIG
# ============================================================
SEEDS = [0, 1, 2, 3, 4]

EPISODES = 30        # fewer episodes because budget is large
GAMMA = 0.99
LR = 1e-3

ENTROPY_BETA = 1e-3
CLIP_GRAD_NORM = 5.0

# ============================================================
# BUDGET CONFIG (10K)
# ============================================================
BUDGET = 10000.0     # ✅ 10K budget
LAMBDA_INIT = 0.002  # scaled down for large budget
LAMBDA_LR = 1e-4

MAX_STEPS = 15000    # allow longer episodes

# ============================================================
# ACTION SPACE
# ============================================================
BID_VALUES = [0.0] + list(np.linspace(10, 150, 20))
NUM_ACTIONS = len(BID_VALUES)

# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_for_seed(seed):

    print(f"\n==============================")
    print(f" Actor–Critic (Budget 10K) | Seed {seed}")
    print(f"==============================")

    set_seed(seed)

    model = ActorCriticNetwork(
        input_dim=4,
        num_actions=NUM_ACTIONS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log_file = LOG_DIR / f"actor_critic_budget10k_seed_{seed}.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "return", "clicks", "cost", "budget_left", "lambda"]
        )

    # ========================================================
    # EPISODE LOOP
    # ========================================================
    for ep in range(1, EPISODES + 1):

        env = RTBEnvironment(
            data_path=str(DATA_FILE),
            budget=BUDGET,
            max_steps=MAX_STEPS,
            lambda_init=LAMBDA_INIT,
        )

        state = to_tensor(env.reset())
        done = False
        ep_return = 0.0

        # ----------------------------------------------------
        # STEP LOOP
        # ----------------------------------------------------
        while not done:

            logits, value = model(state)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()

            bid = BID_VALUES[action.item()]
            next_state, reward, done = env.step(bid)
            next_state = to_tensor(next_state)

            # TD target
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
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CLIP_GRAD_NORM
            )
            optimizer.step()

            state = next_state
            ep_return += reward

        # ====================================================
        # ADAPTIVE LAGRANGIAN UPDATE
        # ====================================================
        overspend = env.cost - env.budget
        env.lambda_penalty = max(
            0.0,
            env.lambda_penalty + LAMBDA_LR * overspend
        )

        print(
            f"Seed {seed} | Ep {ep:03d} | "
            f"Return={ep_return:.3f} | "
            f"Clicks={env.total_clicks} | "
            f"Cost={env.cost:.2f} | "
            f"Lambda={env.lambda_penalty:.5f}"
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

    # ========================================================
    # SAVE MODEL
    # ========================================================
    model_path = ROOT / f"policy_actor_critic_budget10k_seed_{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved: {model_path.name}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":

    print(f"Using device: {DEVICE}")
    print("Actor–Critic with 10K budget started")

    for seed in SEEDS:
        train_for_seed(seed)

    print("\n✅ Training complete for all seeds")
