import random
import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path
import pandas as pd

from simulator.multi_environment import MultiRTBEnvironment
from policy_network import ActorCriticNetwork

# ======================================================
# DEVICE
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# PATHS
# ======================================================
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "final_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATHS = {
    0: ROOT / "data" / "ipinyou" / "1458" / "final_sample_log_with_pctr.txt",
    1: ROOT / "data" / "ipinyou" / "2259" / "final_sample_log_with_pctr.txt",
    2: ROOT / "data" / "ipinyou" / "2821" / "final_sample_log_with_pctr.txt",
}

# ======================================================
# FINAL EXPERIMENT CONFIG (LOCKED)
# ======================================================
NUM_AGENTS = 3
BUDGETS = [1000.0, 1000.0, 1000.0]

MAX_STEPS = 5000
EPISODES = 100
SEEDS = [0, 1, 2, 3, 4]

GAMMA = 0.99
LR = 1e-3
ENTROPY_BETA = 1e-3

BID_VALUES = [0.0] + list(np.linspace(0.1, 5.0, 20))
NUM_ACTIONS = len(BID_VALUES)

RESERVE_PRICE = 0.1
LAMBDA_INIT = 0.005

# ======================================================
# SEED UTILS
# ======================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ======================================================
# MAIN MULTI-SEED TRAINING
# ======================================================
for seed in SEEDS:

    print(f"\n==============================")
    print(f"🚀 FINAL RUN | Seed {seed}")
    print(f"==============================")

    set_seed(seed)

    env = MultiRTBEnvironment(
        data_paths=DATA_PATHS,
        budgets=BUDGETS,
        max_steps=MAX_STEPS,
        lambda_init=LAMBDA_INIT,
        reserve_price=RESERVE_PRICE,
    )

    agents = []
    optimizers = []

    for _ in range(NUM_AGENTS):
        model = ActorCriticNetwork(
            input_dim=4,
            num_actions=NUM_ACTIONS
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        agents.append(model)
        optimizers.append(optimizer)

    logs = []

    # --------------------------------------------------
    # EPISODES
    # --------------------------------------------------
    for ep in range(1, EPISODES + 1):

        states = env.reset()
        states = [
            torch.tensor(s, dtype=torch.float32, device=DEVICE)
            for s in states
        ]

        done = False
        ep_rewards = np.zeros(NUM_AGENTS)

        while not done:

            bids = []
            log_probs = []
            values = []

            # Action selection
            for i in range(NUM_AGENTS):
                logits, value = agents[i](states[i])
                dist = Categorical(logits=logits.squeeze(0))
                action = dist.sample()

                bids.append(BID_VALUES[action.item()])
                log_probs.append(dist.log_prob(action))
                values.append(value.squeeze())

            # Environment step
            next_states, rewards, done = env.step(bids)

            if not done:
                next_states = [
                    torch.tensor(s, dtype=torch.float32, device=DEVICE)
                    for s in next_states
                ]

            # Actor–Critic update
            for i in range(NUM_AGENTS):
                with torch.no_grad():
                    if done:
                        td_target = rewards[i]
                    else:
                        _, next_value = agents[i](next_states[i])
                        td_target = rewards[i] + GAMMA * next_value.squeeze()

                advantage = td_target - values[i]

                actor_loss = -log_probs[i] * advantage.detach()
                critic_loss = advantage.pow(2)
                entropy_loss = -ENTROPY_BETA

                loss = actor_loss + critic_loss + entropy_loss

                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()

                ep_rewards[i] += rewards[i]

            if not done:
                states = next_states

        # Episode log
        logs.append({
            "seed": seed,
            "episode": ep,
            "reward_1458": ep_rewards[0],
            "reward_2259": ep_rewards[1],
            "reward_2821": ep_rewards[2],
            "clicks_1458": env.clicks[0],
            "clicks_2259": env.clicks[1],
            "clicks_2821": env.clicks[2],
            "cost_1458": env.costs[0],
            "cost_2259": env.costs[1],
            "cost_2821": env.costs[2],
        })

        print(
            f"Seed {seed} | Ep {ep:03d} | "
            f"Clicks={env.clicks} | Costs={env.costs}"
        )

    # Save per-seed CSV
    df = pd.DataFrame(logs)
    out_file = OUT_DIR / f"actor_critic_3adv_seed_{seed}.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file}")

print("\n🎉 FINAL MULTI-SEED EXPERIMENT COMPLETED")
