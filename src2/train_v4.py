"""
train_v4.py
5-Agent Actor-Critic v4

Improvements over v3:
1. Enhanced features (13 features in state space)
2. Heterogeneous budgets (realistic advertiser sizes)
3. Enhanced pCTR files from filtered_output/{adv}/enhanced/
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

from simulator.multi_environment_v2 import MultiRTBEnvironmentV2
from policy_network_v2 import ActorCriticNetworkV2

# ======================================================
# DEVICE
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ======================================================
# PATHS — Enhanced pCTR files (13 features)
# ======================================================
ENHANCED_DIR = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output"
)

OUT_DIR = ROOT / "outputs" / "final_experiments_5agents_v4"
MDL_DIR = ROOT / "models" / "5adv_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MDL_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATHS = {
    0: ENHANCED_DIR / "1458" / "enhanced" / "final_sample_log_with_pctr.txt",
    1: ENHANCED_DIR / "2259" / "enhanced" / "final_sample_log_with_pctr.txt",
    2: ENHANCED_DIR / "2821" / "enhanced" / "final_sample_log_with_pctr.txt",
    3: ENHANCED_DIR / "2997" / "enhanced" / "final_sample_log_with_pctr.txt",
    4: ENHANCED_DIR / "3358" / "enhanced" / "final_sample_log_with_pctr.txt",
}

# ======================================================
# CONFIG
# ======================================================
NUM_AGENTS = 5
ADV_IDS    = ["1458", "2259", "2821", "2997", "3358"]

# ✅ IMPROVEMENT 1: Heterogeneous budgets
# Reflects real advertisers with different campaign sizes
BUDGETS = [20000.0, 15000.0, 25000.0, 10000.0, 18000.0]

MAX_STEPS = 10000
EPISODES  = 200
SEEDS     = [0, 1, 2, 3, 4]

GAMMA        = 0.99
LR           = 5e-4
ENTROPY_BETA = 1e-3
CLIP_GRAD    = 1.0
UPDATE_EVERY = 50

# pCTR threshold action space
THRESHOLD_VALUES = list(np.linspace(0.0, 0.9, 51))
NUM_ACTIONS      = len(THRESHOLD_VALUES)

# ✅ IMPROVEMENT 2: 13 features in state space (up from 10)
STATE_DIM = 13

print(f"Action space : {NUM_ACTIONS} pCTR thresholds")
print(f"State dim    : {STATE_DIM} features (enhanced)")
print(f"Budgets      : {BUDGETS} (heterogeneous)")
print(f"Episodes     : {EPISODES} | Seeds: {SEEDS}")
print(f"Core concept : pCTR threshold — agent learns WHEN to bid")


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
# BATCH UPDATE
# ======================================================
def update_agents(agents, optimizers, batch):
    for i in range(NUM_AGENTS):
        if not batch[i]:
            continue

        states_b      = torch.stack([t["state"]      for t in batch[i]])
        actions_b     = torch.stack([t["action"]     for t in batch[i]])
        rewards_b     = torch.tensor([t["reward"]    for t in batch[i]],
                                      dtype=torch.float32, device=DEVICE)
        next_states_b = torch.stack([t["next_state"] for t in batch[i]])
        dones_b       = torch.tensor([t["done"]      for t in batch[i]],
                                      dtype=torch.float32, device=DEVICE)

        logits_b, values_b = agents[i](states_b)
        dist_b      = Categorical(logits=logits_b)
        log_probs_b = dist_b.log_prob(actions_b)
        entropies_b = dist_b.entropy()
        values_b    = values_b.squeeze()

        with torch.no_grad():
            _, next_values_b = agents[i](next_states_b)
            next_values_b    = next_values_b.squeeze()
            td_targets = rewards_b + GAMMA * next_values_b * (1 - dones_b)

        advantages = td_targets - values_b

        actor_loss   = -(log_probs_b * advantages.detach()).mean()
        critic_loss  = advantages.pow(2).mean()
        entropy_loss = -ENTROPY_BETA * entropies_b.mean()

        loss = actor_loss + critic_loss + entropy_loss

        optimizers[i].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agents[i].parameters(), CLIP_GRAD)
        optimizers[i].step()


# ======================================================
# MAIN TRAINING LOOP
# ======================================================
for seed in SEEDS:

    print(f"\n{'='*60}")
    print(f" v4 | 5 Agents | Seed {seed}")
    print(f" Enhanced features (13) + Heterogeneous budgets")
    print(f"{'='*60}")

    set_seed(seed)

    env = MultiRTBEnvironmentV2(
        data_paths    = DATA_PATHS,
        budgets       = BUDGETS,
        max_steps     = MAX_STEPS,
        reserve_price = 1.0,
    )

    agents     = []
    optimizers = []
    for _ in range(NUM_AGENTS):
        model = ActorCriticNetworkV2(
            input_dim   = STATE_DIM,   # ← 13 features
            num_actions = NUM_ACTIONS,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        agents.append(model)
        optimizers.append(optimizer)

    logs = []

    for ep in range(1, EPISODES + 1):

        states = env.reset()
        states = [
            torch.tensor(s, dtype=torch.float32, device=DEVICE)
            for s in states
        ]

        done       = False
        ep_rewards = np.zeros(NUM_AGENTS)
        batch      = [[] for _ in range(NUM_AGENTS)]
        step       = 0

        while not done:

            thresholds = []
            step_data  = [None] * NUM_AGENTS

            for i in range(NUM_AGENTS):
                with torch.no_grad():
                    logits, value = agents[i](states[i])
                dist   = Categorical(logits=logits.squeeze(0))
                action = dist.sample()

                threshold = THRESHOLD_VALUES[action.item()]
                thresholds.append(threshold)

                step_data[i] = {
                    "state" : states[i],
                    "action": action,
                }

            next_states, rewards, done = env.step(thresholds)

            next_states_t = [
                torch.tensor(s, dtype=torch.float32, device=DEVICE)
                for s in (next_states if not done
                          else [np.zeros(STATE_DIM, dtype=np.float32)]
                               * NUM_AGENTS)
            ]

            for i in range(NUM_AGENTS):
                step_data[i]["reward"]     = float(rewards[i])
                step_data[i]["next_state"] = next_states_t[i]
                step_data[i]["done"]       = float(done)
                batch[i].append(step_data[i])
                ep_rewards[i] += float(rewards[i])

            if not done:
                states = next_states_t

            step += 1

            if step % UPDATE_EVERY == 0 or done:
                update_agents(agents, optimizers, batch)
                batch = [[] for _ in range(NUM_AGENTS)]

        # Log
        log_entry = {"seed": seed, "episode": ep}
        for i, adv in enumerate(ADV_IDS):
            log_entry[f"reward_{adv}"] = ep_rewards[i]
            log_entry[f"clicks_{adv}"] = int(env.clicks[i])
            log_entry[f"cost_{adv}"]   = round(float(env.costs[i]), 2)
            log_entry[f"budget_{adv}"] = BUDGETS[i]
        logs.append(log_entry)

        print(
            f"Seed {seed} | Ep {ep:03d} | "
            f"Clicks={env.clicks.tolist()} | "
            f"Costs={[round(float(c), 0) for c in env.costs]} | "
            f"Rewards={[round(float(r), 2) for r in ep_rewards]}"
        )

    # Save CSV
    df = pd.DataFrame(logs)
    out_file = OUT_DIR / f"actor_critic_5adv_v4_seed_{seed}.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file}")

    # Save models
    for i, adv in enumerate(ADV_IDS):
        model_path = MDL_DIR / f"policy_v4_{adv}_seed_{seed}.pt"
        torch.save(agents[i].state_dict(), model_path)
        print(f"✅ Saved model: {model_path.name}")

print("\n🎉 v4 TRAINING COMPLETE!")