"""
train_v6.py
5-Agent Actor-Critic v6

Changes from v5:
1. Replaced advertiser 2821 (broken) with 3386 (better CTR + lower price)
2. Fixed budget assignment — proportional to CTR
3. Uses multi_environment_v5.py (same environment as v5)
"""

import sys
import random
import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

from simulator.multi_environment_v5 import MultiRTBEnvironmentV5
from policy_network_v2 import ActorCriticNetworkV2

# ======================================================
# DEVICE
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ======================================================
# PATHS
# ======================================================
ENHANCED_DIR = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output"
)

OUT_DIR = ROOT / "outputs" / "final_experiments_5agents_v6"
MDL_DIR = ROOT / "models" / "5adv_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MDL_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# V6: 3386 replaces 2821
# ======================================================
DATA_PATHS = {
    0: ENHANCED_DIR / "1458" / "enhanced" / "final_sample_log_with_pctr.txt",
    1: ENHANCED_DIR / "2259" / "enhanced" / "final_sample_log_with_pctr.txt",
    2: ENHANCED_DIR / "3386" / "enhanced" / "final_sample_log_with_pctr.txt",  # ← NEW
    3: ENHANCED_DIR / "2997" / "enhanced" / "final_sample_log_with_pctr.txt",
    4: ENHANCED_DIR / "3358" / "enhanced" / "final_sample_log_with_pctr.txt",
}

# ======================================================
# CONFIG
# ======================================================
NUM_AGENTS = 5

# V6: Updated advertiser IDs
ADV_IDS = ["1458", "2259", "3386", "2997", "3358"]

# V6: Fixed budget assignment — proportional to CTR
# CTR:    1458=0.088%, 2259=0.031%, 3386=0.091%, 2997=0.341%, 3358=0.113%
# Higher CTR → more budget (economically rational)
BUDGETS = [20000.0,   # 1458 — moderate CTR
           12000.0,   # 2259 — lowest CTR → less budget
           20000.0,   # 3386 — similar CTR to 1458
           25000.0,   # 2997 — highest CTR → most budget
           18000.0]   # 3358 — moderate CTR

MAX_STEPS = 10000
EPISODES  = 200
SEEDS     = [0, 1, 2, 3, 4]

GAMMA        = 0.99
LR           = 5e-4
ENTROPY_BETA = 1e-3
CLIP_GRAD    = 1.0
UPDATE_EVERY = 50

# Same as v5 — threshold cap 0.3
THRESHOLD_VALUES = list(np.linspace(0.0, 0.3, 51))
NUM_ACTIONS      = len(THRESHOLD_VALUES)

# 14 features (13 enhanced + urgency)
STATE_DIM = 14

print(f"Action space : {NUM_ACTIONS} thresholds (0.0 to 0.3)")
print(f"State dim    : {STATE_DIM} features")
print(f"Advertisers  : {ADV_IDS}")
print(f"Budgets      : {BUDGETS}")
print(f"Episodes     : {EPISODES} | Seeds: {SEEDS}")
print(f"V6 changes   : 3386 replaces 2821 + budget proportional to CTR")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    print(f" v6 | 5 Agents | Seed {seed}")
    print(f" 3386 replaces 2821 | Budget proportional to CTR")
    print(f"{'='*60}")

    set_seed(seed)

    env = MultiRTBEnvironmentV5(
        data_paths    = DATA_PATHS,
        budgets       = BUDGETS,
        max_steps     = MAX_STEPS,
        reserve_price = 1.0,
        state_dim     = STATE_DIM,
    )

    agents     = []
    optimizers = []
    for _ in range(NUM_AGENTS):
        model = ActorCriticNetworkV2(
            input_dim   = STATE_DIM,
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
                step_data[i] = {"state": states[i], "action": action}

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

        utils = [
            round(float(env.costs[i]) / BUDGETS[i] * 100, 1)
            for i in range(NUM_AGENTS)
        ]

        log_entry = {"seed": seed, "episode": ep}
        for i, adv in enumerate(ADV_IDS):
            log_entry[f"reward_{adv}"]      = ep_rewards[i]
            log_entry[f"clicks_{adv}"]      = int(env.clicks[i])
            log_entry[f"cost_{adv}"]        = round(float(env.costs[i]), 2)
            log_entry[f"budget_{adv}"]      = BUDGETS[i]
            log_entry[f"utilization_{adv}"] = utils[i]
        logs.append(log_entry)

        print(
            f"Seed {seed} | Ep {ep:03d} | "
            f"Clicks={env.clicks.tolist()} | "
            f"Rewards={[round(float(r), 2) for r in ep_rewards]} | "
            f"Util={utils}%"
        )

    df = pd.DataFrame(logs)
    out_file = OUT_DIR / f"actor_critic_5adv_v6_seed_{seed}.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file}")

    for i, adv in enumerate(ADV_IDS):
        model_path = MDL_DIR / f"policy_v6_{adv}_seed_{seed}.pt"
        torch.save(agents[i].state_dict(), model_path)
        print(f"✅ Saved model: {model_path.name}")

print("\n🎉 v6 TRAINING COMPLETE!")