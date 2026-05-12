"""
train_shared.py — AC on TRUE shared auction
MAX_STEPS = 2000 (matches budget depletion)
"""

import sys, random
import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

from simulator.multi_environment_shared import MultiRTBEnvironmentShared
from policy_network_v2 import ActorCriticNetworkV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SHARED_DATA = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/shared_auction_log.txt"
)

OUT_DIR = ROOT / "outputs" / "final_experiments_shared"
MDL_DIR = ROOT / "models" / "5adv_shared"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MDL_DIR.mkdir(parents=True, exist_ok=True)

NUM_AGENTS       = 5
ADV_IDS          = ["1458", "2259", "3386", "2997", "3476"]
BUDGETS          = [18000.0, 14000.0, 2000.0, 20000.0, 8000.0]
MAX_STEPS        = 2000    # ← reduced from 10000
EPISODES         = 200
SEEDS            = [0, 1, 2, 3, 4]
GAMMA            = 0.99
LR               = 5e-4
ENTROPY_BETA     = 1e-3
CLIP_GRAD        = 1.0
UPDATE_EVERY     = 50
THRESHOLD_VALUES = list(np.linspace(0.0, 0.3, 51))
NUM_ACTIONS      = len(THRESHOLD_VALUES)
STATE_DIM        = 14

print(f"Advertisers  : {ADV_IDS}")
print(f"Budgets      : {BUDGETS}")
print(f"Max steps    : {MAX_STEPS} (matches budget depletion)")
print(f"Episodes     : {EPISODES} | Seeds: {SEEDS}")


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
        rewards_b     = torch.tensor(
            [t["reward"] for t in batch[i]],
            dtype=torch.float32, device=DEVICE)
        next_states_b = torch.stack([t["next_state"] for t in batch[i]])
        dones_b       = torch.tensor(
            [t["done"] for t in batch[i]],
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

        advantages   = td_targets - values_b
        actor_loss   = -(log_probs_b * advantages.detach()).mean()
        critic_loss  = advantages.pow(2).mean()
        entropy_loss = -ENTROPY_BETA * entropies_b.mean()
        loss         = actor_loss + critic_loss + entropy_loss

        optimizers[i].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agents[i].parameters(), CLIP_GRAD)
        optimizers[i].step()


for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f" Shared AC | Seed {seed} | MAX_STEPS={MAX_STEPS}")
    print(f"{'='*60}")

    set_seed(seed)

    env = MultiRTBEnvironmentShared(
        shared_data_path = str(SHARED_DATA),
        budgets          = BUDGETS,
        adv_ids          = ADV_IDS,
        max_steps        = MAX_STEPS,
        reserve_price    = 1.0,
        state_dim        = STATE_DIM,
    )

    agents     = []
    optimizers = []
    for _ in range(NUM_AGENTS):
        model = ActorCriticNetworkV2(
            input_dim=STATE_DIM, num_actions=NUM_ACTIONS
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        agents.append(model)
        optimizers.append(optimizer)

    logs = []

    for ep in range(1, EPISODES + 1):
        states   = env.reset()
        states_t = [
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
                    logits, value = agents[i](states_t[i])
                dist      = Categorical(logits=logits.squeeze(0))
                action    = dist.sample()
                threshold = THRESHOLD_VALUES[action.item()]
                thresholds.append(threshold)
                step_data[i] = {"state": states_t[i], "action": action}

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
                states_t = next_states_t

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
            f"Util={utils}%"
        )

    df       = pd.DataFrame(logs)
    out_file = OUT_DIR / f"actor_critic_shared_seed_{seed}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    for i, adv in enumerate(ADV_IDS):
        torch.save(
            agents[i].state_dict(),
            MDL_DIR / f"policy_shared_{adv}_seed_{seed}.pt"
        )
    print(f"Models saved for seed {seed}")

print("\nSHARED AC TRAINING COMPLETE!")