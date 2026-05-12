"""
train_shared_mappo.py — MAPPO on TRUE shared auction
MAX_STEPS = 2000 (matches budget depletion)
"""

import sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

from simulator.multi_environment_shared_mappo import MultiRTBEnvironmentSharedMAPPO
from policy_network_mappo import MAPPOActor, CentralizedCritic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SHARED_DATA = Path(
    "D:/dataset/ipinyou-project/make-ipinyou-data/shared_auction_log.txt"
)

OUT_DIR = ROOT / "outputs" / "final_experiments_shared_mappo"
MDL_DIR = ROOT / "models" / "5adv_shared_mappo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MDL_DIR.mkdir(parents=True, exist_ok=True)

NUM_AGENTS       = 5
ADV_IDS          = ["1458", "2259", "3386", "2997", "3476"]
BUDGETS          = [18000.0, 14000.0, 10000.0, 28000.0, 8000.0]
MAX_STEPS        = 2000    # ← reduced from 10000
EPISODES         = 200
SEEDS            = [0, 1, 2, 3, 4]
GAMMA            = 0.99
LR_ACTOR         = 5e-4
LR_CRITIC        = 1e-3
ENTROPY_BETA     = 0.05
CLIP_GRAD        = 1.0
UPDATE_EVERY     = 50
PPO_EPOCHS       = 2
PPO_EPSILON      = 0.2
LOCAL_DIM        = 14
GLOBAL_DIM       = LOCAL_DIM * NUM_AGENTS
THRESHOLD_VALUES = list(np.linspace(0.0, 0.3, 51))
NUM_ACTIONS      = len(THRESHOLD_VALUES)

print(f"Advertisers  : {ADV_IDS}")
print(f"Budgets      : {BUDGETS}")
print(f"Max steps    : {MAX_STEPS} (matches budget depletion)")
print(f"Global dim   : {GLOBAL_DIM}")
print(f"Entropy beta : {ENTROPY_BETA}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(s):
    if isinstance(s, torch.Tensor):
        return s.clone().detach().float().to(DEVICE)
    return torch.tensor(
        np.array(s, dtype=np.float32),
        dtype=torch.float32, device=DEVICE
    )


def update_mappo(actors, critic, actor_optimizers,
                 critic_optimizer, batch):
    local_states_b  = [[] for _ in range(NUM_AGENTS)]
    global_states_b = []
    actions_b       = [[] for _ in range(NUM_AGENTS)]
    rewards_b       = [[] for _ in range(NUM_AGENTS)]
    old_log_probs_b = [[] for _ in range(NUM_AGENTS)]
    next_global_b   = []
    dones_b         = []

    for transition in batch:
        for i in range(NUM_AGENTS):
            local_states_b[i].append(transition["local_states"][i])
            actions_b[i].append(transition["actions"][i])
            rewards_b[i].append(transition["rewards"][i])
            old_log_probs_b[i].append(transition["log_probs"][i])
        global_states_b.append(transition["global_state"])
        next_global_b.append(transition["next_global"])
        dones_b.append(transition["done"])

    global_states_t = torch.stack([to_tensor(s) for s in global_states_b])
    next_global_t   = torch.stack([to_tensor(s) for s in next_global_b])
    dones_t         = torch.tensor(dones_b, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        values      = critic(global_states_t).squeeze()
        next_values = critic(next_global_t).squeeze()

    for i in range(NUM_AGENTS):
        local_states_t  = torch.stack([to_tensor(s) for s in local_states_b[i]])
        actions_t       = torch.stack(actions_b[i])
        rewards_t       = torch.tensor(rewards_b[i], dtype=torch.float32, device=DEVICE)
        old_log_probs_t = torch.stack(old_log_probs_b[i]).detach()

        td_targets = rewards_t + GAMMA * next_values * (1 - dones_t)
        advantages = (td_targets - values).detach()

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / \
                         (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            log_probs, entropies = actors[i].evaluate_action(
                local_states_t, actions_t)
            ratio      = torch.exp(log_probs - old_log_probs_t)
            surr1      = ratio * advantages
            surr2      = torch.clamp(ratio, 1-PPO_EPSILON, 1+PPO_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() \
                         - ENTROPY_BETA * entropies.mean()

            actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actors[i].parameters(), CLIP_GRAD)
            actor_optimizers[i].step()

    values_new     = critic(global_states_t).squeeze()
    td_targets_all = torch.zeros(len(batch), device=DEVICE)
    for i in range(NUM_AGENTS):
        rewards_t = torch.tensor(rewards_b[i], dtype=torch.float32, device=DEVICE)
        td_targets_all += rewards_t + GAMMA * next_values * (1 - dones_t)
    td_targets_all /= NUM_AGENTS

    critic_loss = nn.MSELoss()(values_new, td_targets_all.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), CLIP_GRAD)
    critic_optimizer.step()


for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f" MAPPO Shared | Seed {seed} | MAX_STEPS={MAX_STEPS}")
    print(f"{'='*60}")

    set_seed(seed)

    env = MultiRTBEnvironmentSharedMAPPO(
        shared_data_path = str(SHARED_DATA),
        budgets          = BUDGETS,
        adv_ids          = ADV_IDS,
        max_steps        = MAX_STEPS,
        reserve_price    = 1.0,
        state_dim        = LOCAL_DIM,
    )

    actors = [
        MAPPOActor(LOCAL_DIM, NUM_ACTIONS).to(DEVICE)
        for _ in range(NUM_AGENTS)
    ]
    critic           = CentralizedCritic(GLOBAL_DIM).to(DEVICE)
    actor_optimizers = [
        torch.optim.Adam(a.parameters(), lr=LR_ACTOR)
        for a in actors
    ]
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=LR_CRITIC)

    logs = []

    for ep in range(1, EPISODES + 1):
        local_states, global_state = env.reset()
        states_t       = [to_tensor(s) for s in local_states]
        global_state_t = to_tensor(global_state)

        done       = False
        ep_rewards = np.zeros(NUM_AGENTS)
        batch      = []
        step       = 0

        while not done:
            thresholds = []
            actions    = []
            log_probs  = []

            for i in range(NUM_AGENTS):
                with torch.no_grad():
                    action, log_prob, _ = actors[i].get_action(states_t[i])
                threshold = THRESHOLD_VALUES[action.item()]
                thresholds.append(threshold)
                actions.append(action)
                log_probs.append(log_prob)

            next_local, next_global, rewards, done = env.step(thresholds)

            next_global_t = to_tensor(next_global)

            batch.append({
                "local_states": [s.cpu().numpy() for s in states_t],
                "global_state": global_state_t.cpu().numpy(),
                "actions":      actions,
                "log_probs":    log_probs,
                "rewards":      rewards,
                "next_global":  next_global_t.cpu().numpy(),
                "done":         float(done),
            })

            for i in range(NUM_AGENTS):
                ep_rewards[i] += float(rewards[i])

            if not done:
                states_t       = [to_tensor(s) for s in next_local]
                global_state_t = next_global_t

            step += 1
            if step % UPDATE_EVERY == 0 or done:
                update_mappo(actors, critic,
                             actor_optimizers, critic_optimizer, batch)
                batch = []

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
    out_file = OUT_DIR / f"mappo_shared_seed_{seed}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    for i, adv in enumerate(ADV_IDS):
        torch.save(
            actors[i].state_dict(),
            MDL_DIR / f"policy_shared_mappo_actor_{adv}_seed_{seed}.pt"
        )
    torch.save(
        critic.state_dict(),
        MDL_DIR / f"policy_shared_mappo_critic_seed_{seed}.pt"
    )
    print(f"Models saved for seed {seed}")

print("\nMAPPO SHARED TRAINING COMPLETE!")