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
OUT_DIR = ROOT / "outputs" / "final_experiments_5agents_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATHS = {
    0: ROOT / "data" / "ipinyou" / "1458" / "final_sample_log_with_pctr.txt",
    1: ROOT / "data" / "ipinyou" / "2259" / "final_sample_log_with_pctr.txt",
    2: ROOT / "data" / "ipinyou" / "2821" / "final_sample_log_with_pctr.txt",
    3: ROOT / "data" / "ipinyou" / "2997" / "final_sample_log_with_pctr.txt",
    4: ROOT / "data" / "ipinyou" / "3358" / "final_sample_log_with_pctr.txt",
}

# ======================================================
# CONFIG
# ======================================================
NUM_AGENTS = 5
BUDGETS = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]

MAX_STEPS = 5000
EPISODES = 200          # ✅ increased from 100

GAMMA = 0.99
LR = 3e-4               # ✅ reduced from 1e-3
ENTROPY_BETA = 5e-3     # ✅ increased from 1e-3
CLIP_GRAD_NORM = 1.0    # ✅ tightened from 5.0

BID_VALUES = [0.0] + list(np.linspace(0.1, 5.0, 20))
NUM_ACTIONS = len(BID_VALUES)

RESERVE_PRICE = 0.1
LAMBDA_INIT = 0.005

ADV_IDS = ["1458", "2259", "2821", "2997", "3358"]

SEEDS = [0, 1, 2, 3, 4]

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
    print(f" FINAL RUN v2 | 5 Agents | Seed {seed}")
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
            num_actions=NUM_ACTIONS,
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

            # ✅ Reward normalization for stability
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            if rewards_tensor.std() > 1e-8:
                rewards = (
                    (rewards_tensor - rewards_tensor.mean()) /
                    (rewards_tensor.std() + 1e-8)
                ).numpy()

            if not done:
                next_states = [
                    torch.tensor(s, dtype=torch.float32, device=DEVICE)
                    for s in next_states
                ]

            # Actor-Critic update
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
                entropy = -ENTROPY_BETA * torch.distributions.Categorical(
                    logits=agents[i](states[i])[0].squeeze(0)
                ).entropy()

                loss = actor_loss + critic_loss + entropy

                optimizers[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agents[i].parameters(), CLIP_GRAD_NORM  # ✅ tighter clipping
                )
                optimizers[i].step()

                ep_rewards[i] += float(rewards[i])

            if not done:
                states = next_states

        # Episode log
        log_entry = {"seed": seed, "episode": ep}
        for i, adv in enumerate(ADV_IDS):
            log_entry[f"reward_{adv}"] = ep_rewards[i]
            log_entry[f"clicks_{adv}"] = env.clicks[i]
            log_entry[f"cost_{adv}"] = env.costs[i]

        logs.append(log_entry)

        print(
            f"Seed {seed} | Ep {ep:03d} | "
            f"Clicks={env.clicks.tolist()} | "
            f"Costs={[round(c, 2) for c in env.costs.tolist()]}"
        )

    # Save per-seed CSV
    df = pd.DataFrame(logs)
    out_file = OUT_DIR / f"actor_critic_5adv_v2_seed_{seed}.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file}")

    # Save per-agent model weights
    for i, adv in enumerate(ADV_IDS):
        model_path = ROOT / f"policy_5adv_v2_{adv}_seed_{seed}.pt"
        torch.save(agents[i].state_dict(), model_path)
        print(f"✅ Saved model: {model_path.name}")

print("\n🎉 FINAL 5-AGENT v2 MULTI-SEED EXPERIMENT COMPLETED")