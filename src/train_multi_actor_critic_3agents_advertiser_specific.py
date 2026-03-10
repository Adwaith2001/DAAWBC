import random
import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path

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

DATA_PATHS = {
    0: ROOT / "data" / "ipinyou" / "1458" / "final_sample_log_with_pctr.txt",
    1: ROOT / "data" / "ipinyou" / "2259" / "final_sample_log_with_pctr.txt",
    2: ROOT / "data" / "ipinyou" / "2821" / "final_sample_log_with_pctr.txt",
}

# ======================================================
# CONFIG
# ======================================================
NUM_AGENTS = 3
BUDGETS = [1000.0, 1000.0, 1000.0]

MAX_STEPS = 5000
EPISODES = 10

GAMMA = 0.99
LR = 1e-3
ENTROPY_BETA = 1e-3

# Calibrated bid grid
BID_VALUES = [0.0] + list(np.linspace(0.1, 5.0, 20))
NUM_ACTIONS = len(BID_VALUES)

# ======================================================
# SEED
# ======================================================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# ======================================================
# ENVIRONMENT
# ======================================================
env = MultiRTBEnvironment(
    data_paths=DATA_PATHS,
    budgets=BUDGETS,
    max_steps=MAX_STEPS,
    lambda_init=0.005,
    reserve_price=0.1,
)

# ======================================================
# AGENTS
# ======================================================
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

# ======================================================
# TRAINING LOOP
# ======================================================
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

        # --------------------------------------------------
        # ACTION SELECTION
        # --------------------------------------------------
        for i in range(NUM_AGENTS):
            logits, value = agents[i](states[i])
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()

            bids.append(BID_VALUES[action.item()])
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())

        # --------------------------------------------------
        # ENV STEP
        # --------------------------------------------------
        next_states, rewards, done = env.step(bids)

        if not done:
            next_states = [
                torch.tensor(s, dtype=torch.float32, device=DEVICE)
                for s in next_states
            ]

        # --------------------------------------------------
        # UPDATE (ACTOR–CRITIC)
        # --------------------------------------------------
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

    # --------------------------------------------------
    # EPISODE SUMMARY
    # --------------------------------------------------
    reward_str = " | ".join(
        [f"Agent{i} Reward={ep_rewards[i]:.2f}" for i in range(NUM_AGENTS)]
    )

    print(
        f"Episode {ep:02d} | "
        f"{reward_str} | "
        f"Costs={env.costs} | "
        f"Clicks={env.clicks}"
    )

print("\n✅ Advertiser-specific 3-agent training completed successfully")
