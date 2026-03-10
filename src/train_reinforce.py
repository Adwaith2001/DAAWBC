# ============================================================
# STANDARD PYTHON LIBRARIES
# ============================================================

import random
# Used to control stochasticity in environment dynamics
# and action sampling for reproducibility across runs

import csv
# Used to log training statistics (episode return, clicks, cost)
# in a structured CSV format for later analysis and plotting

import os
# General OS-level utilities (kept for extensibility)

from pathlib import Path
# Provides robust, platform-independent file path handling


# ============================================================
# NUMERICAL AND MACHINE LEARNING LIBRARIES
# ============================================================

import numpy as np
# Numerical operations such as action discretization

import torch
# Core PyTorch library for tensor operations and neural networks

from torch.distributions import Categorical
# Used to represent a discrete probability distribution over bid actions
# Required for stochastic policy sampling in REINFORCE


# ============================================================
# PROJECT-SPECIFIC MODULES
# ============================================================

from simulator.environment import RTBEnvironment
# Custom RTB simulation environment implementing:
# - impression arrival
# - auction outcome
# - click feedback
# - budget consumption

from policy_network import PolicyNetwork
# Policy (actor) neural network that outputs action probabilities


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

# Automatically select GPU (RTX 3050) if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multiple random seeds used for robustness analysis
# This helps verify that learning trends are not seed-dependent
SEEDS = [0, 1, 2, 3, 4]

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Input dataset path (sampled iPinYou log with cached pCTR)
DATA_FILE = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"

# ------------------------------------------------------------
# Action space definition
# ------------------------------------------------------------
# Discrete bid values (in monetary units)
# Action 0 corresponds to "no-bid"
BID_VALUES = [0.0] + list(np.linspace(10, 150, 20))

# Number of discrete actions available to the policy
NUM_ACTIONS = len(BID_VALUES)

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------
EPISODES = 50                 # Number of training episodes
GAMMA = 0.99                  # Discount factor for future rewards
LR = 1e-3                     # Learning rate for policy optimizer
MAX_STEPS_PER_EP = 10000      # Safety cap on episode length

# ------------------------------------------------------------
# Stabilization and regularization
# ------------------------------------------------------------
ENTROPY_BETA = 1e-3           # Encourages exploration
CLIP_GRAD_NORM = 5.0          # Prevents exploding gradients
LAMBDA_LR = 1e-4              # Learning rate for budget penalty update


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
# Ensures a directory exists to store per-seed training logs


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):
    """
    Ensures deterministic behavior across:
    - Python random
    - NumPy
    - PyTorch (CPU & GPU)

    This is critical for fair comparison across multiple runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x):
    """
    Converts a NumPy state vector into a PyTorch tensor
    and moves it to the correct device (CPU / GPU).
    """
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def discounted_returns(rewards, gamma):
    """
    Computes Monte-Carlo discounted returns for REINFORCE.

    G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...

    These returns act as unbiased estimates of Q(s_t, a_t).
    """
    G = 0.0
    out = []

    # Traverse rewards backward to compute cumulative return
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)

    out.reverse()
    return torch.tensor(out, dtype=torch.float32, device=DEVICE)


# ============================================================
# TRAINING FUNCTION (ONE SEED)
# ============================================================

def train_for_seed(seed):
    """
    Trains a REINFORCE agent for a single random seed.
    """

    print(f"\n==============================")
    print(f" Training with SEED = {seed}")
    print(f"==============================")

    # Fix randomness for this run
    set_seed(seed)

    # --------------------------------------------------------
    # Initialize policy network
    # --------------------------------------------------------
    policy = PolicyNetwork(input_dim=4, num_actions=NUM_ACTIONS).to(DEVICE)

    # Adam optimizer is commonly used for policy gradient methods
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # --------------------------------------------------------
    # CSV logging setup
    # --------------------------------------------------------
    log_file = LOG_DIR / f"reinforce_training_seed_{seed}.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "return", "clicks", "cost", "budget_left", "lambda"]
        )

    # ========================================================
    # EPISODE LOOP
    # ========================================================
    for ep in range(1, EPISODES + 1):

        # ----------------------------------------------------
        # Initialize environment for one episode
        # ----------------------------------------------------
        env = RTBEnvironment(
            data_path=str(DATA_FILE),
            budget=300.0,          # Fixed campaign budget
            max_steps=MAX_STEPS_PER_EP,
            lambda_init=0.013,     # Initial budget penalty
        )

        # Reset environment and get initial state
        state = to_tensor(env.reset())

        # Storage for REINFORCE update
        log_probs = []    # log π(a_t | s_t)
        rewards = []      # immediate rewards
        entropies = []    # policy entropy (exploration metric)

        done = False
        t = 0

        # ====================================================
        # STEP LOOP (IMPRESSION-LEVEL DECISIONS)
        # ====================================================
        while not done:

            # Forward pass: policy outputs unnormalized logits
            logits = policy(state)

            # Convert logits to categorical distribution
            dist = Categorical(logits=logits.squeeze(0))

            # Sample action according to policy πθ
            action = dist.sample()

            # Store log-probability for policy gradient
            log_probs.append(dist.log_prob(action))

            # Store entropy for exploration regularization
            entropies.append(dist.entropy())

            # Convert discrete action → actual bid value
            bid = BID_VALUES[action.item()]

            # Environment transition
            next_state, reward, done = env.step(bid)

            rewards.append(float(reward))
            state = to_tensor(next_state)

            t += 1
            if t >= MAX_STEPS_PER_EP:
                break

        # ====================================================
        # REINFORCE POLICY UPDATE
        # ====================================================

        # Compute discounted Monte-Carlo returns
        returns = discounted_returns(rewards, GAMMA)

        # Normalize returns to reduce variance
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ----------------------------------------------------
        # Policy gradient loss
        # L = - E[ log π(a_t|s_t) * G_t ]
        # ----------------------------------------------------
        loss = (
            -(torch.stack(log_probs) * returns).sum()
            - ENTROPY_BETA * torch.stack(entropies).sum()
        )

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP_GRAD_NORM)

        optimizer.step()

        # ----------------------------------------------------
        # Episode-level statistics
        # ----------------------------------------------------
        ep_return = sum(rewards)

        # Dual update for budget constraint
        overspend = env.cost - env.budget
        env.lambda_penalty += LAMBDA_LR * overspend

        print(
            f"Seed {seed} | Ep {ep:03d} | "
            f"return={ep_return:.3f} | "
            f"clicks={env.total_clicks} | "
            f"cost={env.cost:.2f}"
        )

        # Log episode results
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

    # --------------------------------------------------------
    # Save trained policy for this seed
    # --------------------------------------------------------
    model_path = ROOT / f"policy_reinforce_seed_{seed}.pt"
    torch.save(policy.state_dict(), model_path)
    print(f"✅ Saved policy: {model_path.name}")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(f"Using device: {DEVICE}")
    print("Multi-seed REINFORCE training started")

    for seed in SEEDS:
        train_for_seed(seed)

    print("\n✅ All seeds finished")
