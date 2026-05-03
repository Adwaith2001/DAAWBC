"""
test_environment_v2.py
Tests for RTBEnvironmentV2 and MultiRTBEnvironmentV2

Run from src2/:
    python test_environment_v2.py
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

from simulator.environment_v2 import RTBEnvironmentV2
from simulator.multi_environment_v2 import MultiRTBEnvironmentV2

# ======================================================
# PATHS
# ======================================================
DATA_V2 = ROOT / "data" / "ipinyou_v2"

DATA_PATHS = {
    0: DATA_V2 / "1458" / "final_sample_log_with_pctr.txt",
    1: DATA_V2 / "2259" / "final_sample_log_with_pctr.txt",
    2: DATA_V2 / "2821" / "final_sample_log_with_pctr.txt",
    3: DATA_V2 / "2997" / "final_sample_log_with_pctr.txt",
    4: DATA_V2 / "3358" / "final_sample_log_with_pctr.txt",
}

BUDGETS       = [20000.0] * 5
CLICK_VALUES  = [150.0, 200.0, 200.0, 100.0, 200.0]

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


# ======================================================
# TEST 1: SINGLE AGENT ENVIRONMENT
# ======================================================
section("TEST 1: RTBEnvironmentV2 — Basic Reset")

env   = RTBEnvironmentV2(
    data_path   = str(DATA_PATHS[0]),
    budget      = 20000.0,
    max_steps   = 10000,
    click_value = 150.0,
)
state = env.reset()

print(f"State shape     : {state.shape}  {PASS if state.shape == (10,) else FAIL}")
print(f"State dtype     : {state.dtype}  {PASS if state.dtype == np.float32 else FAIL}")
print(f"Budget ratio    : {state[0]:.2f}  {PASS if state[0] == 1.0 else FAIL}")
print(f"Time ratio      : {state[1]:.2f}  {PASS if state[1] == 1.0 else FAIL}")
print(f"No NaN in state : {PASS if not np.isnan(state).any() else FAIL}")
print(f"State values    : {state}")


# ======================================================
# TEST 2: SINGLE AGENT STEP
# ======================================================
section("TEST 2: RTBEnvironmentV2 — Step")

state = env.reset()
next_state, reward, done = env.step(50.0)

print(f"Next state shape : {next_state.shape}  {PASS if next_state.shape == (10,) else FAIL}")
print(f"Reward type      : {type(reward)}  {PASS if isinstance(reward, float) else FAIL}")
print(f"Done is bool     : {PASS if isinstance(done, bool) else FAIL}")
print(f"Done is False    : {PASS if not done else FAIL}")
print(f"No NaN in state  : {PASS if not np.isnan(next_state).any() else FAIL}")

# Step with zero bid (should not win)
state = env.reset()
_, reward_zero, _ = env.step(0.0)
print(f"Zero bid reward  : {reward_zero}  {PASS if reward_zero == 0.0 else FAIL}")


# ======================================================
# TEST 3: BUDGET CONSTRAINT
# ======================================================
section("TEST 3: RTBEnvironmentV2 — Budget Exhaustion")

env_small = RTBEnvironmentV2(
    data_path   = str(DATA_PATHS[0]),
    budget      = 100.0,
    max_steps   = 100000,
    click_value = 150.0,
)

state = env_small.reset()
done  = False
steps = 0

while not done:
    _, _, done = env_small.step(300.0)
    steps += 1

print(f"Episode ended    : {PASS if done else FAIL}")
print(f"Remaining budget : {env_small.remaining_budget:.2f}  {PASS if env_small.remaining_budget <= 0 else '⚠️  Budget not exhausted'}")
print(f"Total steps      : {steps}")
print(f"Total clicks     : {env_small.total_clicks}")
print(f"Total cost       : {env_small.cost:.2f}")


# ======================================================
# TEST 4: FULL EPISODE
# ======================================================
section("TEST 4: RTBEnvironmentV2 — Full Episode")

env   = RTBEnvironmentV2(
    data_path   = str(DATA_PATHS[0]),
    budget      = 20000.0,
    max_steps   = 1000,
    click_value = 150.0,
)
state = env.reset()
done  = False
total_reward = 0.0

while not done:
    bid = np.random.uniform(0, 300)
    state, reward, done = env.step(bid)
    total_reward += reward

print(f"Episode completed : {PASS if done else FAIL}")
print(f"Total reward      : {total_reward:.2f}")
print(f"Total clicks      : {env.total_clicks}")
print(f"Total cost        : {env.cost:.2f}")
print(f"Budget remaining  : {env.remaining_budget:.2f}")
print(f"Win rate          : {env.win_rate:.4f}")


# ======================================================
# TEST 5: MULTI AGENT RESET
# ======================================================
section("TEST 5: MultiRTBEnvironmentV2 — Basic Reset")

multi_env = MultiRTBEnvironmentV2(
    data_paths   = DATA_PATHS,
    budgets      = BUDGETS,
    click_values = CLICK_VALUES,
    max_steps    = 10000,
    reserve_price= 1.0,
)

states = multi_env.reset()

print(f"Num states       : {len(states)}  {PASS if len(states) == 5 else FAIL}")
print(f"Each state shape : {states[0].shape}  {PASS if states[0].shape == (10,) else FAIL}")
print(f"No NaN in states : {PASS if all(not np.isnan(s).any() for s in states) else FAIL}")
for i, s in enumerate(states):
    print(f"  Agent {i} budget ratio: {s[0]:.2f}  {PASS if s[0] == 1.0 else FAIL}")


# ======================================================
# TEST 6: MULTI AGENT STEP
# ======================================================
section("TEST 6: MultiRTBEnvironmentV2 — Step")

states = multi_env.reset()
bids   = [50.0, 80.0, 30.0, 60.0, 100.0]
next_states, rewards, done = multi_env.step(bids)

print(f"Rewards shape    : {rewards.shape}  {PASS if rewards.shape == (5,) else FAIL}")
print(f"Done is bool     : {PASS if isinstance(done, bool) else FAIL}")
print(f"Done is False    : {PASS if not done else FAIL}")
print(f"Rewards          : {rewards}")
print(f"Only 1 winner    : {PASS if (rewards != 0).sum() <= 1 else FAIL}  (second-price auction)")


# ======================================================
# TEST 7: MULTI AGENT FULL EPISODE
# ======================================================
section("TEST 7: MultiRTBEnvironmentV2 — Full Episode")

states = multi_env.reset()
done   = False
ep_rewards = np.zeros(5)
steps  = 0

while not done:
    bids = [np.random.uniform(1, 300) for _ in range(5)]
    next_states, rewards, done = multi_env.step(bids)
    ep_rewards += rewards
    if not done:
        states = next_states
    steps += 1

print(f"Episode completed : {PASS if done else FAIL}")
print(f"Total steps       : {steps}")
print(f"Clicks per agent  : {multi_env.clicks.tolist()}")
print(f"Costs per agent   : {[round(c, 2) for c in multi_env.costs.tolist()]}")
print(f"Rewards per agent : {[round(r, 2) for r in ep_rewards.tolist()]}")
print(f"Win rates         : {[round(w, 4) for w in multi_env.win_rates.tolist()]}")


# ======================================================
# TEST 8: RESERVE PRICE
# ======================================================
section("TEST 8: MultiRTBEnvironmentV2 — Reserve Price")

states = multi_env.reset()
# All agents bid below reserve price
bids = [0.0] * 5
next_states, rewards, done = multi_env.step(bids)

print(f"All zero rewards : {PASS if (rewards == 0).all() else FAIL}  (all below reserve)")
print(f"Costs unchanged  : {PASS if (multi_env.costs == 0).all() else FAIL}")


# ======================================================
# SUMMARY
# ======================================================
print(f"\n{'='*60}")
print(" ALL TESTS COMPLETED")
print(f"{'='*60}")
print("If all ✅ PASS above, environments are working correctly.")
print("Proceed to policy_network_v2 tests and then train_v3.py")