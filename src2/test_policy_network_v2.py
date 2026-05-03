"""
test_policy_network_v2.py
Tests for ActorCriticNetworkV2

Run from src2/:
    python test_policy_network_v2.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src2"))

print("Importing policy_network_v2...")
from policy_network_v2 import ActorCriticNetworkV2
print("Import successful!")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM   = 10
NUM_ACTIONS = 51

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


# ======================================================
# TEST 1: INITIALIZATION
# ======================================================
section("TEST 1: Network Initialization")

model = ActorCriticNetworkV2(
    input_dim   = INPUT_DIM,
    num_actions = NUM_ACTIONS,
).to(DEVICE)

print(f"Device           : {DEVICE}")
print(f"Model created    : {PASS}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total params     : {total_params:,}  {PASS if total_params > 0 else FAIL}")


# ======================================================
# TEST 2: FORWARD PASS — SINGLE STATE
# ======================================================
section("TEST 2: Forward Pass — Single State")

model.eval()
state = torch.randn(INPUT_DIM).to(DEVICE)

with torch.no_grad():
    logits, value = model(state)

print(f"Input shape      : {state.shape}")
print(f"Logits shape     : {logits.shape}  {PASS if logits.shape == (1, NUM_ACTIONS) else FAIL}")
print(f"Value shape      : {value.shape}   {PASS if value.shape == (1, 1) else FAIL}")
print(f"No NaN in logits : {PASS if not torch.isnan(logits).any() else FAIL}")
print(f"No NaN in value  : {PASS if not torch.isnan(value).any() else FAIL}")


# ======================================================
# TEST 3: FORWARD PASS — BATCH
# ======================================================
section("TEST 3: Forward Pass — Batch of States")

model.train()
batch = torch.randn(32, INPUT_DIM).to(DEVICE)
logits, value = model(batch)

print(f"Batch input shape : {batch.shape}")
print(f"Logits shape      : {logits.shape}  {PASS if logits.shape == (32, NUM_ACTIONS) else FAIL}")
print(f"Value shape       : {value.shape}   {PASS if value.shape == (32, 1) else FAIL}")


# ======================================================
# TEST 4: SINGLE STATE IN TRAIN MODE (LayerNorm fix)
# ======================================================
section("TEST 4: Single State in Train Mode")

model.train()
state = torch.randn(INPUT_DIM).to(DEVICE)

try:
    logits, value = model(state)
    print(f"Single state train mode : {PASS}")
    print(f"No NaN in logits        : {PASS if not torch.isnan(logits).any() else FAIL}")
except Exception as e:
    print(f"Single state train mode : {FAIL}")
    print(f"Error: {e}")


# ======================================================
# TEST 5: ACTION SAMPLING
# ======================================================
section("TEST 5: Action Sampling")

model.eval()
state = torch.randn(INPUT_DIM).to(DEVICE)

with torch.no_grad():
    logits, value = model(state)
    dist     = Categorical(logits=logits.squeeze(0))
    action   = dist.sample()
    log_prob = dist.log_prob(action)
    entropy  = dist.entropy()

print(f"Action           : {action.item()}  {PASS if 0 <= action.item() < NUM_ACTIONS else FAIL}")
print(f"Log prob shape   : {log_prob.shape}  {PASS if log_prob.shape == torch.Size([]) else FAIL}")
print(f"Entropy          : {entropy.item():.4f}  {PASS if entropy.item() > 0 else FAIL}")
print(f"No NaN log prob  : {PASS if not torch.isnan(log_prob).any() else FAIL}")


# ======================================================
# TEST 6: BACKWARD PASS
# ======================================================
section("TEST 6: Backward Pass — Gradient Flow")

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
state     = torch.randn(INPUT_DIM).to(DEVICE)
logits, value = model(state)
dist      = Categorical(logits=logits.squeeze(0))
action    = dist.sample()

loss = -dist.log_prob(action) + value.squeeze().pow(2)
optimizer.zero_grad()
loss.backward()

grad_ok = all(
    p.grad is not None
    for p in model.parameters()
    if p.requires_grad
)
print(f"Gradients exist  : {PASS if grad_ok else FAIL}")
optimizer.step()
print(f"Optimizer step   : {PASS}")


# ======================================================
# TEST 7: GRADIENT CLIPPING
# ======================================================
section("TEST 7: Gradient Clipping")

state     = torch.randn(INPUT_DIM).to(DEVICE)
logits, value = model(state)
dist      = Categorical(logits=logits.squeeze(0))
action    = dist.sample()

loss = -dist.log_prob(action) * 1000
optimizer.zero_grad()
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

after_clip = max(
    p.grad.abs().max().item()
    for p in model.parameters()
    if p.grad is not None
)
print(f"Max grad after clip : {after_clip:.4f}  {PASS if after_clip <= 1.0 + 1e-4 else FAIL}")


# ======================================================
# TEST 8: 5 INDEPENDENT AGENTS
# ======================================================
section("TEST 8: 5 Independent Agent Networks")

agents = [
    ActorCriticNetworkV2(INPUT_DIM, NUM_ACTIONS).to(DEVICE)
    for _ in range(5)
]

all_ok = True
for i, agent in enumerate(agents):
    agent.eval()
    state = torch.randn(INPUT_DIM).to(DEVICE)
    with torch.no_grad():
        logits, value = agent(state)
        dist   = Categorical(logits=logits.squeeze(0))
        action = dist.sample()
    ok = 0 <= action.item() < NUM_ACTIONS
    print(f"  Agent {i} action: {action.item():3d}  {PASS if ok else FAIL}")
    all_ok = all_ok and ok

print(f"All agents OK    : {PASS if all_ok else FAIL}")


# ======================================================
# SUMMARY
# ======================================================
print(f"\n{'='*60}")
print(" ALL POLICY NETWORK TESTS COMPLETED")
print(f"{'='*60}")
print("If all ✅ PASS above, policy network is working correctly.")
print("Proceed to train_v3.py")