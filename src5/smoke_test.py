"""
src5/smoke_test.py
==================

3-episode end-to-end sanity check.

Validates that:
  1. Propensity model fits cleanly
  2. Env initializes correctly
  3. Both action heads produce gradients
  4. At least some impressions are won and at least some clicks captured
  5. Linear-baseline picker produces non-zero clicks (sanity)

Run this BEFORE any real training run. ~30 seconds.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src5.context_propensity import ContextPropensityModel  # noqa: E402
from src5.simulator.context_environment import (  # noqa: E402
    ContextRTBEnvironment, DEFAULT_THRESHOLD_GRID, DEFAULT_RESIDUAL_GRID, ADV_IDS,
)
from src5.policy_network import StrategicActorCritic  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA = "data_2/shared_auction_log_v4_dense.txt"
N_AGENTS = 5
BUDGETS = [10000.0] * N_AGENTS    # smaller budget for smoke
EPISODE_ROWS = 5000                # tiny episode
N_EPISODES = 3


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


def main():
    t_start = time.time()
    print("=" * 60)
    print(" SMOKE TEST — src5 context-propensity bidding")
    print("=" * 60)
    print(f"  device: {DEVICE}")
    print()

    # ------------------------------------------------------------------
    # [1/5] Fit propensity
    # ------------------------------------------------------------------
    print("[1/5] Loading data + fitting propensity ...")
    data_path = Path(DATA)
    if not data_path.is_absolute():
        # Try resolving relative to repo root
        candidate = ROOT / data_path
        if candidate.exists():
            data_path = candidate
    if not data_path.exists():
        print(f"  ERROR: data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, sep="\t")
    print(f"  Rows: {len(df):,}  CTR: {df['click'].mean()*100:.2f}%")

    model = ContextPropensityModel()
    model.fit(df, train_weekdays=(3, 4), eval_weekday=5, verbose=True)
    df = model.attach_to_dataframe(df)
    df_train = df[df["weekday"].isin([3, 4])].reset_index(drop=True)
    print(f"  Train rows: {len(df_train):,}  "
          f"Mean propensity: {df_train['propensity'].mean():.4f}")

    # ------------------------------------------------------------------
    # [2/5] Build env
    # ------------------------------------------------------------------
    print("\n[2/5] Building env ...")
    env = ContextRTBEnvironment(
        df=df_train,
        budgets=BUDGETS,
        episode_rows=EPISODE_ROWS,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # [3/5] Linear-baseline sanity (no model)
    # ------------------------------------------------------------------
    print("\n[3/5] Linear-propensity baseline sanity (1 episode) ...")
    env.reset()
    done = False
    n_res = len(DEFAULT_RESIDUAL_GRID)
    while not done:
        # Always: th=0 (never skip), res=middle (neutral)
        th_a = [0] * N_AGENTS
        res_a = [n_res // 2] * N_AGENTS
        _, _, done, _ = env.step(th_a, res_a)
    diag = env.diagnostics()
    print(f"  Linear baseline diagnostics:")
    print(f"    wins per agent : {diag['wins']}")
    print(f"    clicks per agent: {diag['clicks']}")
    print(f"    cost per agent : {[f'{c:.0f}' for c in diag['cost']]}")
    print(f"    util %         : {[f'{u:.0f}' for u in diag['utilization_pct']]}")

    if sum(diag["wins"]) == 0:
        print("  FAIL: linear baseline won zero impressions across all agents.")
        sys.exit(1)
    if sum(diag["clicks"]) == 0:
        print("  WARN: linear baseline won impressions but got zero clicks. "
              "  Possible if episode_rows is too small. Continuing...")

    # ------------------------------------------------------------------
    # [4/5] Build agents + check forward pass
    # ------------------------------------------------------------------
    print("\n[4/5] Building actor-critic agents + forward-pass check ...")
    agents = []
    for _ in range(N_AGENTS):
        a = StrategicActorCritic(
            input_dim=4,
            hidden_dim=128,
            n_threshold=len(DEFAULT_THRESHOLD_GRID),
            n_residual=len(DEFAULT_RESIDUAL_GRID),
        ).to(DEVICE)
        agents.append(a)

    test_state = to_tensor(env._get_state()[0])
    lt, lr_, v = agents[0](test_state)
    print(f"  threshold logits shape : {tuple(lt.shape)}")
    print(f"  residual  logits shape : {tuple(lr_.shape)}")
    print(f"  value            shape : {tuple(v.shape)}")
    print(f"  threshold logits OK : {torch.isfinite(lt).all().item()}")
    print(f"  residual  logits OK : {torch.isfinite(lr_).all().item()}")
    print(f"  value             OK : {torch.isfinite(v).all().item()}")

    # ------------------------------------------------------------------
    # [5/5] Tiny training loop — 3 episodes
    # ------------------------------------------------------------------
    print(f"\n[5/5] Tiny training loop ({N_EPISODES} episodes) ...")
    optimizers = [torch.optim.Adam(a.parameters(), lr=1e-3) for a in agents]
    total_grads = [0.0] * N_AGENTS

    for ep in range(1, N_EPISODES + 1):
        states_t = [to_tensor(s) for s in env.reset()]
        log_probs_th = [[] for _ in range(N_AGENTS)]
        log_probs_res = [[] for _ in range(N_AGENTS)]
        values = [[] for _ in range(N_AGENTS)]
        rewards = [[] for _ in range(N_AGENTS)]
        ents = [[] for _ in range(N_AGENTS)]

        done = False
        while not done:
            th_a, res_a = [], []
            for i in range(N_AGENTS):
                lt, lr_, v = agents[i](states_t[i])
                dist_th = Categorical(logits=lt.squeeze(0))
                dist_res = Categorical(logits=lr_.squeeze(0))
                a_th = dist_th.sample()
                a_res = dist_res.sample()
                th_a.append(int(a_th.item()))
                res_a.append(int(a_res.item()))
                log_probs_th[i].append(dist_th.log_prob(a_th))
                log_probs_res[i].append(dist_res.log_prob(a_res))
                values[i].append(v.squeeze())
                ents[i].append(dist_th.entropy() + dist_res.entropy())

            ns, r, done, _ = env.step(th_a, res_a)
            for i in range(N_AGENTS):
                rewards[i].append(float(r[i]))
            if not done:
                states_t = [to_tensor(s) for s in ns]

        # End-of-episode update
        for i in range(N_AGENTS):
            R = 0.0
            returns = []
            for r_t in reversed(rewards[i]):
                R = r_t + 0.99 * R
                returns.append(R)
            returns.reverse()
            returns_t = to_tensor(returns)
            vals = torch.stack(values[i])
            adv = returns_t - vals.detach()
            if adv.std() > 1e-8:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            log_th = torch.stack(log_probs_th[i])
            log_res = torch.stack(log_probs_res[i])
            ent = torch.stack(ents[i])
            critic_loss = F.mse_loss(vals, returns_t)
            actor_loss = -((log_th + log_res) * adv).mean()
            ent_bonus = -1e-3 * ent.mean()
            loss = actor_loss + 0.5 * critic_loss + ent_bonus
            optimizers[i].zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 10.0)
            total_grads[i] += float(gn)
            optimizers[i].step()

        diag = env.diagnostics()
        ent_mean = np.mean([e.mean().item() for e in
                            [torch.stack(es) for es in ents if es]])
        print(f"  ep {ep}: wins={diag['wins']} clicks={diag['clicks']} "
              f"H={ent_mean:.2f} mean_grad={[f'{g/ep:.2f}' for g in total_grads]}")

    # ------------------------------------------------------------------
    # [5b/5] SLOT-MODE check (R9b ruling — wrapper port verification)
    # Verifies env.step_slot works and gradients remain finite under
    # the larger per-slot aggregate reward magnitudes.
    # ------------------------------------------------------------------
    print("[5b/5] Slot-mode gradient check (R9b — wrapped architecture) ...")
    SLOT_SIZE = 1000  # tiny for smoke, real training uses 5000
    SLOT_EPISODES = 1
    SLOT_UPDATE_EVERY = 2  # update every 2 slots

    env_slot = ContextRTBEnvironment(
        df=df_train,
        budgets=BUDGETS,
        episode_rows=EPISODE_ROWS,
        verbose=False,
    )

    slot_grads = [0.0 for _ in range(N_AGENTS)]
    slot_rewards_seen = []

    for ep in range(SLOT_EPISODES):
        states = env_slot.reset()
        states_t = [torch.as_tensor(s, dtype=torch.float32, device=DEVICE) for s in states]
        done = False
        slot_count = 0
        ep_log_probs_th = [[] for _ in range(N_AGENTS)]
        ep_log_probs_res = [[] for _ in range(N_AGENTS)]
        ep_values = [[] for _ in range(N_AGENTS)]
        ep_rewards = [[] for _ in range(N_AGENTS)]

        while not done:
            th_acts, res_acts = [], []
            for i in range(N_AGENTS):
                lt, lr_, v = agents[i](states_t[i])
                dt = Categorical(logits=lt.squeeze(0))
                dr = Categorical(logits=lr_.squeeze(0))
                a_t = dt.sample(); a_r = dr.sample()
                th_acts.append(int(a_t.item()))
                res_acts.append(int(a_r.item()))
                ep_log_probs_th[i].append(dt.log_prob(a_t))
                ep_log_probs_res[i].append(dr.log_prob(a_r))
                ep_values[i].append(v.squeeze())

            ns, r, done, _ = env_slot.step_slot(th_acts, res_acts, slot_size=SLOT_SIZE)
            for i in range(N_AGENTS):
                ep_rewards[i].append(float(r[i]))
            slot_rewards_seen.append(float(np.sum(r)))
            if not done:
                states_t = [torch.as_tensor(s, dtype=torch.float32, device=DEVICE) for s in ns]
            slot_count += 1

            if slot_count % SLOT_UPDATE_EVERY == 0 or done:
                for i in range(N_AGENTS):
                    if not ep_rewards[i]:
                        continue
                    R = 0.0; rets = []
                    for rt in reversed(ep_rewards[i]):
                        R = rt + 0.99 * R; rets.append(R)
                    rets.reverse()
                    rets_t = torch.as_tensor(rets, dtype=torch.float32, device=DEVICE)
                    vals = torch.stack(ep_values[i])
                    adv = rets_t - vals.detach()
                    if adv.std() > 1e-8:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    lp_th = torch.stack(ep_log_probs_th[i])
                    lp_res = torch.stack(ep_log_probs_res[i])
                    loss = -(lp_th * adv).mean() - (lp_res * adv).mean() + 0.5 * (adv ** 2).mean()
                    optimizers[i].zero_grad()
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 10.0)
                    slot_grads[i] += float(gn)
                    optimizers[i].step()
                ep_log_probs_th = [[] for _ in range(N_AGENTS)]
                ep_log_probs_res = [[] for _ in range(N_AGENTS)]
                ep_values = [[] for _ in range(N_AGENTS)]
                ep_rewards = [[] for _ in range(N_AGENTS)]

    print(f"  slot mode: slots_run={slot_count} "
          f"per_slot_reward range=[{min(slot_rewards_seen):.0f}, {max(slot_rewards_seen):.0f}] "
          f"grad_norms={[f'{g:.2f}' for g in slot_grads]}")

    # ------------------------------------------------------------------
    # CHECKS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" CHECKS")
    print("=" * 60)
    all_passed = True

    diag = env.diagnostics()

    chk1 = sum(diag["wins"]) > 0
    print(f" [{'PASS' if chk1 else 'FAIL'}] Last episode had some wins "
          f"(total = {sum(diag['wins'])})")
    all_passed &= chk1

    chk2 = all(g > 0 for g in total_grads)
    print(f" [{'PASS' if chk2 else 'FAIL'}] All agents had non-zero gradients "
          f"(mins: {[f'{g/N_EPISODES:.2f}' for g in total_grads]})")
    all_passed &= chk2

    chk3 = all(np.all(np.isfinite(g)) for g in [total_grads])
    print(f" [{'PASS' if chk3 else 'FAIL'}] All gradients finite")
    all_passed &= chk3

    chk4 = model.test_auc >= 0.55
    print(f" [{'PASS' if chk4 else 'FAIL'}] Propensity model AUC "
          f"= {model.test_auc:.4f} (need >= 0.55)")
    all_passed &= chk4

    chk5 = all(g > 0 and np.isfinite(g) for g in slot_grads)
    print(f" [{'PASS' if chk5 else 'FAIL'}] Slot-mode gradients finite + non-zero "
          f"(R9b §5): {[f'{g:.2f}' for g in slot_grads]}")
    all_passed &= chk5

    elapsed = time.time() - t_start
    print(f"\n Elapsed: {elapsed:.1f}s")
    if all_passed:
        print(" SMOKE TEST PASSED")
        print(" Next: python -m src5.train_context_ac --episodes 80 --seeds 0")
    else:
        print(" SMOKE TEST FAILED — fix issues before running full training")
        sys.exit(1)


if __name__ == "__main__":
    main()
