"""
src6 — MAPPO smoke test.

Validates:
  [1/5]  Propensity + env build (delegates to src5)
  [2/5]  Linear-propensity baseline sanity (1 episode)
  [3/5]  Actor + Centralized Critic forward-pass shapes
  [4/5]  Tiny PPO training loop (3 episodes) — gradients finite, non-zero
  [5b/5] Centralized critic gradient check under multi-agent rewards

Run BEFORE launching real training:
    python -m src6.smoke_test_mappo
"""

import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Reuse src5's env (no copying)
from src5.simulator.context_environment import ContextRTBEnvironment  # noqa: F401

from src6.policy_network_mappo import (
    MAPPOActor,
    CentralizedCritic,
    concat_global_state,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADV_IDS = ["1458", "2259", "3386", "2997", "3476"]


def to_t(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def main():
    t0 = time.time()
    print("=" * 60)
    print(" SMOKE TEST — src6 MAPPO")
    print("=" * 60)
    print(f"  device: {DEVICE}")

    # ============================================================
    # [1/5] Build env (replicates src5/train_context_ac.py setup)
    # ============================================================
    print("[1/5] Loading data + fitting propensity (via src6.env_setup) ...")
    try:
        from src6.env_setup import build_train_env
        env, info = build_train_env(verbose=True)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        print(f"  Check that:")
        print(f"    - The data file exists at the path passed to build_train_env")
        print(f"      (default: data_2/shared_auction_log_v4_dense.txt)")
        print(f"    - src5.context_propensity exposes ContextPropensityModel")
        print(f"    - src5.simulator.context_environment exposes ContextRTBEnvironment")
        print(f"  src5 is read-only; no edits needed if imports above resolve.")
        return False

    n_agents = env.n_agents
    print(f"  ContextRTBEnvironment loaded successfully")
    print(f"    n_agents : {n_agents}")
    print(f"    budgets  : {list(env.budgets)}")

    # ============================================================
    # [2/5] Linear-propensity baseline sanity (1 episode)
    # ============================================================
    print("[2/5] Linear-propensity baseline sanity (1 episode) ...")
    env.reset()
    done = False
    n_slots = 0
    # Find indices that correspond to a "linear baseline":
    #   threshold = 0 (don't skip anything)
    #   residual  = middle bin (residual ≈ 0, pure linear bid)
    th_idx = 0
    res_idx = 5  # middle of 11-bin grid for symmetric [-0.3, +0.3]
    while not done and n_slots < 30:
        _, _, done, _ = env.step_slot(
            [th_idx] * n_agents, [res_idx] * n_agents, slot_size=5000,
        )
        n_slots += 1
    try:
        d = env.diagnostics()
        print(f"  Linear baseline diagnostics (1 ep):")
        print(f"    wins per agent  : {[int(x) for x in d['wins']]}")
        print(f"    clicks per agent: {[int(x) for x in d['clicks']]}")
        print(f"    cost per agent  : {[f'{int(x)}' for x in d['cost']]}")
    except Exception as e:
        print(f"  (diagnostics method not available: {e})")

    # ============================================================
    # [3/5] Forward-pass shape check
    # ============================================================
    print("[3/5] Building MAPPO actors + centralized critic, forward check ...")
    state_dim = 4
    n_th = 51
    n_res = 11

    actors = [
        MAPPOActor(state_dim=state_dim, hidden_dim=128, n_threshold=n_th, n_residual=n_res).to(DEVICE)
        for _ in range(n_agents)
    ]
    critic = CentralizedCritic(state_dim=state_dim, n_agents=n_agents, hidden_dim=256).to(DEVICE)

    dummy_state = torch.randn(1, state_dim, device=DEVICE)
    dummy_global = torch.randn(1, state_dim * n_agents, device=DEVICE)

    th_logits, res_logits = actors[0](dummy_state)
    values = critic(dummy_global)

    print(f"  actor[0] threshold logits shape : {tuple(th_logits.shape)}  expected: (1, {n_th})")
    print(f"  actor[0] residual  logits shape : {tuple(res_logits.shape)} expected: (1, {n_res})")
    print(f"  centralized critic values shape : {tuple(values.shape)}     expected: (1, {n_agents})")

    assert th_logits.shape == (1, n_th), "Threshold logits shape mismatch"
    assert res_logits.shape == (1, n_res), "Residual logits shape mismatch"
    assert values.shape == (1, n_agents), "Critic output shape mismatch"
    print(f"  [OK] All shape checks passed")

    # ============================================================
    # [4/5] Tiny PPO training loop (3 episodes)
    # ============================================================
    print("[4/5] Tiny PPO loop (3 episodes) ...")
    actor_opts = [torch.optim.Adam(a.parameters(), lr=3e-4) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    clip_eps = 0.2
    ppo_epochs = 4
    entropy_beta = 0.03
    gamma = 0.99
    lam = 0.95

    last_grad_norms = [0.0] * n_agents

    for ep in range(1, 4):
        # ---- Rollout (1 episode) ----
        states_buf = []
        global_buf = []
        th_a_buf, res_a_buf = [], []
        th_lp_buf, res_lp_buf = [], []
        rew_buf, done_buf = [], []

        states = env.reset()
        done = False
        n_slots = 0
        while not done and n_slots < 30:
            th_a, res_a, th_lp, res_lp = [], [], [], []
            with torch.no_grad():
                for i in range(n_agents):
                    s_t = to_t(states[i]).unsqueeze(0)
                    tl, rl = actors[i](s_t)
                    td = Categorical(logits=tl)
                    rd = Categorical(logits=rl)
                    ta = td.sample()
                    ra = rd.sample()
                    th_a.append(int(ta.item()))
                    res_a.append(int(ra.item()))
                    th_lp.append(float(td.log_prob(ta).item()))
                    res_lp.append(float(rd.log_prob(ra).item()))

            states_buf.append(np.stack([np.asarray(s) for s in states]).astype(np.float32))
            global_buf.append(np.concatenate([np.asarray(s).flatten() for s in states]).astype(np.float32))
            th_a_buf.append(th_a)
            res_a_buf.append(res_a)
            th_lp_buf.append(th_lp)
            res_lp_buf.append(res_lp)

            next_states, slot_rewards, done, _ = env.step_slot(
                th_a, res_a, slot_size=5000,
            )
            rew_buf.append(np.asarray(slot_rewards, dtype=np.float32))
            done_buf.append(float(done))
            if not done:
                states = next_states
            n_slots += 1

        # ---- Stack to arrays ----
        T = len(rew_buf)
        states_T = to_t(np.stack(states_buf))   # (T, n_agents, state_dim)
        global_T = to_t(np.stack(global_buf))   # (T, n_agents*state_dim)
        th_a_T = torch.tensor(np.array(th_a_buf), dtype=torch.long, device=DEVICE)
        res_a_T = torch.tensor(np.array(res_a_buf), dtype=torch.long, device=DEVICE)
        old_th_lp = to_t(np.array(th_lp_buf))
        old_res_lp = to_t(np.array(res_lp_buf))
        rew_np = np.array(rew_buf)              # (T, n_agents)
        done_np = np.array(done_buf)            # (T,)

        # ---- Compute GAE ----
        with torch.no_grad():
            values = critic(global_T).cpu().numpy()
            values_pad = np.concatenate([values, np.zeros((1, n_agents), dtype=np.float32)], axis=0)

        adv = np.zeros((T, n_agents), dtype=np.float32)
        ret = np.zeros((T, n_agents), dtype=np.float32)
        for i in range(n_agents):
            gae = 0.0
            for t in reversed(range(T)):
                nv = values_pad[t + 1, i] * (1.0 - done_np[t])
                delta = rew_np[t, i] + gamma * nv - values_pad[t, i]
                gae = delta + gamma * lam * (1.0 - done_np[t]) * gae
                adv[t, i] = gae
            ret[:, i] = adv[:, i] + values_pad[:-1, i]
            # Normalize
            s = adv[:, i].std()
            if s > 1e-8:
                adv[:, i] = (adv[:, i] - adv[:, i].mean()) / (s + 1e-8)

        adv_T = to_t(adv)
        ret_T = to_t(ret)

        # ---- PPO multi-epoch updates ----
        grad_norms = [0.0] * n_agents
        for epoch in range(ppo_epochs):
            # Critic
            v = critic(global_T)
            c_loss = F.smooth_l1_loss(v, ret_T)
            critic_opt.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            critic_opt.step()

            # Actors
            for i in range(n_agents):
                tl, rl = actors[i](states_T[:, i, :])
                td = Categorical(logits=tl)
                rd = Categorical(logits=rl)
                new_lp = td.log_prob(th_a_T[:, i]) + rd.log_prob(res_a_T[:, i])
                old_lp = old_th_lp[:, i] + old_res_lp[:, i]
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_T[:, i]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_T[:, i]
                policy_loss = -torch.min(surr1, surr2).mean()
                ent = td.entropy().mean() + rd.entropy().mean()
                loss = policy_loss - entropy_beta * ent

                actor_opts[i].zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(actors[i].parameters(), 10.0)
                grad_norms[i] = float(gn.item())
                actor_opts[i].step()

        last_grad_norms = grad_norms
        try:
            d = env.diagnostics()
            wins_str = str([int(x) for x in d['wins']])
            clicks_str = str([int(x) for x in d['clicks']])
        except Exception:
            wins_str = clicks_str = "(unavailable)"
        gn_str = [f"{g:.2f}" for g in grad_norms]
        print(f"  ep {ep}: wins={wins_str} clicks={clicks_str} grad_norms={gn_str}")

    # ============================================================
    # [5b/5] Final centralized-critic gradient check
    # ============================================================
    print("[5b/5] Centralized critic final gradient check ...")
    # Push one synthetic batch through with large reward magnitudes
    # to confirm Huber + critic-side grad clipping keep things bounded.
    big_global = torch.randn(8, state_dim * n_agents, device=DEVICE) * 5.0
    big_returns = torch.randn(8, n_agents, device=DEVICE) * 1000.0  # mimic click-reward scale
    v_out = critic(big_global)
    c_loss = F.smooth_l1_loss(v_out, big_returns)
    critic_opt.zero_grad()
    c_loss.backward()
    critic_gn = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    print(f"  critic grad norm under reward magnitude ~1000: {float(critic_gn.item()):.2f}")
    print(f"  critic loss (Huber, smooth_l1): {float(c_loss.item()):.2f}")

    # ============================================================
    # CHECKS
    # ============================================================
    print("=" * 60)
    print(" CHECKS")
    print("=" * 60)

    try:
        d_last = env.diagnostics()
        total_wins = int(sum(d_last["wins"]))
        if total_wins > 0:
            print(f" [PASS] Last episode had wins (total = {total_wins})")
        else:
            print(f" [WARN] Last episode had ZERO wins — check env reset / state")
    except Exception as e:
        print(f" [WARN] diagnostics() unavailable: {e}")

    all_finite = all(np.isfinite(g) for g in last_grad_norms)
    all_nonzero = all(g > 1e-6 for g in last_grad_norms)
    print(f" [{'PASS' if all_finite else 'FAIL'}] All actor gradients finite")
    print(f" [{'PASS' if all_nonzero else 'FAIL'}] All actor gradients non-zero  (mins: {[f'{g:.2f}' for g in last_grad_norms]})")

    critic_finite = np.isfinite(float(critic_gn.item()))
    print(f" [{'PASS' if critic_finite else 'FAIL'}] Centralized critic gradient finite under large rewards")

    elapsed = time.time() - t0
    print(f" Elapsed: {elapsed:.1f}s")

    if all_finite and all_nonzero and critic_finite:
        print(" SMOKE TEST PASSED")
        print(" Next: python -m src6.train_mappo --episodes 80 --seeds 0")
        return True
    else:
        print(" SMOKE TEST FAILED — fix issues above before training")
        return False


if __name__ == "__main__":
    main()
