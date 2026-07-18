"""
src5/train_context_ac.py
========================

Multi-agent actor-critic trainer using the context-propensity environment.

Layout:
  - 5 independent agents, one StrategicActorCritic each.
  - Each agent has its own optimizer; agents do NOT share gradients.
  - Shared 2nd-price auction (one winner per impression).
  - Training rows: weekday {3,4}. Eval rows: weekday 5 (handled by evaluate_context.py).

Logging: per-episode CSV row with clicks/wins/cost/utilization per agent,
plus action histograms at adaptive snapshots.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Path setup so this script can be run from the repo root
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src5.context_propensity import ContextPropensityModel, FEATURES  # noqa: E402
from src5.simulator.context_environment import (  # noqa: E402
    ContextRTBEnvironment,
    DEFAULT_THRESHOLD_GRID,
    DEFAULT_RESIDUAL_GRID,
    ADV_IDS,
)
from src5.policy_network import StrategicActorCritic  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


# ----------------------------------------------------------------------
def train_one_seed(
    df_train,
    seed: int,
    episodes: int,
    budgets: list,
    episode_rows: int,
    lr: float,
    gamma: float,
    entropy_beta: float,
    update_every: int,
    clip_grad: float,
    slot_size: int,
    click_reward_scale: float,
    output_dir: Path,
    snapshot_episodes: list[int],
):
    set_seed(seed)

    n_agents = len(budgets)
    agents = [
        StrategicActorCritic(input_dim=4, hidden_dim=128,
                              n_threshold=len(DEFAULT_THRESHOLD_GRID),
                              n_residual=len(DEFAULT_RESIDUAL_GRID)).to(DEVICE)
        for _ in range(n_agents)
    ]
    optimizers = [torch.optim.Adam(a.parameters(), lr=lr) for a in agents]

    # Single env shared by all agents
    env = ContextRTBEnvironment(
        df=df_train,
        budgets=budgets,
        episode_rows=episode_rows,
        click_reward_scale=click_reward_scale,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f" src5 | seed {seed} | {episodes} episodes")
    print(f"{'='*60}")
    print(f" Trainer config (per R9e launch-banner discipline):")
    print(f"   slot_size       : {slot_size}  (time-slot wrapper, R9b)")
    print(f"   update_every    : {update_every}  (slots between gradient updates, R9e §2)")
    n_slots_per_ep = episode_rows // slot_size if slot_size > 0 else 1
    print(f"   slots/ep        : {n_slots_per_ep}")
    print(f"   updates/ep      : ~{max(1, n_slots_per_ep // update_every)}")
    print(f"   updates total   : ~{episodes * max(1, n_slots_per_ep // update_every)}  over {episodes} eps")
    print(f"   lr              : {lr}")
    print(f"   gamma           : {gamma}")
    print(f"   entropy_beta    : {entropy_beta}")
    print(f"   clip_grad       : {clip_grad}")
    print(f" R9f algorithmic stabilization:")
    print(f"   advantages      : GAE(gamma={gamma}, tau=0.95)")
    print(f"   critic loss     : Huber (smooth_l1)")
    print(f"   critic weight   : 0.1  (R9f, down from 0.5)")

    log_file = output_dir / f"context_ac_seed_{seed}.csv"
    hist_file = output_dir / f"context_ac_seed_{seed}_action_hist.json"

    fieldnames = ["episode", "seed", "steps", "total_clicks", "total_wins", "total_cost"]
    for adv in ADV_IDS:
        fieldnames += [
            f"clicks_{adv}", f"wins_{adv}", f"cost_{adv}",
            f"util_{adv}", f"lambda_{adv}",
            f"ent_thr_{adv}", f"ent_res_{adv}",
        ]
    with open(log_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)

    action_snapshots = {}
    t_train_start = time.time()
    last_print_time = t_train_start

    for ep in range(1, episodes + 1):
        t_ep_start = time.time()
        states = env.reset()
        states_t = [to_tensor(s) for s in states]

        # Per-agent rollout buffers
        log_probs_th = [[] for _ in range(n_agents)]
        log_probs_res = [[] for _ in range(n_agents)]
        ents_th = [[] for _ in range(n_agents)]
        ents_res = [[] for _ in range(n_agents)]
        values = [[] for _ in range(n_agents)]
        rewards = [[] for _ in range(n_agents)]

        # Snapshot histogram counters
        if ep in snapshot_episodes:
            ep_th_counts = [np.zeros(len(DEFAULT_THRESHOLD_GRID), dtype=np.int64)
                            for _ in range(n_agents)]
            ep_res_counts = [np.zeros(len(DEFAULT_RESIDUAL_GRID), dtype=np.int64)
                             for _ in range(n_agents)]
        else:
            ep_th_counts = None
            ep_res_counts = None

        done = False
        step = 0

        while not done:
            th_actions = []
            res_actions = []
            for i in range(n_agents):
                logits_th, logits_res, v = agents[i](states_t[i])
                dist_th = Categorical(logits=logits_th.squeeze(0))
                dist_res = Categorical(logits=logits_res.squeeze(0))
                a_th = dist_th.sample()
                a_res = dist_res.sample()

                th_actions.append(int(a_th.item()))
                res_actions.append(int(a_res.item()))

                log_probs_th[i].append(dist_th.log_prob(a_th))
                log_probs_res[i].append(dist_res.log_prob(a_res))
                ents_th[i].append(dist_th.entropy())
                ents_res[i].append(dist_res.entropy())
                values[i].append(v.squeeze())

                if ep_th_counts is not None:
                    ep_th_counts[i][a_th.item()] += 1
                    ep_res_counts[i][a_res.item()] += 1

            if slot_size > 0:
                next_states, r, done, _ = env.step_slot(
                    th_actions, res_actions, slot_size=slot_size
                )
            else:
                # Deprecated per-impression fallback (R9b discovered this is
                # computationally infeasible at full episode size).
                next_states, r, done, _ = env.step(th_actions, res_actions)
            for i in range(n_agents):
                rewards[i].append(float(r[i]))
            if not done:
                states_t = [to_tensor(s) for s in next_states]
            step += 1

            # Periodic batched update
            if step % update_every == 0 or done:
                # R9f: GAE replaces Monte Carlo returns for variance reduction.
                # R9f: Huber loss (smooth_l1) replaces MSE for reward-magnitude stability.
                # R9f: Critic weight 0.5 -> 0.1 to prevent critic gradient dominance over actor.
                # These are standard A2C/PPO stabilization choices, not methodology changes.
                gae_tau = 0.95
                for i in range(n_agents):
                    if len(rewards[i]) == 0:
                        continue

                    vals = torch.stack(values[i])
                    rewards_t = to_tensor(rewards[i])

                    # Bootstrap from current state's value if not done; else 0
                    with torch.no_grad():
                        if done:
                            next_value = 0.0
                        else:
                            _, _, v_next = agents[i](states_t[i])
                            next_value = float(v_next.squeeze().item())

                    next_values = torch.cat([vals[1:].detach(), to_tensor([next_value])])

                    # GAE(gamma, tau) advantages, backward-recursive
                    advantages = torch.zeros_like(rewards_t, device=DEVICE)
                    last_gae_lam = 0.0
                    for t_step in reversed(range(len(rewards[i]))):
                        delta = rewards_t[t_step] + gamma * next_values[t_step] - vals[t_step].detach()
                        last_gae_lam = float(delta.item()) + gamma * gae_tau * last_gae_lam
                        advantages[t_step] = last_gae_lam

                    # Critic target = GAE advantage + value (a.k.a. lambda-return)
                    returns_t = advantages + vals.detach()

                    # Normalize advantages for actor (zero-mean unit-std)
                    if advantages.std() > 1e-8:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    log_pi_th = torch.stack(log_probs_th[i])
                    log_pi_res = torch.stack(log_probs_res[i])
                    ent_th = torch.stack(ents_th[i])
                    ent_res = torch.stack(ents_res[i])

                    # Huber loss for the critic (R9f: tolerates large reward spikes)
                    critic_loss = F.smooth_l1_loss(vals, returns_t)

                    actor_loss_th = -(log_pi_th * advantages).mean()
                    actor_loss_res = -(log_pi_res * advantages).mean()
                    entropy_term = -entropy_beta * (ent_th.mean() + ent_res.mean())

                    # Critic weight 0.1 (R9f: down from 0.5 to prevent shared-trunk dominance)
                    loss = actor_loss_th + actor_loss_res + 0.1 * critic_loss + entropy_term

                    optimizers[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agents[i].parameters(), clip_grad)
                    optimizers[i].step()

                # Clear buffers
                log_probs_th = [[] for _ in range(n_agents)]
                log_probs_res = [[] for _ in range(n_agents)]
                ents_th = [[] for _ in range(n_agents)]
                ents_res = [[] for _ in range(n_agents)]
                values = [[] for _ in range(n_agents)]
                rewards = [[] for _ in range(n_agents)]

        # Episode-end diagnostics
        diag = env.diagnostics()
        total_clicks = int(np.sum(diag["clicks"]))
        total_wins = int(np.sum(diag["wins"]))
        total_cost = float(np.sum(diag["cost"]))

        # Final-policy entropy on a sample of states (cheap proxy)
        ent_thr_per_agent = []
        ent_res_per_agent = []
        sample_state = to_tensor(states[0])
        for i in range(n_agents):
            with torch.no_grad():
                lt, lr_, _ = agents[i](sample_state)
                p_th = F.softmax(lt.squeeze(0), dim=-1)
                p_res = F.softmax(lr_.squeeze(0), dim=-1)
                ent_thr_per_agent.append(float(-(p_th * (p_th.clamp_min(1e-12)).log()).sum()))
                ent_res_per_agent.append(float(-(p_res * (p_res.clamp_min(1e-12)).log()).sum()))

        # CSV row
        row = [ep, seed, diag["steps"], total_clicks, total_wins, total_cost]
        for j, adv in enumerate(ADV_IDS):
            row += [
                int(diag["clicks"][j]),
                int(diag["wins"][j]),
                float(diag["cost"][j]),
                float(diag["utilization_pct"][j]),
                float(diag["lambda_final"][j]),
                ent_thr_per_agent[j],
                ent_res_per_agent[j],
            ]
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Snapshot histograms
        if ep_th_counts is not None:
            action_snapshots[str(ep)] = {
                "threshold": [c.tolist() for c in ep_th_counts],
                "residual": [c.tolist() for c in ep_res_counts],
            }

        if ep == 1 or ep % 10 == 0 or ep == episodes:
            ep_secs = time.time() - t_ep_start
            total_secs = time.time() - t_train_start
            print(f"Seed {seed} | Ep {ep:03d} | "
                  f"Clicks={diag['clicks']} | Wins={diag['wins']} | "
                  f"Cost={[f'{c:.0f}' for c in diag['cost']]} | "
                  f"Util={[f'{u:.0f}%' for u in diag['utilization_pct']]} | "
                  f"H_thr_mean={np.mean(ent_thr_per_agent):.2f} | "
                  f"H_res_mean={np.mean(ent_res_per_agent):.2f} | "
                  f"t_ep={ep_secs:.1f}s | t_total={total_secs:.0f}s")

    # Save histograms
    with open(hist_file, "w") as f:
        json.dump(action_snapshots, f)

    # Save models
    for j, adv in enumerate(ADV_IDS):
        torch.save(
            agents[j].state_dict(),
            output_dir / f"context_ac_seed_{seed}_agent_{adv}.pt",
        )

    print(f"Seed {seed} saved to {log_file}")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="data_2/shared_auction_log_v4_dense.txt")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated seed list, e.g. '0,1,2'")
    parser.add_argument("--episode-rows", type=int, default=125_000)
    parser.add_argument("--budgets", type=str,
                        default="50000,50000,50000,50000,50000",
                        help="Comma-separated per-agent budgets")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-beta", type=float, default=1e-2)
    parser.add_argument("--update-every", type=int, default=2,
                        help="Update every N slots. Default 2 per R9e §2 — gives ~12 updates/ep, ~1000 total over 80 ep, enough to distinguish 'can't learn signal' from 'wasn't given enough updates.' (Was 5 in R9b draft.)")
    parser.add_argument("--slot-size", type=int, default=5000,
                        help="Time-slot wrapper size. Default 5000 (= src4). Set 0 to disable wrapper (slow, deprecated).")
    parser.add_argument("--clip-grad", type=float, default=10.0)
    parser.add_argument("--click-reward-scale", type=float, default=None,
                        help="Scale on click bonus in reward. DEFAULT = derive "
                             "from data: mean(market_price)/target_CTR_design. "
                             "Per Round-9 §1, do NOT override this except for "
                             "explicit ablation; the derivation is the pre-"
                             "committed value.")
    parser.add_argument("--output-dir", type=str,
                        default="src5/outputs/context_ac")
    parser.add_argument("--no-learn", action="store_true",
                        help="Don't update policies (frozen-uniform baseline). "
                             "Use this for Run A.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [float(b) for b in args.budgets.split(",")]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Fit propensity model on weekday {3,4} ---
    print("\n[Step 1/3] Fitting context-propensity model")
    print("-" * 60)
    import pandas as pd
    df = pd.read_csv(args.data, sep="\t")
    model = ContextPropensityModel()
    model.fit(df, train_weekdays=(3, 4), eval_weekday=5, verbose=True)

    # --- 2. Attach propensity column to full df, then filter ---
    df_with_p = model.attach_to_dataframe(df)
    df_train = df_with_p[df_with_p["weekday"].isin([3, 4])].reset_index(drop=True)
    print(f"  Training-data rows (weekday 3,4): {len(df_train):,}")

    # --- 3. Train ---
    print("\n[Step 2/3] Multi-agent actor-critic training")
    print("-" * 60)
    snapshot_eps = sorted(set([1, max(1, args.episodes // 4),
                                args.episodes // 2,
                                max(1, 3 * args.episodes // 4),
                                args.episodes]))

    if args.no_learn:
        print("  --no-learn enabled: policies will not be updated (Run A baseline)")
        # Quick hack: zero entropy_beta and lr -> still updates but gradient ~ 0
        # Cleaner: explicit no-learn flag. For now: just set lr to 0.
        args.lr = 0.0

    for seed in seeds:
        train_one_seed(
            df_train=df_train,
            seed=seed,
            episodes=args.episodes,
            budgets=budgets,
            episode_rows=args.episode_rows,
            lr=args.lr,
            gamma=args.gamma,
            entropy_beta=args.entropy_beta,
            update_every=args.update_every,
            clip_grad=args.clip_grad,
            slot_size=args.slot_size,
            click_reward_scale=args.click_reward_scale,
            output_dir=output_dir,
            snapshot_episodes=snapshot_eps,
        )

    print("\n[Step 3/3] DONE")
    print(f"  Outputs: {output_dir}")


if __name__ == "__main__":
    main()
