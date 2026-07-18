"""
src6 — MAPPO trainer.

Multi-Agent PPO (Yu et al., 2022) on the same time-slot-wrapped
context-propensity RTB environment as src5. CTDE: centralized critic at
training, decentralized actors at execution.

Reuses src5's environment, propensity model, and dataset unchanged.
Only the policy parameterization and the optimization algorithm differ.

Launch:
    python -m src6.train_mappo --episodes 80 --seeds 0
    python -m src6.train_mappo --episodes 80 --seeds 1 2 3 4 --entropy-beta 0.03

The trainer mirrors src5's CSV/JSON output schema so existing analysis
scripts work unmodified.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Reuse src5's environment and propensity (no duplication — same physics)
from src5.simulator.context_environment import ContextRTBEnvironment

# MAPPO networks
from src6.policy_network_mappo import (
    MAPPOActor,
    CentralizedCritic,
    concat_global_state,
)


# ============================================================
# Defaults (match src5 R9g winning config where they overlap)
# ============================================================
DEFAULTS = dict(
    episodes=80,
    slot_size=5000,
    episode_rows=125_000,
    n_agents=5,
    state_dim=4,
    n_threshold=51,
    n_residual=11,
    budgets=[50000.0] * 5,
    # Optimization
    lr_actor=3e-4,
    lr_critic=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    # PPO-specific
    clip_eps=0.2,           # PPO clip ratio (standard)
    ppo_epochs=4,           # multi-epoch updates per rollout (standard)
    entropy_beta=0.03,      # matches R9g winning config
    critic_weight=0.5,
    clip_grad=10.0,
    target_kl=0.02,         # early-stop epoch if mean KL exceeds this
    # Bookkeeping
    snapshot_episodes=(1, 20, 40, 60, 80),
)

ADV_IDS = ["1458", "2259", "3386", "2997", "3476"]


# ============================================================
# Utilities
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x, device, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=device)


def compute_gae(rewards, values, dones, gamma, lam):
    """
    Generalized Advantage Estimation, per-agent.

    rewards: (T,)
    values:  (T+1,) — includes bootstrap value at the end (zero on done)
    dones:   (T,) — 1.0 if episode terminated at step t, else 0.0
    returns: advantages (T,), returns (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def env_diag_summary(env, n_agents):
    """
    Pull per-episode env stats. The env's diagnostics() method returns
    keys: clicks, wins, cost, utilization_pct, lambda_final, steps.
    (Confirmed against src5/simulator/context_environment.py — keys match
    src5/train_context_ac.py and src5/evaluate_context.py expectations.)
    """
    d = env.diagnostics()
    return {
        "clicks": [int(x) for x in d["clicks"]],
        "wins": [int(x) for x in d["wins"]],
        "cost": [float(x) for x in d["cost"]],
        "utilization_pct": [float(x) for x in d["utilization_pct"]],
        "lambda_final": [float(x) for x in d["lambda_final"]],
        "steps": int(d.get("steps", 0)),
    }


# ============================================================
# Rollout buffer
# ============================================================
class RolloutBuffer:
    """
    Stores one episode's worth of (state, action, log_prob, reward) tuples
    plus the centralized global states for the critic. Cleared between
    episodes.
    """

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.states = []        # list[T] of (n_agents, state_dim)
        self.global_states = [] # list[T] of (state_dim * n_agents,)
        self.th_actions = []    # list[T] of (n_agents,) int
        self.res_actions = []   # list[T] of (n_agents,) int
        self.th_logp = []       # list[T] of (n_agents,) float
        self.res_logp = []      # list[T] of (n_agents,) float
        self.rewards = []       # list[T] of (n_agents,) float
        self.dones = []         # list[T] of bool

    def add(self, states, global_state, th_a, res_a, th_lp, res_lp, rewards, done):
        self.states.append(np.asarray(states, dtype=np.float32))
        self.global_states.append(np.asarray(global_state, dtype=np.float32))
        self.th_actions.append(np.asarray(th_a, dtype=np.int64))
        self.res_actions.append(np.asarray(res_a, dtype=np.int64))
        self.th_logp.append(np.asarray(th_lp, dtype=np.float32))
        self.res_logp.append(np.asarray(res_lp, dtype=np.float32))
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(float(done))

    def as_arrays(self):
        return dict(
            states=np.stack(self.states),               # (T, n_agents, state_dim)
            global_states=np.stack(self.global_states), # (T, state_dim*n_agents)
            th_actions=np.stack(self.th_actions),       # (T, n_agents)
            res_actions=np.stack(self.res_actions),     # (T, n_agents)
            th_logp=np.stack(self.th_logp),             # (T, n_agents)
            res_logp=np.stack(self.res_logp),           # (T, n_agents)
            rewards=np.stack(self.rewards),             # (T, n_agents)
            dones=np.asarray(self.dones, dtype=np.float32),  # (T,)
        )


# ============================================================
# Action sampling
# ============================================================
@torch.no_grad()
def sample_actions(actors, states_per_agent, device):
    """
    For each agent, sample (threshold, residual) action from its actor.
    Returns lists of int actions and float log-probs, one entry per agent.
    """
    th_actions, res_actions = [], []
    th_logps, res_logps = [], []
    for i, actor in enumerate(actors):
        s = to_tensor(states_per_agent[i], device).unsqueeze(0)  # (1, state_dim)
        th_logits, res_logits = actor(s)
        th_dist = Categorical(logits=th_logits)
        res_dist = Categorical(logits=res_logits)
        th_a = th_dist.sample()
        res_a = res_dist.sample()
        th_actions.append(int(th_a.item()))
        res_actions.append(int(res_a.item()))
        th_logps.append(float(th_dist.log_prob(th_a).item()))
        res_logps.append(float(res_dist.log_prob(res_a).item()))
    return th_actions, res_actions, th_logps, res_logps


# ============================================================
# PPO update step
# ============================================================
def ppo_update(
    actors, critic, actor_opts, critic_opt,
    buffer_arrays, device,
    clip_eps, ppo_epochs, entropy_beta,
    critic_weight, clip_grad, target_kl,
    gamma, gae_lambda,
):
    """
    PPO multi-epoch update on one episode's rollout. Returns a dict of
    per-agent diagnostics (entropy, kl, ratio stats) on the last epoch.
    """
    n_agents = len(actors)

    # ---- Move data to GPU ----
    states_T = to_tensor(buffer_arrays["states"], device)                   # (T, n_agents, state_dim)
    global_T = to_tensor(buffer_arrays["global_states"], device)            # (T, global_dim)
    th_act_T = to_tensor(buffer_arrays["th_actions"], device, torch.long)   # (T, n_agents)
    res_act_T = to_tensor(buffer_arrays["res_actions"], device, torch.long) # (T, n_agents)
    old_th_lp = to_tensor(buffer_arrays["th_logp"], device)                 # (T, n_agents)
    old_res_lp = to_tensor(buffer_arrays["res_logp"], device)               # (T, n_agents)
    rewards_T = buffer_arrays["rewards"]                                    # numpy (T, n_agents)
    dones_T = buffer_arrays["dones"]                                        # numpy (T,)

    T = states_T.shape[0]

    # ---- Compute values with current critic, then GAE ----
    with torch.no_grad():
        values_now = critic(global_T).cpu().numpy()        # (T, n_agents)
        # Bootstrap value at terminal step: 0 (episode ended on done)
        values_padded = np.concatenate([values_now, np.zeros((1, n_agents), dtype=np.float32)], axis=0)

    advantages = np.zeros((T, n_agents), dtype=np.float32)
    returns_np = np.zeros((T, n_agents), dtype=np.float32)
    for i in range(n_agents):
        adv, ret = compute_gae(
            rewards_T[:, i],
            values_padded[:, i],
            dones_T,
            gamma, gae_lambda,
        )
        advantages[:, i] = adv
        returns_np[:, i] = ret

    # Per-agent advantage normalization (PPO stability)
    for i in range(n_agents):
        a = advantages[:, i]
        std = a.std()
        if std > 1e-8:
            advantages[:, i] = (a - a.mean()) / (std + 1e-8)

    advantages_T = to_tensor(advantages, device)  # (T, n_agents)
    returns_T = to_tensor(returns_np, device)     # (T, n_agents)

    # ---- Multi-epoch PPO update ----
    # T is small (e.g., 25 slots), so we do full-batch per epoch.
    diag_ent_thr = np.zeros(n_agents, dtype=np.float32)
    diag_ent_res = np.zeros(n_agents, dtype=np.float32)
    diag_kl_mean = np.zeros(n_agents, dtype=np.float32)
    diag_ratio_max = np.zeros(n_agents, dtype=np.float32)
    epochs_actually_run = ppo_epochs

    for epoch in range(ppo_epochs):
        # ---- Critic update (centralized) ----
        values_pred = critic(global_T)  # (T, n_agents)
        critic_loss = F.smooth_l1_loss(values_pred, returns_T)
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), clip_grad)
        critic_opt.step()

        # ---- Actor updates (per agent, independent) ----
        kl_too_large = False
        for i in range(n_agents):
            th_logits, res_logits = actors[i](states_T[:, i, :])  # (T, n_th), (T, n_res)
            th_dist = Categorical(logits=th_logits)
            res_dist = Categorical(logits=res_logits)

            new_th_lp = th_dist.log_prob(th_act_T[:, i])    # (T,)
            new_res_lp = res_dist.log_prob(res_act_T[:, i]) # (T,)
            new_logp = new_th_lp + new_res_lp
            old_logp = old_th_lp[:, i] + old_res_lp[:, i]

            ratio = torch.exp(new_logp - old_logp)
            adv_i = advantages_T[:, i]

            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_i
            policy_loss = -torch.min(surr1, surr2).mean()

            ent_th = th_dist.entropy().mean()
            ent_res = res_dist.entropy().mean()
            entropy_bonus = ent_th + ent_res

            loss_i = policy_loss - entropy_beta * entropy_bonus

            actor_opts[i].zero_grad()
            loss_i.backward()
            nn.utils.clip_grad_norm_(actors[i].parameters(), clip_grad)
            actor_opts[i].step()

            # KL estimator (Schulman): kl ~= mean((ratio - 1) - log(ratio))
            with torch.no_grad():
                kl_est = ((ratio - 1.0) - (new_logp - old_logp)).mean().item()
                diag_kl_mean[i] = kl_est
                diag_ratio_max[i] = float(ratio.max().item())
                if epoch == ppo_epochs - 1:
                    diag_ent_thr[i] = float(ent_th.item())
                    diag_ent_res[i] = float(ent_res.item())
                if target_kl is not None and kl_est > 1.5 * target_kl:
                    kl_too_large = True

        if kl_too_large:
            epochs_actually_run = epoch + 1
            # On last epoch where any agent exceeded KL, still record diagnostics
            for i in range(n_agents):
                # Make sure entropies got recorded (they're from this last-attempted epoch)
                if diag_ent_thr[i] == 0.0:
                    with torch.no_grad():
                        th_logits, res_logits = actors[i](states_T[:, i, :])
                        diag_ent_thr[i] = float(Categorical(logits=th_logits).entropy().mean().item())
                        diag_ent_res[i] = float(Categorical(logits=res_logits).entropy().mean().item())
            break

    return dict(
        ent_thr=diag_ent_thr,
        ent_res=diag_ent_res,
        kl_mean=diag_kl_mean,
        ratio_max=diag_ratio_max,
        epochs_run=epochs_actually_run,
    )


# ============================================================
# Main training loop
# ============================================================
def train_one_seed(seed, args, train_df_setup_fn=None):
    """
    Train MAPPO for one seed. train_df_setup_fn is an optional callable
    that returns (env_kwargs) — useful for testing. Normally not used.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Step 2/3] MAPPO training | seed {seed}")
    print("-" * 60)
    set_seed(seed)

    # ---- Env setup (replicates src5/train_context_ac.py main(), no src5 edits) ----
    from src6.env_setup import build_train_env
    env, _info = build_train_env(
        data_path=args.data,
        budgets=args.budgets,
        episode_rows=args.episode_rows,
        click_reward_scale=args.click_reward_scale,
        verbose=True,
    )

    n_agents = env.n_agents
    n_th = args.n_threshold
    n_res = args.n_residual
    state_dim = args.state_dim

    # ---- Build actors + centralized critic ----
    actors = [
        MAPPOActor(
            state_dim=state_dim,
            hidden_dim=128,
            n_threshold=n_th,
            n_residual=n_res,
        ).to(device)
        for _ in range(n_agents)
    ]
    critic = CentralizedCritic(
        state_dim=state_dim,
        n_agents=n_agents,
        hidden_dim=256,
    ).to(device)

    actor_opts = [torch.optim.Adam(a.parameters(), lr=args.lr) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    # ---- Launch banner ----
    print("=" * 60)
    print(f" src6 MAPPO | seed {seed} | {args.episodes} episodes")
    print("=" * 60)
    print(" Trainer config:")
    print(f"   slot_size       : {args.slot_size}")
    print(f"   slots/ep        : {args.episode_rows // args.slot_size}")
    print(f"   lr              : {args.lr}")
    print(f"   gamma           : {args.gamma}")
    print(f"   gae_lambda      : {args.gae_lambda}")
    print(f"   clip_eps        : {args.clip_eps}  (PPO)")
    print(f"   ppo_epochs      : {args.ppo_epochs}  (multi-epoch updates per rollout)")
    print(f"   entropy_beta    : {args.entropy_beta}")
    print(f"   critic_weight   : {args.critic_weight}")
    print(f"   clip_grad       : {args.clip_grad}")
    print(f"   target_kl       : {args.target_kl}  (early-stop epoch if exceeded)")
    print(" Architecture (CTDE):")
    print(f"   actors          : 5 x decentralized  (state_dim={state_dim})")
    print(f"   critic          : centralized        (state_dim={state_dim * n_agents} = {state_dim} * {n_agents})")

    # ---- Logging setup ----
    out_dir = Path("src6/outputs/mappo")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"mappo_seed_{seed}.csv"
    hist_path = out_dir / f"mappo_seed_{seed}_action_hist.json"
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    action_hist_snapshots = {}
    log_rows = []
    buffer = RolloutBuffer(n_agents)
    t_start = time.time()

    # ============================================================
    # Episode loop
    # ============================================================
    for ep in range(1, args.episodes + 1):
        ep_t0 = time.time()
        buffer.reset()
        states = env.reset()  # list of n_agents 1D arrays

        # Histogram snapshot tracking
        snapshot_this_ep = ep in args.snapshot_episodes
        if snapshot_this_ep:
            th_hist = np.zeros((n_agents, n_th), dtype=np.int32)
            res_hist = np.zeros((n_agents, n_res), dtype=np.int32)

        # ---- Collect one full-episode rollout ----
        done = False
        n_slots = args.episode_rows // args.slot_size
        for _ in range(n_slots):
            # Sample actions for all agents
            th_a, res_a, th_lp, res_lp = sample_actions(actors, states, device)
            global_state = np.concatenate([np.asarray(s).flatten() for s in states])

            # Histogram
            if snapshot_this_ep:
                for i in range(n_agents):
                    th_hist[i, th_a[i]] += 1
                    res_hist[i, res_a[i]] += 1

            # Step env (slot-wrapped, keyword-arg matches src5's call convention)
            next_states, slot_rewards, done, _ = env.step_slot(
                th_a, res_a, slot_size=args.slot_size,
            )

            buffer.add(
                states=np.stack([np.asarray(s) for s in states]),
                global_state=global_state,
                th_a=th_a,
                res_a=res_a,
                th_lp=th_lp,
                res_lp=res_lp,
                rewards=slot_rewards,
                done=done,
            )

            if done:
                break
            states = next_states

        # ---- PPO update ----
        buffer_arrays = buffer.as_arrays()
        diag = ppo_update(
            actors, critic, actor_opts, critic_opt,
            buffer_arrays, device,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            entropy_beta=args.entropy_beta,
            critic_weight=args.critic_weight,
            clip_grad=args.clip_grad,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        # ---- Per-episode diagnostics ----
        env_diag = env_diag_summary(env, n_agents)
        total_clicks = sum(env_diag["clicks"])
        total_wins = sum(env_diag["wins"])
        total_cost = sum(env_diag["cost"])

        log_row = {
            "episode": ep,
            "seed": seed,
            "steps": env_diag["steps"],
            "total_clicks": int(total_clicks),
            "total_wins": int(total_wins),
            "total_cost": float(total_cost),
            "ppo_epochs_run": int(diag["epochs_run"]),
        }
        for i, adv in enumerate(ADV_IDS):
            log_row[f"clicks_{adv}"] = env_diag["clicks"][i]
            log_row[f"wins_{adv}"] = env_diag["wins"][i]
            log_row[f"cost_{adv}"] = env_diag["cost"][i]
            log_row[f"util_{adv}"] = env_diag["utilization_pct"][i]
            log_row[f"lambda_{adv}"] = env_diag["lambda_final"][i]
            log_row[f"ent_thr_{adv}"] = float(diag["ent_thr"][i])
            log_row[f"ent_res_{adv}"] = float(diag["ent_res"][i])
            log_row[f"kl_{adv}"] = float(diag["kl_mean"][i])
            log_row[f"ratio_max_{adv}"] = float(diag["ratio_max"][i])
        log_rows.append(log_row)

        if snapshot_this_ep:
            action_hist_snapshots[str(ep)] = {
                "threshold": th_hist.tolist(),
                "residual": res_hist.tolist(),
            }

        # Print progress
        if ep == 1 or ep % 10 == 0 or ep == args.episodes:
            t_ep = time.time() - ep_t0
            t_tot = time.time() - t_start
            clicks_str = "[" + ", ".join(str(c) for c in env_diag["clicks"]) + "]"
            wins_str = "[" + ", ".join(str(w) for w in env_diag["wins"]) + "]"
            util_str = "[" + ", ".join(f"'{u:.0f}%'" for u in env_diag["utilization_pct"]) + "]"
            h_thr_m = float(np.mean(diag["ent_thr"]))
            h_res_m = float(np.mean(diag["ent_res"]))
            kl_m = float(np.mean(diag["kl_mean"]))
            print(
                f"Seed {seed} | Ep {ep:03d} | "
                f"Clicks={clicks_str} | Wins={wins_str} | Util={util_str} | "
                f"H_thr_mean={h_thr_m:.2f} | H_res_mean={h_res_m:.2f} | "
                f"KL_mean={kl_m:.4f} | epochs={diag['epochs_run']} | "
                f"t_ep={t_ep:.1f}s | t_total={t_tot:.0f}s"
            )

    # ============================================================
    # Save outputs
    # ============================================================
    if log_rows:
        cols = list(log_rows[0].keys())
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(log_rows)
    with open(hist_path, "w") as f:
        json.dump(action_hist_snapshots, f)

    # Save models — actors per agent + critic
    for i, adv in enumerate(ADV_IDS):
        torch.save(
            actors[i].state_dict(),
            models_dir / f"mappo_seed_{seed}_actor_{adv}.pt",
        )
    torch.save(critic.state_dict(), models_dir / f"mappo_seed_{seed}_critic.pt")

    print(f"\nSeed {seed} saved to {log_path}")
    print(f"  models: {models_dir}/mappo_seed_{seed}_actor_*.pt and _critic.pt")


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="src6 — MAPPO trainer")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--data", type=str,
                        default="data_2/shared_auction_log_v4_dense.txt",
                        help="Path to densified iPinYou log (same as src5 default).")
    parser.add_argument("--budgets", type=lambda s: [float(b) for b in s.split(",")],
                        default=[50000.0] * 5,
                        help="Comma-separated per-agent budgets (matches src5).")
    parser.add_argument("--click-reward-scale", type=float, default=None,
                        help="Override click reward scale. Default None = derive "
                             "from data (mean_market_price / target_CTR_design), "
                             "same as src5 R9g.")
    parser.add_argument("--episodes", type=int, default=DEFAULTS["episodes"])
    parser.add_argument("--slot-size", type=int, default=DEFAULTS["slot_size"])
    parser.add_argument("--episode-rows", type=int, default=DEFAULTS["episode_rows"])
    parser.add_argument("--n-threshold", type=int, default=DEFAULTS["n_threshold"])
    parser.add_argument("--n-residual", type=int, default=DEFAULTS["n_residual"])
    parser.add_argument("--state-dim", type=int, default=DEFAULTS["state_dim"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr_actor"])
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--gae-lambda", type=float, default=DEFAULTS["gae_lambda"])
    parser.add_argument("--clip-eps", type=float, default=DEFAULTS["clip_eps"])
    parser.add_argument("--ppo-epochs", type=int, default=DEFAULTS["ppo_epochs"])
    parser.add_argument("--entropy-beta", type=float, default=DEFAULTS["entropy_beta"])
    parser.add_argument("--critic-weight", type=float, default=DEFAULTS["critic_weight"])
    parser.add_argument("--clip-grad", type=float, default=DEFAULTS["clip_grad"])
    parser.add_argument("--target-kl", type=float, default=DEFAULTS["target_kl"])
    args = parser.parse_args()

    # Stash snapshot episodes as an attribute (not from CLI)
    args.snapshot_episodes = DEFAULTS["snapshot_episodes"]

    for seed in args.seeds:
        train_one_seed(seed, args)

    print("\n[Step 3/3] DONE")
    print(f"  Outputs: src6/outputs/mappo/")


if __name__ == "__main__":
    main()
