"""
src6/evaluate_all.py
====================

Unified multi-seed, multi-policy evaluation.

USAGE (from project root):
    cd D:\\Research Methodology\\DAAWBC\\dynamic_ad_allocation
    python -m src6.evaluate_all

    # Common options:
    python -m src6.evaluate_all --seeds 0,1,2,3,4
    python -m src6.evaluate_all --n-episodes 20
    python -m src6.evaluate_all --skip-mappo
    python -m src6.evaluate_all --skip-ac

POLICIES EVALUATED (matching Cai et al. 2017 "An Effective Budget
Management Framework for Real-Time Bidding in Online Advertising"):

  STATELESS BASELINES (no pacing, no controller, paper-faithful):
    1. const_baseline   bid = C              (Const, C = mean market_price_train)
    2. rand_baseline    bid ~ U(0, max_mp)   (Rand, max_mp from training data)
    3. pctr_baseline    bid = alpha * pCTR   (Lin, alpha calibrated below)

  RL POLICIES (env-based, lambda controller intact — part of what makes
  them work, comparison is "static rule vs RL with adaptive pacing"):
    4. src5_ac          trained src5 independent actor-critic
    5. src6_mappo       trained src6 MAPPO (decentralized actors at eval)

All three baselines run in STANDALONE NumPy loops that replicate the env's
2nd-price auction logic exactly, without using ContextRTBEnvironment. This
guarantees:
  - src5 code is bit-for-bit untouched
  - baselines cannot accidentally inherit env machinery (lambda, etc.)
  - the formulas are visible plain code an examiner can read

OUTPUTS (in <project_root>/outputs/eval_all/):
  - eval_all_per_seed.csv        long form: (seed, policy) rows
  - eval_all_per_seed_table.csv  wide form: per seed, rows=ep, cols=policies
  - eval_all_aggregate.csv       one row per policy, across-seed means
  - eval_all_paired.csv          per-seed paired comparisons + significance

Per-seed × per-policy wide tables (rows = ep0..ep_{N-1} + mean) are also
printed to the console, one per seed, plus an aggregate table.
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src5.context_propensity import ContextPropensityModel  # noqa: E402
from src5.simulator.context_environment import (  # noqa: E402
    ContextRTBEnvironment, DEFAULT_THRESHOLD_GRID, DEFAULT_RESIDUAL_GRID, ADV_IDS,
)
from src5.policy_network import StrategicActorCritic  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESERVE_PRICE = 1.0


# ======================================================================
# Stateless baselines — standalone, NO env, NO lambda controller
# ======================================================================
def calibrate_const(df_train):
    """Const(C): C = mean(market_price) from training split."""
    return float(df_train["market_price"].mean())


def calibrate_rand_max(df_train):
    """Rand: bid ~ Uniform(0, max_mp). max_mp from training split."""
    return float(df_train["market_price"].max())


def calibrate_alpha(df_train, adv_ids):
    """Lin(alpha): alpha = mean(mp) / mean(pctr_raw) from training split."""
    mean_mp = float(df_train["market_price"].mean())
    raw_cols = [f"pctr_raw_{adv}" for adv in adv_ids]
    mean_pctr = float(np.mean([df_train[c].mean() for c in raw_cols]))
    if mean_pctr < 1e-9:
        raise RuntimeError("Mean pctr_raw is ~0; alpha calibration impossible")
    return mean_mp / mean_pctr


def _run_stateless_baseline(
    bid_fn, df_eval, adv_ids, budgets, n_episodes, episode_rows,
    seed_for_random=None, reserve_price=RESERVE_PRICE,
):
    """Run one or more episodes of a stateless baseline.

    bid_fn(t, mp, pctr_row_vec, rng) -> np.array of shape (n_agents,) with
    per-agent bids for impression t. rng is a numpy RandomState for any
    stochastic baselines (Rand). For deterministic ones (Const, Lin), pass
    seed_for_random=None.

    Replicates env's step() auction logic: highest bid wins, pays max(2nd
    bid, market_price, reserve), respects budget hard guard, reads click
    from file (no resampling).
    """
    n_agents = len(budgets)
    available = max(1, len(df_eval) - episode_rows)
    stride = 0 if n_episodes <= 1 else available // (n_episodes - 1)

    mp_all = df_eval["market_price"].values.astype(np.float32)
    click_all = df_eval["click"].values.astype(np.int32)
    pctr_all = np.stack(
        [df_eval[f"pctr_raw_{adv}"].values.astype(np.float32) for adv in adv_ids],
        axis=1,
    )

    results = []
    for ep in range(n_episodes):
        start = ep * stride
        if start + episode_rows > len(df_eval):
            start = len(df_eval) - episode_rows
        start = max(0, start)
        end = min(start + episode_rows, len(df_eval))

        rng = np.random.default_rng(seed_for_random + ep) if seed_for_random is not None else None

        remaining_budget = np.array(budgets, dtype=np.float32)
        clicks = np.zeros(n_agents, dtype=np.int64)
        wins = np.zeros(n_agents, dtype=np.int64)
        cost = np.zeros(n_agents, dtype=np.float64)

        for t in range(start, end):
            mp = float(mp_all[t])
            click = int(click_all[t])
            pctr_row = pctr_all[t]                       # shape (n_agents,)

            bids = bid_fn(t, mp, pctr_row, rng)          # (n_agents,)
            bids = np.where(bids > remaining_budget, 0.0, bids)
            eligible_mask = bids >= reserve_price
            if not eligible_mask.any():
                continue

            eligible_idx = np.where(eligible_mask)[0]
            sorted_idx = eligible_idx[np.argsort(-bids[eligible_idx])]
            top1 = int(sorted_idx[0])
            top1_bid = float(bids[top1])

            if len(sorted_idx) >= 2:
                top2_bid = float(bids[sorted_idx[1]])
                payable = max(top2_bid, mp, reserve_price)
            else:
                payable = max(mp, reserve_price)

            if top1_bid >= payable:
                price_paid = min(payable, float(remaining_budget[top1]))
                remaining_budget[top1] -= price_paid
                cost[top1] += price_paid
                wins[top1] += 1
                clicks[top1] += click

        util_pct = (cost / np.array(budgets, dtype=np.float32)) * 100.0
        results.append({
            "episode": ep,
            "start_row": start,
            "clicks": clicks.tolist(),
            "wins": wins.tolist(),
            "cost": cost.tolist(),
            "utilization_pct": util_pct.tolist(),
            "total_clicks": int(clicks.sum()),
            "total_wins": int(wins.sum()),
            "total_cost": float(cost.sum()),
        })
    return results


# ======================================================================
# Env-based rollout (for AC / MAPPO)
# ======================================================================
def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


def rollout(env: ContextRTBEnvironment, action_picker, n_episodes: int):
    episode_rows = env.episode_rows
    available = max(1, env.n_rows - episode_rows)
    stride = 0 if n_episodes <= 1 else available // (n_episodes - 1)

    results = []
    for ep in range(n_episodes):
        start = ep * stride
        if start + episode_rows > env.n_rows:
            start = env.n_rows - episode_rows
        start = max(0, start)

        env.reset(start_row=start)
        done = False
        states_t = [to_tensor(s) for s in env._get_state()]
        while not done:
            th_a, res_a = action_picker(states_t)
            next_states, _, done, _ = env.step(th_a, res_a)
            if not done:
                states_t = [to_tensor(s) for s in next_states]
        d = env.diagnostics()
        d["episode"] = ep
        d["start_row"] = start
        d["total_clicks"] = int(np.sum(d["clicks"]))
        d["total_wins"] = int(np.sum(d["wins"]))
        d["total_cost"] = float(np.sum(d["cost"]))
        results.append(d)
    return results


def make_trained_ac_picker(agents):
    def pick(states_t):
        th_actions, res_actions = [], []
        for i, agent in enumerate(agents):
            with torch.no_grad():
                lt, lr_, _ = agent(states_t[i])
                p_th = F.softmax(lt.squeeze(0), dim=-1)
                p_res = F.softmax(lr_.squeeze(0), dim=-1)
                th_actions.append(int(torch.argmax(p_th).item()))
                res_actions.append(int(torch.argmax(p_res).item()))
        return th_actions, res_actions
    return pick


def make_trained_mappo_picker(actors):
    def pick(states_t):
        th_actions, res_actions = [], []
        for i, actor in enumerate(actors):
            with torch.no_grad():
                th_logits, res_logits = actor(states_t[i].unsqueeze(0))
                th_actions.append(int(th_logits.argmax(dim=-1).item()))
                res_actions.append(int(res_logits.argmax(dim=-1).item()))
        return th_actions, res_actions
    return pick


# ======================================================================
# Statistics
# ======================================================================
def paired_t_test(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan"), 0, "n<2"
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    if sd < 1e-9:
        return float("inf"), n - 1, "p<0.001"
    se = sd / math.sqrt(n)
    t = mean / se
    df = n - 1
    crit = {1: (12.71, 63.66), 2: (4.303, 9.925), 3: (3.182, 5.841),
            4: (2.776, 4.604), 5: (2.571, 4.032), 6: (2.447, 3.707),
            7: (2.365, 3.499), 8: (2.306, 3.355), 9: (2.262, 3.250)}
    c05, c01 = crit.get(df, (1.96, 2.576))
    a = abs(t)
    if a > c01:
        p = "p < 0.01"
    elif a > c05:
        p = "p < 0.05"
    elif a > c05 * 0.85:
        p = "p < 0.10"
    else:
        p = "p > 0.10"
    return t, df, p


def sign_test_p(n_positive, n_total):
    from math import comb
    k = max(n_positive, n_total - n_positive)
    p_one = sum(comb(n_total, i) for i in range(k, n_total + 1)) / (2 ** n_total)
    return min(1.0, 2 * p_one)


def print_per_seed_table(seed, ep_clicks_by_policy, policy_order):
    """Print a per-seed wide table: rows = ep0..ep_{N-1} + mean, cols = policies."""
    available_policies = [p for p in policy_order if p in ep_clicks_by_policy
                          and ep_clicks_by_policy[p] is not None]
    if not available_policies:
        return
    n_eps = len(next(iter(ep_clicks_by_policy[p] for p in available_policies)))
    # Trim policy names to 14 chars for header readability
    short_names = {p: p[:14] for p in available_policies}
    col_w = max(12, max(len(short_names[p]) for p in available_policies))

    print(f"\n  --- Seed {seed} per-episode clicks ---")
    header = f"  {'ep':>4} | " + " | ".join(f"{short_names[p]:>{col_w}}" for p in available_policies)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ep in range(n_eps):
        row = f"  {ep:>4} | " + " | ".join(
            f"{ep_clicks_by_policy[p][ep]:>{col_w}.1f}" for p in available_policies
        )
        print(row)
    mean_row = f"  {'mean':>4} | " + " | ".join(
        f"{np.mean(ep_clicks_by_policy[p]):>{col_w}.2f}" for p in available_policies
    )
    print("  " + "-" * (len(header) - 2))
    print(mean_row)


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified multi-seed, multi-policy evaluation (Cai et al. 2017 baselines)."
    )
    parser.add_argument("--data", default="data_2/shared_auction_log_v4_dense.txt")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--episode-rows", type=int, default=125_000)
    parser.add_argument("--budgets", default="50000,50000,50000,50000,50000")
    parser.add_argument("--ac-model-dir", default="src5/outputs/context_ac")
    parser.add_argument("--mappo-model-dir", default="src6/outputs/mappo/models")
    parser.add_argument("--skip-ac", action="store_true")
    parser.add_argument("--skip-mappo", action="store_true")
    parser.add_argument("--output-dir", default="outputs/eval_all")
    parser.add_argument("--reference", default="pctr_baseline",
                        choices=["pctr_baseline", "const_baseline", "rand_baseline"],
                        help="Which baseline to use as reference for lift% column")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    budgets = [float(b) for b in args.budgets.split(",")]
    ac_dir = Path(args.ac_model_dir).resolve()
    mappo_dir = Path(args.mappo_model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("UNIFIED EVALUATION (src6/evaluate_all.py)")
    print("Const + Rand + Lin baselines (Cai et al. 2017) + src5 AC + src6 MAPPO")
    print("=" * 78)
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Seeds         : {seeds}")
    print(f"n_episodes    : {args.n_episodes} per (seed × policy)")
    print(f"Data          : {args.data}")
    print(f"AC models     : {ac_dir}")
    print(f"MAPPO models  : {mappo_dir}")
    print(f"Output dir    : {out_dir}")
    print(f"Reference     : {args.reference}")
    print()

    # ------------------------------------------------------------------
    # 1. Read data + fit propensity
    # ------------------------------------------------------------------
    print("[1/5] Reading data + fitting context-propensity model")
    df = pd.read_csv(args.data, sep="\t")
    propensity = ContextPropensityModel()
    propensity.fit(df, train_weekdays=(3, 4), eval_weekday=5, verbose=True)
    df_with_p = propensity.attach_to_dataframe(df)
    df_train = df_with_p[df_with_p["weekday"].isin([3, 4])].reset_index(drop=True)
    df_eval = df_with_p[df_with_p["weekday"].eq(5)].reset_index(drop=True)
    print(f"\n  Train rows (weekday 3,4): {len(df_train):,}")
    print(f"  Eval set (weekday 5)    : {len(df_eval):,} rows, "
          f"{int(df_eval['click'].sum()):,} clicks "
          f"(CTR {df_eval['click'].mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Calibrate baselines (training data only)
    # ------------------------------------------------------------------
    print("\n[2/5] Calibrating baseline constants from training data")
    C = calibrate_const(df_train)
    rand_max = calibrate_rand_max(df_train)
    try:
        alpha = calibrate_alpha(df_train, ADV_IDS)
        pctr_available = True
    except (KeyError, RuntimeError) as e:
        print(f"  [WARN] pctr_baseline unavailable: {e}")
        alpha = None
        pctr_available = False

    print(f"  Const(C)        : bid = {C:.2f}  (mean market_price_train)")
    print(f"  Rand(max_mp)    : bid ~ U(0, {rand_max:.2f})  (max market_price_train)")
    if pctr_available:
        print(f"  Lin(alpha)      : bid = {alpha:.2f} * pctr_raw_{{adv_i}}")
    print("  (All three baselines: NO lambda, NO pacing, NO threshold, NO residual)")

    # ------------------------------------------------------------------
    # 3. Build env (for AC / MAPPO only)
    # ------------------------------------------------------------------
    print("\n[3/5] Building shared evaluation env (for AC / MAPPO)")
    env = ContextRTBEnvironment(
        df=df_eval, budgets=budgets,
        episode_rows=args.episode_rows, verbose=True,
    )
    n_agents = len(budgets)
    n_th = len(DEFAULT_THRESHOLD_GRID)
    n_res = len(DEFAULT_RESIDUAL_GRID)

    # ------------------------------------------------------------------
    # 4. Run stateless baselines (deterministic = single run; Rand = per-seed)
    # ------------------------------------------------------------------
    print("\n[4/5] Running stateless baselines")
    print("-" * 60)

    # Const: deterministic, single run
    print(f"  const_baseline (deterministic, single run)")
    const_bid_fn = lambda t, mp, pctr_row, rng: np.full(n_agents, C, dtype=np.float32)
    const_res = _run_stateless_baseline(
        const_bid_fn, df_eval, ADV_IDS, budgets,
        n_episodes=args.n_episodes, episode_rows=args.episode_rows,
    )
    const_clicks = np.array([r["total_clicks"] for r in const_res], dtype=np.float64)
    const_util = np.array([np.mean(r["utilization_pct"]) for r in const_res])
    print(f"    mean clicks: {const_clicks.mean():.2f} ± {const_clicks.std():.2f}  "
          f"util {const_util.mean():.1f}%")

    # Lin(alpha): deterministic, single run
    if pctr_available:
        print(f"  pctr_baseline (Lin alpha, deterministic, single run)")
        pctr_bid_fn = lambda t, mp, pctr_row, rng: alpha * pctr_row
        pctr_res = _run_stateless_baseline(
            pctr_bid_fn, df_eval, ADV_IDS, budgets,
            n_episodes=args.n_episodes, episode_rows=args.episode_rows,
        )
        pctr_clicks = np.array([r["total_clicks"] for r in pctr_res], dtype=np.float64)
        pctr_util = np.array([np.mean(r["utilization_pct"]) for r in pctr_res])
        print(f"    mean clicks: {pctr_clicks.mean():.2f} ± {pctr_clicks.std():.2f}  "
              f"util {pctr_util.mean():.1f}%")
    else:
        pctr_clicks = np.full(args.n_episodes, float("nan"))

    # Rand: stochastic, per-seed (different draws per seed)

    # ------------------------------------------------------------------
    # 5. Per-seed loop: rand_baseline, src5_ac, src6_mappo
    # ------------------------------------------------------------------
    print("\n[5/5] Running per-seed evaluations")
    print("-" * 60)

    per_seed_rows = []
    per_seed_clicks = {
        "const_baseline": {s: const_clicks.tolist() for s in seeds},
        "rand_baseline": {},
        "pctr_baseline": {s: pctr_clicks.tolist() for s in seeds} if pctr_available else {},
        "src5_ac": {},
        "src6_mappo": {},
    }
    # Diagnostics
    per_seed_util = {
        "const_baseline": {s: const_util.tolist() for s in seeds},
        "rand_baseline": {},
        "pctr_baseline": {s: pctr_util.tolist() for s in seeds} if pctr_available else {},
        "src5_ac": {},
        "src6_mappo": {},
    }
    per_seed_cost = {
        "const_baseline": {s: [np.sum(r["cost"]) for r in const_res] for s in seeds},
        "rand_baseline": {},
        "pctr_baseline": {s: [np.sum(r["cost"]) for r in pctr_res] for s in seeds} if pctr_available else {},
        "src5_ac": {},
        "src6_mappo": {},
    }

    policy_order_print = ["const_baseline", "rand_baseline", "pctr_baseline",
                          "src5_ac", "src6_mappo"]

    for seed in seeds:
        print(f"\n  === SEED {seed} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # const replicated for paired test convenience
        const_mean = float(np.mean(const_clicks))
        const_std = float(np.std(const_clicks))
        per_seed_rows.append({
            "seed": seed, "policy": "const_baseline",
            "mean_clicks": const_mean, "std_clicks": const_std,
            "mean_cost": float(np.mean([np.sum(r["cost"]) for r in const_res])),
            "mean_util_pct": float(const_util.mean()),
            "n_episodes": args.n_episodes,
        })

        # rand: per-seed draws
        print(f"    rand_baseline ...")
        rand_bid_fn = lambda t, mp, pctr_row, rng: rng.uniform(0.0, rand_max, size=n_agents).astype(np.float32)
        rand_res = _run_stateless_baseline(
            rand_bid_fn, df_eval, ADV_IDS, budgets,
            n_episodes=args.n_episodes, episode_rows=args.episode_rows,
            seed_for_random=seed * 1000,
        )
        rand_clicks = np.array([r["total_clicks"] for r in rand_res], dtype=np.float64)
        rand_util = np.array([np.mean(r["utilization_pct"]) for r in rand_res])
        per_seed_clicks["rand_baseline"][seed] = rand_clicks.tolist()
        per_seed_util["rand_baseline"][seed] = rand_util.tolist()
        per_seed_cost["rand_baseline"][seed] = [np.sum(r["cost"]) for r in rand_res]
        per_seed_rows.append({
            "seed": seed, "policy": "rand_baseline",
            "mean_clicks": float(rand_clicks.mean()),
            "std_clicks": float(rand_clicks.std()),
            "mean_cost": float(np.mean([np.sum(r["cost"]) for r in rand_res])),
            "mean_util_pct": float(rand_util.mean()),
            "n_episodes": args.n_episodes,
        })
        print(f"      mean clicks: {rand_clicks.mean():.2f} ± {rand_clicks.std():.2f}")

        # pctr replicated for paired test
        if pctr_available:
            pctr_mean_v = float(np.mean(pctr_clicks))
            pctr_std_v = float(np.std(pctr_clicks))
            per_seed_rows.append({
                "seed": seed, "policy": "pctr_baseline",
                "mean_clicks": pctr_mean_v, "std_clicks": pctr_std_v,
                "mean_cost": float(np.mean([np.sum(r["cost"]) for r in pctr_res])),
                "mean_util_pct": float(pctr_util.mean()),
                "n_episodes": args.n_episodes,
            })

        # src5 AC
        if not args.skip_ac:
            ac_present = all(
                (ac_dir / f"context_ac_seed_{seed}_agent_{adv}.pt").exists()
                for adv in ADV_IDS
            )
            if ac_present:
                print(f"    src5_ac ...")
                agents = []
                for adv in ADV_IDS:
                    a = StrategicActorCritic(
                        input_dim=4, hidden_dim=128,
                        n_threshold=n_th, n_residual=n_res,
                    ).to(DEVICE)
                    a.load_state_dict(torch.load(
                        ac_dir / f"context_ac_seed_{seed}_agent_{adv}.pt",
                        map_location=DEVICE, weights_only=True,
                    ))
                    a.eval()
                    agents.append(a)
                ac_picker = make_trained_ac_picker(agents)
                ac_res = rollout(env, ac_picker, args.n_episodes)
                ac_clicks = np.array([r["total_clicks"] for r in ac_res], dtype=np.float64)
                ac_util = np.array([np.mean(r["utilization_pct"]) for r in ac_res])
                per_seed_clicks["src5_ac"][seed] = ac_clicks.tolist()
                per_seed_util["src5_ac"][seed] = ac_util.tolist()
                per_seed_cost["src5_ac"][seed] = [r["total_cost"] for r in ac_res]
                per_seed_rows.append({
                    "seed": seed, "policy": "src5_ac",
                    "mean_clicks": float(ac_clicks.mean()),
                    "std_clicks": float(ac_clicks.std()),
                    "mean_cost": float(np.mean([r["total_cost"] for r in ac_res])),
                    "mean_util_pct": float(ac_util.mean()),
                    "n_episodes": args.n_episodes,
                })
                if pctr_available:
                    lift_pct = (ac_clicks.mean() - pctr_clicks.mean()) / pctr_clicks.mean() * 100
                    print(f"      mean clicks: {ac_clicks.mean():.2f} ± {ac_clicks.std():.2f}  "
                          f"({lift_pct:+.2f}% vs Lin)")
                else:
                    print(f"      mean clicks: {ac_clicks.mean():.2f} ± {ac_clicks.std():.2f}")
            else:
                print(f"    src5_ac models not found for seed {seed} — skipping")

        # src6 MAPPO
        if not args.skip_mappo:
            try:
                from src6.policy_network_mappo import MAPPOActor
            except ImportError:
                print("    src6.policy_network_mappo not importable — skipping MAPPO")
            else:
                mappo_present = all(
                    (mappo_dir / f"mappo_seed_{seed}_actor_{adv}.pt").exists()
                    for adv in ADV_IDS
                )
                if mappo_present:
                    print(f"    src6_mappo ...")
                    actors = []
                    for adv in ADV_IDS:
                        a = MAPPOActor(
                            state_dim=4, hidden_dim=128,
                            n_threshold=n_th, n_residual=n_res,
                        ).to(DEVICE)
                        a.load_state_dict(torch.load(
                            mappo_dir / f"mappo_seed_{seed}_actor_{adv}.pt",
                            map_location=DEVICE, weights_only=True,
                        ))
                        a.eval()
                        actors.append(a)
                    mappo_picker = make_trained_mappo_picker(actors)
                    m_res = rollout(env, mappo_picker, args.n_episodes)
                    m_clicks = np.array([r["total_clicks"] for r in m_res], dtype=np.float64)
                    m_util = np.array([np.mean(r["utilization_pct"]) for r in m_res])
                    per_seed_clicks["src6_mappo"][seed] = m_clicks.tolist()
                    per_seed_util["src6_mappo"][seed] = m_util.tolist()
                    per_seed_cost["src6_mappo"][seed] = [r["total_cost"] for r in m_res]
                    per_seed_rows.append({
                        "seed": seed, "policy": "src6_mappo",
                        "mean_clicks": float(m_clicks.mean()),
                        "std_clicks": float(m_clicks.std()),
                        "mean_cost": float(np.mean([r["total_cost"] for r in m_res])),
                        "mean_util_pct": float(m_util.mean()),
                        "n_episodes": args.n_episodes,
                    })
                    if pctr_available:
                        lift_pct = (m_clicks.mean() - pctr_clicks.mean()) / pctr_clicks.mean() * 100
                        print(f"      mean clicks: {m_clicks.mean():.2f} ± {m_clicks.std():.2f}  "
                              f"({lift_pct:+.2f}% vs Lin)")
                    else:
                        print(f"      mean clicks: {m_clicks.mean():.2f} ± {m_clicks.std():.2f}")
                else:
                    print(f"    src6_mappo models not found for seed {seed} — skipping")

        # Per-seed wide table for THIS seed
        ep_clicks_by_policy = {
            "const_baseline": const_clicks.tolist(),
            "rand_baseline": rand_clicks.tolist(),
        }
        if pctr_available:
            ep_clicks_by_policy["pctr_baseline"] = pctr_clicks.tolist()
        if seed in per_seed_clicks["src5_ac"]:
            ep_clicks_by_policy["src5_ac"] = per_seed_clicks["src5_ac"][seed]
        if seed in per_seed_clicks["src6_mappo"]:
            ep_clicks_by_policy["src6_mappo"] = per_seed_clicks["src6_mappo"][seed]
        print_per_seed_table(seed, ep_clicks_by_policy, policy_order_print)

    # ------------------------------------------------------------------
    # 6. Aggregate
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("AGGREGATE RESULTS (mean across seeds)")
    print("=" * 78)
    ref_label = {"pctr_baseline": "vs Lin", "const_baseline": "vs Const",
                 "rand_baseline": "vs Rand"}[args.reference]
    print(f"{'Policy':<20} {'Across-seed mean':>17} {'SD of means':>13} "
          f"{ref_label:>11} {'n seeds':>8}")
    print("-" * 78)

    aggregate_rows = []
    ref_clicks_per_seed = per_seed_clicks.get(args.reference, {})
    if ref_clicks_per_seed:
        ref_seed_means = [np.mean(ref_clicks_per_seed[s]) for s in seeds if s in ref_clicks_per_seed]
        ref_across = float(np.mean(ref_seed_means)) if ref_seed_means else float("nan")
    else:
        ref_across = float("nan")

    for policy in policy_order_print:
        seed_means = []
        for s in seeds:
            clicks_list = per_seed_clicks.get(policy, {}).get(s)
            if clicks_list:
                seed_means.append(float(np.mean(clicks_list)))
        if not seed_means:
            continue
        n_s = len(seed_means)
        across_mean = float(np.mean(seed_means))
        across_sd = float(np.std(seed_means, ddof=1)) if n_s > 1 else 0.0
        vs_ref = (across_mean - ref_across) / ref_across * 100 if not math.isnan(ref_across) else float("nan")
        vs_ref_str = f"{vs_ref:>10.2f}%" if not math.isnan(vs_ref) else "     n/a"
        print(f"{policy:<20} {across_mean:>17.2f} {across_sd:>13.2f} "
              f"{vs_ref_str:>11} {n_s:>8d}")
        aggregate_rows.append({
            "policy": policy,
            "across_seed_mean_clicks": across_mean,
            "across_seed_sd_of_means": across_sd,
            "across_seed_se_of_mean": across_sd / math.sqrt(n_s) if n_s > 1 else 0.0,
            f"lift_vs_{args.reference}_pct": vs_ref,
            "n_seeds": n_s,
            "n_episodes_per_seed": args.n_episodes,
        })

    # ------------------------------------------------------------------
    # 7. Paired comparisons
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS (per-seed means, paired by seed index)")
    print("=" * 78)

    paired_rows = []

    def collect_pair(arm_a, arm_b):
        common = sorted(set(per_seed_clicks.get(arm_a, {}).keys()) &
                        set(per_seed_clicks.get(arm_b, {}).keys()))
        diffs = []
        for s in common:
            ma = float(np.mean(per_seed_clicks[arm_a][s]))
            mb = float(np.mean(per_seed_clicks[arm_b][s]))
            diffs.append(ma - mb)
        return common, diffs

    def report_pair(name, arm_a, arm_b):
        common, diffs = collect_pair(arm_a, arm_b)
        if len(diffs) < 2:
            return
        n_pos = sum(1 for d in diffs if d > 0)
        mean_d = sum(diffs) / len(diffs)
        sd_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (len(diffs) - 1))
        t, df, p = paired_t_test(diffs)
        sign_p = sign_test_p(n_pos, len(diffs))
        print(f"\n  {name}  (n={len(common)} paired seeds)")
        print(f"    seeds: {common}")
        print(f"    diffs: {[f'{d:+.2f}' for d in diffs]}")
        print(f"    mean diff: {mean_d:+.2f} clicks  (sd={sd_d:.2f})")
        print(f"    paired t-test: t={t:+.3f}, df={df}, {p}")
        print(f"    sign test: {n_pos}/{len(diffs)} positive, p≈{sign_p:.4f}")
        paired_rows.append({
            "comparison": name,
            "arm_a": arm_a, "arm_b": arm_b,
            "n_paired_seeds": len(common),
            "mean_diff_clicks": mean_d,
            "sd_diff": sd_d,
            "paired_t": t, "df": df, "p_t_label": p,
            "n_positive": n_pos,
            "sign_test_p": sign_p,
            "per_seed_diffs": ";".join(f"{d:+.2f}" for d in diffs),
            "seeds_used": ";".join(str(s) for s in common),
        })

    # AC and MAPPO vs each baseline + against each other
    if per_seed_clicks["src5_ac"]:
        if pctr_available:
            report_pair("src5_ac vs pctr_baseline (Lin)", "src5_ac", "pctr_baseline")
        report_pair("src5_ac vs const_baseline (Const)", "src5_ac", "const_baseline")
        report_pair("src5_ac vs rand_baseline (Rand)", "src5_ac", "rand_baseline")
    if per_seed_clicks["src6_mappo"]:
        if pctr_available:
            report_pair("src6_mappo vs pctr_baseline (Lin)", "src6_mappo", "pctr_baseline")
        report_pair("src6_mappo vs const_baseline (Const)", "src6_mappo", "const_baseline")
        report_pair("src6_mappo vs rand_baseline (Rand)", "src6_mappo", "rand_baseline")
    if per_seed_clicks["src5_ac"] and per_seed_clicks["src6_mappo"]:
        report_pair("src6_mappo vs src5_ac", "src6_mappo", "src5_ac")

    # ------------------------------------------------------------------
    # 8. Write CSVs (long-form, wide-form, aggregate, paired)
    # ------------------------------------------------------------------
    per_seed_path = out_dir / "eval_all_per_seed.csv"
    with open(per_seed_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_seed_rows[0].keys())
        w.writeheader()
        for r in per_seed_rows:
            w.writerow(r)

    # Wide-form per-seed table: one row per (seed, ep), columns = policies
    wide_path = out_dir / "eval_all_per_seed_table.csv"
    wide_cols = ["seed", "ep"] + [p for p in policy_order_print if per_seed_clicks.get(p)]
    with open(wide_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(wide_cols)
        for seed in seeds:
            for ep in range(args.n_episodes):
                row = [seed, ep]
                for p in policy_order_print:
                    if not per_seed_clicks.get(p):
                        continue
                    clicks_list = per_seed_clicks[p].get(seed)
                    val = clicks_list[ep] if clicks_list and ep < len(clicks_list) else ""
                    row.append(val)
                w.writerow(row)
            # Mean row
            row = [seed, "mean"]
            for p in policy_order_print:
                if not per_seed_clicks.get(p):
                    continue
                clicks_list = per_seed_clicks[p].get(seed)
                val = float(np.mean(clicks_list)) if clicks_list else ""
                row.append(val)
            w.writerow(row)
        # Aggregate row across all seeds
        row = ["all", "mean"]
        for p in policy_order_print:
            if not per_seed_clicks.get(p):
                continue
            all_seed_means = []
            for s in seeds:
                cl = per_seed_clicks[p].get(s)
                if cl:
                    all_seed_means.append(float(np.mean(cl)))
            val = float(np.mean(all_seed_means)) if all_seed_means else ""
            row.append(val)
        w.writerow(row)

    aggregate_path = out_dir / "eval_all_aggregate.csv"
    with open(aggregate_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
        w.writeheader()
        for r in aggregate_rows:
            w.writerow(r)

    if paired_rows:
        paired_path = out_dir / "eval_all_paired.csv"
        with open(paired_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=paired_rows[0].keys())
            w.writeheader()
            for r in paired_rows:
                w.writerow(r)
        print(f"\n  Wrote paired CSV   : {paired_path}")

    print("\n" + "=" * 78)
    print(f"  Wrote per-seed CSV     : {per_seed_path}")
    print(f"  Wrote wide-form CSV    : {wide_path}")
    print(f"  Wrote aggregate CSV    : {aggregate_path}")
    print("=" * 78)
    print("\nDone.")


if __name__ == "__main__":
    main()
