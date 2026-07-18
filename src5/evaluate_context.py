"""
src5/evaluate_context.py
========================

Held-out evaluation on weekday 5.

Compares three policies:
  1. Linear-propensity baseline (threshold=0, residual=neutral) — what the
     fitted propensity signal alone delivers, no RL choice.
  2. Uniform-random baseline — sanity check; if the trained policy can't
     beat this, something is broken.
  3. Trained context actor-critic — loaded from the trained model files.

Output: prints comparison table + writes eval_results.csv.
"""
from __future__ import annotations
import argparse
import csv
import sys
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


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


# ----------------------------------------------------------------------
def rollout(env: ContextRTBEnvironment, action_picker, n_episodes: int, slot_size: int = 5000):
    """Runs n_episodes rollouts with DETERMINISTIC stride-based starts.

    Per Round-9 review §3: eval must NOT use random starts, otherwise the
    baseline-vs-RL comparison compares different row windows. We tile the
    eval data with evenly-spaced episode starts, then run each
    deterministically. Both baseline and RL evaluation use the SAME starts.

    R9g fix: eval rollouts use env.step_slot to match the training-time
    decision frequency. Per-impression eval (slot_size=0) is OOD relative
    to a slot-trained policy and was producing distribution-shift artifacts.
    """
    # Stride: evenly divide eval data into n_episodes windows
    episode_rows = env.episode_rows
    available = max(1, env.n_rows - episode_rows)
    if n_episodes <= 1:
        stride = 0
    else:
        stride = available // (n_episodes - 1)

    results = []
    for ep in range(n_episodes):
        start = ep * stride
        # Ensure we don't run past data
        if start + episode_rows > env.n_rows:
            start = env.n_rows - episode_rows
        if start < 0:
            start = 0

        env.reset(start_row=start)
        done = False
        states_t = [to_tensor(s) for s in env._get_state()]
        while not done:
            th_a, res_a = action_picker(states_t)
            # R9g: match training decision frequency (slot vs per-impression)
            if slot_size > 0:
                next_states, _, done, _ = env.step_slot(th_a, res_a, slot_size)
            else:
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


# --- Three action pickers --------------------------------------------
def make_linear_baseline_picker(n_agents: int, n_threshold: int, n_residual: int):
    """Always picks (threshold=0=never skip, residual=middle=neutral)."""
    neutral_res = n_residual // 2  # bin 5 of 11 = +0.0 residual

    def pick(states_t):
        return [0] * n_agents, [neutral_res] * n_agents

    return pick


def make_uniform_picker(n_agents: int, n_threshold: int, n_residual: int):
    """Uniform random over both action heads."""
    def pick(states_t):
        th = [int(np.random.randint(n_threshold)) for _ in range(n_agents)]
        res = [int(np.random.randint(n_residual)) for _ in range(n_agents)]
        return th, res

    return pick


def make_trained_picker(agents):
    """Sample actions from trained policies."""
    def pick(states_t):
        th_actions = []
        res_actions = []
        for i, agent in enumerate(agents):
            with torch.no_grad():
                lt, lr_, _ = agent(states_t[i])
                p_th = F.softmax(lt.squeeze(0), dim=-1)
                p_res = F.softmax(lr_.squeeze(0), dim=-1)
                # Use argmax for evaluation (deterministic)
                th_actions.append(int(torch.argmax(p_th).item()))
                res_actions.append(int(torch.argmax(p_res).item()))
        return th_actions, res_actions

    return pick


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_2/shared_auction_log_v4_dense.txt")
    parser.add_argument("--model-dir", default="src5/outputs/context_ac",
                        help="Directory containing context_ac_seed_*_agent_*.pt files")
    parser.add_argument("--seed", type=int, default=0,
                        help="Which seed's trained model to load")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--episode-rows", type=int, default=125_000)
    parser.add_argument("--slot-size", type=int, default=5000,
                        help="R9g: must match training slot-size. 0 = per-impression (only for "
                             "evaluating policies trained without the wrapper).")
    parser.add_argument("--budgets", type=str,
                        default="50000,50000,50000,50000,50000")
    parser.add_argument("--output", default="src5/outputs/context_ac/eval_results.csv")
    args = parser.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    model_dir = Path(args.model_dir).resolve()

    # --- 1. Refit propensity (must match training-time fit exactly) ---
    print("[1/4] Refitting context-propensity model")
    df = pd.read_csv(args.data, sep="\t")
    model = ContextPropensityModel()
    model.fit(df, train_weekdays=(3, 4), eval_weekday=5, verbose=True)
    df = model.attach_to_dataframe(df)

    # --- 2. Filter to weekday 5 (held-out) ---
    df_eval = df[df["weekday"].eq(5)].reset_index(drop=True)
    print(f"\n[2/4] Eval set (weekday 5): {len(df_eval):,} rows, "
          f"{int(df_eval['click'].sum()):,} clicks "
          f"(CTR {df_eval['click'].mean()*100:.2f}%)")

    # --- 3. Build env (reused across all evaluations) ---
    print("\n[3/4] Building evaluation env")
    env = ContextRTBEnvironment(
        df=df_eval,
        budgets=budgets,
        episode_rows=args.episode_rows,
        verbose=True,
    )
    n_agents = len(budgets)
    n_th = len(DEFAULT_THRESHOLD_GRID)
    n_res = len(DEFAULT_RESIDUAL_GRID)

    # --- 4. Run each policy ---
    print("\n[4/4] Running evaluations")
    print(f"  n_episodes per policy: {args.n_episodes}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_results = {}

    # 4a. Linear-propensity baseline
    print("\n  -- LINEAR-PROPENSITY BASELINE --")
    picker = make_linear_baseline_picker(n_agents, n_th, n_res)
    res_linear = rollout(env, picker, args.n_episodes, args.slot_size)
    all_results["linear_baseline"] = res_linear

    # 4b. Uniform random baseline
    print("  -- UNIFORM-RANDOM BASELINE --")
    picker = make_uniform_picker(n_agents, n_th, n_res)
    res_uniform = rollout(env, picker, args.n_episodes, args.slot_size)
    all_results["uniform_baseline"] = res_uniform

    # 4c. Trained policy (if models present)
    trained_present = all(
        (model_dir / f"context_ac_seed_{args.seed}_agent_{adv}.pt").exists()
        for adv in ADV_IDS
    )
    if trained_present:
        print("  -- TRAINED ACTOR-CRITIC --")
        agents = []
        for adv in ADV_IDS:
            a = StrategicActorCritic(
                input_dim=4, hidden_dim=128, n_threshold=n_th, n_residual=n_res
            ).to(DEVICE)
            a.load_state_dict(torch.load(
                model_dir / f"context_ac_seed_{args.seed}_agent_{adv}.pt",
                map_location=DEVICE, weights_only=True,
            ))
            a.eval()
            agents.append(a)
        picker = make_trained_picker(agents)
        res_trained = rollout(env, picker, args.n_episodes, args.slot_size)
        all_results["trained_ac"] = res_trained
    else:
        print(f"  Trained models not found at {model_dir} — skipping")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'Policy':<22} {'Mean clicks':>14} {'Std':>10} {'Mean cost':>12} "
          f"{'Mean util%':>12}")
    print("=" * 78)

    summary_rows = []
    for name, res in all_results.items():
        clicks = np.array([r["total_clicks"] for r in res])
        cost = np.array([r["total_cost"] for r in res])
        util = np.array([np.mean(r["utilization_pct"]) for r in res])
        print(f"{name:<22} {clicks.mean():>14.1f} {clicks.std():>10.1f} "
              f"{cost.mean():>12.1f} {util.mean():>12.1f}")
        summary_rows.append({
            "policy": name,
            "mean_clicks": clicks.mean(),
            "std_clicks": clicks.std(),
            "mean_cost": cost.mean(),
            "mean_util_pct": util.mean(),
            "n_episodes": len(res),
        })

    print("=" * 78)

    # Improvement over linear baseline
    if "trained_ac" in all_results:
        linear_mean = np.mean([r["total_clicks"] for r in all_results["linear_baseline"]])
        trained_mean = np.mean([r["total_clicks"] for r in all_results["trained_ac"]])
        improvement_pct = (trained_mean - linear_mean) / max(1, linear_mean) * 100
        print(f"\nTrained AC vs linear baseline: {improvement_pct:+.2f}%")
        if improvement_pct > 5:
            print("  -> Clear improvement.")
        elif improvement_pct > 0:
            print("  -> Small positive; check significance with more seeds.")
        else:
            print("  -> No improvement. Investigate training.")

    # Save CSV
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()