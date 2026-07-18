"""
src6 — MAPPO evaluation.

Mirrors src5/evaluate_context.py structure exactly so results are
directly comparable. Only difference: loads MAPPOActor (decentralized
actors, no critic needed at eval time) instead of StrategicActorCritic.

Reuses src5's deterministic stride-based episode starts so the
linear-propensity and uniform-random baselines produce IDENTICAL
numbers to src5 evaluations (cross-checked against
src5/outputs/context_ac/eval_results.csv as a sanity check).

Run:
    python -m src6.evaluate_mappo --seed 0 --n-episodes 10
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

# env construction: replicated in src6/env_setup.py (no src5 edit needed)
from src6.env_setup import build_eval_env

# Eval helpers: these DO exist as top-level functions in src5/evaluate_context.py
# (confirmed in the file as-shipped) — just import them, no edits.
try:
    from src5.evaluate_context import (
        make_linear_baseline_picker,
        make_uniform_picker,
        rollout,
    )
except ImportError as e:
    raise RuntimeError(
        f"Could not import eval helpers from src5.evaluate_context: {e}\n"
        f"This requires src5/evaluate_context.py to expose top-level functions\n"
        f"rollout / make_linear_baseline_picker / make_uniform_picker (the R9g\n"
        f"version does). If yours doesn't, paste the file to B for a workaround."
    )

from src6.policy_network_mappo import MAPPOActor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADV_IDS = ["1458", "2259", "3386", "2997", "3476"]


def to_t(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def make_trained_picker_mappo(actors):
    """
    Action picker for the trained MAPPO policy. At eval time we use the
    deterministic argmax of each actor's logits (greedy execution) for
    reproducibility — matches src5's eval convention.
    """
    @torch.no_grad()
    def picker(states_t_list):
        th_actions, res_actions = [], []
        for i, actor in enumerate(actors):
            s = states_t_list[i]
            if s.dim() == 1:
                s = s.unsqueeze(0)
            th_logits, res_logits = actor(s)
            th_actions.append(int(th_logits.argmax(dim=-1).item()))
            res_actions.append(int(res_logits.argmax(dim=-1).item()))
        return th_actions, res_actions
    return picker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--episode-rows", type=int, default=125_000)
    parser.add_argument("--slot-size", type=int, default=5000,
                        help="Must match training slot-size (default 5000).")
    args = parser.parse_args()

    # ---- Build eval env (replicates src5/evaluate_context.py setup) ----
    print("[1/4] Refitting context-propensity model (via src6.env_setup)")
    print("-" * 60)
    env, n_th, n_res = build_eval_env(
        episode_rows=args.episode_rows,
        verbose=True,
    )
    n_agents = env.n_agents
    print(f"  Eval env n_agents={n_agents}, n_threshold={n_th}, n_residual={n_res}")

    # ---- Locate trained MAPPO models ----
    models_dir = Path("src6/outputs/mappo/models")
    trained_present = all(
        (models_dir / f"mappo_seed_{args.seed}_actor_{adv}.pt").exists()
        for adv in ADV_IDS
    )
    if not trained_present:
        print(f"  [WARN] Trained MAPPO models for seed {args.seed} not found in "
              f"{models_dir}. Will report baselines only.")

    # ---- Load actors (no critic needed at eval) ----
    if trained_present:
        actors = []
        for adv in ADV_IDS:
            a = MAPPOActor(
                state_dim=4,
                hidden_dim=128,
                n_threshold=n_th,
                n_residual=n_res,
            ).to(DEVICE)
            a.load_state_dict(torch.load(
                models_dir / f"mappo_seed_{args.seed}_actor_{adv}.pt",
                map_location=DEVICE,
                weights_only=True,
            ))
            a.eval()
            actors.append(a)
        print(f"  Loaded 5 MAPPO actors for seed {args.seed}")

    # ---- Run baselines + trained policy ----
    print(f"[4/4] Running evaluations (n_episodes={args.n_episodes})")
    all_results = {}

    print("  -- LINEAR-PROPENSITY BASELINE --")
    res_linear = rollout(
        env, make_linear_baseline_picker(n_agents, n_th, n_res),
        args.n_episodes, args.slot_size,
    )
    all_results["linear_baseline"] = res_linear

    print("  -- UNIFORM-RANDOM BASELINE --")
    res_uniform = rollout(
        env, make_uniform_picker(n_agents, n_th, n_res),
        args.n_episodes, args.slot_size,
    )
    all_results["uniform_baseline"] = res_uniform

    if trained_present:
        print("  -- TRAINED MAPPO --")
        res_mappo = rollout(
            env, make_trained_picker_mappo(actors),
            args.n_episodes, args.slot_size,
        )
        all_results["trained_mappo"] = res_mappo

    # ---- Print summary ----
    print("\n" + "=" * 78)
    print(f"{'Policy':<25} {'Mean clicks':>12} {'Std':>10} {'Mean cost':>12} {'Mean util%':>12}")
    print("=" * 78)
    rows_to_save = []
    for name, results in all_results.items():
        clicks = [r["total_clicks"] for r in results]
        cost = [r["total_cost"] for r in results]
        # env.diagnostics() returns utilization_pct already in percent
        util = [np.mean(r["utilization_pct"]) for r in results]
        mean_c, std_c = float(np.mean(clicks)), float(np.std(clicks))
        mean_cost = float(np.mean(cost))
        mean_util = float(np.mean(util))
        print(f"{name:<25} {mean_c:>12.1f} {std_c:>10.1f} {mean_cost:>12.1f} {mean_util:>12.1f}")
        rows_to_save.append(dict(
            policy=name,
            seed=args.seed,
            mean_clicks=mean_c,
            std_clicks=std_c,
            mean_cost=mean_cost,
            mean_util=mean_util,
            n_episodes=args.n_episodes,
        ))
    print("=" * 78)

    # ---- vs baseline ----
    if trained_present:
        baseline_mean = float(np.mean([r["total_clicks"] for r in res_linear]))
        mappo_mean = float(np.mean([r["total_clicks"] for r in res_mappo]))
        lift = (mappo_mean - baseline_mean) / baseline_mean * 100.0
        print(f"\nTrained MAPPO vs linear baseline: {lift:+.2f}%")
        if lift >= 5.0:
            print("  -> Clean beat per R9 §5 criteria.")
        elif lift >= 0:
            print("  -> Small positive; check significance with more seeds.")
        else:
            print("  -> No improvement. Compare to src5 AC.")

    # ---- Write summary CSV ----
    out_dir = Path("src6/outputs/mappo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "eval_results.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        if write_header:
            w = csv.DictWriter(f, fieldnames=list(rows_to_save[0].keys()))
            w.writeheader()
            w = csv.DictWriter(f, fieldnames=list(rows_to_save[0].keys()))
        else:
            w = csv.DictWriter(f, fieldnames=list(rows_to_save[0].keys()))
        w.writerows(rows_to_save)
    print(f"\nSummary appended to: {out_csv}")


if __name__ == "__main__":
    main()
