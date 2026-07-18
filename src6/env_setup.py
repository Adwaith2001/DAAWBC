"""
src6/env_setup.py
=================

Replicates the env-construction recipe from src5/train_context_ac.py
and src5/evaluate_context.py main() functions.

REPLICATES — does not modify. src5 stays bit-for-bit identical.
src6 imports the same src5 classes/functions that src5's own main() uses
(ContextPropensityModel, ContextRTBEnvironment); we just call them in
the same order, packaged as reusable functions for src6 MAPPO scripts.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

# Imports from src5 — same as what src5's own scripts use. No modifications required.
from src5.context_propensity import ContextPropensityModel
from src5.simulator.context_environment import (
    ContextRTBEnvironment,
    DEFAULT_THRESHOLD_GRID,
    DEFAULT_RESIDUAL_GRID,
    ADV_IDS,
)

DEFAULT_DATA_PATH = "data_2/shared_auction_log_v4_dense.txt"


def build_train_env(
    data_path: str = DEFAULT_DATA_PATH,
    budgets=None,
    episode_rows: int = 125_000,
    click_reward_scale=None,
    verbose: bool = True,
):
    """
    Mirror of `[Step 1/3] + [Step 2/3]` env setup in
    src5/train_context_ac.py main(). Returns (env, info_dict).

    Steps:
      1. Read CSV (tab-separated).
      2. Fit ContextPropensityModel on weekday {3,4}.
      3. Attach propensity column to full df.
      4. Filter to weekday {3,4} → training df.
      5. Construct ContextRTBEnvironment on training df.
    """
    if budgets is None:
        budgets = [50000.0] * 5

    if verbose:
        print("\n[Step 1/3] Fitting context-propensity model")
        print("-" * 60)

    df = pd.read_csv(data_path, sep="\t")
    model = ContextPropensityModel()
    model.fit(df, train_weekdays=(3, 4), eval_weekday=5, verbose=verbose)

    df_with_p = model.attach_to_dataframe(df)
    df_train = df_with_p[df_with_p["weekday"].isin([3, 4])].reset_index(drop=True)
    if verbose:
        print(f"  Training-data rows (weekday 3,4): {len(df_train):,}")

    if verbose:
        print("\n[Step 2/3] Building env (src5.simulator.context_environment)")
        print("-" * 60)

    env = ContextRTBEnvironment(
        df=df_train,
        budgets=budgets,
        episode_rows=episode_rows,
        click_reward_scale=click_reward_scale,
        verbose=verbose,
    )

    return env, {
        "train_rows": len(df_train),
        "n_threshold": len(DEFAULT_THRESHOLD_GRID),
        "n_residual": len(DEFAULT_RESIDUAL_GRID),
        "propensity_model": model,  # for sanity-check use only
    }


def build_eval_env(
    data_path: str = DEFAULT_DATA_PATH,
    budgets=None,
    episode_rows: int = 125_000,
    verbose: bool = True,
):
    """
    Mirror of the env setup in src5/evaluate_context.py main().
    Returns (env, n_threshold, n_residual).

    Steps:
      1. Read CSV.
      2. Fit ContextPropensityModel (re-fit deterministically — same data, same weekdays).
      3. Attach propensity column.
      4. Filter to weekday {5} → eval df (held out).
      5. Construct ContextRTBEnvironment on eval df.
    """
    if budgets is None:
        budgets = [50000.0] * 5

    if verbose:
        print("\n[Eval Step 1/2] Refitting context-propensity model")
        print("-" * 60)

    df = pd.read_csv(data_path, sep="\t")
    model = ContextPropensityModel()
    model.fit(df, train_weekdays=(3, 4), eval_weekday=5, verbose=verbose)

    df_with_p = model.attach_to_dataframe(df)
    df_eval = df_with_p[df_with_p["weekday"].eq(5)].reset_index(drop=True)

    if verbose:
        print(f"\n[Eval Step 2/2] Eval set (weekday 5): {len(df_eval):,} rows, "
              f"{int(df_eval['click'].sum()):,} clicks "
              f"(CTR {df_eval['click'].mean()*100:.2f}%)")
        print("-" * 60)

    env = ContextRTBEnvironment(
        df=df_eval,
        budgets=budgets,
        episode_rows=episode_rows,
        verbose=verbose,
    )

    n_th = len(DEFAULT_THRESHOLD_GRID)
    n_res = len(DEFAULT_RESIDUAL_GRID)
    return env, n_th, n_res
