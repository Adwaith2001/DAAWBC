"""
src5/context_propensity.py
==========================

Empirically-fitted context-propensity model.

Replaces the role pCTR played in src4. In src5, the bid value signal is
derived from this propensity score, NOT from pCTR.

Methodological contract:
  - Fit on weekday {3,4} only.
  - Frozen after fit; never updated during RL training.
  - Used to score impressions for bidding; the agent never sees pCTR.

Expanded feature set (decided after AUC gate: 0.6364 vs 0.6229 for basic):
    mp_tier, slotvisibility, slotformat, device_type,
    hour, weekday, slot_w, slot_h
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Pre-committed feature set — frozen before any RL training run.
FEATURES: Sequence[str] = (
    "mp_tier",
    "slotvisibility",
    "slotformat",
    "device_type",
    "hour",
    "weekday",
    "slot_w",
    "slot_h",
)


class ContextPropensityModel:
    """Logistic regression fit on weekday {3,4} contextual features."""

    def __init__(self, n_mp_tiers: int = 5):
        self.n_mp_tiers = n_mp_tiers
        self.model: LogisticRegression | None = None
        self.dummy_columns: list[str] | None = None
        self.train_auc: float | None = None
        self.test_auc: float | None = None
        self.mean_propensity: float | None = None

    # ------------------------------------------------------------------
    def _add_mp_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        if "mp_tier" not in df.columns:
            df = df.copy()
            df["mp_tier"] = pd.qcut(
                df["market_price"],
                self.n_mp_tiers,
                labels=False,
                duplicates="drop",
            )
        return df

    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = [f for f in FEATURES if f in df.columns]
        if len(feats) < len(FEATURES):
            missing = set(FEATURES) - set(feats)
            raise ValueError(f"Missing features in dataframe: {missing}")
        X = pd.get_dummies(df[list(feats)].astype("category"), drop_first=False)
        # Lock column order so train/test alignment is preserved.
        if self.dummy_columns is None:
            self.dummy_columns = X.columns.tolist()
        else:
            X = X.reindex(columns=self.dummy_columns, fill_value=0)
        return X

    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        train_weekdays: Sequence[int] = (3, 4),
        eval_weekday: int = 5,
        verbose: bool = True,
    ) -> "ContextPropensityModel":
        """Fit on `train_weekdays`, evaluate held-out on `eval_weekday`."""
        df = self._add_mp_tier(df)

        train_mask = df["weekday"].isin(train_weekdays).values
        eval_mask = df["weekday"].eq(eval_weekday).values

        if train_mask.sum() == 0:
            raise ValueError(f"No training rows for weekdays {train_weekdays}")
        if eval_mask.sum() == 0:
            raise ValueError(f"No eval rows for weekday {eval_weekday}")

        X = self._build_X(df)
        y = df["click"].values.astype(int)

        if verbose:
            print(f"  Fitting on {train_mask.sum():,} rows (weekday {list(train_weekdays)})")
            print(f"  Held-out on {eval_mask.sum():,} rows (weekday {eval_weekday})")

        self.model = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )
        self.model.fit(X.iloc[train_mask].values, y[train_mask])

        p_train = self.model.predict_proba(X.iloc[train_mask].values)[:, 1]
        p_eval = self.model.predict_proba(X.iloc[eval_mask].values)[:, 1]

        self.train_auc = float(roc_auc_score(y[train_mask], p_train))
        self.test_auc = float(roc_auc_score(y[eval_mask], p_eval))
        self.mean_propensity = float(p_train.mean())

        if verbose:
            print(f"  Train AUC:  {self.train_auc:.4f}")
            print(f"  Held-out AUC: {self.test_auc:.4f}")
            print(f"  Mean propensity (train): {self.mean_propensity:.4f}")

        return self

    # ------------------------------------------------------------------
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized propensity scoring — returns array of shape (n,)."""
        if self.model is None:
            raise RuntimeError("Model not fit yet")
        df = self._add_mp_tier(df)
        X = self._build_X(df)
        return self.model.predict_proba(X.values)[:, 1]

    def score_row(self, row: pd.Series) -> float:
        """Single-row scoring — for compatibility with row-by-row env loops."""
        df = row.to_frame().T
        return float(self.score(df)[0])

    # ------------------------------------------------------------------
    def attach_to_dataframe(
        self,
        df: pd.DataFrame,
        column_name: str = "propensity",
    ) -> pd.DataFrame:
        """Compute propensity for every row and attach as a new column.

        This is the RECOMMENDED workflow — compute once at env load,
        then look up by row index during stepping. Avoids per-step ML calls.
        """
        df = df.copy()
        df[column_name] = self.score(df)
        return df


# ----------------------------------------------------------------------
# Convenience: load file, fit model, attach propensity column, return df
# ----------------------------------------------------------------------
def fit_and_attach_propensity(
    data_path: str | Path,
    train_weekdays: Sequence[int] = (3, 4),
    eval_weekday: int = 5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, ContextPropensityModel]:
    """One-call pipeline: read file -> fit model -> attach propensity."""
    if verbose:
        print(f"Loading: {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    if verbose:
        print(f"  Rows: {len(df):,}")
        print(f"  CTR : {df['click'].mean()*100:.2f}%")

    model = ContextPropensityModel()
    model.fit(df, train_weekdays=train_weekdays, eval_weekday=eval_weekday, verbose=verbose)
    df = model.attach_to_dataframe(df)
    return df, model


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Standalone smoke check.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_2/shared_auction_log_v4_dense.txt")
    args = p.parse_args()

    df, m = fit_and_attach_propensity(args.data)
    print()
    print("Propensity distribution stats:")
    print(df["propensity"].describe())
