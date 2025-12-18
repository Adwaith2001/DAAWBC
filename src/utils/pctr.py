import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load

# Minimal feature config (extend as your real dataset allows)
NUM_FEATS = ["market_price"]
CAT_FEATS = ["weekday", "hour", "siteid", "advertiser"]

def _prep_X(df: pd.DataFrame):
    X = df.copy()
    for c in NUM_FEATS:
        if c not in X.columns:
            X[c] = 0.0
    for c in CAT_FEATS:
        if c in X.columns:
            X[c] = X[c].astype(str)
    use_cols = [c for c in NUM_FEATS + CAT_FEATS if c in X.columns]
    return X[use_cols], use_cols

def fit_pctr_model(train_df: pd.DataFrame, model_path: str = None):
    if "click" not in train_df.columns:
        raise ValueError("Training DataFrame must contain 'click' column.")
    X, use_cols = _prep_X(train_df)
    y = train_df["click"].astype(int).values

    pre = ColumnTransformer([
        ("num", "passthrough", [c for c in NUM_FEATS if c in use_cols]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
               [c for c in CAT_FEATS if c in use_cols]),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    model = Pipeline([("prep", pre), ("clf", clf)])
    model.fit(X, y)

    if model_path:
        dump({"model": model, "use_cols": use_cols}, model_path)
    return model, use_cols

def load_pctr_model(model_path: str):
    obj = load(model_path)
    return obj["model"], obj["use_cols"]

def add_pctr(df: pd.DataFrame, model, use_cols):
    X = df.copy()
    for c in NUM_FEATS:
        if c not in X.columns:
            X[c] = 0.0
    for c in CAT_FEATS:
        if c in X.columns:
            X[c] = X[c].astype(str)
    X = X[[c for c in use_cols if c in X.columns]]
    p = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["pctr"] = np.clip(p, 1e-5, 0.5)
    return out
