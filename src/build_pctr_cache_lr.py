import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils.data_loader import load_ipinyou_logs

def main():
    ROOT = Path(__file__).resolve().parents[1]  # project root (…/dynamic_ad_allocation)
    IN_FILE   = ROOT / "data" / "ipinyou" / "sample_log.txt"
    OUT_FILE  = ROOT / "data" / "ipinyou" / "sample_log_with_pctr.txt"
    MODEL_OUT = ROOT / "data" / "ipinyou" / "pctr_model.joblib"

    df = load_ipinyou_logs(str(IN_FILE))

    X = df[["weekday", "hour", "slot_w", "slot_h", "siteid"]].copy()
    y = df["click"].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["siteid"]),
            ("num", "passthrough", ["weekday", "hour", "slot_w", "slot_h"]),
        ]
    )

    clf = LogisticRegression(
        solver="saga",
        max_iter=1000,
        n_jobs=-1,
        verbose=0,
        class_weight="balanced",
        tol=1e-3
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    df["pctr"] = pipe.predict_proba(X)[:, 1]

    df.to_csv(OUT_FILE, sep="\t", index=False)
    joblib.dump(pipe, MODEL_OUT)

    print(f"✅ Saved: {OUT_FILE}")
    print(f"✅ Saved: {MODEL_OUT}")

if __name__ == "__main__":
    main()
