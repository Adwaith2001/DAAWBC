import joblib
from pathlib import Path

# ============================================================
# SCIKIT-LEARN IMPORTS
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================================================
# PROJECT-SPECIFIC DATA LOADER
# ============================================================

from utils.data_loader import load_ipinyou_logs


def train_pctr_for_advertiser(advertiser_id: str):
    """
    Train a pCTR model for a single advertiser and
    augment its dataset with predicted CTR values.
    """

    print(f"\n🚀 Processing advertiser {advertiser_id}")

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    BASE_DIR = Path(
        "D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output"
    )

    IN_FILE = BASE_DIR / advertiser_id / "final_sample_log.txt"
    OUT_FILE = BASE_DIR / advertiser_id / "final_sample_log_with_pctr.txt"
    MODEL_OUT = BASE_DIR / advertiser_id / "pctr_model.joblib"

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    df = load_ipinyou_logs(str(IN_FILE))

    # ------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------
    X = df[["weekday", "hour", "slot_w", "slot_h", "siteid"]].copy()
    y = df["click"].astype(int)

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["siteid"],
            ),
            (
                "num",
                "passthrough",
                ["weekday", "hour", "slot_w", "slot_h"],
            ),
        ]
    )

    # ------------------------------------------------------------
    # Logistic Regression model
    # ------------------------------------------------------------
    clf = LogisticRegression(
        solver="saga",
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        tol=1e-3,
        verbose=0,
    )

    # ------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------
    pipe = Pipeline(
        [
            ("pre", preprocessor),
            ("clf", clf),
        ]
    )

    pipe.fit(X, y)

    # ------------------------------------------------------------
    # Predict pCTR
    # ------------------------------------------------------------
    df["pctr"] = pipe.predict_proba(X)[:, 1]

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    df.to_csv(OUT_FILE, sep="\t", index=False)
    joblib.dump(pipe, MODEL_OUT)

    print(f"✅ Saved dataset with pCTR: {OUT_FILE}")
    print(f"✅ Saved model: {MODEL_OUT}")


def main():
    """
    Train advertiser-specific pCTR models for multiple advertisers.
    """

    ADVERTISERS = ["1458", "2259", "2821"]

    for adv in ADVERTISERS:
        train_pctr_for_advertiser(adv)

    print("\n🎉 All advertiser pCTR models generated successfully")


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
