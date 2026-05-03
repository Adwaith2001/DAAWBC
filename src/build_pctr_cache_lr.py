import joblib
import shutil
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

# ============================================================
# PROJECT ROOT (for copying files to data/ipinyou/)
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATA_DIR = ROOT / "data" / "ipinyou"


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

    # Project data directory for this advertiser
    PROJECT_ADV_DIR = PROJECT_DATA_DIR / advertiser_id
    PROJECT_ADV_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_OUT_FILE = PROJECT_ADV_DIR / "final_sample_log_with_pctr.txt"

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
        max_iter=2000,       # increased to reduce convergence warnings
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
    # Save outputs to dataset folder
    # ------------------------------------------------------------
    df.to_csv(OUT_FILE, sep="\t", index=False)
    joblib.dump(pipe, MODEL_OUT)
    print(f"✅ Saved dataset with pCTR: {OUT_FILE}")
    print(f"✅ Saved model: {MODEL_OUT}")

    # ------------------------------------------------------------
    # Copy to project data/ipinyou/<advertiser_id>/
    # ------------------------------------------------------------
    shutil.copy(OUT_FILE, PROJECT_OUT_FILE)
    print(f"✅ Copied to project: {PROJECT_OUT_FILE}")


def main():
    """
    Train advertiser-specific pCTR models for multiple advertisers.
    """

    ADVERTISERS = ["1458", "2259", "2821", "2997", "3358"]

    for adv in ADVERTISERS:
        train_pctr_for_advertiser(adv)

    print("\n🎉 All advertiser pCTR models generated successfully")


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()