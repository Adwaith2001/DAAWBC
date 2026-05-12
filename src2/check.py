import pandas as pd
from pathlib import Path

BASE = Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output")

for adv in ["3427", "3476"]:
    path = BASE / adv / "final_sample_log.txt"
    if path.exists():
        df = pd.read_csv(path, sep="\t", nrows=5)
        print(f"{adv} columns: {df.columns.tolist()}")
    else:
        print(f"{adv}: file not found")

print()

for adv in ["1458", "2259", "3386", "2997", "3358"]:
    path = BASE / adv / "enhanced" / "pctr_model_enhanced.joblib"
    print(f"{adv} model: {'EXISTS' if path.exists() else 'MISSING'}")