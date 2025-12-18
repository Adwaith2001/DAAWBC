import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

OUT_FILE = OUT_DIR / "baseline_comparison_multiseed.csv"

results = [
    ("FixedBid", 5.40, 1.85, 298.0),
    ("LinearPCTR", 2.20, 0.75, 299.0),
    ("REINFORCE", 6.40, 4.67, 299.0),
]

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "mean_clicks", "std_clicks", "mean_cost"])
    for r in results:
        writer.writerow(r)

print("✅ Saved:", OUT_FILE)
