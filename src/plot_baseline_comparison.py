import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = ROOT / "results" / "baseline_comparison_multiseed.csv"

PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

OUT_FILE = PLOT_DIR / "baseline_comparison_multiseed.png"

df = pd.read_csv(CSV_FILE)

plt.figure(figsize=(7, 5))
plt.bar(
    df["method"],
    df["mean_clicks"],
    yerr=df["std_clicks"],
    capsize=6
)

plt.ylabel("Clicks (mean ± std)")
plt.title("Multi-Seed Baseline Comparison")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300)
plt.close()

print("✅ Plot saved:", OUT_FILE)
