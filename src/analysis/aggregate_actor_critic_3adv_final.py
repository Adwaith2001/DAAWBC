import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "outputs" / "final_experiments"
OUTPUT_DIR = ROOT / "outputs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5

# ============================================================
# LOAD ALL SEEDS
# ============================================================

dfs = []

for seed in range(N_SEEDS):
    file = INPUT_DIR / f"actor_critic_3adv_seed_{seed}.csv"
    print(f"Loading {file}")
    df = pd.read_csv(file)
    df["seed"] = seed
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# ============================================================
# AUTO-DETECT AGENT COLUMNS
# ============================================================

click_cols = sorted([c for c in data.columns if c.startswith("click")])
cost_cols  = sorted([c for c in data.columns if c.startswith("cost")])

print("Detected click columns:", click_cols)
print("Detected cost columns :", cost_cols)

# ============================================================
# AGGREGATE MEAN ± STD
# ============================================================

agg_rows = []

for ep in sorted(data["episode"].unique()):
    ep_df = data[data["episode"] == ep]
    row = {"episode": ep}

    for i, col in enumerate(click_cols):
        row[f"clicks_a{i}_mean"] = ep_df[col].mean()
        row[f"clicks_a{i}_std"]  = ep_df[col].std()

    for i, col in enumerate(cost_cols):
        row[f"cost_a{i}_mean"] = ep_df[col].mean()
        row[f"cost_a{i}_std"]  = ep_df[col].std()

    agg_rows.append(row)

agg_df = pd.DataFrame(agg_rows)

# ============================================================
# SAVE
# ============================================================

out_file = OUTPUT_DIR / "actor_critic_3adv_aggregated.csv"
agg_df.to_csv(out_file, index=False)

print("✅ Aggregation complete")
print(f"📄 Saved: {out_file}")
