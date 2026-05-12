from utils.data_loader import load_ipinyou_logs
from utils.pctr import fit_pctr_model, add_pctr
from simulator.environment import RTBEnvironment
from simulator.agents import FixedBidAgent, LinearPctrAgent

DATA_DIR = r"D:/Research Methodology/DAAWBC/dynamic_ad_allocation/data/ipinyou"

# 1) load data
df = load_ipinyou_logs(DATA_DIR)

# tiny dataset: use 75% train, 25% test (but with 4 rows, it's just 3/1)
m = max(1, int(0.75 * len(df)))
train_df = df.iloc[:m].reset_index(drop=True)
test_df = df.iloc[m:].reset_index(drop=True)

# 2) train pCTR model and add pCTR to test data
model, use_cols = fit_pctr_model(train_df)
test_df = add_pctr(test_df, model, use_cols)

print("Test data with pCTR:")
print(test_df)

# 3) build impressions list for environment
impressions = [
    {"pctr": float(row.pctr), "market_price": float(row.market_price)}
    for row in test_df.itertuples(index=False)
]

# if test set is empty (just in case), fall back to train set
if not impressions:
    impressions = [
        {"pctr": float(row.pctr), "market_price": float(row.market_price)}
        for row in train_df.assign(pctr=0.1).itertuples(index=False)
    ]

# 4) create environment
avg_market = test_df["market_price"].mean() if len(test_df) else train_df["market_price"].mean()
total_budget = max(100.0, avg_market * 4)

env_fixed = RTBEnvironment(impressions, total_budget=total_budget, time_slots=4, pctr_threshold=0.0)
env_linear = RTBEnvironment(impressions, total_budget=total_budget, time_slots=4, pctr_threshold=0.0)

fixed_agent = FixedBidAgent(bid=max(1.0, avg_market))
linear_agent = LinearPctrAgent(k=avg_market * 500.0, cap=avg_market * 2.0)


def run_env(env, agent, impressions):
    state = env.reset()
    clicks = 0
    i = 0
    done = False

    while not done:
        imp = impressions[i] if i < len(impressions) else None
        bid = agent.act(state, imp)
        state, reward, done, info = env.step(bid)
        clicks += int(reward > 0)
        i += 1

    return {
        "clicks": clicks,
        "cost": round(env.cost, 2),
        "budget_left": round(env.remaining_budget, 2),
    }


res_fixed = run_env(env_fixed, fixed_agent, impressions)
res_linear = run_env(env_linear, linear_agent, impressions)

print("\nFixedBidAgent result:", res_fixed)
print("LinearPctrAgent result:", res_linear)
