import torch
from torch.distributions import Categorical

from utils.data_loader import load_ipinyou_logs
from utils.pctr import fit_pctr_model, add_pctr
from simulator.environment import RTBEnvironment
from simulator.agents import FixedBidAgent, LinearPctrAgent
from policy_network import PolicyNetwork

DATA_DIR = r"D:/Research Methodology/DAAWBC/dynamic_ad_allocation/data/ipinyou"
TIME_SLOTS = 12
PCTR_THRESHOLD = 0.0
BID_GRID = torch.linspace(0.0, 150.0, steps=21)


def build_env():
    df = load_ipinyou_logs(DATA_DIR)
    m = max(1, int(0.75 * len(df)))
    train_df = df.iloc[:m].reset_index(drop=True)
    test_df = df.iloc[m:].reset_index(drop=True)
    if len(test_df) == 0:
        test_df = train_df.copy().reset_index(drop=True)

    model, use_cols = fit_pctr_model(train_df)
    test_df = add_pctr(test_df, model, use_cols)

    impressions = [
        {"pctr": float(row.pctr), "market_price": float(row.market_price)}
        for row in test_df.itertuples(index=False)
    ]

    avg_market = test_df["market_price"].mean() if len(test_df) else 50.0
    total_budget = max(200.0, avg_market * 4.0)

    env = RTBEnvironment(
        impressions=impressions,
        total_budget=total_budget,
        time_slots=TIME_SLOTS,
        pctr_threshold=PCTR_THRESHOLD,
    )
    return env, impressions, avg_market


def run_fixed_and_linear(episodes: int = 10):
    env, impressions, avg_market = build_env()

    fixed = FixedBidAgent(bid=max(1.0, avg_market))
    linear = LinearPctrAgent(k=avg_market * 500.0, cap=avg_market * 2.0)

    def run_agent(agent):
        total_clicks, total_cost = 0, 0.0
        for _ in range(episodes):
            e, imps, _ = build_env()
            s = e.reset()
            i = 0
            done = False
            while not done:
                imp = imps[i] if i < len(imps) else None
                bid = agent.act(s, imp)
                s, r, done, info = e.step(bid)
                total_clicks += int(info.get("click", 0))
                i += 1
            total_cost += e.cost
        return total_clicks, total_cost

    fixed_clicks, fixed_cost = run_agent(fixed)
    linear_clicks, linear_cost = run_agent(linear)

    return (fixed_clicks, fixed_cost), (linear_clicks, linear_cost)


def run_rl_policy(episodes: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env, impressions, avg_market = build_env()
    input_dim = 4
    num_actions = len(BID_GRID)

    policy = PolicyNetwork(input_dim, num_actions).to(device)
    ckpt = torch.load("policy_reinforce.pt", map_location=device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    total_clicks, total_cost = 0, 0.0

    for _ in range(episodes):
        env, impressions, _ = build_env()
        state = env.reset().to(device)
        done = False
        i = 0

        while not done:
            with torch.no_grad():
                logits = policy(state)
                dist = Categorical(logits=logits.squeeze(0))
                action_idx = dist.sample()
                bid = BID_GRID[action_idx.item()]

            imp = impressions[i] if i < len(impressions) else None
            state, reward, done, info = env.step(float(bid))
            state = state.to(device)
            total_clicks += int(info.get("click", 0))
            i += 1

        total_cost += env.cost

    return total_clicks, total_cost


def main():
    episodes = 20  # eval episodes

    print("Evaluating baselines...")
    (fixed_clicks, fixed_cost), (linear_clicks, linear_cost) = run_fixed_and_linear(
        episodes=episodes
    )

    print("\nEvaluating RL policy...")
    rl_clicks, rl_cost = run_rl_policy(episodes=episodes)

    print("\n==== Evaluation over", episodes, "episodes ====")
    print(f"FixedBid    -> clicks={fixed_clicks},  cost={fixed_cost:.2f}")
    print(f"LinearPCTR  -> clicks={linear_clicks}, cost={linear_cost:.2f}")
    print(f"RL Policy   -> clicks={rl_clicks},    cost={rl_cost:.2f}")


if __name__ == "__main__":
    main()
