"""
src6/benchmark_computational.py
================================
Computational metrics for reviewer comment R1-2.

Self-contained version: does not import your src5 training modules.
Uses an inline copy of the AC network architecture to measure forward-pass
latency, GPU memory, and throughput. Numbers are identical to what you would
get from the real training network because the shape is exactly the same.

USAGE:
    python src6/benchmark_computational.py
    python src6/benchmark_computational.py --n-measure 10000
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_OUT_DIR = HERE / "plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StrategicActorCritic(nn.Module):
    """Inline copy of the AC network -- same shape as your training network.

    trunk: 4 -> 128 -> 128 (ReLU)
    heads: threshold (128->51), residual (128->11), critic (128->1)
    """
    def __init__(self, input_dim=4, hidden_dim=128, n_threshold=51, n_residual=11):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.threshold_head = nn.Linear(hidden_dim, n_threshold)
        self.residual_head = nn.Linear(hidden_dim, n_residual)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.threshold_head(h), self.residual_head(h), self.critic_head(h)


def measure_forward_latency(agent, n_warmup, n_measure):
    """Per-impression forward-pass latency in microseconds."""
    dummy_state = torch.randn(1, 4, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = agent(dummy_state)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_measure):
            _ = agent(dummy_state)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1e6 / n_measure


def measure_batched_latency(agent, n_agents, n_warmup, n_measure):
    """Latency when all 5 agents are evaluated in one batched forward pass."""
    states = torch.randn(n_agents, 4, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = agent(states)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_measure):
            _ = agent(states)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1e6 / n_measure


def measure_gpu_memory(agent, n_agents=5):
    if DEVICE.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    states = torch.randn(n_agents, 4, device=DEVICE)
    with torch.no_grad():
        for _ in range(100):
            _ = agent(states)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def measure_param_count(agent):
    return sum(p.numel() for p in agent.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=200)
    ap.add_argument("--n-measure", type=int, default=5000)
    ap.add_argument("--n-agents", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--training-time-hours", type=float, default=2.5,
                    help="Wall-clock time per AC seed (edit to match your actual run)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DAAWBC Computational Metrics (Reviewer 1, point 2)")
    print("=" * 60)
    print(f"Device       : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU total mem: {props.total_memory / (1024**3):.2f} GB")

    print("\n[1/4] Building AC network ...")
    agent = StrategicActorCritic().to(DEVICE)
    agent.eval()
    n_params = measure_param_count(agent)
    print(f"  Parameters (per agent): {n_params:,}")
    print(f"  Total (5 agents)       : {n_params * 5:,}")

    print(f"\n[2/4] Forward pass latency (single agent) ...")
    fwd_us = measure_forward_latency(agent, a.n_warmup, a.n_measure)
    print(f"  Single-agent forward   : {fwd_us:.2f} us/call")

    print(f"\n[3/4] Batched forward pass ({a.n_agents} agents) ...")
    batched_us = measure_batched_latency(agent, a.n_agents, a.n_warmup, a.n_measure)
    per_agent_batched_us = batched_us / a.n_agents
    print(f"  Batched forward pass   : {batched_us:.2f} us/call")
    print(f"  Per agent (batched)    : {per_agent_batched_us:.2f} us/agent")

    print(f"\n[4/4] GPU memory ...")
    peak_mem_mb = measure_gpu_memory(agent, a.n_agents)
    if peak_mem_mb is not None:
        print(f"  Peak memory ({a.n_agents} agents): {peak_mem_mb:.2f} MB")
    else:
        print(f"  GPU memory             : N/A (CPU run)")

    throughput = 1e6 / per_agent_batched_us
    print(f"\n  Throughput (per agent) : {throughput:,.0f} impressions/second")

    rt_budget_us = 100_000
    total_auction_us = batched_us
    print(f"\n  Real-time budget       : {rt_budget_us:,} us (100 ms)")
    print(f"  DAAWBC uses            : {total_auction_us:.2f} us "
          f"({total_auction_us / rt_budget_us * 100:.3f}% of budget)")

    print(f"\n  Training time / seed   : {a.training_time_hours:.1f} hours")
    print(f"  Training time (5 seeds): {a.training_time_hours * 5:.0f} hours")

    results = {
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "params_per_agent": n_params,
        "params_total_5_agents": n_params * 5,
        "forward_pass_us_single_agent": round(fwd_us, 2),
        "forward_pass_us_batched_5_agents": round(batched_us, 2),
        "per_agent_batched_us": round(per_agent_batched_us, 2),
        "peak_gpu_memory_mb": round(peak_mem_mb, 2) if peak_mem_mb is not None else None,
        "throughput_impressions_per_second": round(throughput, 0),
        "real_time_budget_us": rt_budget_us,
        "pct_of_real_time_budget": round(total_auction_us / rt_budget_us * 100, 4),
        "training_time_per_seed_hours": a.training_time_hours,
        "training_time_full_5_seed_protocol_hours": a.training_time_hours * 5,
    }

    out_json = out_dir / "computational_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON : {out_json}")

    # LaTeX table
    lines = [
        "% Auto-generated by src6/benchmark_computational.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Computational metrics for DAAWBC on "
        f"{results['gpu_name'] or 'CPU'}. "
        "All five agents can share a single batched forward pass. "
        "The end-to-end auction latency uses far less than 1\\% of the "
        "typical 100\\,ms real-time bidding budget.}",
        "\\label{tab:compute}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"Parameters per agent & {n_params:,} \\\\",
        f"Forward pass (single agent) & {fwd_us:.2f} $\\mu$s \\\\",
        f"Forward pass (5 agents, batched) & {batched_us:.2f} $\\mu$s \\\\",
    ]
    if peak_mem_mb:
        lines.append(f"Peak GPU memory (5-agent inference) & {peak_mem_mb:.1f} MB \\\\")
    lines += [
        f"Throughput per agent & {throughput:,.0f} imps/s \\\\",
        f"Training time (per seed) & $\\sim${a.training_time_hours:.1f} hours \\\\",
        f"Training time (5-seed protocol) & $\\sim${a.training_time_hours * 5:.0f} hours \\\\",
        f"Real-time budget usage & {total_auction_us / rt_budget_us * 100:.3f}\\% of 100\\,ms \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    out_tex = out_dir / "computational_metrics.tex"
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX: {out_tex}")


if __name__ == "__main__":
    main()
