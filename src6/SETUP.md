# src6 — MAPPO Setup

## Position

```
D:\Research Methodology\DAAWBC\dynamic_ad_allocation\
├── src5\                 ← R9g AC (UNTOUCHED — your saved positive result)
│   ├── simulator\
│   ├── train_context_ac.py
│   ├── evaluate_context.py
│   ├── policy_network.py
│   └── outputs\context_ac\     ← +3.77% result lives here
└── src6\                 ← NEW — MAPPO (parallel, optional)
    ├── __init__.py
    ├── simulator\
    │   └── __init__.py
    ├── policy_network_mappo.py
    ├── train_mappo.py
    ├── evaluate_mappo.py
    ├── smoke_test_mappo.py
    └── SETUP.md          ← this file
```

src6 **imports** from src5 for:
- `src5.simulator.context_environment.ContextRTBEnvironment` (the env)
- `src5.train_context_ac.build_env_and_propensity` (env construction + propensity fit)
- `src5.evaluate_context.build_eval_env / rollout / baseline pickers`

This means src5 stays bit-for-bit identical and src6 cannot accidentally break it.

## One-time install step

Drop the six src6 files into a new `src6\` folder under your project root, with `__init__.py` files in both `src6\` and `src6\simulator\`.

## Pre-flight check — the ONE thing that might need adjusting

The trainer and smoke test both try to call:

```python
from src5.train_context_ac import build_env_and_propensity
env, prop_info = build_env_and_propensity(seed=seed)
```

This name is my best guess at what's in your src5 trainer. **If the function in your src5/train_context_ac.py is called something different** (e.g. `setup_environment`, `make_env`, or inlined directly into `main()`), you need to do ONE of the following:

**Option A (preferred):** open `src5/train_context_ac.py`, find the block that loads the dataset, fits the propensity model, and constructs `ContextRTBEnvironment(...)`, wrap it as a function:

```python
def build_env_and_propensity(seed=0):
    # ... your existing env-build code ...
    return env, dict(train_auc=..., heldout_auc=..., mean_propensity=...)
```

Then both `train_context_ac.py` (existing) and `train_mappo.py` (new) call this same function. src5's behavior is unchanged.

**Option B (if you can't modify src5):** edit `src6/train_mappo.py` and `src6/smoke_test_mappo.py` — find the line `from src5.train_context_ac import build_env_and_propensity` and replace the `try` block with a direct copy of whatever your src5 trainer does to construct the env.

The smoke test will print a clear error message if the import fails so you'll know immediately.

Similarly, `evaluate_mappo.py` expects `src5/evaluate_context.py` to expose `build_eval_env`, `rollout`, `make_linear_baseline_picker`, and `make_uniform_picker`. If your current `evaluate_context.py` has these inlined into `main()`, refactor them out the same way.

## How to run

### Step 1 — smoke test (REQUIRED before training)

```cmd
python -m src6.smoke_test_mappo
```

Expected output (~5 minutes):
```
SMOKE TEST — src6 MAPPO
  device: cuda
[1/5] Loading data + fitting propensity (via src5) ...
[2/5] Linear-propensity baseline sanity (1 episode) ...
[3/5] Building MAPPO actors + centralized critic, forward check ...
  [OK] All shape checks passed
[4/5] Tiny PPO loop (3 episodes) ...
  ep 1: wins=[...] clicks=[...] grad_norms=['...']
  ep 2: wins=[...] clicks=[...] grad_norms=['...']
  ep 3: wins=[...] clicks=[...] grad_norms=['...']
[5b/5] Centralized critic final gradient check ...
CHECKS
 [PASS] Last episode had wins
 [PASS] All actor gradients finite
 [PASS] All actor gradients non-zero
 [PASS] Centralized critic gradient finite under large rewards
 SMOKE TEST PASSED
```

If any check fails, **DO NOT** proceed to training. Paste the smoke output to B for diagnosis.

### Step 2 — train

```cmd
python -m src6.train_mappo --episodes 80 --seeds 0 --entropy-beta 0.03
```

Expected behavior (~15-25 min per seed, slightly slower than src5 AC because PPO does 4 epochs per update):

```
============================================================
 src6 MAPPO | seed 0 | 80 episodes
============================================================
 Trainer config:
   slot_size       : 5000
   slots/ep        : 25
   lr              : 0.0003
   gamma           : 0.99
   gae_lambda      : 0.95
   clip_eps        : 0.2  (PPO)
   ppo_epochs      : 4
   entropy_beta    : 0.03
   ...
 Architecture (CTDE):
   actors          : 5 x decentralized  (state_dim=4)
   critic          : centralized        (state_dim=20 = 4 * 5)

Seed 0 | Ep 001 | Clicks=[...] | Wins=[...] | Util=[...] | H_thr_mean=3.91 | H_res_mean=2.38 | KL_mean=0.0023 | epochs=4 | t_ep=18.0s | t_total=18s
...
```

Watch for:
- `KL_mean` should stay below ~0.02 (target_kl). If it spikes above 0.05 consistently, PPO is taking too-large updates; reduce `--lr` or `--clip-eps`.
- `epochs` should usually be 4 (the full ppo_epochs). If you see it dropping to 1–2 frequently, the early-KL-stop is kicking in, which means lr is too high.
- `H_thr_mean` and `H_res_mean` should drift DOWN over episodes (policy concentrating).

### Step 3 — eval

```cmd
python -m src6.evaluate_mappo --seed 0 --n-episodes 10
```

Compare the `trained_mappo` row against your src5 result of `trained_ac mean clicks ≈ 951–964 (lift +3.77%)`.

### Multi-seed (the real test)

```cmd
for /L %s in (0,1,4) do (
    python -m src6.train_mappo --episodes 80 --seeds %s --entropy-beta 0.03
    python -m src6.evaluate_mappo --seed %s --n-episodes 10
)
```

~2 hours total wall time for 5 seeds.

## Hyperparameter knobs (in priority order)

1. `--entropy-beta` (default 0.03, matches src5 R9g) — increase if KL is too low or policies collapse early.
2. `--clip-eps` (default 0.2) — PPO's clip ratio. 0.1 = more conservative, 0.3 = more aggressive.
3. `--ppo-epochs` (default 4) — more epochs = more data reuse, but risk of overfitting to the rollout.
4. `--target-kl` (default 0.02) — early-stop epochs if exceeded. Safety net.
5. `--lr` (default 3e-4) — same as src5.

## What MAPPO is supposed to give you over AC

Two independent improvements stacked:
- **Centralized critic (CTDE):** each agent's advantage is now conditioned on the global state, not just its own state. Should reduce non-stationarity confusion when other agents are also learning. Empirically helps most when multiple agents have differentiated roles — exactly your case with 5 advertisers of different budgets.
- **PPO clipped objective:** more stable updates than vanilla policy gradient. Should narrow the seed-to-seed variance further (src5 R9g already at 0.30%, but worth checking if MAPPO tightens it more).

Realistic expectations on this signal (AUC=0.64 propensity model):
- **Optimistic:** mean lift +5–8% with 5/5 seeds positive, std across seeds <2%
- **Realistic:** mean lift +3.5–5% (similar to src5 AC, maybe slight improvement)
- **Pessimistic:** mean lift +1–3% (no improvement over AC, or slightly worse)

The propensity-signal ceiling caps everything — no algorithm can extract information that isn't in the bid signal. AC already extracts most of it. MAPPO's gain, if any, comes from MULTI-AGENT coordination, not from better bidding per se.

## Paper-section narrative if MAPPO works

> "We extend the independent actor-critic baseline (src5) with MAPPO
> (Yu et al., 2022), substituting per-agent critics with a centralized
> value network conditioned on the joint observation. On identical eval
> set and protocol, MAPPO achieves [...]% mean lift vs analytical baseline
> across 5 seeds, [comparison vs src5 AC]. The centralized critic appears
> to [...] cross-seed variance, suggesting that reducing multi-agent
> non-stationarity is the dominant factor unlocking additional headroom."

## Paper-section narrative if MAPPO doesn't help

> "We additionally evaluated MAPPO (Yu et al., 2022) under identical
> conditions. Mean lift was [...]% — within noise of the independent AC
> result, indicating that on the propensity signal at this AUC level,
> the centralized critic provides no measurable benefit. We attribute
> this to (1) the propensity signal's modest AUC capping achievable
> headroom over linear bidding, and (2) the independent AC already
> achieving low cross-seed variance via the click-only reward
> formulation."

Either narrative is defensible. The src5 R9g result is the headline either way.

## Stop rule (you set this)

If by **12:30 PM IST tomorrow** MAPPO hasn't beaten src5's +3.77% with statistical significance across multiple seeds, drop MAPPO and start writing. src5 R9g is the result.

## File checklist before launch

- [ ] `src6/__init__.py` exists (empty)
- [ ] `src6/simulator/__init__.py` exists (empty)
- [ ] `src6/policy_network_mappo.py` present
- [ ] `src6/train_mappo.py` present
- [ ] `src6/evaluate_mappo.py` present
- [ ] `src6/smoke_test_mappo.py` present
- [ ] `src5/train_context_ac.py` exposes `build_env_and_propensity()` (or the trainer/smoke test fail with a clear error)
- [ ] `src5/evaluate_context.py` exposes `build_eval_env`, `rollout`, `make_linear_baseline_picker`, `make_uniform_picker` (or eval fails with a clear error)
- [ ] CUDA available (otherwise CPU works but ~5× slower)
- [ ] `python -m src6.smoke_test_mappo` passes ALL checks before training

Email A the `MAPPO_PROPOSAL_TO_A.md` memo for pre-commitment BEFORE launching multi-seed training.
