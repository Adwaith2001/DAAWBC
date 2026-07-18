# src5 — Context-Propensity Bidding (R9f — wrapper + algorithmic stabilization)

Standalone multi-agent actor-critic using **empirically fitted context-
propensity** as the bid signal. **Time-slot wrapper enabled** (R9b).
**GAE + Huber loss + critic weight 0.1** for variance reduction and
gradient stability (R9f).

This is the launch-ready build for the second wrapped attempt, after
the first per-impression smoke run showed entropy stagnation that
Gemini (external consultation) traced to critic-gradient dominance over
the shared trunk.

---

## Review chain summary

| Round | Decision | Status |
|-------|----------|--------|
| R9 | Structural pass + 3 required fixes (click_reward derivation, lambda confound doc, deterministic eval) | ✓ |
| R9b | Time-slot wrapper port required (per-impression infeasible) | ✓ |
| R9c | Wrapper port greenlit | ✓ |
| R9d | Sent actual build for review (artifact-drift fix) | ✓ |
| R9e | Structural pass on wrapped; recommended `update_every=2` | ✓ |
| **R9f** | **GAE + Huber loss + critic weight 0.1 (this build)** | **pending A re-review** |

R9f rationale: after the first wrapped run (NOT performed yet — entropy
stagnation was observed in the per-impression smoke run that user ran
without the wrapper), the algorithmic instability concerns are real and
addressable with standard policy-gradient stabilization. These are not
methodology changes — they don't alter:
- the bid signal (still context-propensity)
- the state shape (still 4-dim)
- the action space (still 51 threshold × 11 residual)
- the wrapper architecture (still slot_size=5000)
- the lambda controller (still per-agent adaptive, still disclosed as confound)
- the reward formula (still click_reward_scale * click - lambda * cost)
- the pre-committed reading framework

They only change how the optimizer digests the same reward signal:
1. **GAE(γ=0.99, τ=0.95)** replaces full Monte Carlo returns. Standard
   variance-reduction technique in policy gradient methods (Schulman+ 2016).
2. **Huber (smooth_l1) loss** replaces MSE for the critic. Standard
   stability technique under large reward magnitudes (Mnih+ 2015 DQN).
3. **Critic weight 0.1** (down from 0.5). Prevents the critic's
   high-variance gradient from dominating the shared trunk and
   destabilizing the actor heads.

What we EXPLICITLY rejected from Gemini's analysis:
- State expansion to include lambda (would change the src4 contrast and
  reframe a disclosed confound as a state input — out of scope without
  A re-review)
- Threshold grid narrowing 0.6 → 0.25 (Gemini's reasoning assumes
  imbalanced RTB; the densified dataset has mean propensity 0.4773)
- Budget cliff change (too marginal to bundle)

---

## Pre-committed configuration (locked at R9f launch)

```
episode_rows       125,000
slot_size          5,000      -> 25 slots/episode  (R9b)
update_every       2          -> ~12 updates/ep    (R9e §2)
total updates      ~1,000 over 80 episodes
budgets            50,000 × 5 agents (symmetric)
lr                 3e-4
gamma              0.99
entropy_beta       0.01
clip_grad          10.0
click_reward_scale None  -> env derives 580.79  (R9 §1)
advantages         GAE(gamma=0.99, tau=0.95)  (R9f)
critic_loss        smooth_l1 (Huber)  (R9f)
critic_weight      0.1  (R9f, was 0.5)
```

---

## Pipeline

### 0. Pre-requisites
- Dense dataset: `data_2/shared_auction_log_v4_dense.txt`
- Conda env `rtb` active.
- Replace any existing `src5/` with this build.

### 1. Smoke test (~5 minutes)
```cmd
cd /d "D:\Research Methodology\DAAWBC\dynamic_ad_allocation"
python -m src5.smoke_test
```

Expect all 5 PASS, including `[5b/5]` slot-mode gradients finite.
The R9f changes should make this MORE likely to pass, not less.

### 2. Train, single seed, 80 episodes (~2-3 hours on RTX 3050)
```cmd
python -m src5.train_context_ac --episodes 80 --seeds 0
```

All defaults are pre-committed. The trainer prints full config in the
launch banner including the R9f changes — verify before walking away.

### 3. AT EPISODE 20 — DIAGNOSTIC GATE (R9b §4)

Paste to A:
```
Episode 1 / 10 / 20:
  H_thr per agent       [5 numbers each]
  H_res per agent       [5 numbers each]
  max_bin_thr per agent [5 numbers each]
  max_bin_res per agent [5 numbers each]
  clicks per agent      [5 numbers each]
```

A judges against src4 non-learning signature.

### 4. Evaluate on held-out (weekday 5)
```cmd
python -m src5.evaluate_context --seed 0 --n-episodes 10
```

---

## Pre-committed reading framework (unchanged from R9b §3)

- ~10-15% — clean beat
- ~50% — match
- ~35-40% — non-learning signature

R9f algorithmic fixes might shift this distribution mildly toward the
positive end (lower probability of non-learning, higher probability
of match-or-beat), but the wrapped architecture's fundamental gradient-
disconnection property persists. Match-or-non-learning is still the
modal outcome.
