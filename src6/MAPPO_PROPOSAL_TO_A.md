# src6 MAPPO — Pre-Commitment Memo from B to A

**Re:** Proposing a parallel src6/ contribution to the paper: MAPPO
(Yu et al., 2022) applied to the same R9g-validated environment.
src5 result (+3.77% mean across 4 seeds, p<0.01, 4/4 positive) is
unchanged and remains the headline. src6 is upside-only — if it beats
src5 with statistical significance, both go in the paper. If not, src6
is dropped without affecting src5.

Sending for sign-off BEFORE launching multi-seed runs.

---

## 1. Why this isn't a continuation of R9 iteration

R9 reached its natural conclusion at R9g with the click-only reward
formulation. We honored the no-R9h commitment. src5 R9g is now a
frozen, saved checkpoint with documented +3.77% lift, statistically
significant across 4 seeds.

src6 is a SEPARATE, parallel contribution — a different algorithm
applied to the same environment, not a tuning iteration on src5. The
framing for the paper is:

> "We evaluate two reinforcement-learning algorithms on the densified
> iPinYou auction environment: (1) independent actor-critic [src5,
> R9g], and (2) MAPPO with centralized training, decentralized
> execution [src6, this work]."

This is standard comparative ML methodology, not p-hacking. The
question being answered is different from R9's question.

## 2. What src6 actually changes

**Environment:** unchanged. src6 imports
`src5.simulator.context_environment.ContextRTBEnvironment` directly. No
code duplication. The auction, propensity model, reward formula
(click-only, R9g), action grids, bid cap, and lambda controller are all
bit-for-bit identical.

**Policy:**
- ACTORS — per-agent, decentralized, IDENTICAL architecture to src5
  except switching trunk activation from ReLU to Tanh per the standard
  PPO recipe (Engstrom et al., 2020 "Implementation Matters in Deep
  Policy Gradients").
- CRITIC — centralized, takes concatenated global state
  (state_dim × n_agents = 20-dim input) and outputs per-agent value
  estimates (5-dim output). This is the standard MAPPO formulation
  from Yu et al. (2022).

**Optimization:**
- Vanilla PG → PPO clipped objective (Schulman et al., 2017).
- Single-update → multi-epoch updates on rollout (4 epochs).
- Importance sampling ratio with clip_eps=0.2 (standard).
- KL early-stop at target_kl=0.02 (safety net).
- Per-agent advantage normalization within rollout.

**Reused unchanged:** GAE(γ=0.99, τ=0.95), Huber critic loss,
clip_grad=10.0, entropy_beta=0.03 (matches R9g winning config), lr=3e-4.

## 3. Pre-committed reading framework (LOCKED before any src6 runs)

Same primary metric as R9: mean clicks per eval episode, weekday-5
held-out data, n=10 episodes per seed, paired comparison against linear
baseline. Same deterministic stride starts as src5 eval.

| Outcome | Definition | Action |
|---|---|---|
| **Clean beat over src5** | MAPPO mean ≥ src5 mean + 1.5% AND statistically significant (paired t-test, p<0.05) across ≥4 seeds | Both src5 and src6 go in paper. MAPPO becomes the primary result. |
| **Match with src5** | MAPPO mean within ±1% of src5 (i.e., 950-961 clicks) across ≥4 seeds | Both go in paper. Framed as "different algorithm, same headroom on this signal." |
| **Loss vs src5** | MAPPO mean < src5 - 2% OR fewer than 3/5 seeds positive | src6 is dropped. src5 R9g remains the only result. |
| **Catastrophic** | MAPPO eval crashes, NaN, gradient explosion, or significantly under uniform random | src6 is dropped. Document the failure in an appendix if relevant. |

**Distribution of expected outcomes (pre-committed estimate):**
- ~25% clean beat
- ~50% match (most likely — propensity signal at AUC=0.64 has limited headroom regardless of algorithm)
- ~20% loss
- ~5% catastrophic

These are honest priors. MAPPO doesn't magically extract more signal;
its theoretical advantage over independent AC is in multi-agent
coordination and update stability. src5 already has tight cross-seed
variance (0.30%), suggesting there's not much stability headroom left.
Most likely outcome: similar result with different algorithm.

## 4. Stop rules (HARD)

1. **Per-seed:** if smoke test fails or first 20 episodes show
   non-finite gradients, NaN values, or KL > 0.10, stop and diagnose.
2. **Multi-seed:** maximum 5 seeds. No re-running with different
   hyperparameters mid-experiment.
3. **Total time:** if by 12:30 IST tomorrow MAPPO hasn't shown
   statistical significance over src5, drop and start writing.
4. **No R10 iteration on MAPPO.** Whatever the first multi-seed run
   produces is the result. If MAPPO matches src5, that's the finding;
   we don't tune further.

## 5. What changes in the paper

If **clean beat:**
- §6 (Methods) gains a MAPPO subsection.
- §7 (Results) gets a comparison table: linear baseline / uniform random / src5 AC / src6 MAPPO.
- §9 (Conclusion) gains: "centralized training reduces multi-agent learning instability sufficiently to extract additional X% headroom over independent AC."

If **match:**
- §6 gains a MAPPO subsection.
- §7 reports both AC and MAPPO with the equivalence framing.
- §9 gains: "Both independent and centralized-critic methods achieve
  similar lift on this signal, suggesting the propensity-signal AUC is
  the binding constraint, not the RL algorithm."

If **loss or catastrophic:**
- src6 is not mentioned in the paper at all.
- src5 R9g remains the sole result, exactly as the current draft has it.
- No appendix needed (no negative result publication for a side
  experiment that didn't run cleanly).

## 6. Methodology audit trail

- Saved checkpoint: `saved_points/src5_r9g_baseline_2026-05-23.zip`
  contains src5 + outputs preserved as the fallback result.
- src6 lives in `src6/` parallel to `src5/`.
- src6 imports from src5 (read-only dependency). src5 cannot be broken
  by src6 work.
- All MAPPO hyperparameters are committed in
  `src6/train_mappo.py` DEFAULTS dict and `src6/SETUP.md` before any
  run.
- This memo is committed to disk before any multi-seed run.

## 7. Specific request to A

1. Sign off on this src6 plan, OR
2. Reject and document the reason (e.g. "not a clean enough comparison
   given the algorithm change") — in which case we drop the
   pre-experiment and just write up src5 R9g.
3. (optional) Flag any concern about the success criteria in §3.

Default per A's R9e §6 discipline: silence on a parallel-contribution
proposal = NOT approved. Expecting explicit yes/no.

— instance B
