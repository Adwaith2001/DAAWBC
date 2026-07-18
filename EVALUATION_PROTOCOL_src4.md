# Pre-Committed Evaluation Protocol — src4 (time-slot arm)

**Status: LOCKED. Written before the 30-episode × 3-seed run and before
any of its results were observed.** Mirrors src3's EVALUATION_PROTOCOL.md.
Same honesty contract: metrics and conditions are fixed here so they
cannot be selected after seeing which favour the learned policy. Any
post-hoc change is a disclosed deviation, not a silent substitution.

## 1. Locked configuration (fixed BEFORE the run)

These are methodological choices, committed now so they are not
result-chasing knobs:

- `slot_size = 5000`  (≈3.75 expected clicks/slot; chosen for gradient
  SNR, not because it produced a good number — it has not been run)
- `slots_per_episode = 25`  (preserves ~10 independent data chunks)
- `UPDATE_EVERY = 5`  (the shipped default of 25 gave 1 update/episode —
  a defect; 5 is required for any gradient flow)
- budgets = 25 × src3 base = [450K,350K,200K,500K,200K]  (per-impression
  spend power held ≈ constant vs src3; same for both auction modes to
  keep the isolated-vs-market_baseline comparison clean)
- inner env = byte-identical src3 merged env (verified via diff);
  composition guarantees the controlled comparison

## 2. Primary metric

**Sampled clicks per episode**, averaged over evaluation episodes and
seeds. Identical field-standard metric to src3. Reported with a **formal
uniform-random (thr,res)-per-slot baseline in the same table** — because
the random policy already scores ~31 clicks/ep on the dense stream, the
honest question is not "RL > 0" but "RL > random, with entropy movement
and rising CTR-on-wins."

## 3. Secondary metrics (alongside, never instead)

CTR-on-wins, eCPC, total cost — every condition, every method, whichever
way they cut. Same rule as src3 §2.

## 4. Mechanism evidence

Per-head entropy trajectories (H_thr, H_res), impression-weighted action-
occupancy histograms (labelled as such — they count impressions-under-bin,
not decisions-under-bin), budget-utilisation and skip-rate trajectories.

## 5. Conditions (fixed)

| # | config | isolates |
|---|--------|----------|
| E | `--auction-mode market_baseline` (else default) | does slot aggregation break the v2 market-floor Nash collapse? |
| F | defaults (isolated + sampled + both) | first clean time-slot learnability result |
| G | `--policy-mode residual_only --reward-mode expected_click` | most-favourable time-slot config |
| H | slot_size sweep {2000, 5000}, else = F | reward-density-per-decision as the variable (the time-slot analog of src3's --max-steps data-horizon test) |

## 6. Pre-committed interpretation (both outcomes publishable)

- **If learned policy beats the random baseline with entropy movement and
  rising CTR-on-wins** → time-slot aggregation recovers learnability that
  per-impression src3 never achieved. This is the positive arm of the
  thesis. Frame as *learnability recovered under aggregation*, NOT
  *state-of-the-art clicks*.
- **If learned policy ≈ random baseline (no entropy movement / flat
  CTR-on-wins)** → even with a dense aggregated reward, on-policy MARL in
  this shared-auction setting does not learn. Combined with src3's
  characterised failures this is the strong negative result: the barrier
  is not solely sparsity but the multi-agent shared-auction structure.
  Equally publishable; pre-committed as such here.

Neither outcome is discarded. The learnability framing in the positive
case is committed NOW precisely so it cannot be called post-hoc spin.

## 7. Scope exclusions

- MAPPO time-slot: out of scope (future work); AC suffices for the
  learnability claim.
- src4↔Wu-2018 fidelity: src4 is multi-agent; Wu is single-advertiser.
  src4 takes only Wu's slot-aggregation concept, in our architecture.
  This is stated so a reviewer cannot call it a misrepresentation.

---

*Locked prior to the src4 30-ep × 3-seed run. Mirrors src3
EVALUATION_PROTOCOL.md. Post-hoc changes are disclosed deviations.*
